#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Select the best sample from evaluation results based on multi-objective optimization.

Usage Examples:
    # 1. Standard Global Search (Default) - No constraints, saves to details.json
    python scripts/select_best_sample.py -i results/eval.jsonl
    
    # 2. Specify Output Folder (Recommended)
    python scripts/select_best_sample.py -i results/eval.jsonl --copy-pdb-to best_pdbs
    
    # 3. Enable Constraints (filter by physical limits)
    python scripts/select_best_sample.py -i results/eval.jsonl --use-constraints
    
    # 4. Per-Entry Search (instead of global)
    python scripts/select_best_sample.py -i results/eval.jsonl --per-entry

Metrics and their optimization directions:
- AAR (%)↑ - higher is better (Amino Acid Recovery)
- C-RMS↓ - lower is better (C_RMSD)
- L-RMS↓ - lower is better (L_RMSD) 
- dG↓ - lower is better (binding energy)
- IMP (%)↑ - higher is better (ddG <= 0, improvement rate)
- C_in (%)↓ - lower is better (Clash_inner)
- C_out (%)↓ - lower is better (Clash_outer)
- Seq.↑ - higher is better (Sequence Diversity)
- Str.↑ - higher is better (Struct Diversity)

Strategy: Weighted normalized score with optional constraints
"""

import argparse
import json
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import shutil
from pathlib import Path
import os


@dataclass
class MetricConfig:
    """Configuration for a single metric."""
    name: str  # Key name in JSONL
    direction: str  # 'max' (higher is better) or 'min' (lower is better)
    weight: float = 1.0  # Weight for scoring
    constraint_min: Optional[float] = None  # Minimum acceptable value
    constraint_max: Optional[float] = None  # Maximum acceptable value


# Default metric configurations based on the paper table
DEFAULT_METRICS = [
    MetricConfig("AAR", "max", weight=1.0),          # AAR(%)↑
    MetricConfig("C_RMSD(CA)", "min", weight=1.0),   # C-RMS↓
    MetricConfig("L_RMSD(CA)", "min", weight=1.2),   # L-RMS↓
    MetricConfig("ddG", "min", weight=0.25),            # ΔG↓
    MetricConfig("ddG <= 0 (IMP)", "max", weight=0.25),  # IMP(%)↑
    MetricConfig("Clash_inner", "min", weight=0.8),  # C_in(%)↓
    MetricConfig("Clash_outer", "min", weight=0.8),  # C_out(%)↓
]

# Physical constraints for valid samples
PHYSICAL_CONSTRAINTS = {
    "Clash_inner": {"max": 0.02},    # Low internal clash
    "Clash_outer": {"max": 0.02},    # Low external clash
    "C_RMSD(CA)": {"max": 10.0},     # Reasonable C-RMSD
    "L_RMSD(CA)": {"max": 5.0},      # Reasonable L-RMSD
}


def normalize_values(values: List[float], direction: str) -> np.ndarray:
    """
    Normalize values to [0, 1] range.
    For 'max' direction: higher original values → higher normalized values
    For 'min' direction: lower original values → higher normalized values
    """
    arr = np.array(values, dtype=float)
    
    # Handle NaN and inf
    valid_mask = np.isfinite(arr)
    if not valid_mask.any():
        return np.zeros_like(arr)
    
    valid_vals = arr[valid_mask]
    vmin, vmax = valid_vals.min(), valid_vals.max()
    
    if vmax - vmin < 1e-10:
        # All values are the same
        return np.ones_like(arr) * 0.5
    
    normalized = (arr - vmin) / (vmax - vmin)
    
    if direction == "min":
        normalized = 1 - normalized
    
    # Set invalid values to 0
    normalized[~valid_mask] = 0
    
    return normalized


def check_constraints(sample_idx: int, data: Dict, constraints: Dict) -> Tuple[bool, List[str]]:
    """
    Check if a sample satisfies all physical constraints.
    Returns (passed, list of violation messages).
    """
    violations = []
    
    for metric_name, limits in constraints.items():
        if metric_name not in data:
            continue
        
        metric_data = data[metric_name]
        if "individual" not in metric_data:
            continue
        
        individual = metric_data["individual"]
        if sample_idx >= len(individual):
            continue
        
        value = individual[sample_idx]
        
        if "max" in limits and value > limits["max"]:
            violations.append(f"{metric_name}={value:.4f} > {limits['max']}")
        if "min" in limits and value < limits["min"]:
            violations.append(f"{metric_name}={value:.4f} < {limits['min']}")
    
    return len(violations) == 0, violations


def compute_sample_scores(
    data: Dict,
    metrics_config: List[MetricConfig],
    constraints: Optional[Dict] = None,
    apply_constraints: bool = True
) -> Tuple[np.ndarray, List[bool]]:
    """
    Compute weighted normalized scores for all samples.
    
    Returns:
        scores: Array of scores for each sample
        valid_mask: List of booleans indicating if sample passes constraints
    """
    # Determine number of samples
    n_samples = None
    for mc in metrics_config:
        if mc.name in data and "individual" in data[mc.name]:
            n_samples = len(data[mc.name]["individual"])
            break
    
    if n_samples is None:
        return np.array([]), []
    
    scores = np.zeros(n_samples)
    total_weight = 0.0
    valid_mask = [True] * n_samples
    
    # Check constraints
    if apply_constraints and constraints:
        for idx in range(n_samples):
            passed, _ = check_constraints(idx, data, constraints)
            valid_mask[idx] = passed
    
    # Compute weighted scores
    for mc in metrics_config:
        if mc.name not in data:
            continue
        
        metric_data = data[mc.name]
        if "individual" not in metric_data:
            continue
        
        individual = metric_data["individual"]
        
        # Handle boolean metrics (like "ddG <= 0 (IMP)")
        if isinstance(individual[0], bool):
            # For booleans, strictly use 1.0 for True and 0.0 for False
            # Do NOT normalize relative to min/max of the batch
            vals = np.array([1.0 if v else 0.0 for v in individual])
            scores += vals * mc.weight
        else:
            normalized = normalize_values(individual, mc.direction)
            scores += normalized * mc.weight
        
        total_weight += mc.weight
    
    if total_weight > 0:
        scores /= total_weight
    
    # Apply constraint penalty
    for idx in range(n_samples):
        if not valid_mask[idx]:
            scores[idx] = -1  # Invalid samples get negative score
    
    return scores, valid_mask


def select_best_sample(
    data: Dict,
    metrics_config: Optional[List[MetricConfig]] = None,
    constraints: Optional[Dict] = None,
    apply_constraints: bool = True,
    top_k: int = 1
) -> List[Dict]:
    """
    Select the best sample(s) based on weighted normalized score.
    
    Args:
        data: Parsed JSONL entry
        metrics_config: List of MetricConfig, uses DEFAULT_METRICS if None
        constraints: Physical constraints dict, uses PHYSICAL_CONSTRAINTS if None
        apply_constraints: Whether to apply physical constraints
        top_k: Number of top samples to return
    
    Returns:
        List of dicts with sample info: {'idx': int, 'score': float, 'metrics': dict}
    """
    if metrics_config is None:
        metrics_config = DEFAULT_METRICS
    if constraints is None:
        constraints = PHYSICAL_CONSTRAINTS
    
    scores, valid_mask = compute_sample_scores(
        data, metrics_config, constraints, apply_constraints
    )
    
    if len(scores) == 0:
        return []
    
    # Get top-k indices
    sorted_indices = np.argsort(scores)[::-1]  # Descending
    
    results = []
    for idx in sorted_indices[:top_k]:
        idx = int(idx)
        sample_metrics = {}
        
        for mc in metrics_config:
            if mc.name in data and "individual" in data[mc.name]:
                value = data[mc.name]["individual"][idx]
                sample_metrics[mc.name] = value
        
        results.append({
            "idx": idx,
            "score": float(scores[idx]),
            "valid": valid_mask[idx],
            "metrics": sample_metrics
        })
    
    return results


def copy_selected_pdb(
    entry_id: str,
    sample_idx: int,
    input_json_path: str,
    output_dir: str
) -> None:
    """
    Copy the Rosetta PDB file for the selected sample.
    Assumes structure: {input_dir}/candidates/{entry_id}/{sample_idx}_rosetta.pdb
    """
    input_dir = Path(input_json_path).parent
    # Base candidate directory for this entry
    base_cand_dir = input_dir / "candidates" / entry_id
    
    # Target filename
    target_filename = f"{sample_idx}_rosetta.pdb"
    src_path = base_cand_dir / target_filename
    
    if not src_path.exists():
        # Check subdirectories (e.g. antibodies might have HCDR3 subdir)
        found = False
        if base_cand_dir.exists():
            for item in base_cand_dir.iterdir():
                if item.is_dir():
                    potential_path = item / target_filename
                    if potential_path.exists():
                        src_path = potential_path
                        found = True
                        break
        
        if not found:
            print(f"Warning: PDB file not found at {src_path} or immediate subdirectories")
            return
        
    dest_dir = Path(output_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # Sanitize entry_id to handle slashes (e.g. "1fe8_A_H_L/HCDR3" -> "1fe8_A_H_L_HCDR3")
    safe_entry_id = entry_id.replace("/", "_").replace("\\", "_")
    
    dest_path = dest_dir / f"{safe_entry_id}_{sample_idx}_rosetta.pdb"
    
    try:
        shutil.copy2(str(src_path), str(dest_path))
        # print(f"Copied {src_path} to {dest_path}")
    except Exception as e:
        print(f"Error copying {src_path}: {e}")


def process_jsonl(
    input_path: str,
    output_path: Optional[str] = None,
    metrics_config: Optional[List[MetricConfig]] = None,
    constraints: Optional[Dict] = None,
    apply_constraints: bool = True,
    top_k: int = 1,
    verbose: bool = True,
    copy_pdb_to: Optional[str] = None
) -> List[Dict]:
    """
    Process evaluation JSONL file and select best samples for each entry.
    
    Returns:
        List of results for each entry
    """
    if metrics_config is None:
        metrics_config = DEFAULT_METRICS
    if constraints is None:
        constraints = PHYSICAL_CONSTRAINTS
    
    all_results = []
    
    with open(input_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        data = json.loads(line.strip())
        entry_id = data.get("id", "unknown")
        
        best_samples = select_best_sample(
            data, metrics_config, constraints, apply_constraints, top_k
        )
        
        if copy_pdb_to and best_samples:
            for sample in best_samples:
                copy_selected_pdb(
                    entry_id, 
                    sample['idx'], 
                    input_path, 
                    copy_pdb_to
                )
        
        result = {
            "id": entry_id,
            "ref_dG": data.get("ref_dG"),
            "best_samples": best_samples
        }
        all_results.append(result)
        
        if verbose and best_samples:
            best = best_samples[0]
            print(f"\n{'='*60}")
            print(f"Entry: {entry_id}")
            print(f"Best sample idx: {best['idx']} (score: {best['score']:.4f}, valid: {best['valid']})")
            print("Metrics:")
            for metric_name, value in best['metrics'].items():
                direction = next((m.direction for m in metrics_config if m.name == metric_name), "?")
                arrow = "↑" if direction == "max" else "↓"
                if isinstance(value, bool):
                    print(f"  {metric_name}: {value} {arrow}")
                elif isinstance(value, float):
                    print(f"  {metric_name}: {value:.4f} {arrow}")
                else:
                    print(f"  {metric_name}: {value} {arrow}")
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        if verbose:
            print(f"\nResults saved to: {output_path}")
    
    return all_results


def find_global_best_sample(
    input_path: str,
    metrics_config: Optional[List[MetricConfig]] = None,
    constraints: Optional[Dict] = None,
    apply_constraints: bool = True,
    verbose: bool = True,
    copy_pdb_to: Optional[str] = None
) -> Dict:
    """
    Find the single best sample across ALL entries (global optimization).
    Normalizes metrics globally across the entire dataset before scoring.
    
    Returns:
        Dict with global best sample info
    """
    if metrics_config is None:
        metrics_config = DEFAULT_METRICS
    if constraints is None:
        constraints = PHYSICAL_CONSTRAINTS
    
    all_candidates = []
    
    with open(input_path, 'r') as f:
        lines = f.readlines()
    
    # 1. Collect all raw samples
    for line in lines:
        data = json.loads(line.strip())
        entry_id = data.get("id", "unknown")
        
        # Determine number of samples from the first available metric
        n_samples = None
        for mc in metrics_config:
            if mc.name in data and "individual" in data[mc.name]:
                n_samples = len(data[mc.name]["individual"])
                break
        
        if n_samples is None:
            continue
            
        for idx in range(n_samples):
            # Check physical constraints
            is_valid = True
            if apply_constraints and constraints:
                passed, _ = check_constraints(idx, data, constraints)
                is_valid = passed
            
            # Extract metrics
            cand_metrics = {}
            for mc in metrics_config:
                if mc.name in data and "individual" in data[mc.name]:
                    cand_metrics[mc.name] = data[mc.name]["individual"][idx]
            
            all_candidates.append({
                "entry_id": entry_id,
                "sample_idx": idx,
                "metrics": cand_metrics,
                "valid": is_valid,
                "ref_dG": data.get("ref_dG")
            })

    if not all_candidates:
        return {"total_samples": 0, "global_best": None, "top_10": []}

    # 2. Compute global scores with global normalization
    n_total = len(all_candidates)
    final_scores = np.zeros(n_total)
    total_weight = 0.0

    for mc in metrics_config:
        # Extract values for this metric across all candidates
        # Use NaN if metric is missing for a candidate
        values = []
        for cand in all_candidates:
             val = cand["metrics"].get(mc.name, np.nan)
             # Handle booleans
             if isinstance(val, bool):
                 values.append(1.0 if val else 0.0)
             else:
                 values.append(val)
        
        if values:
            is_bool = isinstance(values[0], (bool, np.bool_))
            # Just in case we have mixed types or converted 1.0/0.0, check first element from raw extraction
            # But here values come from boolean check above
            # We need to detect if this metric is boolean based on data
            
            # Check a non-nan value from candidates to determine type
            first_val = next((c["metrics"][mc.name] for c in all_candidates if mc.name in c["metrics"]), None)
            is_bool_metric = isinstance(first_val, bool)
        else:
            is_bool_metric = False

        if is_bool_metric:
             # Strict 1.0/0.0
             normalized = np.array([1.0 if v else 0.0 for v in values])
        else:
             # Normalize globally
             values_arr = np.array(values, dtype=float)
             normalized = normalize_values(values_arr, mc.direction)
        
        final_scores += normalized * mc.weight
        total_weight += mc.weight
    
    if total_weight > 0:
        final_scores /= total_weight
        
    # 3. Apply validation mask and assign scores
    for i, cand in enumerate(all_candidates):
        if not cand["valid"]:
            cand["score"] = -1.0
        else:
            cand["score"] = float(final_scores[i])

    # 4. Sort and select
    all_candidates.sort(key=lambda x: x["score"], reverse=True)
    
    global_best = all_candidates[0] if all_candidates else None
    
    if verbose and global_best:
        print(f"\n{'='*60}")
        print(f"GLOBAL BEST SAMPLE (across all {len(all_candidates)} samples)")
        print(f"{'='*60}")
        print(f"Entry ID: {global_best['entry_id']}")
        print(f"Sample Index: {global_best['sample_idx']}")
        print(f"Score: {global_best['score']:.4f}")
        print(f"Valid: {global_best['valid']}")
        print(f"Reference dG: {global_best['ref_dG']}")
        print("\nMetrics (Raw):")
        for mc in metrics_config:
            if mc.name in global_best['metrics']:
                value = global_best['metrics'][mc.name]
                arrow = "↑" if mc.direction == "max" else "↓"
                if isinstance(value, bool):
                    print(f"  {mc.name}: {value} {arrow}")
                elif isinstance(value, float):
                    print(f"  {mc.name}: {value:.4f} {arrow}")
                else:
                    print(f"  {mc.name}: {value} {arrow}")
        
        print(f"\n{'='*60}")
        print("TOP 10 SAMPLES")
        print(f"{'='*60}")
        for i, cand in enumerate(all_candidates[:10]):
            print(f"{i+1}. {cand['entry_id']} sample#{cand['sample_idx']} "
                  f"score={cand['score']:.4f} valid={cand['valid']}")
        
        if copy_pdb_to:
            print(f"\nCopying top 10 PDBs to {copy_pdb_to}...")
            # Copy top 10
            for cand in all_candidates[:10]:
                copy_selected_pdb(
                    cand['entry_id'],
                    cand['sample_idx'],
                    input_path,
                    copy_pdb_to
                )
    
    return {
        "total_samples": len(all_candidates),
        "global_best": global_best,
        "top_10": all_candidates[:10]
    }


def summarize_results(results: List[Dict], verbose: bool = True) -> Dict:
    """Compute summary statistics across all entries."""
    all_scores = []
    all_valid = []
    all_metrics = defaultdict(list)
    
    for r in results:
        if r["best_samples"]:
            best = r["best_samples"][0]
            all_scores.append(best["score"])
            all_valid.append(best["valid"])
            for k, v in best["metrics"].items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    all_metrics[k].append(v)
                elif isinstance(v, bool):
                    all_metrics[k].append(1.0 if v else 0.0)
    
    summary = {
        "total_entries": len(results),
        "mean_score": float(np.mean(all_scores)) if all_scores else 0,
        "valid_ratio": sum(all_valid) / len(all_valid) if all_valid else 0,
        "metric_means": {k: float(np.mean(v)) for k, v in all_metrics.items()}
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        print(f"Total entries: {summary['total_entries']}")
        print(f"Mean score: {summary['mean_score']:.4f}")
        print(f"Valid sample ratio: {summary['valid_ratio']:.2%}")
        print("\nMean metrics of selected samples:")
        for k, v in summary['metric_means'].items():
            print(f"  {k}: {v:.4f}")
    
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select best samples from peptide evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--input", "-i", type=str, required=True,
        help="Path to input JSONL file (e.g., eval_report_lap_cads.jsonl)"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Path to output JSON file (optional)"
    )
    parser.add_argument(
        "--top-k", "-k", type=int, default=1,
        help="Number of top samples to select per entry (default: 1)"
    )
    parser.add_argument(
        "--use-constraints", action="store_true",
        help="Enable physical constraints filtering (Default: False)"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress verbose output"
    )
    parser.add_argument(
        "--per-entry", action="store_true",
        help="Perform LOCAL per-entry search instead of default GLOBAL search"
    )
    parser.add_argument(
        "--copy-pdb-to", "-c", type=str, default=None,
        help="Directory to copy the best Rosetta PDB files to"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Determine target directory for PDBs (and now the main result JSON)
    target_dir_str = args.copy_pdb_to
    if not target_dir_str:
        # Default: modify input filename/folder
        # e.g. "results/data.jsonl" -> "data_best_samples"
        input_path = Path(args.input)
        target_dir_str = f"{input_path.stem}_best_samples"
        print(f"No output directory specified. Using default: {target_dir_str}")

    # Use strict copy_pdb_to for functions called below
    # We override/augment the args to ensure they use this targeted directory
    # Use strict copy_pdb_to for functions called below
    # We override/augment the args to ensure they use this targeted directory
    real_copy_to = target_dir_str

    if not args.per_entry:
        # Default: Global search
        # constraints=True only if --use-constraints is passed
        result = find_global_best_sample(
            input_path=args.input,
            apply_constraints=args.use_constraints,
            verbose=not args.quiet,
            copy_pdb_to=real_copy_to
        )
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nResults saved to: {args.output}")
            
        final_data = result
    else:
        # Per-entry search (if --per-entry specified)
        results = process_jsonl(
            input_path=args.input,
            output_path=args.output,
            apply_constraints=args.use_constraints,
            top_k=args.top_k,
            verbose=not args.quiet,
            copy_pdb_to=real_copy_to
        )
        summarize_results(results, verbose=not args.quiet)
        final_data = results

    # Always save 'details.json' in the target directory
    if final_data:
        dest_dir = Path(real_copy_to)
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        dest_json_path = dest_dir / "details.json"
        
        try:
            with open(dest_json_path, 'w') as f:
                json.dump(final_data, f, indent=2)
            print(f"Details JSON saved to: {dest_json_path}")
        except Exception as e:
            print(f"Error saving details.json to {dest_json_path}: {e}")
