# -*- coding:utf-8 -*-
"""
Compare diversity between baseline and strategy JSONL files.
Finds entries where strategy improves both diversities, then extracts diverse samples.
"""

import argparse
import json
import random
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

METRICS = [("AAR", "max"), ("C_RMSD(CA)", "min"), ("L_RMSD(CA)", "min"), 
           ("dG", "min"), ("Clash_inner", "min"), ("Clash_outer", "min")]


@dataclass
class Result:
    id: str
    baseline_seq_div: float
    baseline_struct_div: float
    strategy_seq_div: float
    strategy_struct_div: float
    seq_gain: float
    struct_gain: float
    combined_gain: float
    strategy_high_div_samples: list = None
    baseline_low_div_samples: list = None


def load_jsonl(path: str) -> dict:
    data = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                data[entry["id"]] = entry
    return data


def load_results_jsonl(path: str) -> dict:
    data = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                sample = json.loads(line)
                data.setdefault(sample["id"], []).append(sample)
    return data


def get_metric_avg(data: dict) -> dict:
    """Compute dataset averages for metrics."""
    avgs = {}
    for key, agg in METRICS:
        vals = []
        for entry in data.values():
            if key in entry and isinstance(entry[key], dict) and "individual" in entry[key]:
                v = [x for x in entry[key]["individual"] if x and not np.isnan(x)]
                if v:
                    vals.append(max(v) if agg == "max" else min(v))
        avgs[key] = float(np.mean(vals)) if vals else None
    return avgs


def score_sample(sample: dict, eval_entry: dict, avgs: dict) -> tuple:
    """
    Score a sample by how well it meets metric requirements.
    Returns (num_metrics_passed, total_score, sample_dict)
    Higher score = better sample.
    """
    n = sample.get("n", -1)
    if n < 0:
        return (0, -float('inf'), sample)
    
    passed = 0
    score = 0.0
    
    for key, direction in METRICS:
        if key not in eval_entry or "individual" not in eval_entry.get(key, {}):
            continue
        indiv = eval_entry[key]["individual"]
        if n >= len(indiv) or indiv[n] is None or np.isnan(indiv[n]):
            continue
        
        val, avg = indiv[n], avgs.get(key)
        if avg is None or avg == 0:
            continue
        
        # Normalize: positive = better than avg, negative = worse
        if direction == "max":
            diff = (val - avg) / abs(avg)  # Higher is better
            if val >= avg:
                passed += 1
        else:
            diff = (avg - val) / abs(avg)  # Lower is better
            if val <= avg:
                passed += 1
        
        score += diff
    
    return (passed, score, sample)


def filter_samples_relaxed(samples: list, eval_entry: dict, avgs: dict, min_samples: int = 5) -> list:
    """
    Filter samples with progressive relaxation.
    First tries strict filtering, then relaxes to select best available.
    """
    # Score all samples
    scored = [score_sample(s, eval_entry, avgs) for s in samples]
    
    # Sort by (num_passed DESC, score DESC)
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    
    # Check how many pass all metrics
    num_metrics = len(METRICS)
    strict_passed = [s for s in scored if s[0] >= num_metrics - 1]  # Allow 1 metric to fail
    
    if len(strict_passed) >= min_samples:
        return [s[2] for s in strict_passed]
    
    # Relaxed: take top samples by score
    return [s[2] for s in scored[:max(min_samples * 2, 20)]]  # Return more candidates for diversity selection



def select_diverse_subset(samples: list, overall_div: float, k: int = 5, mode: str = "high"):
    """Random select k samples, validate diversity condition."""
    if len(samples) <= k:
        return samples, overall_div, [s.get("n") for s in samples]
    
    from evaluation.diversity import seq_diversity
    seqs = [s.get("gen_seq", "") for s in samples]
    best_idx, best_div = None, None
    
    for _ in range(10):
        idx = random.sample(range(len(samples)), k)
        try:
            div, _ = seq_diversity([seqs[i] for i in idx], n_workers=1)
        except:
            continue
        if mode == "high" and (best_div is None or div > best_div):
            best_idx, best_div = idx, div
        elif mode == "low" and (best_div is None or div < best_div):
            best_idx, best_div = idx, div
    
    final_idx = best_idx or list(range(k))
    return [samples[i] for i in final_idx], best_div or overall_div, [samples[i].get("n") for i in final_idx]


def find_diversity_gains(baseline: dict, strategy: dict, verbose: bool = False) -> tuple:
    """Find entries where strategy improves both diversities."""
    avgs = get_metric_avg(strategy)
    base_avgs = get_metric_avg(baseline)
    results = []
    
    common_ids = set(baseline) & set(strategy)
    
    if verbose:
        print(f"\n=== Dataset Statistics ===")
        print(f"Baseline entries: {len(baseline)}")
        print(f"Strategy entries: {len(strategy)}")
        print(f"Common entries: {len(common_ids)}")
        print(f"\nStrategy metric averages:")
        for k, v in avgs.items():
            if v is not None:
                print(f"  {k}: {v:.4f}")
    
    for eid in common_ids:
        b, s = baseline[eid], strategy[eid]
        b_seq, b_str = b.get("Sequence Diversity", 0), b.get("Struct Diversity", 0)
        s_seq, s_str = s.get("Sequence Diversity", 0), s.get("Struct Diversity", 0)
        
        if s_seq <= b_seq or s_str <= b_str:
            continue
        
        # Check metrics meet average
        ok = True
        for key, direction in METRICS:
            if key not in s or "individual" not in s.get(key, {}):
                continue
            v = [x for x in s[key]["individual"] if x and not np.isnan(x)]
            if not v:
                continue
            val = max(v) if direction == "max" else min(v)
            avg = avgs.get(key)
            if avg and ((direction == "max" and val < avg) or (direction == "min" and val > avg)):
                ok = False
                break
        
        if ok:
            results.append(Result(
                id=eid,
                baseline_seq_div=b_seq,
                baseline_struct_div=b_str,
                strategy_seq_div=s_seq,
                strategy_struct_div=s_str,
                seq_gain=s_seq - b_seq,
                struct_gain=s_str - b_str,
                combined_gain=(s_seq - b_seq) + (s_str - b_str)
            ))
    
    return sorted(results, key=lambda x: x.combined_gain, reverse=True), avgs, base_avgs


def main():
    parser = argparse.ArgumentParser(description="Compare diversity between datasets")
    parser.add_argument("--baseline", "-b", required=True, help="Baseline eval_report.jsonl")
    parser.add_argument("--strategy", "-s", required=True, help="Strategy eval_report.jsonl")
    parser.add_argument("--output", "-o", default="diversity.json", help="Output JSON file")
    parser.add_argument("--top-k", "-k", type=int, default=5, help="Samples per entry")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    baseline_path, strategy_path = Path(args.baseline), Path(args.strategy)
    
    print("=" * 60)
    print("Diversity Comparison Script")
    print("=" * 60)
    print(f"\nInput files:")
    print(f"  Baseline: {args.baseline}")
    print(f"  Strategy: {args.strategy}")
    
    # Load data
    print(f"\nLoading eval reports...")
    baseline = load_jsonl(args.baseline)
    strategy = load_jsonl(args.strategy)
    print(f"  Baseline: {len(baseline)} entries")
    print(f"  Strategy: {len(strategy)} entries")
    
    # Find diversity gains
    print(f"\nFinding entries with BOTH Seq & Struct diversity gains...")
    results, strat_avgs, base_avgs = find_diversity_gains(baseline, strategy, args.verbose)
    
    print(f"\n=== Results ===")
    print(f"Qualifying entries: {len(results)}")
    
    if results:
        print(f"\nTop entries by combined diversity gain:")
        for i, r in enumerate(results[:10]):
            print(f"  {i+1}. {r.id}")
            print(f"      Seq:    {r.baseline_seq_div:.3f} -> {r.strategy_seq_div:.3f} (+{r.seq_gain:.3f})")
            print(f"      Struct: {r.baseline_struct_div:.3f} -> {r.strategy_struct_div:.3f} (+{r.struct_gain:.3f})")
    
    # Load sample files
    strat_results_path = strategy_path.parent / "results.jsonl"
    base_results_path = baseline_path.parent / "results.jsonl"
    
    strat_samples = load_results_jsonl(str(strat_results_path)) if strat_results_path.exists() else {}
    base_samples = load_results_jsonl(str(base_results_path)) if base_results_path.exists() else {}
    
    print(f"\nLoading sample results...")
    print(f"  Strategy samples: {sum(len(v) for v in strat_samples.values())} total")
    print(f"  Baseline samples: {sum(len(v) for v in base_samples.values())} total")
    
    # Extract diverse samples
    if results:
        print(f"\n=== Extracting Diverse Samples ===")
        print(f"Strategy: selecting {args.top_k} HIGH diversity samples per entry (with metric filter)")
        print(f"Baseline: selecting {args.top_k} LOW diversity samples per entry (no filter)")
        
        all_high_div, all_low_div = [], []
        
        for r in tqdm(results, desc="Processing entries"):
            # Strategy: high diversity with relaxed metric filter
            if r.id in strat_samples:
                filtered = filter_samples_relaxed(strat_samples[r.id], strategy.get(r.id, {}), strat_avgs, args.top_k)
                sel, div, indices = select_diverse_subset(filtered, r.strategy_seq_div, args.top_k, "high")
                r.strategy_high_div_samples = [{
                    "entry_id": r.id,
                    "sample_n": s.get("n"),
                    "gen_seq": s.get("gen_seq"),
                    "gen_pdb": s.get("gen_pdb"),
                    "subset_diversity": div
                } for s in sel]
                all_high_div.extend(sel)
            
            # Baseline: low diversity without filter
            if r.id in base_samples:
                sel, div, indices = select_diverse_subset(base_samples[r.id], r.baseline_seq_div, args.top_k, "low")
                r.baseline_low_div_samples = [{
                    "entry_id": r.id,
                    "sample_n": s.get("n"),
                    "gen_seq": s.get("gen_seq"),
                    "gen_pdb": s.get("gen_pdb"),
                    "subset_diversity": div
                } for s in sel]
                all_low_div.extend(sel)
        
        print(f"\nExtraction complete:")
        print(f"  Strategy high-diversity samples: {len(all_high_div)}")
        print(f"  Baseline low-diversity samples: {len(all_low_div)}")
    
    # Build output
    output = {
        "config": {
            "baseline_file": str(args.baseline),
            "strategy_file": str(args.strategy),
            "top_k": args.top_k
        },
        "statistics": {
            "baseline_entries": len(baseline),
            "strategy_entries": len(strategy),
            "common_entries": len(set(baseline) & set(strategy)),
            "qualifying_entries": len(results)
        },
        "metric_averages": {
            "strategy": {k: v for k, v in strat_avgs.items() if v is not None},
            "baseline": {k: v for k, v in base_avgs.items() if v is not None}
        },
        "results": [asdict(r) for r in results]
    }
    
    # Save
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n=== Output ===")
    print(f"All results saved to: {args.output}")
    print("=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
