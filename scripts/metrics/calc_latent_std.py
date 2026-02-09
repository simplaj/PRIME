
import os
import argparse
import yaml
import torch
import numpy as np
from tqdm import tqdm

from utils.random_seed import setup_seed
from utils import register as R
from data import create_dataset, create_dataloader
import models

def parse():
    parser = argparse.ArgumentParser(description='Compute AE Latent Statistics')
    parser.add_argument('--config', type=str, required=True, help='Path to AE train.yaml')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to AE checkpoint (.ckpt)')
    parser.add_argument('--num_batches', type=int, default=50, help='Number of batches to estimate stats')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size override')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0], help='gpu to use')
    return parser.parse_args()

def main(args):
    # Load config
    config = yaml.safe_load(open(args.config, 'r'))
    
    # Setup Device
    device = torch.device(f'cuda:{args.gpus[0]}' if torch.cuda.is_available() and args.gpus[0] >= 0 else 'cpu')
    print(f"Using device: {device}")

    # Load Dataset
    print("Loading dataset...")
    # Override batch size
    if 'dataloader' in config and 'train' in config['dataloader']:
        config['dataloader']['train']['batch_size'] = args.batch_size
    train_set, _, _ = create_dataset(config['dataset'])
    train_loader = create_dataloader(train_set, config['dataloader']['train'], 1) # world_size=1

    # Load Model
    print("Loading model...")
    model = torch.load(args.ckpt, map_location='cpu')
    model.to(device)
    model.eval()

    # Collect latent statistics
    all_Zh = []
    
    print(f"Collecting statistics over {args.num_batches} batches...")
    with torch.no_grad():
        for i, batch in tqdm(enumerate(train_loader), total=args.num_batches):
            if i >= args.num_batches:
                break
            
            # Move batch to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            # Encode
            # Look at ldm.py to see how it encodes. It calls self.autoencoder.EPTencode
            # But the model here IS the autoencoder (CondIterAutoEncoder).
            # So we check CondIterAutoEncoder.EPTencode or .encode
            
            # CondIterAutoEncoder.EPTencode signature:
            # X, S, A, atom_positions, block_lengths, lengths, segment_ids, generate_mask ...
            
            # The batch keys usually match these names.
            # Let's try calling encode directly if pretrain=False logic isn't needed or handle it.
            # EPTencode is used in LDM, so let's stick to that for consistency.
            
            Zh, Zx, _, _, _, _, _, _ = model.EPTencode(
                batch['X'], batch['S'], batch['A'], batch['atom_positions'],
                batch['block_lengths'], batch['lengths'], batch['chain_ids'], 
                batch['generate_mask'], deterministic=False
            )
            # Zh: [Nblock, d_latent]
            
            # We want statistics of Zh
            all_Zh.append(Zh.cpu())

    all_Zh = torch.cat(all_Zh, dim=0)
    
    print(f"\nStats calculated on {len(all_Zh)} blocks.")
    mean = torch.mean(all_Zh).item()
    std = torch.std(all_Zh).item()
    
    print(f"Zh Mean: {mean:.6f} (Expected ~0.0)")
    print(f"Zh Std : {std:.6f}  (Expected ~1.0 for Standard Normal)")
    
    scale_factor = 1.0 / std
    print(f"\nRecommended Scale Factor: {scale_factor:.6f}")
    print(f"Use this value as 'latent_scale' in your LDM config/code.")

if __name__ == '__main__':
    setup_seed(42)
    main(parse())
