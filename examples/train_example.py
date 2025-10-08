#!/usr/bin/env python3
"""
ICR-Net Training Example
"""

import os
import sys
import yaml

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.icr_net import ICRNet
from datasets.pair_dataset import H5PairDataset, h5_pair_collate_fn
from utils.metrics import calculate_metrics, print_metrics

def main():
    

    config_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'configs', 'icr_net.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    

    config['clean_data_path'] = '/path/to/clean/data'
    config['corrupt_data_path'] = '/path/to/corrupt/data'
    
    print("ICR-Net Training Example")
    print("=" * 50)
    
    print("Creating ICR-Net model...")
    model = ICRNet(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    

    print("Creating dummy dataset...")
    

    import numpy as np
    import torch
    

    clean_video = np.random.randint(0, 255, (16, 224, 224, 3), dtype=np.uint8)
    clean_label = 0  # Real
    

    corrupt_video = np.random.randint(0, 255, (16, 224, 224, 3), dtype=np.uint8)
    corrupt_label = 0  # Real
    

    clean_tensor = torch.from_numpy(clean_video).float().permute(3, 0, 1, 2).unsqueeze(0) / 255.0
    corrupt_tensor = torch.from_numpy(corrupt_video).float().permute(3, 0, 1, 2).unsqueeze(0) / 255.0
    label_tensor = torch.tensor([clean_label], dtype=torch.long)

    batch_data = {
        'image_clean': clean_tensor,
        'image_corr': corrupt_tensor,
        'label': label_tensor,
        'video_id': ['dummy_video_001']
    }
    
    print("Batch data created:")
    print(f"  Clean video shape: {batch_data['image_clean'].shape}")
    print(f"  Corrupt video shape: {batch_data['image_corr'].shape}")
    print(f"  Label: {batch_data['label'].item()}")
    
    # Forward pass
    print("\nRunning forward pass...")
    model.eval()
    with torch.no_grad():
        pred_dict = model(batch_data, inference=False)
    
    print("Forward pass completed:")
    print(f"  Prediction probability: {pred_dict['prob'].item():.4f}")
    print(f"  Clean clip embedding shape: {pred_dict['out_clean']['clip_embed'].shape}")
    print(f"  Corrupt clip embedding shape: {pred_dict['out_corr']['clip_embed'].shape}")
    

    print("\nCalculating losses...")
    loss_dict = model.get_losses(batch_data, pred_dict)
    
    print("Losses calculated:")
    for loss_name, loss_value in loss_dict.items():
        print(f"  {loss_name}: {loss_value.item():.4f}")
    

    print("\nCalculating metrics...")
    dummy_probs = [pred_dict['prob'].item()]
    dummy_labels = [batch_data['label'].item()]
    
    metrics = calculate_metrics(np.array(dummy_labels), np.array(dummy_probs))
    print_metrics(metrics, "  ")
    
    print("\n" + "=" * 50)
    print("ICR-Net training example completed successfully!")
    print("\nTo run actual training, use:")
    print("python scripts/train.py --config src/configs/icr_net.yaml --train_corruption packet_loss --train_severity 3")

if __name__ == '__main__':
    main()
