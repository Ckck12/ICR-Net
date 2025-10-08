#!/usr/bin/env python3
"""
ICR-Net Batch Testing Script
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from tqdm import tqdm
import logging
import json
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.icr_net import ICRNet
from datasets.pair_dataset import H5PairDataset, h5_pair_collate_fn
from utils.metrics import calculate_metrics, print_metrics, save_metrics

def setup_logging(log_dir):

    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'batch_testing.log')),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def load_config(config_path):

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_model(config, checkpoint_path):

    model = ICRNet(config)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print("No checkpoint provided, using random initialization")
    
    if config['cuda'] and torch.cuda.is_available():
        model = model.cuda()
    
    model.eval()
    return model

def create_test_data_loader(config, corruption_type, severity):

    clean_test_path = os.path.join(
        config['clean_data_path'], 
        f"clean_ff++_test.h5"
    )
    

    corrupt_test_path = os.path.join(
        config['corrupt_data_path'],
        f"center_corruption_test_{corruption_type}_real_{severity}.h5"
    )
    

    test_dataset = H5PairDataset(
        clean_data_path=clean_test_path,
        corrupt_data_path=corrupt_test_path,
        corruption_type=corruption_type,
        severity=severity,
        split='test',
        clip_size=config['clip_size'],
        resolution=config['resolution'],
        augmentation=False
    )
    

    test_loader = DataLoader(
        test_dataset,
        batch_size=config['test_batchSize'],
        shuffle=False,
        num_workers=config['workers'],
        collate_fn=h5_pair_collate_fn,
        pin_memory=True
    )
    
    return test_loader

def test_model_batch(model, test_loader, device, logger):

    model.eval()
    
    all_probs = []
    all_labels = []
    all_video_ids = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Batch Testing')
        
        for batch in pbar:

            for key in ['image_clean', 'image_corr', 'label']:
                batch[key] = batch[key].to(device)
            
            # Forward pass (inference mode)
            pred_dict = model(batch, inference=True)
            

            probs = pred_dict['prob'].detach().cpu().numpy()
            labels = batch['label'].detach().cpu().numpy()
            video_ids = batch['video_id']
            
            all_probs.extend(probs)
            all_labels.extend(labels)
            all_video_ids.extend(video_ids)
            

            pbar.set_postfix({'Processed': len(all_probs)})
    

    metrics = calculate_metrics(np.array(all_labels), np.array(all_probs))
    
    logger.info(f"Batch Test Results:")
    print_metrics(metrics, "  ")
    
    return {
        'metrics': metrics,
        'probs': all_probs,
        'labels': all_labels,
        'video_ids': all_video_ids
    }

def main():
    parser = argparse.ArgumentParser(description='ICR-Net Batch Testing')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--weights', type=str, required=True, help='Model weights path')
    parser.add_argument('--test_corruption', type=str, required=True, help='Test corruption type')
    parser.add_argument('--test_severity', type=int, required=True, help='Test corruption severity')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    
    args = parser.parse_args()
    

    config = load_config(args.config)
    

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"batch_test_{args.test_corruption}_sev{args.test_severity}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    logger = setup_logging(output_dir)
    logger.info("Starting ICR-Net batch testing")
    logger.info(f"Config: {args.config}")
    logger.info(f"Weights: {args.weights}")
    logger.info(f"Corruption: {args.test_corruption}, Severity: {args.test_severity}")
    

    device = torch.device('cuda' if config['cuda'] and torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    

    model = load_model(config, args.weights)
    logger.info(f"Model loaded with {sum(p.numel() for p in model.parameters())} parameters")
    

    test_loader = create_test_data_loader(config, args.test_corruption, args.test_severity)
    logger.info(f"Test samples: {len(test_loader.dataset)}")
    

    results = test_model_batch(model, test_loader, device, logger)
    

    results_path = os.path.join(output_dir, 'test_results.npz')
    np.savez(results_path, 
             probs=results['probs'],
             labels=results['labels'],
             video_ids=results['video_ids'])
    

    metrics_path = os.path.join(output_dir, 'metrics.json')
    save_metrics(results['metrics'], metrics_path)
    

    detailed_results = {
        'config': args.config,
        'weights': args.weights,
        'corruption': args.test_corruption,
        'severity': args.test_severity,
        'timestamp': timestamp,
        'metrics': results['metrics'],
        'num_samples': len(results['probs'])
    }
    
    detailed_path = os.path.join(output_dir, 'detailed_results.json')
    with open(detailed_path, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    logger.info(f"Results saved to {output_dir}")
    logger.info("Batch testing completed!")

if __name__ == '__main__':
    main()
