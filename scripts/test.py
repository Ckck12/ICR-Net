#!/usr/bin/env python3
"""
ICR-Net Testing Script
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import logging
import cv2
from PIL import Image

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.icr_net import ICRNet
from datasets.pair_dataset import H5PairDataset, h5_pair_collate_fn
from utils.metrics import calculate_metrics

def setup_logging(log_dir):

    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'testing.log')),
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

def test_model(model, test_loader, device, logger):

    model.eval()
    
    all_probs = []
    all_labels = []
    all_video_ids = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')
        
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
            

            pbar.set_postfix({'Batch': len(all_probs)})
    

    metrics = calculate_metrics(np.array(all_labels), np.array(all_probs))
    
    logger.info(f"Test Results:")
    logger.info(f"  ACC: {metrics['acc']:.4f}")
    logger.info(f"  AUC: {metrics['auc']:.4f}")
    logger.info(f"  EER: {metrics['eer']:.4f}")
    logger.info(f"  AP: {metrics['ap']:.4f}")
    
    return {
        'metrics': metrics,
        'probs': all_probs,
        'labels': all_labels,
        'video_ids': all_video_ids
    }

def test_single_video(model, video_path, device, config):

    model.eval()
    

    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    
    if len(frames) < config['clip_size']:

        repeat_times = (config['clip_size'] // len(frames)) + 1
        frames = (frames * repeat_times)[:config['clip_size']]
    

    processed_frames = []
    for frame in frames[:config['clip_size']]:
        # BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize
        frame = cv2.resize(frame, (config['resolution'], config['resolution']))
        # Normalize
        frame = frame.astype(np.float32) / 255.0
        processed_frames.append(frame)
    
    # [T, H, W, C] -> [C, T, H, W]
    video_tensor = np.transpose(processed_frames, (3, 0, 1, 2))
    video_tensor = torch.from_numpy(video_tensor).unsqueeze(0).to(device)
    

    with torch.no_grad():
        data_dict = {'image_corr': video_tensor}
        pred_dict = model(data_dict, inference=True)
        prob = pred_dict['prob'].item()
    
    return prob

def test_single_image(model, image_path, device, config):

    model.eval()
    
 
    image = Image.open(image_path).convert('RGB')
    image = image.resize((config['resolution'], config['resolution']))
    image = np.array(image).astype(np.float32) / 255.0
    

    frames = [image] * config['clip_size']
    
    # [T, H, W, C] -> [C, T, H, W]
    video_tensor = np.transpose(frames, (3, 0, 1, 2))
    video_tensor = torch.from_numpy(video_tensor).unsqueeze(0).to(device)
    

    with torch.no_grad():
        data_dict = {'image_corr': video_tensor}
        pred_dict = model(data_dict, inference=True)
        prob = pred_dict['prob'].item()
    
    return prob

def main():
    parser = argparse.ArgumentParser(description='ICR-Net Testing')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--weights', type=str, required=True, help='Model weights path')
    parser.add_argument('--test_corruption', type=str, default=None, help='Test corruption type')
    parser.add_argument('--test_severity', type=int, default=3, help='Test corruption severity')
    parser.add_argument('--input_video', type=str, default=None, help='Single video path for testing')
    parser.add_argument('--input_image', type=str, default=None, help='Single image path for testing')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    
    args = parser.parse_args()
    

    config = load_config(args.config)
    

    os.makedirs(args.output_dir, exist_ok=True)
    

    logger = setup_logging(args.output_dir)
    logger.info("Starting ICR-Net testing")
    

    device = torch.device('cuda' if config['cuda'] and torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    

    model = load_model(config, args.weights)
    logger.info(f"Model loaded with {sum(p.numel() for p in model.parameters())} parameters")
    

    if args.input_video:
        logger.info(f"Testing single video: {args.input_video}")
        prob = test_single_video(model, args.input_video, device, config)
        logger.info(f"Deepfake probability: {prob:.4f}")
        return
    
    if args.input_image:
        logger.info(f"Testing single image: {args.input_image}")
        prob = test_single_image(model, args.input_image, device, config)
        logger.info(f"Deepfake probability: {prob:.4f}")
        return
    

    if not args.test_corruption:
        logger.error("Please specify --test_corruption for batch testing")
        return
    

    test_loader = create_test_data_loader(config, args.test_corruption, args.test_severity)
    logger.info(f"Test samples: {len(test_loader.dataset)}")
    

    results = test_model(model, test_loader, device, logger)
    

    results_path = os.path.join(args.output_dir, f'test_results_{args.test_corruption}_sev{args.test_severity}.npz')
    np.savez(results_path, 
             probs=results['probs'],
             labels=results['labels'],
             video_ids=results['video_ids'],
             metrics=results['metrics'])
    
    logger.info(f"Results saved to {results_path}")
    logger.info("Testing completed!")

if __name__ == '__main__':
    main()
