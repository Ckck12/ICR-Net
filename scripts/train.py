#!/usr/bin/env python3
"""
ICR-Net Training Script
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import logging
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.icr_net import ICRNet
from datasets.pair_dataset import H5PairDataset, h5_pair_collate_fn
from utils.metrics import calculate_metrics

def setup_logging(log_dir):
    """Setup logging"""
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def load_config(config_path):
    """Load config file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_data_loaders(config, corruption_type, severity):
    """Create data loaders"""
    clean_train_path = os.path.join(
        config['clean_data_path'], 
        f"clean_ff++_train.h5"
    )
    clean_val_path = os.path.join(
        config['clean_data_path'], 
        f"clean_ff++_val.h5"
    )
    
    corrupt_train_path = os.path.join(
        config['corrupt_data_path'],
        f"center_corruption_train_{corruption_type}_real_{severity}.h5"
    )
    corrupt_val_path = os.path.join(
        config['corrupt_data_path'],
        f"center_corruption_val_{corruption_type}_real_{severity}.h5"
    )
    
    train_dataset = H5PairDataset(
        clean_data_path=clean_train_path,
        corrupt_data_path=corrupt_train_path,
        corruption_type=corruption_type,
        severity=severity,
        split='train',
        clip_size=config['clip_size'],
        resolution=config['resolution'],
        augmentation=config.get('use_data_augmentation', True)
    )
    
    val_dataset = H5PairDataset(
        clean_data_path=clean_val_path,
        corrupt_data_path=corrupt_val_path,
        corruption_type=corruption_type,
        severity=severity,
        split='val',
        clip_size=config['clip_size'],
        resolution=config['resolution'],
        augmentation=False
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train_batchSize'],
        shuffle=True,
        num_workers=config['workers'],
        collate_fn=h5_pair_collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['test_batchSize'],
        shuffle=False,
        num_workers=config['workers'],
        collate_fn=h5_pair_collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader

def create_model(config):
    """Create model"""
    model = ICRNet(config)
    
    if config['cuda'] and torch.cuda.is_available():
        model = model.cuda()
    
    return model

def create_optimizer(model, config):
    """Create optimizer"""
    optimizer_config = config['optimizer']
    
    if optimizer_config['type'] == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=optimizer_config['adam']['lr'],
            betas=(optimizer_config['adam']['beta1'], optimizer_config['adam']['beta2']),
            eps=optimizer_config['adam']['eps'],
            weight_decay=optimizer_config['adam']['weight_decay']
        )
    elif optimizer_config['type'] == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=optimizer_config['sgd']['lr'],
            momentum=optimizer_config['sgd']['momentum'],
            weight_decay=optimizer_config['sgd']['weight_decay']
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_config['type']}")
    
    return optimizer

def create_scheduler(optimizer, config):
    """Create learning rate scheduler"""
    if config['lr_scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['lr_T_max'],
            eta_min=config['lr_eta_min']
        )
    else:
        scheduler = None
    
    return scheduler

def train_epoch(model, train_loader, optimizer, device, logger):
    """Train one epoch"""
    model.train()
    
    total_loss = 0.0
    total_cls_loss = 0.0
    total_pred_loss = 0.0
    total_con_loss = 0.0
    total_sc_loss = 0.0
    
    all_probs = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc='Training')
    
    for batch_idx, batch in enumerate(pbar):
        for key in ['image_clean', 'image_corr', 'label']:
            batch[key] = batch[key].to(device)
        
        optimizer.zero_grad()
        pred_dict = model(batch, inference=False)
        
        loss_dict = model.get_losses(batch, pred_dict)
        loss = loss_dict['overall']
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_cls_loss += loss_dict['cls'].item()
        total_pred_loss += loss_dict['pred'].item()
        total_con_loss += loss_dict['con'].item()
        total_sc_loss += loss_dict['sc'].item()
        
        probs = pred_dict['prob'].detach().cpu().numpy()
        labels = batch['label'].detach().cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels)
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Cls': f'{loss_dict["cls"].item():.4f}',
            'Pred': f'{loss_dict["pred"].item():.4f}',
            'Con': f'{loss_dict["con"].item():.4f}',
            'SC': f'{loss_dict["sc"].item():.4f}'
        })
    
    num_batches = len(train_loader)
    avg_loss = total_loss / num_batches
    avg_cls_loss = total_cls_loss / num_batches
    avg_pred_loss = total_pred_loss / num_batches
    avg_con_loss = total_con_loss / num_batches
    avg_sc_loss = total_sc_loss / num_batches
    
    metrics = calculate_metrics(np.array(all_labels), np.array(all_probs))
    
    logger.info(f"Train - Loss: {avg_loss:.4f}, "
                f"Cls: {avg_cls_loss:.4f}, Pred: {avg_pred_loss:.4f}, "
                f"Con: {avg_con_loss:.4f}, SC: {avg_sc_loss:.4f}, "
                f"ACC: {metrics['acc']:.4f}, AUC: {metrics['auc']:.4f}")
    
    return {
        'loss': avg_loss,
        'cls_loss': avg_cls_loss,
        'pred_loss': avg_pred_loss,
        'con_loss': avg_con_loss,
        'sc_loss': avg_sc_loss,
        'metrics': metrics
    }

def validate_epoch(model, val_loader, device, logger):
    """Validate one epoch"""
    model.eval()
    
    total_loss = 0.0
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        
        for batch in pbar:
            for key in ['image_clean', 'image_corr', 'label']:
                batch[key] = batch[key].to(device)
            
            pred_dict = model(batch, inference=False)
            
            loss_dict = model.get_losses(batch, pred_dict)
            loss = loss_dict['overall']
            
            total_loss += loss.item()
            
            probs = pred_dict['prob'].detach().cpu().numpy()
            labels = batch['label'].detach().cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels)
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(val_loader)
    
    metrics = calculate_metrics(np.array(all_labels), np.array(all_probs))
    
    logger.info(f"Val - Loss: {avg_loss:.4f}, "
                f"ACC: {metrics['acc']:.4f}, AUC: {metrics['auc']:.4f}, "
                f"EER: {metrics['eer']:.4f}")
    
    return {
        'loss': avg_loss,
        'metrics': metrics
    }

def save_checkpoint(model, optimizer, scheduler, epoch, metrics, save_path):
    """Save checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics
    }
    
    torch.save(checkpoint, save_path)

def main():
    parser = argparse.ArgumentParser(description='ICR-Net Training')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--train_corruption', type=str, required=True, help='Training corruption type')
    parser.add_argument('--train_severity', type=int, default=3, help='Training corruption severity')
    parser.add_argument('--output_dir', type=str, default='./checkpoints', help='Output directory')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    args = parser.parse_args()
    

    config = load_config(args.config)
    
    output_dir = os.path.join(args.output_dir, f"{args.train_corruption}_sev{args.train_severity}")
    os.makedirs(output_dir, exist_ok=True)
    

    logger = setup_logging(output_dir)
    logger.info(f"Starting ICR-Net training for {args.train_corruption} severity {args.train_severity}")
    

    device = torch.device('cuda' if config['cuda'] and torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
 
    train_loader, val_loader = create_data_loaders(config, args.train_corruption, args.train_severity)
    logger.info(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")
    
 
    model = create_model(config)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    

    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    

    writer = SummaryWriter(os.path.join(output_dir, 'tensorboard'))
    

    start_epoch = 0
    best_auc = 0.0
    
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_auc = checkpoint['metrics']['auc']
        logger.info(f"Resumed from epoch {start_epoch}, best AUC: {best_auc:.4f}")

    for epoch in range(start_epoch, config['nEpochs']):
        logger.info(f"Epoch {epoch+1}/{config['nEpochs']}")
        

        train_results = train_epoch(model, train_loader, optimizer, device, logger)
        
  
        val_results = validate_epoch(model, val_loader, device, logger)
        
       
        if scheduler:
            scheduler.step()
        
     
        writer.add_scalar('Train/Loss', train_results['loss'], epoch)
        writer.add_scalar('Train/ACC', train_results['metrics']['acc'], epoch)
        writer.add_scalar('Train/AUC', train_results['metrics']['auc'], epoch)
        writer.add_scalar('Val/Loss', val_results['loss'], epoch)
        writer.add_scalar('Val/ACC', val_results['metrics']['acc'], epoch)
        writer.add_scalar('Val/AUC', val_results['metrics']['auc'], epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
   
        if val_results['metrics']['auc'] > best_auc:
            best_auc = val_results['metrics']['auc']
            best_model_path = os.path.join(output_dir, 'best_model.pth')
            save_checkpoint(model, optimizer, scheduler, epoch, val_results['metrics'], best_model_path)
            logger.info(f"New best model saved with AUC: {best_auc:.4f}")
        
      
        if (epoch + 1) % config['save_epoch'] == 0:
            checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch+1}.pth')
            save_checkpoint(model, optimizer, scheduler, epoch, val_results['metrics'], checkpoint_path)
    
    writer.close()
    logger.info("Training completed!")

if __name__ == '__main__':
    main()
