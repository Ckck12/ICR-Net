"""
ICR-Net (Integrity-aware Contrastive Residual Network) Implementation
Deepfake detection using clean/corrupt pair learning with integrity assessment
"""

import os
import datetime
import logging
import numpy as np
from sklearn import metrics
from typing import Union, Dict, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)

class SpatialEncoder(nn.Module):
    """ResNet34-based Spatial Encoder"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        from torchvision.models import resnet34
        self.backbone = resnet34(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        with torch.no_grad():
            test_input = torch.randn(1, 3, 224, 224)
            test_features = self.backbone(test_input)
            self.feature_dim = test_features.view(-1).size(0)
        
    def forward(self, x):
        """
        Args:
            x: [B, C, T, H, W] - video clip
        Returns:
            S: [B, T, D] - frame embedding sequence
        """
        B, C, T, H, W = x.shape
        
        frames_flat = x.permute(0, 2, 1, 3, 4).contiguous().view(B*T, C, H, W)
        features = self.backbone(frames_flat)
        features = features.view(B*T, -1)
        features = features.view(B, T, 512)
        
        return features
    

class IntegrityGRU(nn.Module):
    """GRU module for integrity assessment"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.feature_dim = config.get('feature_dim', 512)
        self.hidden_dim = config.get('gru_hidden_dim', 512)
        
        self.gru = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=False
        )
        
        self.pred_head = nn.Linear(self.hidden_dim, self.feature_dim)
        self.lambda_alpha = config.get('lambda_alpha', 5.0)
        
    def forward(self, S):
        """
        Args:
            S: [B, T, D] - frame embeddings
        Returns:
            S_hat: [B, T, D] - predicted frame embeddings
            alpha: [B, T] - integrity scores
        """
        B, T, D = S.shape
        
        gru_out, _ = self.gru(S)
        
        S_hat = torch.zeros(B, T, D, device=S.device)
        
        if T > 1:
            S_hat[:, 1:] = self.pred_head(gru_out[:, :-1])
        
        alpha = torch.ones(B, T, device=S.device)
        
        if T > 1:
            pred_error = torch.norm(S[:, 1:] - S_hat[:, 1:], dim=2)
            alpha[:, 1:] = torch.exp(-self.lambda_alpha * pred_error / D)
        
        return S_hat, alpha

class ResidualPredictor(nn.Module):
    """1D-CNN based residual predictor"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.feature_dim = config.get('feature_dim', 512)
        
        self.conv1 = nn.Conv1d(self.feature_dim, self.feature_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(self.feature_dim, self.feature_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.norm = nn.LayerNorm(self.feature_dim)
        
    def forward(self, S):
        """
        Args:
            S: [B, T, D] - frame embeddings
        Returns:
            r: [B, T, D] - residual predictions
        """
        x = S.permute(0, 2, 1)
        
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.tanh(x)
        
        r = x.permute(0, 2, 1)
        r = self.norm(r)
        
        return r

class ContrastiveProjection(nn.Module):
    """Projection head for contrastive learning"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.feature_dim = config.get('feature_dim', 512)
        self.proj_dim = config.get('proj_dim', 256)
        
        self.proj_head = nn.Sequential(
            nn.Linear(self.feature_dim, self.proj_dim),
            nn.ReLU(),
            nn.Linear(self.proj_dim, self.proj_dim)
        )
        
    def forward(self, S_tilde):
        """
        Args:
            S_tilde: [B, T, D] - corrected frame embeddings
        Returns:
            z: [B, proj_dim] - clip embeddings
        """
        clip_embed = S_tilde.mean(dim=1)
        z = self.proj_head(clip_embed)
        z = F.normalize(z, p=2, dim=1)
        
        return z

class FrameClassifier(nn.Module):
    """Frame-wise classifier"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.feature_dim = config.get('feature_dim', 512)
        
        self.clf_head = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        
    def forward(self, S_tilde):
        """
        Args:
            S_tilde: [B, T, D] - corrected frame embeddings
        Returns:
            frame_logits: [B, T, 1] - frame-wise logits
            clip_prob: [B] - clip probability
        """
        B, T, D = S_tilde.shape
        
        frame_logits = self.clf_head(S_tilde)
        clip_prob = frame_logits.mean(dim=1).squeeze(-1)
        
        return frame_logits, clip_prob

class ICRNet(nn.Module):
    """ICR-Net main model"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.lambda_cls = config.get('lambda_cls', 1.0)
        self.lambda_pred = config.get('lambda_pred', 1.0)
        self.lambda_con = config.get('lambda_con', 0.5)
        self.lambda_sc = config.get('lambda_sc', 0.01)
        self.tau = config.get('contrastive_temperature', 0.1)
        
        self.spatial_encoder = SpatialEncoder(config)
        self.integrity_gru = IntegrityGRU(config)
        self.residual_predictor = ResidualPredictor(config)
        self.contrastive_projection = ContrastiveProjection(config)
        self.frame_classifier = FrameClassifier(config)
    
    def forward_single_view(self, x):
        """
        Forward pass for single view (clean or corrupt)
        
        Args:
            x: [B, C, T, H, W] - video clip
        Returns:
            dict: all intermediate results and final predictions
        """
        S = self.spatial_encoder(x)
        S_hat, alpha = self.integrity_gru(S)
        r = self.residual_predictor(S)
        
        m = (1 - alpha).unsqueeze(-1)
        S_tilde = S + m * r
        
        frame_logits, clip_prob = self.frame_classifier(S_tilde)
        z = self.contrastive_projection(S_tilde)
        
        return {
            'S': S,
            'S_hat': S_hat,
            'alpha': alpha,
            'r': r,
            'S_tilde': S_tilde,
            'frame_logits': frame_logits,
            'clip_prob': clip_prob,
            'clip_embed': z
        }
    
    def forward(self, data_dict: dict, inference=False) -> dict:
        """
        Forward pass for training or inference
        
        Args:
            data_dict: {'image_clean': ..., 'image_corr': ..., 'label': ..., 'video_id': ...}
            inference: True for inference mode
        """
        if inference:
            if 'image_corr' in data_dict:
                out = self.forward_single_view(data_dict['image_corr'])
                pred_dict = {
                    'cls': out['clip_prob'].unsqueeze(-1),
                    'prob': out['clip_prob'],
                    'feat': out['clip_embed']
                }
                return pred_dict
            elif 'image' in data_dict:
                out = self.forward_single_view(data_dict['image'])
                pred_dict = {
                    'cls': out['clip_prob'].unsqueeze(-1),
                    'prob': out['clip_prob'],
                    'feat': out['clip_embed']
                }
                return pred_dict
            else:
                raise ValueError("Inference mode requires 'image_corr' or 'image' key in data_dict")
        else:
            if 'image_clean' in data_dict and 'image_corr' in data_dict:
                out_clean = self.forward_single_view(data_dict['image_clean'])
                out_corr = self.forward_single_view(data_dict['image_corr'])
                
                avg_prob = (out_clean['clip_prob'] + out_corr['clip_prob']) / 2
                
                pred_dict = {
                    'cls': avg_prob.unsqueeze(-1),
                    'prob': avg_prob,
                    'feat': torch.cat([out_clean['clip_embed'], out_corr['clip_embed']], dim=0),
                    'out_clean': out_clean,
                    'out_corr': out_corr
                }
                return pred_dict
            else:
                raise ValueError("Training mode requires 'image_clean' and 'image_corr' keys")
    
    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        """ICR-Net multi-loss function calculation"""
        label = data_dict['label']
        
        if 'out_clean' in pred_dict and 'out_corr' in pred_dict:
            out_clean = pred_dict['out_clean']
            out_corr = pred_dict['out_corr']
            
            L_cls = 0.5 * (
                F.binary_cross_entropy(out_clean['clip_prob'], label.float()) +
                F.binary_cross_entropy(out_corr['clip_prob'], label.float())
            )
            
            L_pred = 0.5 * (
                F.mse_loss(out_clean['S'][:, 1:], out_clean['S_hat'][:, 1:]) +
                F.mse_loss(out_corr['S'][:, 1:], out_corr['S_hat'][:, 1:])
            )
            
            L_sc = 0.5 * (
                (out_clean['alpha'] * torch.norm(out_clean['r'], dim=2)).mean() +
                (out_corr['alpha'] * torch.norm(out_corr['r'], dim=2)).mean()
            )
            
            L_con = self._compute_contrastive_loss(
                out_clean['clip_embed'], out_corr['clip_embed'], 
                data_dict['video_id'], self.tau
            )
            
            total_loss = (
                self.lambda_cls * L_cls +
                self.lambda_pred * L_pred +
                self.lambda_con * L_con +
                self.lambda_sc * L_sc
            )
            
            loss_dict = {
                'overall': total_loss,
                'cls': L_cls,
                'pred': L_pred,
                'con': L_con,
                'sc': L_sc
            }
            
        else:
            prob = pred_dict['prob']
            L_cls = F.binary_cross_entropy(prob, label.float())
            loss_dict = {'overall': L_cls, 'cls': L_cls}
        
        return loss_dict
    
    def _compute_contrastive_loss(self, z_clean, z_corr, video_ids, tau):
        """Supervised Contrastive Loss calculation"""
        B = z_clean.size(0)
        
        z_all = torch.cat([z_clean, z_corr], dim=0)
        
        pair_ids = []
        for i, vid in enumerate(video_ids):
            pair_ids.extend([i, i])
        
        pair_ids = torch.tensor(pair_ids, device=z_all.device)
        
        z_norm = F.normalize(z_all, p=2, dim=1)
        sim_matrix = torch.mm(z_norm, z_norm.t()) / tau
        
        pos_mask = pair_ids.unsqueeze(0) == pair_ids.unsqueeze(1)
        pos_mask.fill_diagonal_(False)
        
        if pos_mask.sum() > 0:
            pos_sim = sim_matrix[pos_mask]
            neg_sim = sim_matrix[~pos_mask]
            
            pos_exp = torch.exp(pos_sim)
            neg_exp = torch.exp(neg_sim)
            
            loss = -torch.log(pos_exp / (pos_exp + neg_exp.sum()))
            return loss.mean()
        else:
            return torch.tensor(0.0, device=z_all.device)
