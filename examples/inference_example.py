#!/usr/bin/env python3
"""
ICR-Net Inference Example
"""

import os
import sys
import yaml
import torch
import numpy as np
from PIL import Image
import cv2

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.icr_net import ICRNet

def load_model(config_path, weights_path=None):

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model = ICRNet(config)
    
    if weights_path and os.path.exists(weights_path):
        checkpoint = torch.load(weights_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded weights from {weights_path}")
    else:
        print("No weights provided, using random initialization")
    
    model.eval()
    return model, config

def preprocess_image(image_path, config):

    image = Image.open(image_path).convert('RGB')
    image = image.resize((config['resolution'], config['resolution']))
    image = np.array(image).astype(np.float32) / 255.0
    frames = [image] * config['clip_size']
    
    # [T, H, W, C] -> [C, T, H, W]
    video_tensor = np.transpose(frames, (3, 0, 1, 2))
    video_tensor = torch.from_numpy(video_tensor).unsqueeze(0)
    
    return video_tensor

def preprocess_video(video_path, config):

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
    video_tensor = torch.from_numpy(video_tensor).unsqueeze(0)
    
    return video_tensor

def predict_deepfake(model, input_tensor, device):
    input_tensor = input_tensor.to(device)
    
    with torch.no_grad():
        data_dict = {'image_corr': input_tensor}
        pred_dict = model(data_dict, inference=True)
        prob = pred_dict['prob'].item()
    
    return prob

def main():
    
    print("ICR-Net Inference Example")
    print("=" * 50)
    

    config_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'configs', 'icr_net.yaml')
    weights_path = None  
    
    model, config = load_model(config_path, weights_path)
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Using device: {device}")
    

    print("\nCreating dummy input...")
    

    dummy_image = np.random.randint(0, 255, (config['resolution'], config['resolution'], 3), dtype=np.uint8)
    frames = [dummy_image.astype(np.float32) / 255.0] * config['clip_size']
    input_tensor = np.transpose(frames, (3, 0, 1, 2))
    input_tensor = torch.from_numpy(input_tensor).unsqueeze(0)
    
    print(f"Input tensor shape: {input_tensor.shape}")
    

    print("\nRunning inference...")
    prob = predict_deepfake(model, input_tensor, device)
    
    print(f"Deepfake probability: {prob:.4f}")
    
    if prob > 0.5:
        print("Prediction: FAKE (Deepfake detected)")
    else:
        print("Prediction: REAL (No deepfake detected)")
    
    print("\n" + "=" * 50)
    print("ICR-Net inference example completed!")
    
    print("\nTo run inference on actual files:")
    print("python scripts/test.py --config src/configs/icr_net.yaml --weights path/to/model.pth --input_video video.mp4")
    print("python scripts/test.py --config src/configs/icr_net.yaml --weights path/to/model.pth --input_image image.jpg")

if __name__ == '__main__':
    main()
