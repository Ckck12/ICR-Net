

import numpy as np
from sklearn import metrics
from typing import Dict, Tuple

def calculate_metrics(labels: np.ndarray, probs: np.ndarray) -> Dict[str, float]:



    assert len(labels) == len(probs), f"Labels and probs length mismatch: {len(labels)} vs {len(probs)}"
    

    unique_labels = np.unique(labels)
    assert set(unique_labels).issubset({0, 1}), f"Labels should be 0 or 1, got {unique_labels}"
    

    assert np.all(probs >= 0) and np.all(probs <= 1), "Probs should be in [0, 1] range"
    

    pred_labels = (probs > 0.5).astype(int)
    acc = np.mean(pred_labels == labels)
    
 
    try:
        auc = metrics.roc_auc_score(labels, probs)
    except ValueError as e:
        print(f"Warning: AUC calculation failed: {e}")
        auc = 0.5  # Random performance
    

    try:
        eer = calculate_eer(labels, probs)
    except Exception as e:
        print(f"Warning: EER calculation failed: {e}")
        eer = 0.5  # Random performance
    
  
    try:
        ap = metrics.average_precision_score(labels, probs)
    except ValueError as e:
        print(f"Warning: AP calculation failed: {e}")
        ap = 0.5  # Random performance
    
    return {
        'acc': float(acc),
        'auc': float(auc),
        'eer': float(eer),
        'ap': float(ap)
    }

def calculate_eer(labels: np.ndarray, probs: np.ndarray) -> float:


    fpr, tpr, thresholds = metrics.roc_curve(labels, probs)
    

    fnr = 1 - tpr
    

    eer_idx = np.nanargmin(np.absolute(fnr - fpr))
    eer = fpr[eer_idx]
    
    return float(eer)

def calculate_metrics_per_video(video_ids: list, labels: np.ndarray, probs: np.ndarray) -> Dict[str, float]:

    unique_videos = list(set(video_ids))
    video_probs = []
    video_labels = []
    
    for video_id in unique_videos:
  
        video_mask = np.array([vid == video_id for vid in video_ids])
        video_prob = np.mean(probs[video_mask])
        video_label = labels[video_mask][0]  
        
        video_probs.append(video_prob)
        video_labels.append(video_label)
    
    video_probs = np.array(video_probs)
    video_labels = np.array(video_labels)
    
    return calculate_metrics(video_labels, video_probs)

def print_metrics(metrics: Dict[str, float], prefix: str = ""):

    print(f"{prefix}ACC: {metrics['acc']:.4f}")
    print(f"{prefix}AUC: {metrics['auc']:.4f}")
    print(f"{prefix}EER: {metrics['eer']:.4f}")
    print(f"{prefix}AP:  {metrics['ap']:.4f}")

def save_metrics(metrics: Dict[str, float], save_path: str):

    import json
    
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Metrics saved to {save_path}")

def load_metrics(load_path: str) -> Dict[str, float]:

    import json
    
    with open(load_path, 'r') as f:
        metrics = json.load(f)
    
    return metrics
