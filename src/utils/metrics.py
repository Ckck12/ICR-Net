"""
ICR-Net용 메트릭 계산 유틸리티
"""

import numpy as np
from sklearn import metrics
from typing import Dict, Tuple

def calculate_metrics(labels: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    """
    딥페이크 탐지 메트릭 계산
    
    Args:
        labels: 실제 라벨 (0: real, 1: fake)
        probs: 예측 확률 (0~1, 높을수록 fake)
    
    Returns:
        Dict containing ACC, AUC, EER, AP
    """
    # 라벨과 확률이 같은 길이인지 확인
    assert len(labels) == len(probs), f"Labels and probs length mismatch: {len(labels)} vs {len(probs)}"
    
    # 라벨이 0과 1만 포함하는지 확인
    unique_labels = np.unique(labels)
    assert set(unique_labels).issubset({0, 1}), f"Labels should be 0 or 1, got {unique_labels}"
    
    # 확률이 [0, 1] 범위에 있는지 확인
    assert np.all(probs >= 0) and np.all(probs <= 1), "Probs should be in [0, 1] range"
    
    # ACC 계산
    pred_labels = (probs > 0.5).astype(int)
    acc = np.mean(pred_labels == labels)
    
    # AUC 계산
    try:
        auc = metrics.roc_auc_score(labels, probs)
    except ValueError as e:
        print(f"Warning: AUC calculation failed: {e}")
        auc = 0.5  # Random performance
    
    # EER 계산
    try:
        eer = calculate_eer(labels, probs)
    except Exception as e:
        print(f"Warning: EER calculation failed: {e}")
        eer = 0.5  # Random performance
    
    # AP (Average Precision) 계산
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
    """
    Equal Error Rate (EER) 계산
    
    Args:
        labels: 실제 라벨 (0: real, 1: fake)
        probs: 예측 확률 (0~1, 높을수록 fake)
    
    Returns:
        EER 값
    """
    # FPR, TPR, thresholds 계산
    fpr, tpr, thresholds = metrics.roc_curve(labels, probs)
    
    # EER: FPR = FNR = 1 - TPR인 지점
    fnr = 1 - tpr
    
    # FPR과 FNR이 가장 가까운 지점 찾기
    eer_idx = np.nanargmin(np.absolute(fnr - fpr))
    eer = fpr[eer_idx]
    
    return float(eer)

def calculate_metrics_per_video(video_ids: list, labels: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    """
    비디오별 메트릭 계산 (같은 비디오의 프레임들을 평균)
    
    Args:
        video_ids: 비디오 ID 리스트
        labels: 실제 라벨
        probs: 예측 확률
    
    Returns:
        비디오 레벨 메트릭
    """
    unique_videos = list(set(video_ids))
    video_probs = []
    video_labels = []
    
    for video_id in unique_videos:
        # 해당 비디오의 모든 프레임 찾기
        video_mask = np.array([vid == video_id for vid in video_ids])
        video_prob = np.mean(probs[video_mask])
        video_label = labels[video_mask][0]  # 같은 비디오는 같은 라벨
        
        video_probs.append(video_prob)
        video_labels.append(video_label)
    
    video_probs = np.array(video_probs)
    video_labels = np.array(video_labels)
    
    return calculate_metrics(video_labels, video_probs)

def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """
    메트릭 출력
    
    Args:
        metrics: 메트릭 딕셔너리
        prefix: 출력 접두사
    """
    print(f"{prefix}ACC: {metrics['acc']:.4f}")
    print(f"{prefix}AUC: {metrics['auc']:.4f}")
    print(f"{prefix}EER: {metrics['eer']:.4f}")
    print(f"{prefix}AP:  {metrics['ap']:.4f}")

def save_metrics(metrics: Dict[str, float], save_path: str):
    """
    메트릭을 파일로 저장
    
    Args:
        metrics: 메트릭 딕셔너리
        save_path: 저장 경로
    """
    import json
    
    with open(save_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Metrics saved to {save_path}")

def load_metrics(load_path: str) -> Dict[str, float]:
    """
    파일에서 메트릭 로드
    
    Args:
        load_path: 로드 경로
    
    Returns:
        메트릭 딕셔너리
    """
    import json
    
    with open(load_path, 'r') as f:
        metrics = json.load(f)
    
    return metrics
