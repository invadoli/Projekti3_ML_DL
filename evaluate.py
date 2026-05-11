import torch

def calculate_metrics(pred, target, threshold=0.6):
    pred = (pred > threshold).float()
    
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()
    
    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
    
    intersection = tp
    union = tp + fp + fn
    iou = intersection / (union + 1e-7)
    
    return precision.item(), recall.item(), f1.item(), iou.item()