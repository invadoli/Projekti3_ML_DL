import torch
import numpy as np

def calculate_metrics(pred, target, threshold=0.5):
    pred = (pred > threshold).float()
    target = (target > 0.5).float()

    tp = (pred * target).sum().item()
    fp = (pred * (1 - target)).sum().item()
    fn = ((1 - pred) * target).sum().item()
    tn = ((1 - pred) * (1 - target)).sum().item()

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    iou = tp / (tp + fp + fn + 1e-6)
    
    return iou, precision, recall, f1

def evaluate_model(model, loader, device):
    model.eval()
    all_iou, all_p, all_r, all_f1 = [], [], [], []
    
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            
            iou, p, r, f1 = calculate_metrics(outputs, masks)
            all_iou.append(iou); all_p.append(p); all_r.append(r); all_f1.append(f1)
            
    print(f"Final Metrics:\nIoU: {np.mean(all_iou):.4f}\nPrecision: {np.mean(all_p):.4f}\nRecall: {np.mean(all_r):.4f}\nF1-Score: {np.mean(all_f1):.4f}")