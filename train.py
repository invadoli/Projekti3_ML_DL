import torch
import torch.nn as nn

def hybrid_loss(pred, target):
    bce = nn.BCELoss()(pred, target)
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    iou = 1 - (inter / (union + 1e-7))
    return bce + 0.5 * iou

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for imgs, masks in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = hybrid_loss(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)