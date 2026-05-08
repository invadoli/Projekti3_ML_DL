import torch
import torch.optim as optim
import torch.nn as nn
import os

def dice_iou_loss(pred, target):
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    bce = nn.BCELoss()(pred, target)
    return bce + 0.5 * (1 - iou) 

def train_model(model, train_loader, val_loader, epochs=20, device='cuda'):
    model.to(device)
    # Adam optimizer with learning rate = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training 
        model.train()
        train_loss = 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = dice_iou_loss(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation 
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = dice_iou_loss(outputs, masks)
                val_loss += loss.item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train:.4f}, Val Loss: {avg_val:.4f}")

        # Save Best Model 
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, 'best_sod_model.pth')
            print("Best model saved!")