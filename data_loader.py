import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class SODDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
        self.masks = sorted([f for f in os.listdir(mask_dir) if f.endswith(('.jpg', '.png'))])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L") 

        if self.transform:
            state = torch.get_rng_state()
            image = self.transform(image)
            torch.set_rng_state(state)
            mask = self.transform(mask)

        return image, mask

def get_transforms():
    #Resizing to 224x224, Normalizing 0-1, and Augmentations
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5), 
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor(), 
    ])

def get_loader(img_dir, mask_dir, batch_size=16, shuffle=True):
    dataset = SODDataset(img_dir, mask_dir, transform=get_transforms())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)