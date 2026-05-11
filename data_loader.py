import os
import torch
import random
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image

class DUTSDataset(Dataset):
    def __init__(self, img_dir, mask_dir, size=(224, 224), is_train=True):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.size = size
        self.is_train = is_train
        self.images = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        mask_name = img_name.replace(".jpg", ".png")
        
        image = Image.open(os.path.join(self.img_dir, img_name)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, mask_name)).convert("L")

        image = TF.resize(image, self.size)
        mask = TF.resize(mask, self.size)

        if self.is_train:
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            
            if random.random() > 0.5:
                angle = random.randint(-10, 10)
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle)

        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        image = TF.normalize(image, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        return image, mask