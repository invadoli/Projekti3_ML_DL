import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class DUTSDataset(Dataset):
    def __init__(self, img_dir, mask_dir, size=(224, 224)):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        mask_name = img_name.replace(".jpg", ".png")
        image = self.transform(Image.open(os.path.join(self.img_dir, img_name)).convert("RGB"))
        mask = self.mask_transform(Image.open(os.path.join(self.mask_dir, mask_name)).convert("L"))
        return image, mask