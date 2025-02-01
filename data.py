# Loading Dataset

import torch
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [
            os.path.join(root_dir, fname)
            for fname in os.listdir(root_dir)
            if fname.endswith(('.jpg', '.png', '.jpeg'))
        ]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


if __name__ == '__main__':
    # Transform image for standard processing
    transform = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),
    ])

    dataset = CustomDataset(root_dir = "images", transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)