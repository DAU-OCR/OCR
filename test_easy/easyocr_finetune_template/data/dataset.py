
import os
import cv2
import torch
from torch.utils.data import Dataset

class OCRDataset(Dataset):
    def __init__(self, label_file, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.samples = []

        with open(label_file, 'r', encoding='utf-8') as f:
            for line in f:
                path, label = line.strip().split(',', 1)
                self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        full_path = os.path.join(self.img_dir, os.path.basename(img_path))

        img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (160, 32))  # config 기준 크기
        img = img.astype('float32') / 255.0
        img = torch.tensor(img).unsqueeze(0)  # [1, H, W]

        return img, label
