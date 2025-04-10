import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class OCRDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform

        self.image_names = sorted([
            fname for fname in os.listdir(self.image_dir) 
            if fname.lower().endswith(('.jpg', '.png'))
        ])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = os.path.join(self.image_dir, image_name)
        label_path = os.path.join(self.label_dir, os.path.splitext(image_name)[0] + '.txt')

        image = Image.open(image_path).convert('L')  # 흑백
        if self.transform:
            image = self.transform(image)

        with open(label_path, 'r', encoding='utf-8') as f:
            label = f.read().strip()

        return image, label

class OCRLabelOnlyDataset(Dataset):
    def __init__(self, label_dir):
        self.label_dir = label_dir
        self.label_names = sorted([
            fname for fname in os.listdir(label_dir)
            if fname.endswith('.txt')
        ])

    def __len__(self):
        return len(self.label_names)

    def __getitem__(self, idx):
        label_path = os.path.join(self.label_dir, self.label_names[idx])
        with open(label_path, 'r', encoding='utf-8') as f:
            label = f.read().strip()
        return label
