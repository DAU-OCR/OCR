import os
from PIL import Image
from torch.utils.data import Dataset
import torch
import json

class LicensePlateDataset(Dataset):
    def __init__(self, label_path, image_dir, charset_path, transform=None):
        self.image_dir = image_dir
        self.samples = []
        self.transform = transform

        # 라벨 로드
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f:
                img_name, label = line.strip().split(',')
                self.samples.append((img_name, label))

        # charset 로드
        with open(charset_path, 'r', encoding='utf-8') as f:
            self.charset = json.load(f)['char_list']
        
        # 문자 → 인덱스 매핑 딕셔너리
        self.char2idx = {char: idx + 1 for idx, char in enumerate(self.charset)}  # CTC에서 0은 blank

    def __len__(self):
        return len(self.samples)

    def encode_label(self, label):
        return [self.char2idx[c] for c in label]

    def __getitem__(self, idx):
        img_name, label = self.samples[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # 이미지 로딩
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # 정수 인덱스 시퀀스로 변환
        label_idx = self.encode_label(label)
        label_tensor = torch.tensor(label_idx, dtype=torch.long)

        return image, label_tensor, len(label_tensor)
