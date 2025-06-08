# dataset.py - OCRDataset 및 OCRLabelOnlyDataset 정의

import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# =====================================
# OCRDataset 클래스
# =====================================
class OCRDataset(Dataset):
    """
    이미지와 레이블 텍스트를 함께 제공하는 Dataset 클래스
    학습 및 검증 시 사용
    """
    
    def __init__(self, image_dir, label_dir, transform=None):
        """
        image_dir: 이미지 폴더 경로
        label_dir: 이미지에 대응되는 텍스트 라벨(.txt) 폴더 경로
        transform: 이미지 전처리 함수
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform

        # jpg 또는 png 확장자를 가진 이미지 이름 목록 정렬
        self.image_names = sorted([
            fname for fname in os.listdir(self.image_dir) 
            if fname.lower().endswith(('.jpg', '.png'))
        ])

    def __len__(self):
        # 전체 샘플 개수
        return len(self.image_names)

    def __getitem__(self, idx):
        """
        이미지와 그에 대응되는 텍스트 레이블을 반환
        return: (이미지 Tensor, 문자열 레이블)
        """
        image_name = self.image_names[idx]  # 예: '00123.jpg'
        image_path = os.path.join(self.image_dir, image_name)

        # 이미지 이름과 동일한 이름의 .txt 파일에서 라벨 읽기
        label_path = os.path.join(self.label_dir, os.path.splitext(image_name)[0] + '.txt')

        # 흑백 이미지로 불러오기 (L: 1채널)
        image = Image.open(image_path).convert('L')

        # 전처리 적용 (ex: 리사이즈, 정규화)
        if self.transform:
            image = self.transform(image)

        # 라벨 텍스트 파일 읽기
        with open(label_path, 'r', encoding='utf-8') as f:
            label = f.read().strip()

        return image, label

# =====================================
# OCRLabelOnlyDataset 클래스
# =====================================
class OCRLabelOnlyDataset(Dataset):
    """
    이미지 없이 텍스트 레이블(.txt)만 불러오는 Dataset 클래스
    샘플링 전략(예: 접두어 균형)을 만들 때 사용
    """
    def __init__(self, label_dir):
        """
        label_dir: 라벨 텍스트 파일(.txt)들이 들어있는 폴더 경로
        """
        self.label_dir = label_dir

        # 모든 .txt 파일 이름 정렬
        self.label_names = sorted([
            fname for fname in os.listdir(label_dir)
            if fname.endswith('.txt')
        ])

    def __len__(self):
        # 전체 라벨 수
        return len(self.label_names)

    def __getitem__(self, idx):
        """
        텍스트 레이블만 반환
        return: 문자열 라벨
        """
        label_path = os.path.join(self.label_dir, self.label_names[idx])
        with open(label_path, 'r', encoding='utf-8') as f:
            label = f.read().strip()
        return label
