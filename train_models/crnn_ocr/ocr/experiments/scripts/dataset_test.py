from utils.dataset import OCRDataset

dataset = OCRDataset(
    image_dir='data/images/train',
    label_dir='data/labels/train'
)

for i in range(20):
    img, label = dataset[i]
    print(f"[샘플 {i}] 라벨: {label}")
