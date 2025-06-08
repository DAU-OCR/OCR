from collections import Counter
from utils.dataset import OCRDataset

dataset = OCRDataset(
    image_dir='data/images/train',
    label_dir='data/labels/train'
)

counter = Counter()
for _, label in dataset:
    counter[label] += 1

print("라벨 종류 수:", len(counter))
print(counter.most_common(10))
