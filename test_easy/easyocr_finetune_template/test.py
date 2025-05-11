label_path = "./data/train_label.txt"

with open(label_path, 'r', encoding='utf-8') as f:
    for line in f:
        path, text = line.strip().split(',', 1)
        if len(text) > 16:
            print(f"❌ 라벨이 너무 깁니다: {path} ({len(text)}자) → {text}")
