import sys
sys.path.append("C:/Users/HOME/Desktop/testyolo/EasyOCR")
from easyocr.model.recognition import RecognitionModel
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import wandb
import json
import Levenshtein
from torch.optim.lr_scheduler import LambdaLR
import os
from PIL import ImageOps

from easyocr.model.recognition import RecognitionModel
from dataset import LicensePlateDataset

# ✅ 하이퍼파라미터
LABEL_PATH = 'C:/Users/HOME/Desktop/testyolo/yolov5/train/TrainLabel/train_label.txt'
IMG_DIR = 'C:/Users/HOME/Desktop/testyolo/yolov5/train/TrainingImg'
EVAL_LABEL_PATH = 'C:/Users/HOME/Desktop/testyolo/yolov5/train/evalLabel/eval_label.txt'
EVAL_IMG_DIR = 'C:/Users/HOME/Desktop/testyolo/yolov5/train/evalImg'
CHARSET_PATH = 'charset_korean.json'
PRETRAINED_PATH = 'korean_g2.pth'
CHECKPOINT_DIR = 'C:/Users/HOME/Desktop/testyolo/yolov5/train/ocr_model_checkpoints'

EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
LOG_INTERVAL = 10
WARMUP_EPOCHS = 5

# ✅ 비율 유지 + 패딩 전처리
class ResizeWithPadding:
    def __init__(self, size, fill_color=(0, 0, 0)):
        self.size = size
        self.fill_color = fill_color

    def __call__(self, img):
        img.thumbnail((self.size[1], self.size[0]), Image.Resampling.LANCZOS)
        delta_w = self.size[1] - img.size[0]
        delta_h = self.size[0] - img.size[1]
        padding = (
            delta_w // 2,
            delta_h // 2,
            delta_w - (delta_w // 2),
            delta_h - (delta_h // 2)
        )
        new_img = ImageOps.expand(img, padding, fill=self.fill_color)
        return new_img

transform = transforms.Compose([
    ResizeWithPadding((32, 128)),
    transforms.ToTensor(),
])

# ✅ charset 로드
with open(CHARSET_PATH, 'r', encoding='utf-8') as f:
    charset = json.load(f)['char_list']
idx2char = {i + 1: c for i, c in enumerate(charset)}

# ✅ 디코딩 함수
def decode_prediction(preds):
    pred_indices = preds.argmax(2).permute(1, 0).tolist()
    decoded = []
    for seq in pred_indices:
        string = ""
        prev = -1
        for idx in seq:
            if idx != prev and idx != 0:
                string += idx2char.get(idx, "")
            prev = idx
        decoded.append(string)
    return decoded

# ✅ 정확도, 편집거리

def calculate_accuracy(preds, labels):
    return sum([p == l for p, l in zip(preds, labels)]) / len(labels)

def calculate_avg_edit_distance(preds, labels):
    return sum([Levenshtein.distance(p, l) for p, l in zip(preds, labels)]) / len(labels)

# ✅ collate_fn
def collate_fn(batch):
    images, labels, label_lengths = zip(*batch)
    images = torch.stack(images)
    label_lengths = torch.tensor(label_lengths, dtype=torch.long)
    targets = torch.cat(labels)
    return images, targets, label_lengths, labels

# ✅ Dataset/Dataloader
train_dataset = LicensePlateDataset(LABEL_PATH, IMG_DIR, CHARSET_PATH, transform)
eval_dataset = LicensePlateDataset(EVAL_LABEL_PATH, EVAL_IMG_DIR, CHARSET_PATH, transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# ✅ 모델/손실함수/optimizer/scheduler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RecognitionModel(lang_list=['korean'], network='standard').to(device)
model.load_state_dict(torch.load(PRETRAINED_PATH, map_location=device))

ctc_loss = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

def lr_lambda(epoch):
    if epoch < WARMUP_EPOCHS:
        return (epoch + 1) / WARMUP_EPOCHS
    else:
        cosine_epoch = epoch - WARMUP_EPOCHS
        total_cosine_epochs = EPOCHS - WARMUP_EPOCHS
        return 0.5 * (1 + torch.cos(torch.tensor(cosine_epoch / total_cosine_epochs * 3.141592)))

scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

# ✅ wandb 초기화
wandb.init(project="easyocr-licenseplate-finetune", config={
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "lr_schedule": "warmup + cosine decay",
    "log_interval": LOG_INTERVAL
})

# ✅ 체크포인트 저장 준비
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
best_acc = 0.0
global_step = 0

# ✅ 학습 루프
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for step, (images, targets, target_lengths, _) in enumerate(train_loader):
        images = images.to(device)
        targets = targets.to(device)

        logits = model(images)
        log_probs = logits.log_softmax(2)
        input_lengths = torch.full((logits.size(1),), logits.size(0), dtype=torch.long)

        loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        global_step += 1

        if global_step % LOG_INTERVAL == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"[Epoch {epoch+1} | Step {global_step}] Loss: {loss.item():.4f} | LR: {current_lr:.6f}")
            wandb.log({
                "step": global_step,
                "train_loss": loss.item(),
                "learning_rate": current_lr
            })

    scheduler.step()

    # ✅ 평가 루프
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, _, _, labels_tensor in eval_loader:
            images = images.to(device)
            logits = model(images)
            preds = decode_prediction(logits.cpu())
            labels = [''.join([idx2char[idx.item()] for idx in label]) for label in labels_tensor]
            all_preds.extend(preds)
            all_labels.extend(labels)

    acc = calculate_accuracy(all_preds, all_labels)
    edit_dist = calculate_avg_edit_distance(all_preds, all_labels)

    print(f"[Epoch {epoch+1}] \U0001f4ca Val Accuracy: {acc:.4f}, Edit Distance: {edit_dist:.4f}")
    wandb.log({
        "epoch": epoch + 1,
        "val_accuracy": acc,
        "val_edit_distance": edit_dist
    })

    # ✅ 모델 저장
    torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/epoch_{epoch+1}.pth")
    torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/latest_model.pth")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), f"{CHECKPOINT_DIR}/best_model.pth")

wandb.finish()
print(f"\n✅ 학습 완료. 모든 모델은 {CHECKPOINT_DIR}에 저장됨.")