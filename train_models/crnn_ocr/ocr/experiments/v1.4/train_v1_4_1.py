# train_v1.4.1.py - CRNN + CTC OCR 학습 스크립트

# 기본 라이브러리 및 PyTorch 관련 모듈 불러오기
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import wandb    # Weights & Biases: 실험 로깅 및 시각화 도구
import random
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter
from torchvision import transforms

# 사용자 정의 모듈 불러오기
from utils.dataset import OCRDataset, OCRLabelOnlyDataset   # 이미지/레이블 데이터셋 클래스
from utils.label_encoder import LabelEncoder    # 텍스트를 숫자로 인코딩/디코딩 클래스
from model.crnn import CRNN # CRNN 모델 정의

# 모델 저장 경로 수정
ocr_dir = os.path.dirname(os.path.abspath(__file__))
crnn_dir = os.path.abspath(os.path.join(ocr_dir, os.pardir))
bestmodel_dir = os.path.join(crnn_dir, "bestModel")
os.makedirs(bestmodel_dir, exist_ok=True)
best_path = os.path.join(bestmodel_dir, "ocr_best_model.pth")

# ---------------------------
# 문자 단위 정확도 계산 함수
# ---------------------------
def char_accuracy(gt, pred):
    """
    gt: 실제 텍스트 (ground truth)
    pred: 예측 텍스트
    동일한 위치의 문자 일치 비율을 백분율로 반환
    """
    match = sum(g == p for g, p in zip(gt, pred))
    return match / max(len(gt), len(pred)) * 100

# ---------------------------
# Beam Search 디코딩 함수
# ---------------------------
def decode_beam_search(logits, encoder, beam_width=3):
    """
    logits: [W, B, C] - 시간축, 배치, 클래스
    encoder: 인코더 객체
    beam_width: top-k 후보 유지 수
    """
    log_probs = logits.log_softmax(2).cpu().detach().numpy()  # [W, B, C]
    log_probs = np.transpose(log_probs, (1, 0, 2))  # [B, W, C]

    results = []
    for seq in log_probs:
        paths = [([], 0)]  # 초기 시퀀스와 점수
        for t in seq:
            new_paths = []
            topk = np.argsort(t)[-beam_width:][::-1]  # 확률 상위 k개 선택
            for path, score in paths:
                for k in topk:
                    new_path = path + [k]
                    new_score = score + t[k]
                    new_paths.append((new_path, new_score))
            paths = sorted(new_paths, key=lambda x: x[1], reverse=True)[:beam_width]
        best_path = paths[0][0]
        decoded = encoder.decode_ctc_standard(best_path)
        results.append(decoded)
    return results

# ---------------------------
# 모델 평가 함수 (Beam Search 기반)
# ---------------------------
def evaluate(model, val_loader, encoder, criterion):
    model.eval()
    total = 0
    correct = 0
    val_loss = 0.0

    with torch.no_grad():
        for images, texts in val_loader:
            images = images.to(device)
            outputs = model(images).log_softmax(2)

            # 라벨 인코딩
            targets = [torch.tensor(encoder.encode(t), dtype=torch.long) for t in texts]
            target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
            targets_concat = torch.cat(targets).to(device)
            input_lengths = torch.full(size=(images.size(0),), fill_value=outputs.size(0), dtype=torch.long).to(device)

            loss = criterion(outputs, targets_concat, input_lengths, target_lengths)
            val_loss += loss.item()

            # # Beam Search 디코딩 및 정확도 평가
            decoded_texts = decode_beam_search(outputs, encoder, beam_width=5)
            for pred_text, gt_text in zip(decoded_texts, texts):
                total += 1
                if pred_text == gt_text:
                    correct += 1

    acc = correct / total * 100
    avg_loss = val_loss / total
    return acc, avg_loss

# ---------------------------
# 균형잡힌 샘플러 생성 함수 (접두어 기반)
# ---------------------------
def create_prefix_sampler(dataset, prefix_len=3):
    label_list = [dataset[i] for i in range(len(dataset))]  # 모든 레이블 읽기
    prefix_list = [label[:prefix_len] for label in label_list]  # 접두어 추출
    freq = Counter(prefix_list)
    weights = [1.0 / freq[label[:prefix_len]] for label in label_list]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return sampler

# ---------------------------
# 메인 학습 루틴
# ---------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # wandb 초기화
    wandb.init(
        project="crnn-ocr",
        name="train_v1.4.1",
        dir="wandb_logs",
        config={
            "batch_size": 32,
            "lr": 0.0005,
            "epochs": 100,
            "hidden_size": 512,
            "decoder": "ctc_standard"
        }
    )
    config = wandb.config

    # 이미지 전처리 정의
    transform = transforms.Compose([
        transforms.Resize((32, 128)),
        transforms.RandomAffine(degrees=1, translate=(0.01, 0.01)),
        ## 학습시 '37바'에 묶여버리는 현상의 원인으로 파악됨
        #transforms.ColorJitter(brightness=0.2, contrast=0.2),
        #transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        #transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 학습/검증 데이터셋 및 라벨 전용 데이터셋 로딩
    train_dataset = OCRDataset( image_dir='data/images/train', label_dir='data/labels/train', transform=transform )
    val_dataset = OCRDataset( image_dir='data/images/val', label_dir='data/labels/val', transform=transform )
    label_dataset = OCRLabelOnlyDataset( label_dir='data/labels/train' )
    
    # # 샘플러 설정 및 데이터로더 정의
    sampler = create_prefix_sampler(label_dataset, prefix_len=3)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    # 인코더 및 모델, 손실함수, 옵티마이저 구성
    encoder = LabelEncoder()
    num_classes = encoder.num_classes()
    model = CRNN(num_classes=num_classes, hidden_size=config.hidden_size).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    # 학습률 스케줄러
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.lr,
        steps_per_epoch=len(train_loader),
        epochs=config.epochs,
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=25.0
    )

    best_acc = 0.0
    log = []
    pred_samples = []
    patience = 20  # 성능이 개선되지 않아도 기다릴 최대 epoch 수
    counter = 0    # 현재 patience 카운터

    # 학습 루프
    try:
        for epoch in range(config.epochs):
            model.train()
            running_loss = 0.0
            for batch_idx, (images, texts) in enumerate(train_loader):
                images = images.to(device)

                targets = [torch.tensor(encoder.encode(t), dtype=torch.long) for t in texts]
                target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
                targets_concat = torch.cat(targets).to(device)

                batch_size = images.size(0)
                output = model(images)
                input_lengths = torch.full(size=(batch_size,), fill_value=output.size(0), dtype=torch.long).to(device)

                loss = criterion(output.log_softmax(2), targets_concat, input_lengths, target_lengths)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                running_loss += loss.item()

            # 검증 평가
            val_acc, val_loss = evaluate(model, val_loader, encoder, criterion)

            # 샘플 예측 및 정확도 측정
            avg_char_acc = 0
            pred_samples.clear()
            with torch.no_grad():
                sample_indices = random.sample(range(len(train_dataset)), 10)
                for idx in sample_indices:
                    sample_img, sample_text = train_dataset[idx]
                    sample_img = sample_img.unsqueeze(0).to(device)
                    pred = model(sample_img)
                    pred_std = encoder.decode_ctc_standard(pred.softmax(2).argmax(2)[:, 0].cpu().numpy())
                    pred_beam = decode_beam_search(pred, encoder, beam_width=5)[0]
                    acc = char_accuracy(sample_text, pred_std)
                    avg_char_acc += acc
                    pred_samples.append((sample_text, pred_std, pred_beam, round(acc, 2)))
            avg_char_acc /= len(sample_indices)

            print(f"[Epoch {epoch+1}] Train Loss: {running_loss / len(train_loader):.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}% | Char Acc (평균): {avg_char_acc:.4f}%")
            
            # 모델 저장
            if not os.path.exists('checkpoints'):
                os.makedirs('checkpoints')
            torch.save(model.state_dict(), f'checkpoints/crnn_epoch_{epoch+1}.pth')
            print(f"✅ 모델 저장됨: checkpoints/crnn_epoch_{epoch+1}.pth")

            if val_acc > best_acc:
                best_acc = val_acc
                counter = 0  # 개선되었으므로 리셋
                torch.save(model.state_dict(), best_path)
                print(f"🌟 향상 모델 갱신됨! 저장됨: {best_path}")
            else:
                counter += 1
                print(f"⏳ 향상되지 않음 (counter: {counter}/{patience})")
                if counter >= patience:
                    print("⛔️ Early stopping triggered!")
                    break

            # 예시 디코딩 출력 유지
            for sample_text, pred_std, pred_beam, acc in pred_samples[:3]:
                print(f"[샘플] GT: {sample_text}")
                print(f"  └ Pred (표준 CTC): {pred_std} | Char Acc: {acc:.4f}%")
                print(f"  └ Pred (Beam Search): {pred_beam}")

            # wandb 로깅
            wandb.log({
                "epoch": epoch + 1,
                "loss": running_loss / len(train_loader),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "char_accuracy": avg_char_acc,
                "lr": optimizer.param_groups[0]["lr"]
            })
            log.append((epoch + 1, running_loss / len(train_loader), val_loss, val_acc, avg_char_acc))

    except KeyboardInterrupt:
        print("\n⛔️ 학습 중단됨! 마지막 상태 저장 중...")
        torch.save(model.state_dict(), 'checkpoints/last_interrupted.pth')
        print("📂 마지막 상태 저장됨: checkpoints/last_interrupted.pth")

    # CSV로 학습 로그 저장
    wandb.finish()
    df = pd.DataFrame(log, columns=['epoch', 'train_loss', 'val_loss', 'val_acc', 'char_accuracy'])
    df.to_csv('train_logs/train_log.csv', index=False)
    print("📄 학습 로그 저장됨: train_log.csv")