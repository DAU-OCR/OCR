import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import wandb
import random
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter
from torchvision import transforms
from utils.dataset import OCRDataset
from utils.label_encoder import LabelEncoder
from model.crnn import CRNN

# 문자 단위 정확도
def char_accuracy(gt, pred):
    match = sum(g == p for g, p in zip(gt, pred))
    return match / max(len(gt), len(pred)) * 100

# 평가 함수
def evaluate(model, val_loader, encoder):
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for images, texts in val_loader:
            images = images.to(device)
            outputs = model(images)
            pred_indices = outputs.softmax(2).argmax(2)  # [W, B]
            pred_indices = pred_indices[:, 0].cpu().numpy()
            pred_text = encoder.decode_ctc_standard(pred_indices)
            gt_text = texts[0]
            total += 1
            if pred_text == gt_text:
                correct += 1
    acc = correct / total * 100
    return acc

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    wandb.init(
        project="crnn-ocr",
        name="train_v1.3",
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

    train_dataset = OCRDataset(
        image_dir='data/images/train',
        label_dir='data/labels/train',
        transform=transform
    )

    val_dataset = OCRDataset(
        image_dir='data/images/val',
        label_dir='data/labels/val',
        transform=transform
    )
    
    # WeightedSampler 적용
    label_list = [train_dataset[i][1] for i in range(len(train_dataset))]
    label_freq = Counter(label_list)
    weights = [1.0 / label_freq[label] for label in label_list]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    encoder = LabelEncoder()
    num_classes = encoder.num_classes()

    model = CRNN(num_classes=num_classes, hidden_size=config.hidden_size).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)


    best_acc = 0.0
    log = []
    patience = 20  # 성능이 개선되지 않아도 기다릴 최대 epoch 수
    counter = 0    # 현재 patience 카운터

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

                running_loss += loss.item()
            
            scheduler.step(running_loss)

            if not os.path.exists('checkpoints'):
                os.makedirs('checkpoints')

            val_acc = evaluate(model, val_loader, encoder)
            print(f"[Epoch {epoch+1}] Loss: {running_loss / len(train_loader):.4f} | Val Acc: {val_acc:.4f}%")

            torch.save(model.state_dict(), f'checkpoints/crnn_epoch_{epoch+1}.pth')
            print(f"✅ 모델 저장됨: checkpoints/crnn_epoch_{epoch+1}.pth")

            if val_acc > best_acc:
                best_acc = val_acc
                counter = 0  # 개선되었으므로 리셋
                torch.save(model.state_dict(), 'checkpoints/best_model.pth')
                print(f"🌟 향상 모델 갱신됨! 저장됨: checkpoints/best_model.pth")
            else:
                counter += 1
                print(f"⏳ 향상되지 않음 (counter: {counter}/{patience})")
                if counter >= patience:
                    print("⛔️ Early stopping triggered!")
                    break

            # 예시 디코딩 출력 (무작위 샘플 3개 출력)
            model.eval()
            with torch.no_grad():
                sample_indices = random.sample(range(len(train_dataset)), 3)
                for idx in sample_indices:
                    sample_img, sample_text = train_dataset[idx]
                    sample_img = sample_img.unsqueeze(0).to(device)
                    pred = model(sample_img)
                    pred_indices = pred.softmax(2).argmax(2)
                    pred_indices = pred_indices[:, 0].cpu().numpy()
                    decoded_std = encoder.decode_ctc_standard(pred_indices)
                    decoded_raw = encoder.decode_keep_repeats(pred_indices)
                    acc = char_accuracy(sample_text, decoded_std)
                    print(f"[샘플] GT: {sample_text}")
                    print(f"  └ Pred (표준 CTC): {decoded_std} | Char Acc: {acc:.4f}%")
                    print(f"  └ Pred (중복 유지): {decoded_raw}")

            avg_loss = running_loss / len(train_loader)
            wandb.log({
                "epoch": epoch + 1,
                "loss": avg_loss,
                "val_acc": val_acc,
                "char_accuracy": acc,
                "lr": optimizer.param_groups[0]["lr"]
            })
            log.append((epoch + 1, avg_loss, val_acc))

    except KeyboardInterrupt:
        print("\n⛔️ 학습 중단됨! 마지막 상태 저장 중...")
        torch.save(model.state_dict(), 'checkpoints/last_interrupted.pth')
        print("📂 마지막 상태 저장됨: checkpoints/last_interrupted.pth")

    wandb.finish()

    df = pd.DataFrame(log, columns=['epoch', 'loss', 'val_acc'])
    df.to_csv('train_log.csv', index=False)
    print("📄 학습 로그 저장됨: train_log.csv")
