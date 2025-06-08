import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms

from utils.dataset import OCRDataset
from utils.label_encoder import LabelEncoder
from model.crnn import CRNN


# 💡 평가 함수는 전역에 둬도 괜찮음
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
            pred_text = encoder.decode(pred_indices)
            gt_text = texts[0]
            total += 1
            if pred_text == gt_text:
                correct += 1
    acc = correct / total * 100
    print(f"✅ 검증 정확도: {acc:.2f}% ({correct}/{total})")
    return acc


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 데이터 및 변환
    transform = transforms.Compose([
        transforms.Resize((32, 128)),
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

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    # 2. 라벨 인코더
    encoder = LabelEncoder()
    num_classes = encoder.num_classes()

    # 3. 모델, 손실함수, 옵티마이저
    model = CRNN(num_classes=num_classes).to(device)
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 4. 학습 루프
    best_acc = 0.0
    log = []

    try:
        for epoch in range(5):
            model.train()
            running_loss = 0.0
            for images, texts in train_loader:
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

            print(f"[Epoch {epoch+1}] Loss: {running_loss:.4f}")

            if not os.path.exists('checkpoints'):
                os.makedirs('checkpoints')

            torch.save(model.state_dict(), f'checkpoints/crnn_epoch_{epoch+1}.pth')
            print(f"✅ 모델 저장됨: checkpoints/crnn_epoch_{epoch+1}.pth")

            val_acc = evaluate(model, val_loader, encoder)

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), 'checkpoints/best_model.pth')
                print(f"🌟 최고 모델 갱신됨! 저장됨: checkpoints/best_model.pth")

            log.append((epoch + 1, running_loss, val_acc))

            # 예시 디코딩 출력
            model.eval()
            with torch.no_grad():
                sample_img, sample_text = train_dataset[0]
                sample_img = sample_img.unsqueeze(0).to(device)
                pred = model(sample_img)
                pred_indices = pred.softmax(2).argmax(2)
                pred_indices = pred_indices[:, 0].cpu().numpy()
                decoded = encoder.decode(pred_indices)
                print(f"GT: {sample_text}, Predicted: {decoded}")

    except KeyboardInterrupt:
        print("\n⛔️ 학습 중단됨! 마지막 상태 저장 중...")
        torch.save(model.state_dict(), 'checkpoints/last_interrupted.pth')
        print("💾 마지막 상태 저장됨: checkpoints/last_interrupted.pth")

    # 로그 저장
    df = pd.DataFrame(log, columns=['epoch', 'loss', 'val_acc'])
    df.to_csv('train_log.csv', index=False)
    print("📄 학습 로그 저장됨: train_log.csv")
