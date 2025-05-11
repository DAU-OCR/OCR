import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import OneCycleLR
import wandb
from torch.amp import autocast, GradScaler

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from easyocr.recognition import get_recognizer
from utils.label_converter import CTCLabelConverter
from data.dataset import OCRDataset

def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def evaluate(model, eval_loader, converter, device, criterion=None):
    model.eval()
    total = 0
    correct = 0
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in eval_loader:
            if images.size(0) != len(labels):
                print(f"⚠️ 이미지 개수와 라벨 수 불일치: {images.size(0)} != {len(labels)}")
                continue

            images = images.to(device)

            with autocast(device.type):
                preds = model(images)
                log_probs = preds.log_softmax(2)  # [T, B, C]

            B = log_probs.shape[1]  # 실제 배치 크기
            T = log_probs.shape[0]

            preds_idx = log_probs.argmax(2).permute(1, 0).tolist()  # [B, T]
            pred_texts = converter.decode(preds_idx)

            if criterion:
                encoded, lengths = converter.encode(labels)
                targets = torch.tensor(encoded, dtype=torch.long).to(device)
                target_lengths = torch.tensor(lengths, dtype=torch.long).to(device)

                # 정확한 B, T 재계산
                B = log_probs.shape[1]
                T = log_probs.shape[0]

                input_lengths = torch.full(size=(B,), fill_value=T, dtype=torch.long).to(device)

                # 디버깅용 출력
                print("========== CTC 디버깅 ==========")
                print(f"log_probs.shape: {log_probs.shape}")  # [T, B, C]
                print(f"targets.shape: {targets.shape}")
                print(f"input_lengths: {input_lengths.tolist()} (len={len(input_lengths)})")
                print(f"target_lengths: {target_lengths.tolist()} (len={len(target_lengths)})")
                print(f"expected batch_size: {B}")
                print("=================================")

                # 길이 불일치 방어 코드
                if target_lengths.size(0) != B:
                    print(f"⚠️ [evaluate] target_lengths 길이 보정: {target_lengths.size(0)} → {B}")
                    target_lengths = target_lengths[:B]
                    targets = targets[:target_lengths.sum()]

                print("==== CTC 디버깅 출력 ====")
                print("targets shape:", targets.shape)
                print("targets flattened shape:", targets.view(-1).shape)
                print("target_lengths:", target_lengths.tolist())
                print("sum of target_lengths:", target_lengths.sum().item())
                print("input_lengths:", input_lengths.tolist())
                print("=========================")

                loss = criterion(log_probs.permute(1, 0, 2), targets, input_lengths, target_lengths)
                total_loss += loss.item()

            for pred, gt in zip(pred_texts, labels):
                total += 1
                if pred == gt:
                    correct += 1

    model.train()
    acc = correct / total if total >  0 else 0.0
    avg_loss = total_loss / len(eval_loader) if criterion else None
    return acc, avg_loss




def main():
    config = load_config('./configs/config.yaml')
    characters = open(config['Global']['character_dict_path'], encoding='utf-8').read().splitlines()
    converter = CTCLabelConverter(characters)
    device = torch.device(config['Global']['device'])

    wandb.init(project=config['wandb']['project'], name=config['wandb']['name'], config=config)

    recog_network = config['Model']['network_arch']
    network_params = {
        'input_channel': config['Model']['input_channel'],
        'output_channel': config['Model']['output_channel'],
        'hidden_size': config['Model']['hidden_size']
    }
    model_path = config['Model']['model_path']
    separator_list = {}
    dict_list = {}

    model, _ = get_recognizer(
        recog_network,
        network_params,
        characters,
        separator_list,
        dict_list,
        model_path,
        device=device,
        quantize=False
    )
    model = model.to(device)

    train_dataset = OCRDataset(config['Dataset']['train_label_file'], config['Dataset']['train_img_dir'])
    train_loader = DataLoader(train_dataset, batch_size=config['Global']['batch_size'], shuffle=True)

    eval_dataset = OCRDataset(config['Dataset']['eval_label_file'], config['Dataset']['eval_img_dir'])
    eval_loader = DataLoader(eval_dataset, batch_size=config['Global']['batch_size'], shuffle=False)

    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['Optimizer']['max_lr'],
        weight_decay=config['Optimizer']['weight_decay']
    )

    scheduler = OneCycleLR(
        optimizer,
        max_lr=config['Optimizer']['max_lr'],
        steps_per_epoch=len(train_loader),
        epochs=config['Global']['epochs'],
        pct_start=0.1,
        anneal_strategy='cos'
    )

    scaler = GradScaler(enabled=config['Global'].get('use_amp', False))

    best_acc = 0
    patience = config.get('EarlyStopping', {}).get('patience', 10)
    wait = 0

    for epoch in range(config['Global']['epochs']):
        for i, (images, texts) in enumerate(train_loader):
            images = images.to(device)
            encoded, lengths = converter.encode(texts)
            targets = torch.tensor(encoded, dtype=torch.long).to(device)
            target_lengths = torch.tensor(lengths, dtype=torch.long).to(device)
            batch_size = images.size(0)

            with autocast(device.type, enabled=config['Global'].get('use_amp', False)):
                preds = model(images)
                preds = preds.log_softmax(2).permute(1, 0, 2)
                T, B, _ = preds.size()
                input_lengths = torch.full((B,), T, dtype=torch.long).to(device)  # ✅ 이 줄이 필수!

                if target_lengths.size(0) != B:
                    print(f"❌ 길이 불일치: input_lengths를 잘라냅니다. B={B}, target_lengths={target_lengths.size(0)}")
                    input_lengths = input_lengths[:target_lengths.size(0)]
                    targets = targets[:target_lengths.sum()]
                    preds = preds[:, :target_lengths.size(0), :]

                # 이 때 반드시 B가 batch_size여야 함
                # → 하지만 마지막 배치는 batch_size보다 작을 수 있음

                # 따라서 target_lengths의 길이와 input_lengths의 길이가 다르면 자르는 것이 아니라, input_lengths도 동기화 필요
                if target_lengths.size(0) != B:
                    print(f"❌ 길이 불일치: input_lengths를 잘라냅니다. B={B}, target_lengths={target_lengths.size(0)}")
                    input_lengths = input_lengths[:target_lengths.size(0)]
                    targets = targets[:target_lengths.sum()]
                    preds = preds[:, :target_lengths.size(0), :]

                if len(lengths) != B:
                    print("⚠️ 길이 불일치: 잘라냅니다.")
                    target_lengths = target_lengths[:B]
                    targets = targets[:sum(target_lengths)]

                print(f"preds.shape: {preds.shape}")
                print(f"targets.shape: {targets.shape}")
                print(f"input_lengths.shape: {input_lengths.shape}")
                print(f"target_lengths.shape: {target_lengths.shape}")

                # CTC 디버깅: input_lengths가 target보다 짧은 경우 확인
                # 디버깅용 코드: input_lengths보다 label 길이가 긴 경우 확인
                # input_lengths < target_lengths 방지 처리
                for idx in range(B):
                    if input_lengths[idx] < target_lengths[idx]:
                        print(
                            f"❌ CTC 불가: input_length({input_lengths[idx]}) < target_length({target_lengths[idx]}) → 샘플 제거")
                        input_lengths[idx] = 0
                        target_lengths[idx] = 0

                valid_indices = [i for i in range(B) if input_lengths[i] > 0 and target_lengths[i] > 0]
                if not valid_indices:
                    print("⚠️ 유효한 샘플 없음 → 학습 스킵")
                    continue

                input_lengths = input_lengths[valid_indices]
                target_lengths = target_lengths[valid_indices]
                preds = preds[:, valid_indices, :]
                targets = targets.view(-1)

                loss = criterion(preds, targets, input_lengths, target_lengths)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if i % config['Logging']['log_interval'] == 0:
                step = epoch * len(train_loader) + i
                print(f"Epoch [{epoch+1}/{config['Global']['epochs']}], Step [{i}], Loss: {loss.item():.4f}")
                wandb.log({
                    "train/loss": loss.item(),
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/step": step
                })

        eval_acc, eval_loss = evaluate(model, eval_loader, converter, device, criterion)
        print(f"[Eval] Epoch {epoch+1}: Accuracy = {eval_acc:.4f}, Loss = {eval_loss:.4f}")
        wandb.log({
            "eval/acc": eval_acc,
            "eval/loss": eval_loss,
            "eval/epoch": epoch + 1
        })

        if eval_acc > best_acc:
            best_acc = eval_acc
            wait = 0
            torch.save(model.state_dict(), config['Global']['save_model_path'])
            print(f"New best model saved to {config['Global']['save_model_path']}")
        else:
            wait += 1
            print(f"No improvement. Wait = {wait}/{patience}")
            if wait >= patience:
                print("Early stopping triggered.")
                return

if __name__ == '__main__':
    main()
