import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from model.crnn import CRNN
from utils.label_encoder import LabelEncoder

# Beam Search 디코딩 (simple version)
def decode_beam_search(logits, encoder, beam_width=3):
    log_probs = logits.log_softmax(2).cpu().detach().numpy()  # [W, B, C]
    log_probs = np.transpose(log_probs, (1, 0, 2))  # [B, W, C]

    results = []
    for seq in log_probs:
        paths = [([], 0)]  # (sequence, score)
        for t in seq:
            new_paths = []
            topk = np.argsort(t)[-beam_width:][::-1]  # top-k indices
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델과 인코더 준비
encoder = LabelEncoder()
num_classes = encoder.num_classes()
model = CRNN(num_classes=num_classes, hidden_size=512).to(device)
model.load_state_dict(torch.load("checkpoints/best_model.pth"))
model.eval()

# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((32, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 테스트할 이미지 경로
image_path = "test_images/test6.jpg"  # <- 여기에 테스트용 이미지 파일 경로를 넣으세요

image = Image.open(image_path).convert('L')
image = transform(image).unsqueeze(0).to(device)

# 예측
with torch.no_grad():
    output = model(image)
    pred_ctc = output.softmax(2).argmax(2)[:, 0].cpu().numpy()
    pred_text_ctc = encoder.decode_ctc_standard(pred_ctc)
    pred_text_beam = decode_beam_search(output, encoder, beam_width=5)[0]

print("✅ 예측 결과")
print(f"표준 CTC 디코딩: {pred_text_ctc}")
print(f"Beam Search 디코딩: {pred_text_beam}")
