# ───────────────────────────────────────────────────────────────────────────────
# 파일명: yolo_ocr_t1.py
#   • 
# ───────────────────────────────────────────────────────────────────────────────

import sys
import os
import re
import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path

# ──────────── (1) project_root 및 yolov5 모듈 검색 경로 추가 ────────────
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root / "yolov5"))
sys.path.insert(1, str(project_root))
# ────────────────────────────────────────────────────────────────────────────

from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.augmentations import letterbox
from yolov5.utils.general import non_max_suppression, scale_boxes
from ocr.model.crnn import CRNN
from ocr.utils.label_encoder import LabelEncoder

# -------------------- 전처리 및 OCR 관련 유틸 함수 --------------------
def remove_bolt_like_circles(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                               param1=50, param2=30, minRadius=5, maxRadius=15)
    if circles is not None:
        circles = np.uint16(np.around(circles[0, :]))
        for (x, y, r) in circles:
            cv2.circle(image, (x, y), r + 2, (255, 255, 255), -1)
    return image

def correct_skew(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    if lines is None:
        return image
    angles = [np.degrees(theta - np.pi / 2) for rho, theta in lines[:, 0]]
    median_angle = np.median(angles)
    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), median_angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def preprocess_enhanced(image):
    img = image.copy()
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l2 = cv2.createCLAHE(2.0, (8, 8)).apply(l)
    img = cv2.merge((l2, a, b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    gaussian = cv2.GaussianBlur(img, (0, 0), 3)
    img = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)
    img = cv2.bilateralFilter(img, 9, 75, 75)
    img = remove_bolt_like_circles(img)
    return img

def resize_with_padding(image, target_size=(128, 32)):
    h, w = image.shape[:2]
    target_w, target_h = target_size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))
    padded = np.ones((target_h, target_w), dtype=np.uint8) * 255
    pad_left = (target_w - new_w) // 2
    pad_top = (target_h - new_h) // 2
    padded[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    return padded

def preprocess_image(image, target_width=128, target_height=32):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = resize_with_padding(image, (target_width, target_height))
    image = image.astype(np.float32) / 255.0
    image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    return image

def decode_prediction(preds, label_encoder):
    preds = preds.permute(1, 0, 2)
    out = torch.nn.functional.log_softmax(preds, dim=2)
    _, out_best = out.max(2)
    out_best = out_best.transpose(1, 0).contiguous().view(-1)
    pred_standard = label_encoder.decode_ctc_standard(out_best.cpu().numpy())
    pred_keep_repeat = label_encoder.decode_keep_repeats(out_best.cpu().numpy())
    print(f"--CTC 표준 디코딩: {pred_standard}")
    print(f"--반복 보존 디코딩: {pred_keep_repeat}")
    return pred_standard

def crnn_ocr(image, crnn_model, label_encoder, device):
    input_tensor = preprocess_image(image).to(device)
    with torch.no_grad():
        preds = crnn_model(input_tensor)
    text = decode_prediction(preds, label_encoder)
    return text

def get_korean_font(size=20):
    font_paths = ['C:/Windows/Fonts/malgun.ttf', '/usr/share/fonts/truetype/nanum/NanumGothic.ttf']
    for path in font_paths:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()

# -------------------- 메인 루틴 --------------------
def load_crnn_model(model_path, num_classes, device):
    model = CRNN(img_height=32, num_channels=1, num_classes=num_classes, hidden_size=512)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def main():
    yolo_model_path = 'bestModel/yoloBestModel.pt'
    crnn_model_path = 'bestModel/ocrBestModel_142.pth'
    test_image_dir = 'images/test_images_kr'
    label_encoder = LabelEncoder()
    charset = label_encoder.get_charset()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    yolo_model = DetectMultiBackend(yolo_model_path, device=device)
    crnn_model = load_crnn_model(crnn_model_path, num_classes=len(charset)+1, device=device)

    output_dir = 'images/ocr_results'
    os.makedirs(output_dir, exist_ok=True)
    font = get_korean_font()

    results = {}

    for filename in os.listdir(test_image_dir):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        
        # ① 원본 파일명에서 확장자 제거
        stem = Path(filename).stem  # e.g. "01가1234_1"
        # ② 끝에 "_숫자" 패턴 제거
        clean_name = re.sub(r'_[0-9]+$', '', stem)  # → "01가1234"

        image_path = os.path.join(test_image_dir, filename)
        original_image = cv2.imread(image_path)
        if original_image is None:
            continue

        enhanced_img = preprocess_enhanced(original_image)
        image_resized = letterbox(enhanced_img, new_shape=640)[0]
        image_resized = image_resized[:, :, ::-1].transpose(2, 0, 1)
        image_resized = np.ascontiguousarray(image_resized, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_resized).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = yolo_model(image_tensor)
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)[0]

        if pred is not None and len(pred):
            pred[:, :4] = scale_boxes(image_tensor.shape[2:], pred[:, :4], original_image.shape).round()
            boxes = pred
        else:
            boxes = []

        vis_img = original_image.copy()
        vis_pil = Image.fromarray(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(vis_pil)
        lines = []

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box[:4])
            cropped = original_image[y1:y2, x1:x2]
            cropped = correct_skew(cropped)
            text = crnn_ocr(cropped, crnn_model, label_encoder, device)
            print(f"[{filename}] → 최종 OCR 결과: {text}")
            lines.append(text)
            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
            draw.text((x1, y1 - 25), text, font=font, fill=(255, 0, 0))

        vis_img = cv2.cvtColor(np.array(vis_pil), cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_dir, f"{Path(filename).stem}_vis.jpg"), vis_img)

        with open(os.path.join(output_dir, f"{Path(filename).stem}_ocr.txt"), 'w', encoding='utf-8') as f:
            if lines:
                for line in lines:
                    f.write(line + '\n')
            else:
                f.write("❌ 인식 실패\n")

        # 첫 번째(prediction)만, 없으면 빈 문자열
        results[clean_name] = lines[0] if lines else ''
    
    # ────────────────────────────────────────────────────────────────────
    # (6) 결과 수집 후 DataFrame 생성 및 정답률 계산
    # ────────────────────────────────────────────────────────────────────
    # results: { ground_truth_str: [pred1, pred2, …, predN] }

    # results: { ground_truth_str: predicted_str }
    # DataFrame 생성
    df = pd.DataFrame.from_dict(
        results, 
        orient='index', 
        columns=['prediction']
    )
    df.index.name = 'ground_truth'

    # 정답 여부 컬럼 추가
    df['correct'] = df.index == df['prediction']

    # 전체 이미지 수 대비 맞춘 비율 계산
    total = len(df)
    correct_count = df['correct'].sum()
    accuracy = correct_count / total * 100

    # 'correct' 컬럼 지우고, 마지막 행에 정답률 추가
    df = df.drop(columns=['correct'])
    df.loc['정답률'] = [f"{accuracy:.2f}%"]

    # ────────────────────────────────────────────────────────────────────
    # (7) CSV로 저장 (utf-8-sig로 BOM 추가)
    # ────────────────────────────────────────────────────────────────────
    # CSV로 저장 (utf-8-sig BOM 포함)
    csv_path = os.path.join(output_dir, 'ocr_v1.4.2_results_type_1.csv')
    df.to_csv(csv_path, encoding='utf-8-sig')
    print(f"✔ OCR 결과 및 정답률이 포함된 CSV로 저장했습니다: {csv_path}")

if __name__ == '__main__':
    main()