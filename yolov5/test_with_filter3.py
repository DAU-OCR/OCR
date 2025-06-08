import os
import cv2
import torch
import easyocr
from pathlib import Path
import numpy as np

# YOLOv5 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'custom', path='weights/best.pt', force_reload=False)

# EasyOCR: 로컬 ko.pth 사용
reader = easyocr.Reader(['ko'], gpu=False,
                        model_storage_directory='custom_weights_easyOCR',
                        download_enabled=False)

# 폴더 설정
image_folder = 'images/korean'
save_crop_folder = 'cropped'
ocr_text_folder = 'ocr_results'
visual_folder = 'visualized'
os.makedirs(save_crop_folder, exist_ok=True)
os.makedirs(ocr_text_folder, exist_ok=True)
os.makedirs(visual_folder, exist_ok=True)

# 필터 기준
MIN_AREA = 2000
MIN_ASPECT_RATIO = 1.2
MIN_CONFIDENCE = 0.4

def preprocess_plate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ✅ 대비 향상: CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # ✅ 선명도 향상: Unsharp Mask
    gaussian = cv2.GaussianBlur(gray, (9, 9), 10.0)
    gray = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)

    # ✅ 노이즈 제거: Bilateral Filter (가장자리 유지)
    gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    return gray  # threshold는 적용 X (easyOCR 성능 저하)

# 이미지 루프
for fname in os.listdir(image_folder):
    if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue

    image_path = os.path.join(image_folder, fname)
    img = cv2.imread(image_path)
    if img is None:
        print(f"[오류] 이미지 로드 실패: {fname}")
        continue

    results = model(img)
    detections = results.xyxy[0]

    filtered = []
    for det in detections:
        x1, y1, x2, y2, conf, cls = det.tolist()
        w, h = x2 - x1, y2 - y1
        area = w * h
        aspect_ratio = w / h if h != 0 else 0

        if area >= MIN_AREA and aspect_ratio >= MIN_ASPECT_RATIO and conf >= MIN_CONFIDENCE:
            filtered.append((area, (int(x1), int(y1), int(x2), int(y2))))

    if not filtered:
        print(f"❌ {fname}: 조건에 맞는 번호판 없음")
        continue

    # 가장 큰 박스 선택
    filtered.sort(reverse=True)
    _, (x1, y1, x2, y2) = filtered[0]

    cropped = img[y1:y2, x1:x2]
    preprocessed = preprocess_plate(cropped)

    crop_fname = f"{Path(fname).stem}_plate.jpg"
    crop_path = os.path.join(save_crop_folder, crop_fname)
    cv2.imwrite(crop_path, cropped)

    # OCR 수행
    ocr_result = reader.readtext(preprocessed)
    txt_path = os.path.join(ocr_text_folder, f"{Path(fname).stem}_ocr.txt")

    print(f"\n🔍 파일: {fname} | OCR 결과:")
    lines = []
    with open(txt_path, 'w', encoding='utf-8') as f:
        if ocr_result:
            for (_, text, conf) in ocr_result:
                line = f"{text} (신뢰도: {conf:.2f})"
                lines.append(line)
                print(f"➡️ {line}")
                f.write(line + '\n')
        else:
            print("❌ 인식 실패")
            f.write("❌ 인식 실패\n")

    # 시각화
    vis_img = img.copy()
    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for i, line in enumerate(lines):
        text_y = y1 - 10 - i * 20
        if text_y < 0: text_y = y1 + 20 + i * 20
        cv2.putText(vis_img, line, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    visual_path = os.path.join(visual_folder, f"{Path(fname).stem}_vis.jpg")
    cv2.imwrite(visual_path, vis_img)
