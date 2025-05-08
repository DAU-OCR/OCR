import os
import cv2
import torch
import easyocr
from pathlib import Path
import numpy as np

# 모델 로드 (YOLOv5 구조로 학습된 best.pt)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='weights/best.pt', force_reload=False)
# 기존 Reader: EasyOCR 기본 모델 사용
# reader = easyocr.Reader(['ko'], gpu=False)

# ⛔ 아래처럼 교체하면 Hugging Face에서 받은 로컬 ko.pth 사용 가능
#   1. model_storage_directory는 ko.pth가 위치한 폴더
#   2. download_enabled=False를 설정해야 로컬 로드만 시도함
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
MIN_AREA = 5000
MIN_ASPECT_RATIO = 1.8

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

        if area >= MIN_AREA and aspect_ratio >= MIN_ASPECT_RATIO:
            filtered.append((area, (int(x1), int(y1), int(x2), int(y2))))

    if not filtered:
        print(f"❌ {fname}: 조건에 맞는 번호판 없음")
        continue

    # 가장 큰 박스 하나 선택
    filtered.sort(reverse=True)
    _, (x1, y1, x2, y2) = filtered[0]

    # Crop 저장
    cropped = img[y1:y2, x1:x2]
    crop_fname = f"{Path(fname).stem}_plate.jpg"
    crop_path = os.path.join(save_crop_folder, crop_fname)
    cv2.imwrite(crop_path, cropped)

    # OCR 수행
    ocr_result = reader.readtext(cropped)
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

    # 시각화 (박스 + OCR 결과)
    vis_img = img.copy()
    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for i, line in enumerate(lines):
        text_y = y1 - 10 - i * 20
        if text_y < 0: text_y = y1 + 20 + i * 20
        cv2.putText(vis_img, line, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # 시각화 이미지 저장
    visual_path = os.path.join(visual_folder, f"{Path(fname).stem}_vis.jpg")
    cv2.imwrite(visual_path, vis_img)
