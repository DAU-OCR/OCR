import os
import cv2
import torch
import easyocr
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# YOLOv5 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'custom', path='weights/best.pt', force_reload=False)

# EasyOCR Reader - 로컬 커스텀 모델 사용할 경우 경로 설정
reader = easyocr.Reader(['ko'], gpu=False,
                        model_storage_directory='custom_weights_easyOCR',
                        download_enabled=False)

# 디렉토리 설정
image_folder = 'images/korean'
save_crop_folder = 'cropped'
ocr_text_folder = 'ocr_results'
visual_folder = 'visualized'
os.makedirs(save_crop_folder, exist_ok=True)
os.makedirs(ocr_text_folder, exist_ok=True)
os.makedirs(visual_folder, exist_ok=True)

# 필터 기준
MIN_AREA = 3000
MIN_ASPECT_RATIO = 1.6

# 전처리 함수
def preprocess(image):
    img = image.copy()

    # ✅ 대비 향상 (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l2 = clahe.apply(l)
    img = cv2.merge((l2, a, b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)

    # ✅ 샤프닝 (Unsharp masking)
    gaussian = cv2.GaussianBlur(img, (0, 0), 3)
    img = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)

    # ✅ 노이즈 제거 (bilateral filter)
    img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

    return img

# 한글 표시용 폰트 로드
def get_korean_font(size=20):
    font_paths = ['C:/Windows/Fonts/malgun.ttf', '/usr/share/fonts/truetype/nanum/NanumGothic.ttf']
    for path in font_paths:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    print("⚠️ 한글 폰트를 찾을 수 없음. 기본 폰트 사용")
    return ImageFont.load_default()

font = get_korean_font()

# 이미지 순회
for fname in os.listdir(image_folder):
    if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    image_path = os.path.join(image_folder, fname)
    img = cv2.imread(image_path)
    if img is None:
        print(f"[오류] 이미지 로드 실패: {fname}")
        continue

    pre_img = preprocess(img)
    results = model(pre_img)
    detections = results.xyxy[0]

    # 필터링
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

    # 가장 큰 박스 선택
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

    # 시각화 (PIL로 한글 텍스트 렌더링)
    vis_img = img.copy()
    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    vis_pil = Image.fromarray(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(vis_pil)
    for i, line in enumerate(lines):
        text_y = y1 - 25 - i * 25
        if text_y < 0: text_y = y2 + 25 + i * 25
        draw.text((x1, text_y), line, font=font, fill=(255, 0, 0))

    vis_img = cv2.cvtColor(np.array(vis_pil), cv2.COLOR_RGB2BGR)
    visual_path = os.path.join(visual_folder, f"{Path(fname).stem}_vis.jpg")
    cv2.imwrite(visual_path, vis_img)
