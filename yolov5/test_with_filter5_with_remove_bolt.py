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
reader = easyocr.Reader(
    lang_list=['ko'],
    gpu=False,
    model_storage_directory='custom_weights_easyOCR',
    user_network_directory='custom_weights_easyOCR',
    recog_network='korean_g2',
    download_enabled=False
)

# 디렉토리 설정
image_folder = 'images/korean'
save_crop_folder = 'cropped'
ocr_text_folder = 'ocr_results'
visual_folder = 'visualized'
os.makedirs(save_crop_folder, exist_ok=True)
os.makedirs(ocr_text_folder, exist_ok=True)
os.makedirs(visual_folder, exist_ok=True)

# 필터 기준
MIN_AREA = 1400
MIN_ASPECT_RATIO = 1.0

# ● 볼트 제거 함수
def remove_bolt_like_circles(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
        param1=50, param2=30, minRadius=5, maxRadius=15
    )

    if circles is not None:
        circles = np.uint16(np.around(circles[0, :]))
        for (x, y, r) in circles:
            cv2.circle(image, (x, y), r + 2, (255, 255, 255), -1)

    return image

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

    # ✅ ● 제거
    img = remove_bolt_like_circles(img)

    return img

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

def filter_small_text(results, min_w=20, min_h=20):
    filtered = []
    for (bbox, text, conf) in results:
        (tl, tr, br, bl) = bbox
        w = np.linalg.norm(np.array(tr) - np.array(tl))
        h = np.linalg.norm(np.array(tl) - np.array(bl))
        if w >= min_w and h >= min_h:
            filtered.append((bbox, text, conf))
    return filtered

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
    print(f"\n📦 {fname} 내 탐지된 박스 수: {len(detections)}")

    for i, det in enumerate(detections):
        x1, y1, x2, y2, conf, cls = det.tolist()
        w, h = x2 - x1, y2 - y1
        area = w * h
        aspect_ratio = w / h if h != 0 else 0
        print(f"  - Box {i+1}: 면적={int(area)}, 비율={aspect_ratio:.2f}, conf={conf:.2f}")

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

    filtered.sort(reverse=True)
    _, (x1, y1, x2, y2) = filtered[0]

    cropped = img[y1:y2, x1:x2]
    cropped = correct_skew(cropped)
    crop_fname = f"{Path(fname).stem}_plate.jpg"
    cv2.imwrite(os.path.join(save_crop_folder, crop_fname), cropped)

    ocr_result = reader.readtext(cropped)
    ocr_result = filter_small_text(ocr_result)
    ocr_result = [(bbox, text, conf) for (bbox, text, conf) in ocr_result if conf >= 0.1]

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

    vis_img = img.copy()
    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    vis_pil = Image.fromarray(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(vis_pil)
    for i, line in enumerate(lines):
        text_y = y1 - 25 - i * 25
        if text_y < 0: text_y = y2 + 25 + i * 25
        draw.text((x1, text_y), line, font=font, fill=(255, 0, 0))

    vis_img = cv2.cvtColor(np.array(vis_pil), cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(visual_folder, f"{Path(fname).stem}_vis.jpg"), vis_img)
