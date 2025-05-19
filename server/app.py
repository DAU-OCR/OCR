from flask import Flask, request, jsonify, send_file, send_from_directory
import os
import io
import re
import cv2
import numpy as np
import pandas as pd
from PIL import Image as PILImage, ImageOps, ImageDraw, ImageFont
from werkzeug.utils import secure_filename
import easyocr
import torch
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Alignment
from datetime import datetime

# --- Pillow 호환성 설정: ANTIALIAS 지원 ---
try:
    PILImage.ANTIALIAS = PILImage.Resampling.LANCZOS
except AttributeError:
    pass

# --- Flask 애플리케이션 설정 ---
UPLOAD_FOLDER = 'uploads'                       # 업로드된 파일 저장 디렉토리
CROP_FOLDER = os.path.join(UPLOAD_FOLDER, 'cropped')   # 전처리된 번호판 이미지 저장 디렉토리
VISUAL_FOLDER = os.path.join(UPLOAD_FOLDER, 'visual')  # 시각화 이미지 저장 디렉토리
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CROP_FOLDER, exist_ok=True)
os.makedirs(VISUAL_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# In-memory 저장소
records = []

# --- CORS 헤더 추가 함수 ---
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response

# --- 헬스체크 엔드포인트 ---
@app.route('/', methods=['GET'])
def index():
    return '서버가 실행 중입니다', 200

# --- 업로드된 파일 서빙 엔드포인트 ---
@app.route('/uploads/<path:filename>', methods=['GET'])
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# --- YOLOv5 모델 초기화 ---
yolo_model = torch.hub.load(
    'yolov5', 'custom',
    path='custom_weights/best.pt',
    source='local',
    trust_repo=True
)
yolo_model.to('cpu')
yolo_model.conf = 0.5  # 검출 최소 신뢰도 설정

# --- EasyOCR Reader 초기화 (한글 OCR) ---
reader = easyocr.Reader(
    lang_list=['ko'],
    gpu=False,
    model_storage_directory='custom_weights',
    user_network_directory='custom_weights',
    detect_network='craft',          # 문자 영역 검출 네트워크
    recog_network='korean_g2',       # 사용자 훈련 모델
    download_enabled=False           # 이미 로컬에 모델이 있어야 함
)

# 한글 폰트 로드 함수
font_paths = [
    'C:/Windows/Fonts/malgun.ttf',
    '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
]
def get_korean_font(size=20):
    for path in font_paths:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()

font = get_korean_font()

# --- bolt-like circle 제거 함수 ---
def remove_bolt_like_circles(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT,
        dp=1.2, minDist=20,
        param1=50, param2=30,
        minRadius=5, maxRadius=15
    )
    if circles is not None:
        for (x, y, r) in np.uint16(np.around(circles[0])):
            cv2.circle(image, (x, y), r + 2, (255, 255, 255), -1)
    return image

# --- 번호판 전처리 함수 ---
def preprocess_plate(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    lab = cv2.merge((l2, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    gaussian = cv2.GaussianBlur(img, (0,0), 3)
    img = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)
    img = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
    img = remove_bolt_like_circles(img)
    return img

# --- 기울기(skew) 보정 함수 ---
def correct_skew(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
    if lines is None:
        return image
    angles = [np.degrees(theta - np.pi/2) for _, theta in lines[:,0]]
    angle = np.median(angles)
    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

# --- 작은 글씨 필터 함수 ---
def filter_small_text(results, min_w=20, min_h=20):
    filtered = []
    for bbox, text, conf in results:
        tl, tr, br, bl = bbox
        w = np.linalg.norm(np.array(tr) - np.array(tl))
        h = np.linalg.norm(np.array(tl) - np.array(bl))
        if w >= min_w and h >= min_h:
            filtered.append((bbox, text, conf))
    return filtered

# --- 번호판 검출 및 ROI 추출 함수 ---
def detect_plate(img):
    dets = yolo_model(img).xyxy[0].cpu().numpy()
    if dets.size == 0:
        return None
    idx = np.argmax(dets[:,4])
    x1, y1, x2, y2, *_ = dets[idx]
    return img[int(y1):int(y2), int(x1):int(x2)], (int(x1),int(y1),int(x2),int(y2))

# --- 이미지 업로드 및 처리 엔드포인트 ---
@app.route('/upload', methods=['POST', 'OPTIONS'])
def upload():
    if request.method == 'OPTIONS':
        return '', 200
    files = request.files.getlist('images')
    if not files:
        return jsonify({'error': '이미지를 제공해주세요'}), 400

    for file in files:
        fn = secure_filename(file.filename)
        save_path = os.path.join(UPLOAD_FOLDER, fn)
        file.save(save_path)
        img = cv2.imread(save_path)
        if img is None:
            records.append({'image': f'/uploads/{fn}', 'raw': '', 'plate': '', 'matched': False})
            continue

        detect = detect_plate(img)
        if not detect:
            records.append({'image': f'/uploads/{fn}', 'raw': '', 'plate': '', 'matched': False})
            continue

        roi, (x1, y1, x2, y2) = detect
        proc = preprocess_plate(roi)
        skewed = correct_skew(proc)

        ocr_res = reader.readtext(skewed)
        ocr_res = filter_small_text(ocr_res)
        raw_text = ''.join(t for (_, t, _) in sorted(ocr_res, key=lambda x: x[2], reverse=True))

        pattern = r"(\d{2,3}[가-힣]\d{4})"
        m = re.search(pattern, raw_text)
        plate = m.group(1) if m else ''
        matched = bool(m)
        if not matched:
            for L in (7, 8):
                for i in range(len(raw_text) - L + 1):
                    sub = raw_text[i:i+L]
                    if re.fullmatch(pattern, sub):
                        plate, matched = sub, True
                        break
                if matched:
                    break

        # 크롭된 번호판 이미지 저장 (내부 처리용)
        crop_name = f"cropped_{fn}"
        cv2.imwrite(os.path.join(CROP_FOLDER, crop_name), skewed)

        # 시각화 이미지 저장
        vis = img.copy()
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        vis_pil = PILImage.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(vis_pil)
        for i, (_, text, conf) in enumerate(ocr_res):
            y = y1 - 25 - i*25 if y1 > 50 else y2 + 25 + i*25
            draw.text((x1, y), f"{text} ({conf:.2f})", font=font, fill=(255, 0, 0))
        vis_name = f"vis_{fn}"
        vis_pil.save(os.path.join(VISUAL_FOLDER, vis_name))

        # 레코드 저장 (원본, 크롭, 시각화 경로 포함)
        records.append({
            'image': f'/uploads/{fn}',
            'crop': f'/uploads/cropped/{crop_name}',
            'visual': f'/uploads/visual/{vis_name}',
            'raw': raw_text,
            'plate': plate,
            'matched': matched
        })

    return jsonify({'status': 'ok'}), 200

# --- 결과 조회 엔드포인트 ---
@app.route('/results', methods=['GET'])
def get_results():
    return jsonify(records), 200

# --- Excel 다운로드 엔드포인트 (매칭된 데이터만, 원본 사진 삽입) ---
@app.route('/download', methods=['GET'])
def download_excel():
    data = [r for r in records if r['matched']]
    if not data:
        return jsonify({'error': '매칭된 데이터가 없습니다'}), 400

    # DataFrame 구성
    df = pd.DataFrame(data)
    df.insert(0, '연번', range(1, len(df) + 1))
    df = df[['연번', 'raw', 'plate']]
    df.columns = ['연번', '차량번호1', '차량번호2']
    df.insert(1, '차량사진', '')

    # Excel에 삽입
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='plates')
        ws = writer.sheets['plates']
        ws.column_dimensions['B'].width = 20
        ws.column_dimensions['C'].width = 25
        ws.column_dimensions['D'].width = 25

        for idx, r in enumerate(data, start=2):
            # 원본 업로드 이미지 삽입
            img_path = os.path.join(UPLOAD_FOLDER, os.path.basename(r['image']))
            if os.path.exists(img_path):
                pil_img = PILImage.open(img_path)
                pil_img = ImageOps.exif_transpose(pil_img)
                pil_img.thumbnail((140, 140))
                bio = io.BytesIO()
                pil_img.save(bio, format='PNG')
                bio.seek(0)
                img = XLImage(bio)
                img.width, img.height = 160, 140
                ws.add_image(img, f'B{idx}')
                ws.row_dimensions[idx].height = 105
            for col in [1, 3, 4]:
                ws.cell(row=idx, column=col).alignment = Alignment(horizontal='center', vertical='center')

    buffer.seek(0)
    fname_param = request.args.get('filename')
    today = datetime.now().strftime('%Y-%m-%d')
    base = secure_filename(fname_param) if fname_param else f"{today} 차량번호판인식"
    download_name = f"{base}.xlsx"

    return send_file(
        buffer,
        as_attachment=True,
        download_name=download_name,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

# --- 결과 초기화 엔드포인트 ---
@app.route('/reset', methods=['POST'])
def reset():
    records.clear()
    return jsonify({'status': 'ok'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
