from flask import Flask, request, jsonify, send_file, send_from_directory
import os
import io
import re
import cv2
import numpy as np
import pandas as pd
from PIL import Image as PILImage, ImageOps
from werkzeug.utils import secure_filename
import easyocr
import torch
from openpyxl.drawing.image import Image as XLImage
from datetime import datetime

# Pillow 호환성 설정
PILImage.ANTIALIAS = PILImage.Resampling.LANCZOS

# 설정
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# In-memory 저장소
records = []

# CORS 처리
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response
app.after_request(add_cors_headers)

# 헬스체크 엔드포인트
@app.route('/', methods=['GET'])
def index():
    return 'Server is running', 200

# 업로드된 이미지 서빙
@app.route('/uploads/<filename>', methods=['GET'])
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# EasyOCR 및 Yolo 모델 초기화
reader = easyocr.Reader(
    ['ko'], gpu=False,
    model_storage_directory='custom_weights',
    user_network_directory='custom_weights',
    detect_network='craft', recog_network='finetuned'
)
yolo_model = torch.hub.load(
    'yolov5', 'custom', path='custom_weights/best.pt', source='local', trust_repo=True
)
yolo_model.to('cpu')
yolo_model.conf = 0.5

# 전처리 함수 (CLAHE, 언샵 마스크, Bilateral Filter)
def preprocess_plate(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    gaussian = cv2.GaussianBlur(gray, (9,9), 10.0)
    gray = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)
    gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    return gray

# 번호판 ROI 검출 함수
def detect_plate(img):
    results = yolo_model(img)
    det = results.xyxy[0].cpu().numpy()
    if det.size == 0:
        return None
    idx = np.argmax(det[:,4])
    x1, y1, x2, y2, *_ = det[idx]
    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
    return img[y1:y2, x1:x2]

# 이미지 업로드 및 OCR 처리
@app.route('/upload', methods=['POST','OPTIONS'])
def upload():
    if request.method == 'OPTIONS':
        return '', 200
    files = request.files.getlist('images')
    if not files:
        return jsonify({'error':'No images provided'}), 400
    for file in files:
        fn = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], fn)
        file.save(save_path)
        img = cv2.imread(save_path)
        if img is None:
            record = {'image': f'/uploads/{fn}', 'raw':'', 'plate':'', 'matched':False}
        else:
            roi = detect_plate(img)
            if roi is None:
                record = {'image': f'/uploads/{fn}', 'raw':'', 'plate':'', 'matched':False}
            else:
                proc = preprocess_plate(roi)
                ocr_res = reader.readtext(proc)
                raw = ''.join(t for (_,t,_) in sorted(ocr_res, key=lambda x: x[2], reverse=True))
                pattern = r"(\d{2,3}[가-힣]\d{4})"
                m = re.search(pattern, raw)
                matched = bool(m)
                plate = m.group(1) if m else ''
                if not matched:
                    for L in (7,8):
                        for i in range(len(raw)-L+1):
                            sub = raw[i:i+L]
                            if re.fullmatch(pattern, sub):
                                plate, matched = sub, True
                                break
                        if matched: break
                record = {'image': f'/uploads/{fn}', 'raw': raw, 'plate': plate, 'matched': matched}
        records.append(record)
    return jsonify({'status':'ok'}), 200

# 결과 조회 (모두 반환)
@app.route('/results', methods=['GET'])
def get_results():
    return jsonify(records), 200

# Excel 다운로드 (matched=True인 데이터만, 이미지 임베드, 기본 파일명에 날짜 포함)
@app.route('/download', methods=['GET'])
def download_excel():
    data = [r for r in records if r['matched']]
    if not data:
        return jsonify({'error':'No matched data to download'}), 400

    # DataFrame 생성 및 컬럼 구성
    df = pd.DataFrame(data)
    df.insert(0, '연번', range(1, len(df)+1))
    df = df[['연번','raw','plate']]
    df.columns = ['연번','차량번호1','차량번호2']
    df.insert(1, '차량사진', '')

    # Excel Writer
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='plates')
        ws = writer.sheets['plates']
        from openpyxl.styles import Alignment
        # 열 폭 설정
        ws.column_dimensions['B'].width = 20
        ws.column_dimensions['C'].width = 25
        ws.column_dimensions['D'].width = 25
        # 이미지 삽입 및 셀 설정
        for idx, row in enumerate(data, start=2):
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], os.path.basename(row['image']))
            if os.path.exists(img_path):
                pil_img = PILImage.open(img_path)
                pil_img = ImageOps.exif_transpose(pil_img)
                pil_img = pil_img.resize((140,140))
                bio = io.BytesIO()
                pil_img.save(bio, format='PNG')
                bio.seek(0)
                img = XLImage(bio)
                img.width, img.height = 160, 140
                ws.add_image(img, f'B{idx}')
                ws.row_dimensions[idx].height = 105
            # 가운데 정렬
            for col in [1,3,4]:
                ws.cell(row=idx, column=col).alignment = Alignment(horizontal='center', vertical='center')

    buffer.seek(0)
    # 파일명 결정
    fname_param = request.args.get('filename')
    if fname_param:
        base = secure_filename(fname_param)
    else:
        today = datetime.now().strftime('%Y-%m-%d')
        base = f"{today} 차량번호판인식"
    download_name = f"{base}.xlsx"

    return send_file(
        buffer,
        as_attachment=True,
        download_name=download_name,
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

# 결과 초기화
@app.route('/reset', methods=['POST'])
def reset():
    records.clear()
    return jsonify({'status':'ok'}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
