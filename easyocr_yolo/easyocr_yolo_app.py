from flask import Flask, request, jsonify
import cv2
import numpy as np
from PIL import Image
# Pillow 호환성
Image.ANTIALIAS = Image.Resampling.LANCZOS
import easyocr
import torch

app = Flask(__name__)

# CORS 처리
@app.after_request
def add_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response

# 헬스체크 엔드포인트
@app.route('/', methods=['GET'])
def index():
    return 'Server is running', 200

# EasyOCR Reader 초기화 (한국어 custom 모델)
reader = easyocr.Reader(
    ['ko'],  # 한국어 인식
    gpu=False,
    model_storage_directory='custom_weights',
    user_network_directory='custom_weights',
    detect_network='craft',
    recog_network='finetuned'
)

# YOLOv5 모델 로드 via torch.hub (로컬 레포)
# server/yolov5 디렉터리에 복제된 hubconf.py 참조
# custom_weights/best.pt 파일을 custom_weights 폴더에 두세요
import torch

yolo_model = torch.hub.load(
    'yolov5',        # 로컬 경로: 서버 프로젝트 내 yolov5 폴더
    'custom',        # custom model 호출
    path='custom_weights/best.pt',
    source='local',  # 로컬에서 불러옴
    trust_repo=True  # 로컬 repo 신뢰
)
# CPU 모드 설정
yolo_model.to('cpu')
# 신뢰도 임계값 설정
yolo_model.conf = 0.5

yolo_model.conf = 0.5

# 번호판 검출 함수
def detect_plate(img):
    """
    torch.hub YOLOv5 모델로 번호판 영역 검출 후 ROI 반환
    """
    results = yolo_model(img)  # DetectMultiBackend instance
    det = results.xyxy[0].cpu().numpy()  # [[x1,y1,x2,y2,conf,class]]
    if det.size == 0:
        return None
    idx = np.argmax(det[:,4])
    x1, y1, x2, y2, *_ = det[idx]
    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
    return img[y1:y2, x1:x2]

# 업로드 및 OCR 엔드포인트
@app.route('/upload', methods=['POST', 'OPTIONS'])
def upload():
    # 프리플라이트 요청 처리
    if request.method == 'OPTIONS':
        return '', 200

    # 이미지 파일 수신
    file = request.files.get('image')
    if not file:
        return jsonify({'error': 'No image provided'}), 400

    # OpenCV로 이미지 디코딩
    data = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'plate': '인식실패'}), 200

    # 1) YOLO로 번호판 영역 검출
    plate_roi = detect_plate(img)
    if plate_roi is None:
        return jsonify({'plate': '인식실패'}), 200

    # 2) EasyOCR로 텍스트 추출
    results = reader.readtext(plate_roi)
    if not results:
        return jsonify({'plate': '인식실패'}), 200

    # 신뢰도 순으로 정렬 후 텍스트 결합
    texts = [text for (_, text, _) in sorted(results, key=lambda x: x[2], reverse=True)]
    plate_text = ''.join(texts)

    return jsonify({'plate': plate_text}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
