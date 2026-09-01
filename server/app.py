import os
# torch/cv2/easyocr가 import되기 전에 설정돼야 하는 환경변수 (models.py에서도
# 동일하게 설정하지만, 이 파일을 직접 실행하는 진입점이므로 여기서도 가장 먼저 설정한다)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from flask import Flask
from flask_cors import CORS

from config import UPLOAD_FOLDER, JSON_AS_ASCII, ensure_directories

ensure_directories()

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ✅ [추가] JSON 응답 시 한글 유니코드 이스케이프를 비활성화 (가장 중요)
app.config['JSON_AS_ASCII'] = JSON_AS_ASCII

@app.after_request
def cors(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response

# routes를 import하는 시점에 모델(YOLO/EasyOCR/CRNN)이 전부 로드된다
# (routes -> ocr_pipeline -> model_loader 순으로 import 연쇄).
from routes import bp
app.register_blueprint(bp)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
