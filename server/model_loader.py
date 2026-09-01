"""
모델 로딩 전담 모듈.

이 모듈을 import하는 시점에 YOLO/EasyOCR/CRNN 가중치가 전부 메모리에 로드된다
(원래 app.py에 있던 "모델 로딩" 섹션과 동일한 시점/순서를 유지).
무거운 라이브러리(torch, cv2, easyocr)를 이 모듈이 처음 import하므로,
CUDA/스레드 관련 환경변수는 그 import보다 먼저 설정해야 한다.

파일명 주의: 'models.py'로 짓지 말 것 - yolov5를 sys.path에 추가한 뒤
yolov5/hubconf.py가 자기 자신의 models/ 패키지를 `from models.common import ...`
로 import하는데, 이 파일이 'models'라는 이름을 먼저 sys.modules에 채가면
yolov5 쪽 import가 깨진다 (실제로 겪은 문제).
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys
import warnings

import cv2
import torch
import easyocr

from config import BASE_DIR, resource_path

warnings.filterwarnings("ignore", category=FutureWarning)
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

# yolov5 로컬 소스를 import 경로에 추가 (vendored copy)
sys.path.insert(0, os.path.join(BASE_DIR, 'yolov5'))
from utils.general import non_max_suppression, scale_boxes
from utils.augmentations import letterbox

# --- YOLO (번호판 탐지) ---
yolo_model = torch.hub.load(os.path.join(BASE_DIR, 'yolov5'), 'custom',
                             path=resource_path(os.path.join('custom_weights_easyOCR', 'best.pt')),
                             source='local',
                             device='cpu', # 이 부분을 추가합니다.
                             verbose=False)

# --- EasyOCR (문자 인식, 한글/영문 두 모델) ---
model_path = resource_path('custom_weights_easyOCR')
reader1 = easyocr.Reader(['ko'], gpu=False,
                         model_storage_directory=model_path,
                         user_network_directory=model_path,
                         recog_network='korean_g2', download_enabled=False)
reader2 = easyocr.Reader(['en'], gpu=False,
                         model_storage_directory=model_path,
                         user_network_directory=model_path,
                         recog_network='best_acc', download_enabled=False)

# --- CRNN 모델 로딩 ---
device = torch.device('cpu')
crnn_model_path = resource_path(os.path.join('CRNN_model', 'ocrBestModel_142.pth'))

# CRNN_model 폴더를 시스템 경로에 추가
sys.path.insert(0, resource_path('CRNN_model'))

from preprocess import load_crnn_model, run_crnn_ocr
from label_encoder import LabelEncoder

label_encoder = LabelEncoder()
charset = label_encoder.get_charset()
num_classes = len(charset) + 1 # CTC blank 토큰을 위해 +1

crnn_model = load_crnn_model(crnn_model_path, num_classes=num_classes, device=device)
# --- CRNN 모델 로딩 종료 ---
