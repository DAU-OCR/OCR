import os
import sys

# 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
CROP_FOLDER = os.path.join(UPLOAD_FOLDER, 'cropped')
VISUAL_FOLDER = os.path.join(UPLOAD_FOLDER, 'visual')
WARPED_FOLDER = os.path.join(UPLOAD_FOLDER, 'warped')

# OCR 파이프라인 파라미터
MIN_AREA = 1400
MIN_ASPECT_RATIO = 1.0
resize1 = (100, 32)
resize2 = (200, 60)

# Flask 설정
JSON_AS_ASCII = False


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)


def ensure_directories():
    for d in [UPLOAD_FOLDER, CROP_FOLDER, VISUAL_FOLDER, WARPED_FOLDER]:
        os.makedirs(d, exist_ok=True)
