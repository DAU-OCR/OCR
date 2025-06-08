import sys
import os
import cv2
import torch
from pathlib import Path
import easyocr

# YOLOv5 루트 디렉터리 등록
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# YOLO 관련 모듈 import
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import (
    non_max_suppression, scale_boxes, check_img_size, cv2
)
from utils.torch_utils import select_device

# 경로 설정
weights = 'weights/best.pt'
source = 'images/korean'
save_crop_dir = 'cropped'
save_txt_dir = 'ocr_results'
os.makedirs(save_crop_dir, exist_ok=True)
os.makedirs(save_txt_dir, exist_ok=True)

# 디바이스 선택 (GPU 사용 안함)
device = select_device('cpu')

# 모델 로드
model = DetectMultiBackend(weights, device=device)
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size(640, s=stride)

# EasyOCR 초기화
reader = easyocr.Reader(['ko'], gpu=False)

# 이미지 로더
dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)

# 이미지별 추론 루프
for path, img, im0s, vid_cap, s in dataset:
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0
    if img.ndim == 3:
        img = img[None]

    # YOLO 추론
    pred = model(img, augment=False, visualize=False)
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

    p = Path(path)
    im0 = im0s.copy()

    for i, det in enumerate(pred):
        if len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0.shape).round()

            for j, (*xyxy, conf, cls) in enumerate(det):
                x1, y1, x2, y2 = map(int, xyxy)
                cropped = im0[y1:y2, x1:x2]
                crop_path = os.path.join(save_crop_dir, f"{p.stem}_plate{j + 1}.jpg")
                cv2.imwrite(crop_path, cropped)

                ocr_result = reader.readtext(cropped)
                print(f"\n🔍 파일: {p.name} | 번호판 {j + 1}")
                result_txt = os.path.join(save_txt_dir, f"{p.stem}_plate{j + 1}.txt")

                with open(result_txt, 'w', encoding='utf-8') as f:
                    if ocr_result:
                        for (_, text, conf_score) in ocr_result:
                            line = f"{text} (신뢰도: {conf_score:.2f})"
                            print(f"➡️ 인식 결과: {line}")
                            f.write(line + '\n')
                    else:
                        print("❌ 인식 실패")
                        f.write("❌ 인식 실패\n")
