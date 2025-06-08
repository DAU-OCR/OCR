import os
import sys
import cv2
import torch
import re
from pathlib import Path
import easyocr

# YOLOv5 로컬 레포에서 import 경로 세팅
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
sys.path.append(str(ROOT))

from models.common import DetectMultiBackend
from utils.general import (non_max_suppression, scale_boxes, xyxy2xywh)
from utils.dataloaders import LoadImages
from utils.torch_utils import select_device

# 경로 설정
weights_path = 'weights/best.pt'
source_dir = 'images/korean'
save_crop_dir = 'cropped'
save_ocr_dir = 'ocr_results'
save_vis_dir = 'visualized'

# OCR 필터 조건
MIN_HEIGHT = 20            # 너무 작은 텍스트 박스 무시
MIN_CONFIDENCE = 0.5       # 신뢰도 낮은 결과 무시
TEXT_PATTERN = re.compile(r'[0-9가-힣]{2,}')  # 숫자+한글 위주 텍스트만 허용

# 디렉토리 생성
os.makedirs(save_crop_dir, exist_ok=True)
os.makedirs(save_ocr_dir, exist_ok=True)
os.makedirs(save_vis_dir, exist_ok=True)

# 디바이스 선택
device = select_device('cpu')

# 모델 로드
model = DetectMultiBackend(weights_path, device=device)
stride, names, pt = model.stride, model.names, model.pt
imgsz = 640

# OCR Reader
reader = easyocr.Reader(['ko'], gpu=False)

# 이미지 로딩
dataset = LoadImages(source_dir, img_size=imgsz, stride=stride, auto=pt)

# 추론 루프
for path, img, im0s, vid_cap, _ in dataset:
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img, augment=False, visualize=False)
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)

    for det in pred:
        p = Path(path)
        filename = p.name
        im0 = im0s.copy()

        if len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0.shape).round()

            for j, (*xyxy, conf, cls) in enumerate(det):
                x1, y1, x2, y2 = map(int, xyxy)
                cropped_img = im0[y1:y2, x1:x2]

                # Crop 저장
                crop_filename = f"{p.stem}_plate{j+1}.jpg"
                crop_path = os.path.join(save_crop_dir, crop_filename)
                cv2.imwrite(crop_path, cropped_img)

                # OCR 수행
                ocr_result = reader.readtext(cropped_img)

                # 시각화용 복사본
                vis_img = cropped_img.copy()

                # OCR 결과 출력/필터링/시각화
                ocr_txt_path = os.path.join(save_ocr_dir, f"{p.stem}_plate{j+1}_ocr.txt")
                with open(ocr_txt_path, 'w', encoding='utf-8') as f:
                    print(f"\n🔍 파일: {filename} | 번호판 {j + 1}")
                    for (bbox, text, conf_score) in ocr_result:
                        y_coords = [point[1] for point in bbox]
                        box_height = max(y_coords) - min(y_coords)

                        if box_height < MIN_HEIGHT or conf_score < MIN_CONFIDENCE:
                            continue
                        if not TEXT_PATTERN.match(text):
                            continue

                        # 결과 저장
                        result_str = f"{text} (신뢰도: {conf_score:.2f})"
                        print(f"➡️ 인식 결과: {result_str}")
                        f.write(result_str + '\n')

                        # 시각화 (bbox + 텍스트)
                        pts = [tuple(map(int, point)) for point in bbox]
                        for i in range(4):
                            cv2.line(vis_img, pts[i], pts[(i+1)%4], (0,255,0), 2)
                        cv2.putText(vis_img, text, pts[0], cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8, (0, 0, 255), 2, cv2.LINE_AA)

                # 시각화 이미지 저장
                vis_path = os.path.join(save_vis_dir, f"{p.stem}_plate{j+1}_vis.jpg")
                cv2.imwrite(vis_path, vis_img)
