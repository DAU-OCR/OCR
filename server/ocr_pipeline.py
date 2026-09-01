import os
import traceback
from datetime import datetime

import cv2
import numpy as np
import torch
from PIL import Image as PILImage

from config import UPLOAD_FOLDER, CROP_FOLDER, VISUAL_FOLDER, MIN_AREA, MIN_ASPECT_RATIO, resize1, resize2
from model_loader import (
    yolo_model, reader1, reader2, crnn_model, label_encoder, device,
    non_max_suppression, scale_boxes, letterbox, run_crnn_ocr,
)
from text_utils import get_filtered_ocr, apply_plate_selection_logic, is_valid_plate, VALID_HANGUL_CHARS
from image_utils import get_plate_corners, get_plate_corners_threshold, warp_perspective
from state import records


def detect_plate(image):
    # Preprocessing
    img0 = image.copy()
    img, ratio, (dw, dh) = letterbox(img0, 640, stride=32, auto=True)
    img = img.transpose((2, 0, 1))[::-1] # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0
    if len(img.shape) == 3:
        img = img[None]

    # Inference
    # pred = yolo_model(img, augment=False, visualize=False)    수정
    pred = yolo_model(img, augment=False ) # visualize 옵션 제거
    if isinstance(pred, (list, tuple)) and len(pred) in (2, 3): # yolov5 returns a tuple (pred, feature_maps) or (pred, protos, feature_maps)
        pred = pred[0]

    # NMS
    pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False, max_det=1000)

    detections = []
    for i, det in enumerate(pred): # per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], img0.shape).round()
            detections = det.cpu().numpy()

    if len(detections) == 0:
        return None

    filtered = []
    for *xyxy, conf, cls in detections.tolist():
        x1, y1, x2, y2 = map(int, xyxy)
        area = (x2 - x1) * (y2 - y1)
        ratio_bbox = (x2 - x1) / (y2 - y1 + 1e-5)
        if area > MIN_AREA and ratio_bbox > MIN_ASPECT_RATIO:
            filtered.append((area, (x1, y1, x2, y2)))

    if not filtered:
        return None

    _, (x1, y1, x2, y2) = max(filtered)
    return image[y1:y2, x1:x2], (x1, y1, x2, y2)

def process_file_and_record(f, fname, image):
    """
    OCR 처리 및 records에 결과 저장 로직을 담은 헬퍼 함수
    /upload 와 /upload-batch 에서 재사용
    """
    try:
        result = {
            'image': f'/uploads/{fname}',
            'timestamp': datetime.now().isoformat(),
            'matched': False
        }

        # 디버그 폴더 생성
        debug_subdir = os.path.join(UPLOAD_FOLDER, 'debug', os.path.splitext(fname)[0])
        os.makedirs(debug_subdir, exist_ok=True)

        detected = detect_plate(image)
        if not detected:
            records.append(result)
            return

        plate_img, (x1, y1, x2, y2) = detected

        # 디버그: YOLO 검출 영역 저장
        cv2.imwrite(os.path.join(debug_subdir, '1_detected_yolo.png'), plate_img)

        # 패딩 추가
        pad = 20
        x1 = max(x1 - pad, 0)
        y1 = max(y1 - pad, 0)
        x2 = min(x2 + pad, image.shape[1])
        y2 = min(y2 + pad, image.shape[0])
        plate_img = image[y1:y2, x1:x2]

        # 디버그: 패딩 후 이미지 저장
        cv2.imwrite(os.path.join(debug_subdir, '2_padded.png'), plate_img)

        # 디버그: 보정 전 이미지 저장
        cv2.imwrite(os.path.join(debug_subdir, '3_before_correction.png'), plate_img)

        # 보정 시도 1
        corners = get_plate_corners(plate_img, fname=os.path.splitext(fname)[0], save_debug=True, debug_dir=debug_subdir)
        if corners is None:
            corners = get_plate_corners_threshold(plate_img, fname=os.path.splitext(fname)[0], save_debug=True, debug_dir=debug_subdir)


        if corners is not None:
            # 디버그: 검출된 코너 시각화
            corner_img = plate_img.copy()
            cv2.polylines(corner_img, [np.int32(corners)], True, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(debug_subdir, '4_detected_corners.png'), corner_img)

            # 보정 수행
            warped_img = warp_perspective(plate_img, corners)

            # 디버그: 보정된 결과 이미지 저장
            cv2.imwrite(os.path.join(debug_subdir, '5_corrected_warped.png'), warped_img)

            # 보정된 이미지로 교체
            plate_img = warped_img

        # OCR 리사이즈 후 입력 저장
        t1_input = cv2.resize(plate_img, resize1)
        t2_input = cv2.resize(plate_img, resize2)
        cv2.imwrite(os.path.join(debug_subdir, '6_ocr_model1_input.png'), t1_input)
        cv2.imwrite(os.path.join(debug_subdir, '7_ocr_model2_input.png'), t2_input)

        # OCR 수행
        t1, c1 = get_filtered_ocr(reader1, plate_img, resize1)
        t2, c2 = get_filtered_ocr(reader2, plate_img, resize2)
        t3, c3 = run_crnn_ocr(plate_img, crnn_model, label_encoder, device)
        c3 = round(c3, 2)

        selected, reason = apply_plate_selection_logic(t1, c1, t2, c2, t3, c3, VALID_HANGUL_CHARS)
        matched = is_valid_plate(selected)

        if not matched:
            selected = '인식 실패'

        crop_name = f"crop_{fname}"
        vis_name = f"vis_{fname}"
        cv2.imwrite(os.path.join(CROP_FOLDER, crop_name), plate_img)
        PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).save(os.path.join(VISUAL_FOLDER, vis_name))

        # 개발자용 디버깅 이미지 저장: YOLO 박스 포함된 원본 이미지
        DEV_VISUAL_FOLDER = os.path.join(UPLOAD_FOLDER, 'dev_visual')
        os.makedirs(DEV_VISUAL_FOLDER, exist_ok=True)
        dev_vis_path = os.path.join(DEV_VISUAL_FOLDER, f'dev_{fname}')
        dev_img = image.copy()
        cv2.rectangle(dev_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        PILImage.fromarray(cv2.cvtColor(dev_img, cv2.COLOR_BGR2RGB)).save(dev_vis_path)

        result.update({
            'crop': f'/uploads/cropped/{crop_name}',
            'visual': f'/uploads/visual/{vis_name}',
            'text1': t1, 'conf1': c1,
            'text2': t2, 'conf2': c2,
            'text3': t3, 'conf3': c3,
            'plate': selected, 'reason': reason,
            'matched': matched
        })
        records.append(result)

    except Exception as e:
        print(f"Error processing file {fname}: {e}")
        traceback.print_exc()
        # 오류 발생 시 실패 결과만 기록하고 계속 진행
        records.append({
            'image': f'/uploads/{fname}',
            'timestamp': datetime.now().isoformat(),
            'matched': False,
            'plate': '처리 오류',
            'reason': f'Internal Error: {str(e)}'
        })
