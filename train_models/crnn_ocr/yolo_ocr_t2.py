# ───────────────────────────────────────────────────────────────────────────────
# 파일명: yolo_ocr_t2.py
#   • 
# ───────────────────────────────────────────────────────────────────────────────

import sys
import os
import cv2
import torch
import numpy as np
from pathlib import Path

# ──────────── (1) project_root 및 yolov5 모듈 검색 경로 추가 ────────────
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root / "yolov5"))
sys.path.insert(1, str(project_root))
# ────────────────────────────────────────────────────────────────────────────

# 이제 yolov5 내부 모듈을 안전하게 import 할 수 있습니다.
from PIL import Image  # CSV 버전에서는 ImageDraw, ImageFont는 사용하지 않지만, PIL을 미리 import 해 두면 혹시 쓸 때 편합니다.
from torchvision import transforms

from yolov5.models.common        import DetectMultiBackend
from yolov5.utils.augmentations import letterbox
from yolov5.utils.general       import non_max_suppression, scale_boxes

from ocr.model.crnn         import CRNN
from ocr.utils.label_encoder import LabelEncoder

import pandas as pd 
import re

# ───────────────────────────────────────────────────────────────────────────────
# 아래부터는 “전처리 및 OCR 관련 유틸 함수” (원본 yolo_ocr_CSV.py 내용 그대로)
# ───────────────────────────────────────────────────────────────────────────────

def remove_bolt_like_circles(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                               param1=50, param2=30, minRadius=5, maxRadius=15)
    if circles is not None:
        circles = np.uint16(np.around(circles[0, :]))
        for (x, y, r) in circles:
            cv2.circle(image, (x, y), r + 2, (255, 255, 255), -1)
    return image

def correct_skew(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(bw > 0))
    if len(coords) < 10:
        return image
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def preprocess_enhanced(image):
    img = image.copy()
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l2 = cv2.createCLAHE(2.0, (8, 8)).apply(l)
    img = cv2.merge((l2, a, b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    gaussian = cv2.GaussianBlur(img, (0, 0), 3)
    img = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)
    img = cv2.bilateralFilter(img, 9, 75, 75)
    img = remove_bolt_like_circles(img)
    return img

def resize_with_padding(image, target_size=(128, 32)):
    h, w = image.shape[:2]
    target_w, target_h = target_size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))
    padded = np.ones((target_h, target_w), dtype=np.uint8) * 255
    pad_left = (target_w - new_w) // 2
    pad_top = (target_h - new_h) // 2
    padded[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    return padded

def preprocess_image(image, target_width=128, target_height=32):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = resize_with_padding(image, (target_width, target_height))
    image = image.astype(np.float32) / 255.0
    image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    return image

# Plate pattern regex for post-processing
#  optional 2글자 지역명(한글) + 2~3숫자 + 1글자 한글 + 3~4숫자
plate_regex = re.compile(r"^(?:[가-힣]{2})?\d{2,3}[가-힣]\d{3,4}$")
# 전역 beam 디코더 초기화
_temp_enc = LabelEncoder()
try:
    from ctcdecode import CTCBeamDecoder
    _temp_enc = LabelEncoder()
    beam_decoder = CTCBeamDecoder(
        _temp_enc.get_charset(),
        beam_width=10,
        blank_id=_temp_enc.blank_idx,
        log_probs_input=True
    )
    USE_BEAM = True
except ImportError:
    print("⚠️ ctcdecode 모듈 미설치: Beam Search 대신 Greedy만 사용합니다.")
    USE_BEAM = False

def decode_prediction(preds, label_encoder):
    preds = preds.permute(1, 0, 2)
    log_probs = torch.nn.functional.log_softmax(preds, dim=2)
    _, out_best = log_probs.max(2)
    out_best = out_best.transpose(1, 0).contiguous().view(-1)
    pred_std = label_encoder.decode_ctc_standard(out_best.cpu().numpy())
    # 1) Greedy
    _, out_best = log_probs.max(2)
    out_best = out_best.transpose(1, 0).contiguous().view(-1)
    greedy = label_encoder.decode_ctc_standard(out_best.cpu().numpy())

    # 2) Beam Search (가능한 경우에만)
    if USE_BEAM:
        beam_results, beam_scores, _, out_lens = beam_decoder.decode(log_probs.cpu().numpy())
        beam = label_encoder.decode_ctc_standard(beam_results[0][0][:out_lens[0][0]])
        final_raw = beam if beam_scores[0][0] > 0 else greedy
    else:
        final_raw = greedy
    text = final_raw.replace('B','8').replace('I','1')
    m = plate_regex.match(text)
    return m.group(0) if m else text

# ———————————————————————————————————————————————
def ocr_with_ensemble(crop_img, crnn_model, label_encoder, device):
    best_text, best_score = None, -1e9
    for a in (-10, 0, 10):
        # 1) 기울기 보정
        deskewed = correct_skew(crop_img)
        # 2) 앵글 회전
        (h,w) = deskewed.shape[:2]
        M = cv2.getRotationMatrix2D((w/2,h/2), a, 1.0)
        rotated = cv2.warpAffine(deskewed, M, (w,h),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)

        # 3) OCR
        input_t = preprocess_image(rotated).to(device)
        with torch.no_grad():
            preds = crnn_model(input_t)
        logp = torch.nn.functional.log_softmax(preds.permute(1,0,2), dim=2)
        score = logp.max(2)[0].sum().item()
        text = decode_prediction(preds, label_encoder)

        if score > best_score:
            best_score, best_text = score, text
    return best_text

def crnn_ocr(image, crnn_model, label_encoder, device):
    input_tensor = preprocess_image(image).to(device)
    with torch.no_grad():
        preds = crnn_model(input_tensor)
    text = decode_prediction(preds, label_encoder)
    return text

# CSV 버전에서는 글자를 화면에 뿌리지는 않으므로 get_korean_font()는 생략 가능.
# 원래 yolo_ocr_CSV.py 에 포함되어 있었지만, CSV만 생성하므로 아래는 굳이 호출되지 않습니다.
def get_korean_font(size=20):
    font_paths = ['C:/Windows/Fonts/malgun.ttf', '/usr/share/fonts/truetype/nanum/NanumGothic.ttf']
    for path in font_paths:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()

def load_crnn_model(model_path, num_classes, device):
    model = CRNN(img_height=32, num_channels=1, num_classes=num_classes, hidden_size=512)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def parse_ground_truth(filename):
    """
    파일명 예: '01beo0101.jpg' → '01버0101' 처럼 roman→한글 매핑
    필요에 따라 mapping dict를 확장하세요.
    """
    mapping = {
        'beo': '버',
        'ga': '가',
        'na': '나',
        'da': '다',
        # … 필요에 따라 추가
    }
    stem = Path(filename).stem
    for rom, kor in mapping.items():
        stem = stem.replace(rom, kor)
    return stem

# ───────────────────────────────────────────────────────────────────────────────
# main() 함수: yolo_crnn_ocr_integrated.py와 동일한 형태로 작성
# ───────────────────────────────────────────────────────────────────────────────

def main():
    # • YOLO 및 CRNN 모델 경로
    yolo_model_path = 'bestModel/yoloBestModel.pt'
    crnn_model_path = 'bestModel/ocrBestModel_142.pth'
    # • 테스트할 이미지 폴더 (여기에 있는 모든 이미지에 대해 OCR을 반복 수행)
    test_image_dir = 'images/test_images_kr'
    # • 결과 CSV를 저장할 폴더
    output_dir = 'images/ocr_results'
    os.makedirs(output_dir, exist_ok=True)

    # • LabelEncoder 및 CRNN, YOLO 모델 로드
    label_encoder = LabelEncoder()
    charset = label_encoder.get_charset()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    yolo_model = DetectMultiBackend(yolo_model_path, device=device)
    crnn_model = load_crnn_model(crnn_model_path, num_classes=len(charset)+1, device=device)

    num_runs = 50  # OCR을 반복 수행할 횟수
    # { ground_truth (문자열) : [run1_pred, run2_pred, …, run50_pred] }
    results = {}

    # ────────────────────────────────────────────────────────────────────────────
    # 1) 테스트 이미지 루프
    # ────────────────────────────────────────────────────────────────────────────
    for filename in os.listdir(test_image_dir):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        image_path = os.path.join(test_image_dir, filename)
        img = cv2.imread(image_path)
        if img is None:
            continue

        # 2) ground-truth 파싱 (파일명 → 한글 텍스트)
        gt = parse_ground_truth(filename)

        # 3) YOLO detection (한 번만 수행)
        enhanced = preprocess_enhanced(img)
        img640 = letterbox(enhanced, new_shape=640)[0]
        img640 = img640[:, :, ::-1].transpose(2, 0, 1)
        img640 = np.ascontiguousarray(img640, dtype=np.float32) / 255.0
        tensor640 = torch.from_numpy(img640).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = yolo_model(tensor640)
        pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)[0]
        if pred is None or not len(pred):
            # 박스가 없으면 OCR 시도하지 않고, 빈 문자열 50번 저장
            results[gt] = [''] * num_runs
            continue

        # 4) 첫 번째 바운딩박스만 사용 (필요시 여러 바운딩박스로 확장 가능)
        x1, y1, x2, y2 = map(int, scale_boxes(tensor640.shape[2:], pred[:, :4], img.shape).round()[0])
        crop = img[y1:y2, x1:x2]
        crop = correct_skew(crop)

        # 5) OCR을 num_runs 번 반복해서 예측 결과 수집
        # run_preds = []
        # for _ in range(num_runs):
        #     text = crnn_ocr(crop, crnn_model, label_encoder, device)
        #     run_preds.append(text)
        # results[gt] = run_preds

        # 5) 앙상블 + 빔 탐색을 이용한 OCR 반복
        run_preds = []
        for _ in range(num_runs):
            text = ocr_with_ensemble(crop, crnn_model, label_encoder, device)
            run_preds.append(text)
        results[gt] = run_preds

    # ────────────────────────────────────────────────────────────────────────────
    # 6) DataFrame 생성 및 정답률 계산
    # ────────────────────────────────────────────────────────────────────────────
    # 인덱스 = ground_truth, 컬럼 = '1', '2', …, '50'
    df = pd.DataFrame.from_dict(
        results,
        orient='index',
        columns=[str(i+1) for i in range(num_runs)]
    )

    # 각 컬럼마다 ground_truth와 예측이 일치한 비율을 계산
    accuracies = []
    total = len(df)
    for col in df.columns:
        correct = (df[col] == df.index).sum()
        accuracies.append(f"{correct/total*100:.2f}%")
    # 맨 아래에 '정답률' 행 추가
    df.loc['정답률'] = accuracies

    # CSV로 저장 (utf-8-sig로 BOM 추가)
    csv_path = os.path.join(output_dir, 'ocr_v1.4.2_results_type_2.csv')
    df.to_csv(csv_path,
              index_label='', 
              encoding='utf-8-sig')
    print(f"✔ OCR 결과를 CSV(utf-8-sig)로 저장했습니다: {csv_path}")


if __name__ == '__main__':
    main()
