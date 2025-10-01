import cv2
import torch
import numpy as np

# crnn.py가 동일한 디렉토리에 있다고 가정합니다.
from crnn import CRNN

# -------------------- 전처리 및 OCR 유틸리티 함수 --------------------

def remove_bolt_like_circles(image):
    """번호판 이미지에서 볼트와 유사한 원형 노이즈를 제거합니다."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                               param1=50, param2=30, minRadius=5, maxRadius=15)
    if circles is not None:
        circles = np.uint16(np.around(circles[0, :]))
        for (x, y, r) in circles:
            # 원을 흰색으로 채웁니다.
            cv2.circle(image, (x, y), r + 2, (255, 255, 255), -1)
    return image

def correct_skew(image):
    """번호판 이미지의 기울어짐을 보정합니다."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    
    if lines is None:
        return image  # 선이 감지되지 않으면 원본 이미지를 반환합니다.

    angles = [np.degrees(theta - np.pi / 2) for rho, theta in lines[:, 0]]
    median_angle = np.median(angles)
    
    # 과도한 회전을 방지합니다.
    if abs(median_angle) > 15:
        return image

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    
    # 검은색 테두리가 생기지 않도록 배경과 유사한 색으로 채웁니다.
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def preprocess_enhanced(image):
    """OCR을 위해 일련의 필터를 적용하여 이미지를 개선합니다."""
    img = image.copy()
    
    # 1. CLAHE (명암 대비 제한 적응 히스토그램 평활화)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    img = cv2.merge((l2, a, b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    
    # 2. 샤프닝
    gaussian = cv2.GaussianBlur(img, (0, 0), 3)
    img = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)
    
    # 3. 노이즈 감소
    img = cv2.bilateralFilter(img, 9, 75, 75)
    
    # 4. 볼트 제거
    img = remove_bolt_like_circles(img)
    
    return img

def resize_with_padding(image, target_size=(128, 32)):
    """가로세로 비율을 유지하면서 패딩을 추가하여 이미지 크기를 조절합니다."""
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # 스케일링 비율 계산
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # 이미지 리사이즈
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # 흰색 배경의 새 이미지 생성
    padded = np.ones((target_h, target_w), dtype=np.uint8) * 255
    
    # 패딩 계산
    pad_left = (target_w - new_w) // 2
    pad_top = (target_h - new_h) // 2
    
    # 리사이즈된 이미지를 중앙에 붙여넣기
    padded[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    
    return padded

def preprocess_for_crnn(image, target_width=128, target_height=32):
    """이미지를 CRNN 모델에 적합한 텐서로 변환합니다."""
    # 그레이스케일로 변환
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    # 리사이즈 및 패딩
    image = resize_with_padding(image, (target_width, target_height))
    
    # 정규화 및 텐서 변환
    image = image.astype(np.float32) / 255.0
    image = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)  # 배치 및 채널 차원 추가
    
    return image

def decode_prediction(preds, label_encoder):
    """CRNN 모델의 원시 출력을 텍스트로 디코딩하고 신뢰도 점수를 계산합니다."""
    # [T, B, C] -> [B, T, C]
    preds = preds.permute(1, 0, 2)
    
    # 신뢰도 점수 계산
    probs = torch.nn.functional.softmax(preds, dim=2)
    max_probs, _ = probs.max(dim=2)
    confidence = max_probs.mean().item()

    # 최적의 경로 탐색
    out = torch.nn.functional.log_softmax(preds, dim=2)
    _, out_best = out.max(2)
    
    out_best = out_best.transpose(1, 0).contiguous().view(-1)
    
    # 레이블 인코더를 사용하여 디코딩
    pred_text = label_encoder.decode_ctc_standard(out_best.cpu().numpy())
    
    return pred_text, confidence

def run_crnn_ocr(image, crnn_model, label_encoder, device):
    """
    CRNN 모델을 사용하여 주어진 이미지에 대해 전체 OCR 프로세스를 실행합니다.
    전체 전처리 파이프라인을 포함합니다.
    """
    # 1. 전체 전처리
    processed_img = preprocess_enhanced(image)
    processed_img = correct_skew(processed_img)

    # 2. CRNN을 위한 텐서 준비
    input_tensor = preprocess_for_crnn(processed_img).to(device)
    
    # 3. 추론 실행
    with torch.no_grad():
        preds = crnn_model(input_tensor)
        
    # 4. 예측 디코딩
    text, confidence = decode_prediction(preds, label_encoder)
    
    return text, confidence

def load_crnn_model(model_path, num_classes, device):
    """.pth 파일에서 CRNN 모델을 로드합니다."""
    model = CRNN(img_height=32, num_channels=1, num_classes=num_classes, hidden_size=512)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model
