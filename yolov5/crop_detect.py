import torch
import cv2
import os
import easyocr

# 모델 로드 (YOLOv5 공식 구조로 학습된 best.pt)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='weights/best.pt', force_reload=True)

# OCR Reader (한글)
reader = easyocr.Reader(['ko'], gpu=False)

# 입력 이미지 폴더
image_folder = 'images/korean'

# 저장 폴더 생성
cropped_dir = 'cropped'
ocr_output_dir = 'ocr_results'
os.makedirs(cropped_dir, exist_ok=True)
os.makedirs(ocr_output_dir, exist_ok=True)

# 이미지 순회
for filename in os.listdir(image_folder):
    if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    image_path = os.path.join(image_folder, filename)
    image = cv2.imread(image_path)

    if image is None:
        print(f"[오류] 이미지 로드 실패: {filename}")
        continue

    # YOLOv5 추론
    results = model(image)

    for i, (*box, conf, cls) in enumerate(results.xyxy[0]):  # x1, y1, x2, y2
        x1, y1, x2, y2 = map(int, box)
        cropped_plate = image[y1:y2, x1:x2]

        # Crop 이미지 저장
        crop_name = f"{os.path.splitext(filename)[0]}_plate{i+1}.jpg"
        cv2.imwrite(os.path.join(cropped_dir, crop_name), cropped_plate)

        # EasyOCR 수행
        ocr_result = reader.readtext(cropped_plate)

        # 결과 출력
        print(f"\n🔍 파일: {filename} | 번호판 {i + 1}")

        # OCR 결과 저장
        ocr_filename = f"{os.path.splitext(filename)[0]}_plate{i+1}_ocr.txt"
        ocr_filepath = os.path.join(ocr_output_dir, ocr_filename)
        with open(ocr_filepath, 'w', encoding='utf-8') as f:
            if ocr_result:
                for (bbox, text, conf_score) in ocr_result:
                    line = f"{text} (신뢰도: {conf_score:.2f})"
                    print(f"➡️ 인식 결과: {line}")
                    f.write(line + '\n')
            else:
                print("❌ 인식 실패")
                f.write("❌ 인식 실패\n")
