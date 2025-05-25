import os
import lmdb
import cv2
import numpy as np
import shutil
from tqdm import tqdm

def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    return img is not None

def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k.encode(), v)

def createDataset(outputPath, imagePath, labelPath):
    print(f"📂 LMDB 생성 경로: {outputPath}")
    print(f"📂 이미지 폴더: {imagePath}")
    print(f"📄 라벨 파일: {labelPath}")

    # 기존 LMDB 폴더 제거 후 생성
    if os.path.exists(outputPath):
        shutil.rmtree(outputPath)
    os.makedirs(outputPath)

    # 라벨 파일 읽기
    with open(labelPath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # LMDB 환경 열기
    env = lmdb.open(outputPath, map_size=5 * 1024 * 1024 * 1024)  # 5GB 테스트용

    cache = {}
    cnt = 1

    for line in tqdm(lines):
        parts = line.strip().split(',')
        if len(parts) != 2:
            print(f"⚠ 잘못된 라벨 줄: {line.strip()}")
            continue
        img_name, label = parts
        img_fp = os.path.join(imagePath, img_name)
        print(f"🧪 처리 중: {img_fp}")

        if not os.path.exists(img_fp):
            print(f"❌ 이미지 없음: {img_fp}")
            continue

        with open(img_fp, 'rb') as f:
            imageBin = f.read()

        if not checkImageIsValid(imageBin):
            print(f"❌ 유효하지 않은 이미지: {img_fp}")
            continue

        imageKey = f'image-{cnt:09d}'
        labelKey = f'label-{cnt:09d}'
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode('utf-8')

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
        cnt += 1

    # 마지막 캐시 쓰기
    cache['num-samples'] = str(cnt - 1).encode()
    writeCache(env, cache)

    print(f"\n✅ 총 {cnt - 1}개 샘플 LMDB 저장 완료 → {outputPath}")

if __name__ == '__main__':
    output_path = 'C:/Users/HOME/Desktop/testyolo/lmdb/eval_lmdb_1'  # ✅ 새로운 경로로 설정
    image_folder = 'C:/Users/HOME/Desktop/testyolo/yolov5/train/evalImg'
    label_file = 'C:/Users/HOME/Desktop/testyolo/yolov5/train/evalLabel/eval_label.txt'
    createDataset(output_path, image_folder, label_file)
