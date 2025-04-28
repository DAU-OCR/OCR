import os
import cv2
import numpy as np
import paddle
from omegaconf import OmegaConf
from PaddleOCR.ppocr.modeling.architectures import build_model
from PaddleOCR.ppocr.data import create_operators
from PaddleOCR.ppocr.postprocess import build_post_process
from PaddleOCR.ppocr.utils.save_load import load_model

# 1. Config 로드
config_path = "./config.yml"
cfg = OmegaConf.load(config_path)

# 2. Recognition 전처리 파이프라인 (RecResizeImg 사용)
# RecResizeImg 내부에서 크기 조정 및 정규화, 채널 변환(C×H×W)을 처리하므로,
# 추가 NormalizeImage/ToCHWImage는 불필요합니다。
transforms = [
    {'DecodeImage': {'img_mode': 'BGR'}},
    {'RecResizeImg': {'image_shape': [3, 32, 320]}},
    {'KeepKeys': {'keep_keys': ['image', 'valid_ratio']}}
]
transform_ops = create_operators(transforms, global_config=cfg['Global'])

# 3. 모델 빌드 및 가중치 로드
model = build_model(cfg['Architecture'])
load_model(cfg, model)
model.eval()

# 4. Post-process 객체 생성
post_process = build_post_process(cfg['PostProcess'], global_config=cfg['Global'])

# 5. 디버깅: character dict 길이 및 out_channels 확인
if hasattr(post_process, 'character'):
    print(f"Character dict length: {len(post_process.character)} (blank 포함 여부 확인)")
print(f"Head out_channels: {cfg['Architecture']['Head']['out_channels']}")


def infer(image_path, model, post_process, ops):
    assert os.path.exists(image_path), f"이미지를 찾을 수 없음: {image_path}"

    # 이미지 바이트 로드
    with open(image_path, 'rb') as f:
        img_bytes = f.read()
    data = {'image': img_bytes}

    # 전처리
    for i, op in enumerate(ops):
        data = op(data)
        while isinstance(data, list):
            if len(data) == 0:
                raise ValueError(f"Transform #{i} 반환값이 빈 리스트입니다.")
            data = data[0]
        if isinstance(data, np.ndarray):
            data = {'image': data}
        if not isinstance(data, dict):
            raise TypeError(f"[Transform #{i}] 예상 dict, 실제 {type(data)}")

    # 디버깅: 전처리된 텐서 정보 출력
    img_arr = data['image']
    print(f"🖼 전처리 후 배열 shape: {img_arr.shape}, min: {img_arr.min()}, max: {img_arr.max()}")

    img_tensor = paddle.to_tensor(img_arr[np.newaxis, :], dtype='float32')

    preds = model(img_tensor)
    if isinstance(preds, (list, tuple)):
        print("📦 preds tuple shapes:", [p.shape for p in preds])
        preds = preds[0]
    else:
        print("📦 preds shape:", preds.shape)

    # 후처리: preds만 입력
    results = post_process(preds)
    print("🔍 결과:", results)
    print("📊 Raw index:", paddle.argmax(preds, axis=2).numpy())


if __name__ == '__main__':
    test_image = './test_image8.jpg'
    infer(test_image, model, post_process, transform_ops)
