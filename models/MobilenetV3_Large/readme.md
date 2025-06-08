# 📘 트러블슈팅 기록: MobileNetV3 Large (aug 0.5, with pretrained model)

## 🧪 실험 정보

- **Backbone**: MobileNetV3 Large
- **Augmentation**: `aug_prob = 0.5`
- **Pretrained Model**: 사용 (`korean_PP-OCRv3_rec_train`)
- **Epoch**: 150
- **Optimizer**: AdamW + LinearWarmupCosine
- **OCR 구조**: CRNN + CTC

---

## 📍 현상 요약

- 학습 초반 `acc = 0`, `edit distance ≒ 0.15`에서 수렴
- **15,000 step** 즈음부터 미세한 향상 시작
- 빠르게 수렴하며 성능 개선 여지 적음
- 최종 `eval/acc ≒ 0.6` 수준에서 멈춤

---

## 🔍 문제 분석

- MobileNetV3 Large는 연산량이 적은 구조로, **복잡한 한글 패턴**에 대한 학습력이 부족
- 특히 **한글로 시작하는 번호판**이나 **지명이 포함된 번호판**에서 인식 성능이 낮음
- 증강 비율을 낮추었지만, **모델 용량 자체의 한계**가 성능 정체에 큰 영향을 준 것으로 보임

---

## 🔁 개선 방안 및 다음 실험 방향

- **ResNet34 / ResNet50** 기반 구조로 변경해 학습
- PaddleOCR 내 **SAR**, **ViT** 계열 아키텍처 실험
- 또는 아래 외부 OCR 프레임워크도 고려:
  - [EasyOCR](https://github.com/JaidedAI/EasyOCR)
  - [ClovaOCR](https://clova.ai)
  - Custom CRNN + Attention 구조 직접 구현

---

> 해당 결과는 한글 번호판 인식 최적화를 위한 실험의 일부입니다.