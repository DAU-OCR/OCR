# 🚗 불법주차 차량 OCR 자동 분석 시스템

> 사하구청과의 협업을 통해 개발된, 청원 기반 불법주차 신고 사진을 자동 분석하고 차량 번호를 추출해 엑셀로 정리하는 로컬 웹 기반 프로그램입니다.

---

## 📌 프로젝트 개요

- **프로젝트명:** 불법주차 신고 이미지 자동 분석 시스템
- **주요 기능:**
  - 차량 번호판 영역 자동 탐지 (YOLOv5)
  - OCR 기반 번호판 텍스트 인식 (EasyOCR / PaddleOCR)
  - 가장 잘 보이는 사진 자동 선택 (OCR confidence 기반)
  - 결과를 엑셀로 자동 저장 (`연번`, `사진`, `차량번호1`, `차량번호2`)
  - Streamlit 기반 로컬 웹 UI (Windows 전용)
  - CPU/GPU 환경 자동 감지 및 설정

---

## 💻 실행 환경

- 운영체제: Windows 10 이상
- Python 3.8+
- GPU가 있을 경우 PyTorch + CUDA 사용 가능
- 인터넷 연결 불필요 (로컬 실행 기반)

---

## 🔧 사용 기술 스택

| 범주         | 기술                           |
|--------------|--------------------------------|
| 언어         | Python 3.8+                    |
| 웹 UI        | Streamlit                      |
| OCR          | EasyOCR, PaddleOCR (옵션)      |
| 객체 탐지    | YOLOv5                         |
| 이미지 처리  | OpenCV                         |
| 엑셀 저장    | pandas, openpyxl               |
| 하드웨어 감지| PyTorch (torch.cuda 지원 여부) |

---

## 🚀 설치 및 실행 방법

```bash
# 1. 가상환경 생성 (선택)
python -m venv venv
source venv/Scripts/activate

# 2. 필수 라이브러리 설치
pip install -r requirements.txt

# 3. 앱 실행
streamlit run app.py
```
---

## 📂 프로젝트 구조

  project_root/ ├── app.py # Streamlit 메인 앱 ├── detector/ # YOLOv5 기반 번호판 탐지 모듈 ├── ocr/ # EasyOCR 기반 텍스트 추출 모듈 ├── utils/ # 엑셀 저장, 이미지 정리, 보조 함수 등 ├── examples/ # 테스트용 이미지 샘플 └── requirements.txt # 설치 필요 패키지 목록

---

## 📈 기능 예시

📸 사진 여러 장 업로드 → 가장 선명한 사진 1장 자동 선택

🔍 차량 번호 자동 인식 → 두 개의 번호까지 추출

📊 엑셀 다운로드 → 연번/사진/차량번호1/차량번호2 자동 정리

⚙️ 하드웨어 상황에 따라 CPU/GPU 자동 전환

---

## 🙋 프로젝트 팀원

    김수윤 팀장

    김륜영

    이승민

    천영기

    서인범

---

## 🔍 참고 자료
Easy Korean License Plate Detector

AIHub 자동차 번호판 데이터셋

EasyOCR GitHub

YOLOv5 GitHub

Streamlit Docs

---

## 📜 라이선스
본 프로젝트는 MIT License 하에 배포됩니다.

