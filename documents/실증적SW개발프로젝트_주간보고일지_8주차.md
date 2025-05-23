# 📝 실증적 SW 개발 프로젝트 주간보고서 (8주차)

- **작성일**: 2025년 4월 28일  
- **팀명**: OCR  
- **활동일시**: 2025년 4월 28일  
- **장소**: 동아대학교 Complex room  
- **참석자**: 김수윤, 김륜영, 천영기, 이승민, 서인범  
- **특이사항**: 시험기간  

---

## ✅ 이번 주 진행사항

### 1. 개발 내용

#### 🔹 Web
- UI 구조 설계 (2%)  
- UX 구상 (20%)  
- 보안 정책 검토 (0%)  
- UI 개발 (2%)  
- React 구조 파악 및 파일 리팩토링  

#### 🔹 Paddle OCR
- Paddle OCR 구조 리서치 (70%)  
- 모델 학습 (ACC: 80%)  
  → 인식률 향상을 위한 실험 중  

#### 🔹 CRNN 방식의 OCR
- 기본 구조 조사 (50%)  
  → 성능 향상용 정보 탐색 중  
- CRNN 모델 훈련 및 검증 (ACC: 94%)  
  → 실제 사용 중 정확도 확인  
- 모델 개선 (50%)  
  → 추가적인 개선 방안 모색  

#### 🔹 DataSet
- YOLO 포맷 변환 스크립트 (0%)  
- 학습셋/검증셋 분리 (0%)  

#### 🔹 이미지 전처리
- 노이즈 제거 및 대비 향상 (10%)  
- 다양한 조도/해상도 이미지 테스트 케이스 확보 (0%)  
- 번호판 영역 크롭 후 grayscale 변환 (10%)  

---

## 👥 팀원별 활동내용

- **김수윤 (팀장)**  
  - Paddle OCR 이용 모델 학습 및 트러블슈팅  
  - 현재 학습 방법의 문제점 파악 및 새로운 학습 방법 고안  
  - Notion에 각종 트러블슈팅 기록  

- **김륜영 (Paddle OCR 담당)**  
  - OCR 모델 학습 및 실험  
  - 불필요 이미지 제거 및 라벨링 정리  
  - 학습 전략 Notion 정리  

- **천영기 (CRNN OCR 담당)**  
  - CRNN 모델 개선 방향 고민  
  - 초기 버전 모델 실제 적용 작업 진행 및 정확도 확인  

- **이승민 (시스템 연결 및 테스트)**  
  - PaddleOCR 모델 테스트 진행  
  - 커스텀 PaddleOCR 모델로 번호판 인식 테스트  
  - 문자 인식 실패 원인 분석 및 데이터 특성 확인  
  - CRNN 모델 테스트 환경 구축  

- **서인범 (Web 담당)**  
  - React 사용법 숙지  
  - Streamlit 한계 파악 후 React 전환  
  - UX 구상 및 자료 조사  
  - 사이트 로고 변경 및 구조 설계  

---

## 🔜 다음 주 계획

### Web  
- UI 구조 설계  
- UX 구상  
- 보안 정책 검토  
- UI 개발  

### Paddle OCR  
- 구조 리서치  
- 모델 학습 – ACC 향상 목표  
- 모델 인식 테스트  

### CRNN + CTC OCR  
- 성능 향상 관련 자료 조사  
- 실사용 성능 확인 및 개선  

### DataSet  
- YOLO 포맷 변환 스크립트 작성  
- 학습셋/검증셋 분리  

### 이미지 전처리  
- 테스트 이미지 확보  
- 다양한 이미지 전처리 실험  

---

## 📌 주요 결과물

- CRNN 기반 OCR 모델 정확도 94% 유지  
- PaddleOCR 기반 모델 학습 진척  
- React 기반 웹 전환 기초 구조 완료  
