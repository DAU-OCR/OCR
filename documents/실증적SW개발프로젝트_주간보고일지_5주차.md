# 실증적 SW 개발 프로젝트 주간보고서 (5주차)

- **작성일**: 2025년 4월 7일  
- **팀명**: OCR  
- **활동일시**: 2025년 4월 6일  
- **장소**: 동아대학교 Complex room  
- **참석자**: 김수윤, 김륜영, 천영기, 이승민, 서인범  
- **특이사항**:
  - 사하구청 담당자 및 멘토 교수와의 초기 미팅 진행
  - AI Hub 차량 번호판 데이터셋 다운로드 완료 및 정제 작업 시작

---

## 이번 주 진행사항

### 1. 개발 내용

#### Web
- UI 구조 설계 (0%)
- UX 구상 (20%)
- 보안 정책 검토 (0%)
- UI 개발 (0%)

#### Easy OCR
- 라이브러리 설치 및 테스트 (100%)
- YOLO 연동 구조 설계 분석 (100%)

#### Paddle OCR
- 구조 리서치 (70%)
- 라이브러리 설치 및 테스트 (0%)
- 실행 환경 구성 (10%)
- 모델 학습 (1%)

#### CRNN 방식의 OCR
- 기본 구조 조사 (50%)
- 학습 데이터 준비 (10%)
- 모델 설계 및 CTC Loss 연결 (0%)
- 모델 훈련 및 검증 (0%)
- 모델 개선 (0%)

#### DataSet
- 데이터셋 다운로드 및 구조 분석 (100%)
- YOLO 포맷 변환 스크립트 (0%)
- 학습셋/검증셋 분리 (0%)

#### 이미지 전처리
- 노이즈 제거 및 대비 향상 (10%)
- 번호판 영역 크롭 및 grayscale 변환 (10%)
- 다양한 테스트 케이스 확보 (0%)

---

## 2. 팀원별 활동

### 김수윤 (팀장)
- 이미지 전처리 조사 및 구현  
  - 노이즈 제거 및 대비 향상 실험  
  - 번호판 영역 크롭  
  - Base Augmentations 조사
- 데이터셋 확인
- Streamlit 구성  
  - 기본 UI 및 기능 연동 시험
- 기존 모델 테스트 (EasyOCR)  
  - 정확도 측정

### 김륜영 (OCR 담당)
- 기존 모델 테스트 (EasyOCR)  
- 데이터셋 확인  
- Paddle OCR Label 사용법 학습  
- Paddle OCR 학습 방법 숙지 및 학습 진행  
- 기존 프로젝트 실행 방법 Notion에 정리  
- 기존 프로젝트 문제점 분석 및 개선 방안 Notion에 정리

### 천영기 (DataSet 담당)
- OCR 기초 학습 자료 수집  
- CRNN 방식의 OCR 데이터셋 구조 분석  
- 데이터셋 전처리 및 입력 양식 확인  
- CRNN 방식의 OCR 예제 코드 분석

### 이승민 (OCR 담당)
- 기존 모델 테스트 (EasyOCR)  
  - CPU 환경에서 테스트  
  - 정확도 측정  
  - 번호판 detect → OCR 추출까지 파이프라인 점검  
  - 이미지 전처리 유무 인식률 변화 확인  
- 데이터셋 확인  
- Streamlit 연동 테스트  
  - zip 파일 업로드 및 번호판 Excel 파일 추출 확인

### 서인범 (Web 담당)
- Streamlit 사용법 숙지  
  - 파일 업로드 및 업로드된 파일 Excel 파일로 추출 확인  
  - Streamlit에 CSS 파일 연동 확인  
- UX 구상  
  - 인터넷 페이지 자료 조사

---

## 다음 주 계획

### Web
- UI 구조 설계  
- UX 구상  
- 보안 정책 검토  
- UI 개발

### Easy OCR (폐기 예정)
- 기본 기능 확인 및 구조 설계 분석

### Paddle OCR
- 구조 리서치 및 모델 학습 지속  
- 테스트 준비

### CRNN + CTC OCR
- 기본 구조 구현 및 학습 데이터 정의  
- 초기 코드 구조화

### DataSet
- YOLO 포맷 변환 스크립트 작성  
- 학습셋/검증셋 분리

### 이미지 전처리
- 노이즈 제거 및 대비 향상 지속  
- 다양한 테스트 케이스 확보  
- 번호판 영역 grayscale 변환
