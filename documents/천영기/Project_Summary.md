# OCR 프로젝트 종합 요약 보고서

이 문서는 차량 번호판 OCR 프로젝트의 전반적인 개발 과정, 주요 기술적 난관 및 해결책, 그리고 CRNN 모델 학습 및 개선 과정을 요약합니다.

## 1. 프로젝트 개요 및 목표

*   **핵심 목표**: 차량 번호판 이미지를 인식(OCR)하고, 그 결과를 관리하며, JSON/Excel 파일로 다운로드하고, 키보드 탐색 및 수정이 가능한 애플리케이션 개발.
*   **최종 배포 목표**: React/Electron 프론트엔드와 Python/Flask 백엔드를 하나의 실행 파일(`.exe`)로 패키징하여 배포.
*   **기술 스택**:
    *   **프론트엔드**: React.js, Electron
    *   **백엔드**: Python, Flask, PyInstaller
    *   **OCR 모델**: YOLOv5 (탐지), EasyOCR, CRNN (인식)

## 2. 백엔드 개발 (Python OCR 서버)

### 2.1. PyInstaller 빌드 안정화 및 문제 해결

개발 환경에서 정상 작동하던 Python 서버가 PyInstaller로 `.exe` 빌드 시 다양한 런타임 오류를 발생시켰습니다.

*   **주요 문제**: `ModuleNotFoundError`, `AttributeError`, `_pickle.UnpicklingError`, CUDA/디바이스 관련 오류.
*   **원인**: 복잡한 라이브러리 의존성 충돌(특히 `opencv-python` 버전), PyInstaller의 패키징 누락(특히 `numpy`, `torch`, `ultralytics` 하위 모듈), `torch.hub.load`의 `force_reload=True` 옵션 충돌, PyTorch의 보안 정책 변경.
*   **해결**:
    *   불필요한 패키지(`torchaudio`, `craft-text-detector`) 제거 및 핵심 라이브러리(torch, numpy, opencv-python, easyocr)의 호환되는 버전 명시적 재설치.
    *   `.spec` 파일 대폭 수정: `PyInstaller.utils.hooks.collect_all`을 사용하여 모든 핵심 라이브러리의 코드, 데이터, 바이너리를 강제로 포함. `yolov5` 등 로컬 모듈도 `datas`에 추가.
    *   YOLOv5 `AutoShape` 비정상 동작 해결을 위해 이미지 전처리 및 후처리(NMS) 수동 구현.
    *   CPU만 사용하도록 `os.environ['CUDA_VISIBLE_DEVICES'] = ''` 및 `device='cpu'` 명시.
    *   `_pickle.UnpicklingError` 해결을 위해 `weights_only=False` 옵션 추가.
    *   UX 개선: 백엔드 CMD 창 숨기기, 불필요한 로그 제거.

### 2.2. OCR 성능 및 기능 개선

*   **결과 선택 로직**: 초기 신뢰도 기반 선택의 한계를 극복하기 위해 하이브리드 로직 도입.
    1.  **다수결 원칙**: 3개 모델 중 2개 이상이 동일한 유효 번호판 제시 시 최우선 채택.
    2.  **모델1(EasyOCR-ko) 우선 원칙**: 다수결이 없으면, 모델1 결과가 유효할 경우 채택.
    3.  **가중치 비교**: 위 조건에 해당하지 않으면, CRNN 모델 신뢰도에 0.9 페널티 적용 후 가장 높은 점수 채택.
*   **엑셀 저장 시 이미지 회전 문제**: 이미지 EXIF 메타데이터로 인한 회전 문제 `Pillow`의 `ImageOps.exif_transpose`로 자동 보정. 이미지 크기 리사이즈 로직 개선 (가로 너비 기준 비율 유지).

## 3. 프론트엔드 개발 (React Electron 앱)

### 3.1. Electron 빌드 및 패키징 문제 해결

*   **주요 문제**: `ocr_server.exe`가 최종 패키지에 누락, `makensis.exe` 빌드 실패, `electron.cjs` 경로 문제, `app.asar` 파일 잠금.
*   **원인**: `electron-builder`의 `extraFiles` 설정 미흡, `makensis.exe`의 시스템 환경 의존적 오류, `electron.cjs`의 코드 손상 및 잘못된 경로 로직, 프로세스 종료 문제.
*   **해결**:
    *   `package.json`의 `extraFiles` 설정 수정하여 `ocr_server.exe`를 최종 빌드 폴더 루트에 복사.
    *   `electron.cjs` 파일 전체 재작성: 손상된 코드 정리, `app.isPackaged`로 개발/프로덕션 환경 구분하여 `ocr_server.exe` 경로 올바르게 찾도록 수정.
    *   `makensis.exe` 빌드 실패 우회를 위해 `win.target`을 `["zip"]`으로 변경하여 압축 파일 형태의 무설치 버전으로 빌드.
    *   앱 종료 시 `taskkill` 명령어를 `will-quit` 이벤트에 추가하여 Python 프로세스 강제 종료.

### 3.2. 패키징 후 런타임 오류 및 UX 개선

*   **주요 문제**: 흰 화면 및 JS/CSS 로드 실패, React Router 경로 오류, API 요청 및 이미지 로드 실패, 백엔드 CMD 창 표시, "인식 실패" 텍스트 동작, 모달 이미지 표시, Excel 다운로드 실패.
*   **원인**: `vite.config.js` `base` 설정 누락, `BrowserRouter` 사용, API 요청 및 이미지 경로의 절대/상대 경로 문제, `windowsHide` 옵션 누락, `ResultsPage.jsx` 로직 오류.
*   **해결**:
    *   `vite.config.js`에 `base: './'` 설정 추가.
    *   `BrowserRouter`를 `HashRouter`로 변경.
    *   모든 API 요청 및 동적 이미지 경로에 `http://localhost:5000` 전체 URL 명시. 정적 아이콘 경로도 상대 경로로 수정.
    *   `electron.cjs`의 `spawn` 함수에 `windowsHide: true` 옵션 추가.
    *   `ResultsPage.jsx` 및 `FailedPage.jsx`의 `plate` 값 초기화 및 입력 필드 로직 수정.
    *   `ResultsPage.jsx` 및 `FailedPage.jsx`에서 모달 이미지 `src`에 `http://localhost:5000` 접두사 추가.
    *   `app.py`에 Excel 다운로드 엔드포인트 구현 및 `ResultsPage.jsx`에 `axios.defaults.baseURL` 설정.
*   **UX 개선**: 이미지 상세 팝업 내 키보드 방향키(↑, ↓)로 이전/다음 이미지 탐색 기능 추가.

## 4. CRNN 모델 학습 및 개선 과정

CRNN 모델의 성능을 높이기 위해 체계적인 가설 설정과 검증 과정을 거쳤습니다.

*   **초기 문제**: 높은 검증 정확도(94%) 대비 낮은 실제 성능(63%) (과적합).
*   **해결 전략**: 데이터 중심 개선(Data-Centric Improvement) 접근법 채택.
*   **주요 가설 및 해결**:
    1.  **데이터셋 결함**: `verify_dataset.py`로 무결성 검증, 문제 없음 확인.
    2.  **모델 표현력 부족**: `hidden_size`를 256에서 512로 증대, `ReduceLROnPlateau`를 `OneCycleLR`로 변경하여 '미니 한글 데이터셋'에서 96.39% 정확도 달성.
    3.  **불완전한 문자셋**: `VIRTUAL_CHARSET`에 한글 자음/모음(자모) 누락 발견, `JAMO_CHARS` 추가하여 완전한 문자셋 구축.
    4.  **데이터 불균형**: `analyze_dataset.py`로 분석, 완벽하게 균형 잡힌 데이터셋임을 확인.
    5.  **문자셋 간 간섭 현상**: 한글, 영문, 숫자 등 특성이 다른 문자셋 동시 학습 시 시각적 유사성으로 인한 혼란(예: 'B'와 '비', '0'과 'ㅇ'). '한글 전용 데이터셋' 생성으로 간섭 요인(영문) 제거 시도.
    6.  **부적절한 학습률**: '한글 전용 데이터셋' 학습 시 손실 정체 현상 발생, `max_lr`을 0.001에서 0.0001로 낮춰 안정적인 학습 유도.

## 5. 최종 결론

프로젝트는 Python 백엔드와 Electron 프론트엔드를 성공적으로 통합하여 모든 핵심 기능을 구현했습니다. PyInstaller 및 Electron 빌드 과정에서 발생한 수많은 복합적인 문제들을 해결하여 안정적인 두 개의 실행 파일(`ocr-web.exe`, `ocr_server.exe`)을 완성했습니다. 다만, Electron 앱이 백엔드 서버를 자식 프로세스로 자동 실행할 때 발생하는 `forrtl: error (200)`과 같은 낮은 수준의 라이브러리 호환성 충돌은 코드 수정으로 해결하기 어려워, 두 프로그램을 개별적으로 실행하는 방식으로 최종 배포되었습니다. CRNN 모델 학습 과정에서도 데이터셋 및 모델 아키텍처에 대한 체계적인 분석과 개선을 통해 성능 향상을 위한 기반을 마련했습니다.
