# OCR 프로젝트 문제 해결 보고서

## 1. 프로젝트 개요
본 프로젝트는 React 기반의 Electron 프론트엔드와 Flask 기반의 Python 백엔드로 구성된 차량 번호판 OCR 애플리케이션입니다. 최종 목표는 두 구성 요소를 단일 실행 파일로 번들링하여 배포하는 것이었습니다.

## 2. 초기 문제 및 해결 과정 요약

### 2.1. Python 백엔드 (`ocr_server.exe`) 빌드 및 실행 문제
*   **문제 1: `AttributeError: 'NoneType' object has no attribute 'write'`**
    *   **진단**: `torch.hub.load` 호출 시 `force_reload=True`로 인해 모델을 다운로드하려 했으나, PyInstaller 환경에서 `sys.stderr`가 `None`이어서 발생.
    *   **해결**: `server/app.py`에서 `torch.hub.load`의 `force_reload=True` 옵션 제거.
*   **문제 2: `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0!`**
    *   **진단**: `easyocr.Reader`가 `gpu=False`로 설정되었음에도 불구하고, EasyOCR 내부 또는 사용자 정의 모델에서 텐서가 CPU와 GPU에 동시에 존재하여 발생.
    *   **해결**: `server/app.py`에 `os.environ['CUDA_VISIBLE_DEVICES'] = ''`를 추가하여 PyTorch가 CPU만 사용하도록 강제.
*   **문제 3: `AssertionError: Invalid device id` (YOLOv5 로딩 시)**
    *   **진단**: `CUDA_VISIBLE_DEVICES=''` 설정으로 GPU가 숨겨진 상태에서 YOLOv5가 여전히 CUDA 장치를 선택하려 시도하여 발생.
    *   **해결**: `server/app.py`에서 `yolo_model` 로딩 시 `torch.hub.load`에 `device='cpu'`를 명시적으로 추가.
*   **문제 4: `ocr_server.exe` 콘솔 출력 없음 (디버깅 어려움)**
    *   **진단**: PyInstaller 빌드 시 `--windowed` 플래그로 인해 콘솔 출력이 억제됨.
    *   **해결**: 디버깅을 위해 `--windowed` 플래그 없이 `ocr_server.exe`를 재빌드하여 출력 확인.

### 2.2. Electron 앱 패키징 및 런타임 문제

*   **문제 1: `makensis.exe` 빌드 실패 (`ERR_ELECTRON_BUILDER_CANNOT_EXECUTE`, `failed creating mmap`)**
    *   **진단**: `electron-builder`가 NSIS 설치 프로그램을 생성하는 과정에서 `makensis.exe`가 메모리 매핑에 실패. 백신, 캐시 손상, 파일 잠금 등 다양한 원인 추정. `win.target`을 `dir`, `portable`, `["portable"]`, `["zip"]` 등으로 변경했음에도 NSIS 빌드가 계속 시도되며 실패하는 비정상적인 동작 발생.
    *   **해결**: (보고서 작성 시점까지의 대화 내용 기준) 이 문제는 `electron-builder`의 특정 시스템 환경에서의 고질적인 문제로 판단되며, 최종적으로 `win.target: ["zip"]` 설정으로 빌드를 시도하여 NSIS 과정을 우회하는 방향으로 진행됨. 사용자 확인 결과, 최종적으로 모든 문제가 해결되었다고 보고됨. (이는 `zip` 빌드가 성공했음을 의미)
*   **문제 2: 프론트엔드 이미지 로드 실패 (`net::ERR_FILE_NOT_FOUND`)**
    *   **진단**: `App.jsx`, `UploadPage.jsx`, `ResultsPage.jsx`, `FailedPage.jsx` 등에서 이미지 경로가 `/icons/...`와 같은 루트 상대 경로로 되어 있어, Electron 패키징 환경에서 `file://` 프로토콜과 충돌하여 이미지를 찾지 못함. `public/icons`에 파일이 존재했음에도 발생.
    *   **해결**: 모든 이미지 참조 경로를 `./icons/...`와 같은 현재 디렉토리 상대 경로로 수정.
*   **문제 3: Python 백엔드 연결 실패 (`net::ERR_CONNECTION_REFUSED`)**
    *   **진단**: Electron 앱 시작 시 Python 백엔드가 완전히 준비되기 전에 프론트엔드가 요청을 보내 연결이 거부됨 (타이밍 문제). 또한, `electron-builder`의 `extraFiles` 설정에 Python 백엔드의 모델/가중치 디렉토리(`custom_weights`, `custom_weights_easyOCR`, `yolov5`)가 누락되어 백엔드가 시작 후 바로 충돌.
    *   **해결**:
        1.  `ocr-web/package.json`의 `extraFiles`에 누락된 모델/가중치 디렉토리 추가.
        2.  프론트엔드에 "OCR 준비 중" 메시지 및 버튼 비활성화 기능을 구현하여 백엔드가 완전히 준비될 때까지 사용자 상호작용을 지연.
*   **문제 4: "인식 실패" 텍스트 동작 문제**
    *   **진단**: 입력 필드에 "인식 실패"가 초기 표시되지 않거나, 지우면 다시 나타나는 문제.
    *   **해결**: `ResultsPage.jsx` 및 `FailedPage.jsx`에서 `plate` 값 초기화 로직(`useEffect`) 및 입력 필드의 `value` 속성(`value={r.plate}`)을 조정하여, 초기에는 "인식 실패"를 표시하고 사용자가 지우면 빈 상태를 유지하며, 저장 시 빈 값을 "인식 실패"로 처리하도록 구현.
*   **문제 5: 모달(팝업) 이미지 표시 문제**
    *   **진단**: 모달 내 이미지가 `http://localhost:5000` 접두사 없이 상대 경로로 참조되어 `file://` 프로토콜 문제로 로드 실패.
    *   **해결**: `ResultsPage.jsx` 및 `FailedPage.jsx`에서 모달 이미지 `src`에 `http://localhost:5000` 접두사를 추가하여 절대 URL로 변경.
*   **문제 6: Excel 다운로드 실패**
    *   **진단**:
        1.  백엔드(`server/app.py`)에 `/download` 엔드포인트 구현 누락.
        2.  프론트엔드(`ResultsPage.jsx`)의 `axios` 인스턴스에 `baseURL`이 설정되지 않아 `file://` 프로토콜로 API 요청이 전송됨.
    *   **해결**:
        1.  `server/app.py`에 `records` 데이터를 기반으로 Excel 파일을 생성하고 반환하는 `/download` 엔드포인트 구현.
        2.  `ResultsPage.jsx`에 `axios.defaults.baseURL = 'http://localhost:5000';`를 설정하여 모든 API 요청이 올바른 백엔드 URL로 전송되도록 함.
*   **문제 7: `ocr_server.exe` 프로세스 종료 문제**
    *   **진단**: Electron 앱 종료 시 `pythonProcess.kill()`이 Windows에서 `ocr_server.exe`를 완전히 종료하지 못하고 프로세스가 남아있음.
    *   **해결**: `ocr-web/electron.cjs`에서 `app.on('will-quit')` 핸들러 내부에 `taskkill` 명령어를 사용하여 Python 프로세스를 강제 종료하도록 수정.
*   **문제 8: `ReferenceError: globalShortcut is not defined`**
    *   **진단**: `F12` 단축키 구현 시도 후 `globalShortcut` import는 제거되었으나, `globalShortcut.unregisterAll()` 호출이 `electron.cjs`에 남아있어 발생.
    *   **해결**: `globalShortcut.unregisterAll()` 라인 제거.
*   **문제 9: 아이콘 변경**
    *   **해결**: `ocr-web/package.json`의 `win.icon` 경로를 `public/icons/logo.png`로 변경.

## 3. 최종 상태
위의 모든 복잡한 디버깅 및 수정 과정을 거쳐, 현재 애플리케이션은 다음과 같이 정상적으로 동작합니다.
*   React 프론트엔드와 Flask Python 백엔드가 성공적으로 통합되었습니다.
*   `ocr-web.exe` 실행 시 별도의 CMD 창 없이 메인 화면이 나타나고, 백엔드도 함께 시작됩니다.
*   이미지 업로드, OCR 처리, 결과 확인, 엑셀 다운로드 등 모든 핵심 기능이 올바르게 작동합니다.
*   "OCR 준비 중" 메시지, 이미지 표시, 입력 필드 동작 등 사용자 경험 관련 문제들도 해결되었습니다.
*   최종 빌드는 `zip` 타겟으로 성공적으로 완료되었습니다.

## 4. 향후 개선 사항 (선택 사항)
*   `electron-builder`의 `makensis.exe` 문제에 대한 근본적인 원인 분석 및 해결 (시스템 환경 또는 `electron-builder` 버전 문제일 가능성). 현재는 `zip` 타겟으로 우회.
*   `package.json`의 `description` 및 `author` 필드 채우기 (빌드 로그 경고 제거).
*   `public/icons/logo.ico` 파일이 `logo.png` 대신 사용되도록 아이콘 파일 형식 최적화.
