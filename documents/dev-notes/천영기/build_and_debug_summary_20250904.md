# Electron 앱 빌드 및 디버깅 과정 요약 (2025-09-04)

이 문서는 Electron 앱의 최종 빌드를 완성하고, 그 과정에서 발생한 여러 문제를 해결한 과정을 기록합니다.

## 최종 목표
- React 프론트엔드와 Python 백엔드(`ocr_server.exe`)를 포함하는 단일 실행 파일(`ocr-web.exe`) 형태의 배포판 생성.
- 사용자가 앱을 실행했을 때, 별도의 설정 없이 모든 기능이 올바르게 동작하도록 보장.

---

## 문제 해결 과정

### 1. 흰 화면 및 JS/CSS 로드 실패
- **문제**: 빌드 후 앱 실행 시, 흰 화면만 나타나고 개발자 도구에 `net::ERR_FILE_NOT_FOUND` 오류가 표시됨. (JS, CSS 파일을 찾지 못함)
- **원인**: `vite.config.js`의 `base` 설정이 없어, `index.html` 내의 리소스 경로가 절대 경로(`/`)로 빌드됨. Electron의 `file://` 프로토콜에서는 상대 경로(`./`)가 필요함.
- **해결**: `ocr-web/vite.config.js` 파일에 `base: './'` 설정을 추가하여 모든 리소스 경로를 상대 경로로 변경.

### 2. React Router 경로 오류
- **문제**: 앱 화면은 표시되나, "No routes matched location" 오류가 발생하며 페이지 내용을 찾지 못함.
- **원인**: `react-router-dom`이 웹서버 환경용 `BrowserRouter`로 설정되어 있어, 로컬 파일 경로를 웹 URL처럼 해석하려고 시도함.
- **해결**: `ocr-web/src/App.jsx` 파일에서 라우터를 `BrowserRouter`에서 Electron 환경에 적합한 `HashRouter`로 변경.

### 3. 빌드 시 "Access is denied" 오류
- **문제**: `npm run electron:build` 실행 중, `d3dcompiler_47.dll` 파일 삭제에 실패하며 "Access is denied" 오류 발생.
- **원인**: 이전에 실행했던 `ocr-web.exe` 프로세스가 완전히 종료되지 않고 백그라운드에 남아 파일을 점유하고 있었음.
- **해결**: 사용자가 직접 작업 관리자(`Ctrl+Shift+Esc`)를 통해 실행 중인 `ocr-web.exe` 프로세스를 강제 종료.

### 4. API 요청 및 이미지 로드 실패 (가장 중요)
- **문제**: 앱 실행 후 이미지 업로드 시, API 요청(예: `/upload`, `/results`)과 아이콘 및 결과 이미지 로드에 `net::ERR_FILE_NOT_FOUND` 오류 발생.
- **원인**:
    1.  **API 요청**: `vite.config.js`의 `proxy` 설정은 개발 환경에만 적용되므로, 빌드된 앱에서는 API 요청이 백엔드 서버(`http://localhost:5000`)가 아닌 로컬 파일 경로로 잘못 전송됨.
    2.  **이미지 경로**: 정적 아이콘과 서버로부터 받는 동적 이미지 모두 파일 경로 문제로 로드되지 않음.
- **해결**:
    - `UploadPage.jsx`, `ResultsPage.jsx`, `FailedPage.jsx` 세 개의 파일에서 API를 요청하는 모든 부분을 찾아, `http://localhost:5000`으로 시작하는 전체 URL로 수정.
    - 정적 아이콘(`upload.png`, `download.png`)의 경로를 상대 경로(`./icons/...`)로 수정.
    - 서버로부터 받는 동적 이미지 경로 앞에 `http://localhost:5000`을 추가하여 전체 URL을 만들어주도록 수정.

### 5. 백엔드 서버 CMD 창 표시
- **문제**: 앱 실행 시, 백엔드 서버인 `ocr_server.exe`의 검은색 CMD 창이 함께 나타남.
- **원인**: Electron의 메인 프로세스 파일(`electron.cjs`)에서 서버를 실행할 때, 창을 숨기는 옵션이 없었음.
- **해결**: `ocr-web/electron.cjs` 파일의 `spawn` 함수에 `windowsHide: true` 옵션을 추가하여 백엔드 서버가 보이지 않는 백그라운드에서 실행되도록 조치.

---

## 최종 상태
위의 모든 디버깅 과정을 거쳐, 현재 애플리케이션은 다음과 같이 정상적으로 동작합니다.
- `ocr-web.exe` 실행 시 별도의 CMD 창 없이 메인 화면이 나타남.
- 이미지 업로드, OCR 처리, 결과 확인, 엑셀 다운로드 등 모든 기능이 올바르게 작동.
