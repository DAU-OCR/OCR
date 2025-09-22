# OCR 프로젝트 실행 가이드

이 프로젝트는 Python으로 작성된 백엔드 서버와 React로 작성된 프론트엔드 웹 애플리케이션으로 구성됩니다. 서비스를 완전히 실행하려면 두 부분을 각각 구동해야 합니다.

---

## 1. 백엔드 서버 실행 (Python)

백엔드 서버는 이미지 처리, OCR, API 제공을 담당합니다.

### 단계

1.  **`server` 디렉토리로 이동합니다.**
    ```bash
    cd server
    ```

2.  **Python 가상 환경을 생성하고 활성화합니다.** (권장)
    ```bash
    # 가상환경 생성
    python -m venv venv

    # Windows에서 활성화
    .\venv\Scripts\activate

    # macOS / Linux에서 활성화
    # source venv/bin/activate
    ```

3.  **필요한 라이브러리를 설치합니다.**
    아래 명령어를 실행하여 모든 종속성을 설치합니다.
    ```bash
    pip install flask werkzeug opencv-python torch torchvision torchaudio easyocr numpy pandas openpyxl gitpython matplotlib pillow psutil pyyaml requests scipy thop tqdm ultralytics seaborn
    ```

4.  **백엔드 서버를 시작합니다.**
    ```bash
    python app.py
    ```
    서버가 시작되면 터미널에 `Running on http://...` 메시지가 나타나며, 프론트엔드의 요청을 받을 준비가 된 상태입니다.

---

## 2. 프론트엔드 웹 실행 (React)

프론트엔드는 사용자가 이미지를 업로드하고 결과를 확인할 수 있는 웹 인터페이스를 제공합니다.

**주의:** 백엔드 서버를 먼저 실행한 후, **새로운 터미널**을 열어 프론트엔드를 실행하세요.

### 단계

1.  **`ocr-web` 디렉토리로 이동합니다.**
    ```bash
    cd ocr-web
    ```

2.  **필요한 라이브러리를 설치합니다.** (최초 실행 시 또는 의존성 변경 시 한 번만)
    ```bash
    npm install
    ```

3.  **프론트엔드 개발 서버를 시작합니다.**
    ```bash
    npm run dev
    ```
    서버가 시작되면 터미널에 `Local: http://localhost:xxxx` 와 같은 주소가 나타납니다.

---

## 3. 서비스 이용

1.  백엔드와 프론트엔드 서버가 모두 실행 중인지 확인합니다.
2.  웹 브라우저를 열고 프론트엔드 개발 서버가 알려준 주소(예: `http://localhost:5173`)로 접속하여 서비스를 이용합니다.
