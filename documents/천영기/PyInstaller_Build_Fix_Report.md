# PyInstaller 빌드 문제 해결 보고서

## 1. 문제 개요

개발 환경에서는 정상적으로 동작하던 Python 백엔드 서버를 `PyInstaller`를 사용해 단일 실행 파일(`.exe`)로 빌드하는 과정에서, 실행 시 `ModuleNotFoundError` 등 다양한 런타임 오류가 발생하는 문제가 발생했습니다. 이 보고서는 해당 문제들의 원인을 진단하고 해결하는 전체 과정을 기록합니다.

---

## 2. 핵심 원인 분석

문제의 근원은 크게 두 가지였습니다.

### 2.1. 복잡한 라이브러리 의존성 충돌 (Dependency Hell)
- **초기 문제:** 사용자의 Python 환경에 설치된 여러 라이브러리(`torchaudio`, `ultralytics`, `craft-text-detector`)들이 서로 다른 버전의 `opencv-python`을 요구하는 등, 의존성이 복잡하게 꼬여 있었습니다.
- **원인:** 이로 인해 `pip`이 모든 라이브러리의 요구사항을 동시에 만족시키는 안정적인 라이브러리 조합을 찾는 것이 불가능했습니다.
- **결론:** 프로젝트에 불필요한 `torchaudio`와 `craft-text-detector`가 충돌의 주된 원인이었음을 파악했습니다.

### 2.2. PyInstaller의 패키징 누락
- **초기 문제:** 라이브러리 버전 충돌을 해결한 후에도, `numpy`, `utils`, `ultralytics`, `seaborn` 등 특정 모듈을 찾지 못한다는 `ModuleNotFoundError`가 순차적으로 발생했습니다.
- **원인:** `PyInstaller`의 정적 분석 기능이 프로젝트에서 사용되는 모든 라이브러리와 그 하위 모듈들을 완벽하게 탐지하지 못하고, 최종 실행 파일에 일부를 누락시켰습니다.
- **결론:** 단순히 `hiddenimports`에 추가하는 것만으로는 부족했으며, 라이브러리의 모든 구성 요소를 강제로 포함시키는 더 강력한 방법이 필요했습니다.

---

## 3. 최종 해결 과정

위 두 가지 핵심 원인을 해결하기 위해 아래와 같은 단계를 순차적으로 진행했습니다.

### 1단계: 라이브러리 환경 정리

1.  **불필요한 패키지 삭제:** 모든 충돌의 시작점이었던, 프로젝트에 불필요한 `craft-text-detector`와 `torchaudio`를 `pip uninstall`로 완전히 삭제했습니다.

2.  **정확한 버전으로 재설치:** `torch`의 CPU 전용 버전을 명시하고, `ultralytics`와 호환되는 `opencv-python` 버전을 지정하는 등, 모든 핵심 라이브러리들의 버전을 명시하여 아래의 명령어로 재설치했습니다. 이를 통해 안정적이고 예측 가능한 라이브러리 환경을 구축했습니다.
    ```bash
    pip install ^
      --index-url https://download.pytorch.org/whl/cpu ^
      --extra-index-url https://pypi.org/simple ^
      numpy==1.26.4 torch==2.3.0+cpu torchvision==0.18.0+cpu ^
      opencv-python==4.10.0.84 easyocr==1.7.1
    ```

### 2단계: `spec` 파일 대폭 수정

`PyInstaller`가 라이브러리를 누락하는 문제를 근본적으로 해결하기 위해, `ocr_server_patched.spec` 파일을 아래와 같은 전략으로 수정했습니다.

- **`collect_all`을 통한 전체 수집:** `hiddenimports` 대신, `PyInstaller.utils.hooks.collect_all` 함수를 사용하여 문제가 발생했던 모든 핵심 라이브러리(`numpy`, `torch`, `torchvision`, `easyocr`, `cv2`, `ultralytics`, `seaborn`)의 코드, 데이터 파일, 바이너리 등 모든 구성 요소를 강제로 수집하여 빌드에 포함시켰습니다.
- **로컬 모듈 포함:** `yolov5`와 같이 프로젝트 내부에 있는 로컬 파이썬 폴더도 `datas`에 추가하여 패키징에 누락되지 않도록 조치했습니다.

**최종 `spec` 파일의 핵심 로직:**
```python
# ...
pkgs = ['numpy', 'torch', 'torchvision', 'easyocr', 'cv2', 'ultralytics', 'seaborn']

datas = []
binaries = []
hiddenimports = []

for pkg in pkgs:
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception:
        pass

# 로컬 폴더 추가
for p in ['custom_weights', 'custom_weights_easyOCR', 'CRNN_model', 'yolov5']:
    if os.path.exists(p):
        datas.append((p, p))
# ...
```

### 3단계: 최종 빌드

수정된 `ocr_server_patched.spec` 파일을 사용하여 `pyinstaller` 명령어를 실행하여 최종적으로 모든 런타임 오류가 해결된 `ocr_server.exe`를 성공적으로 빌드했습니다.

---

## 4. 결론 및 권장 사항

- **핵심 해결책:** 이번 문제의 해결책은 **(1) 명시적인 버전 관리를 통해 깨끗한 Python 환경을 구성**하고, **(2) `collect_all`을 활용한 강력한 `spec` 파일을 작성**하여 PyInstaller의 자동 분석에 의존하지 않는 것이었습니다.

- **향후 권장 사항:** `torch`, `numpy`와 같이 복잡한 라이브러리를 포함하는 프로젝트를 `PyInstaller`로 빌드할 때는, 문제가 발생하기 전에 선제적으로 이 보고서에서 사용된 `collect_all` 전략을 `.spec` 파일에 적용하는 것을 적극 권장합니다. 이는 빌드 과정의 안정성을 크게 높여주고 디버깅 시간을 단축시킬 수 있습니다.
