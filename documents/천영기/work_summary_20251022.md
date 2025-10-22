# 작업 요약 (2025-10-22)

## 1. 드래그 앤 드롭 기능 구현
- **문제점**: 기존 웹 UI(`ocr-web`)에서 파일 선택 버튼만 작동하고 드래그 앤 드롭 기능이 미구현되어 사용자 편의성이 낮았습니다.
- **해결**: 
    - `ocr-web/src/pages/UploadPage.jsx` 파일에 드래그 앤 드롭 관련 상태(`isDragging`) 및 이벤트 핸들러(`handleDragOver`, `handleDragLeave`, `handleDrop`)를 추가했습니다.
    - 파일 처리 로직을 `handleFiles` 함수로 분리하여 파일 입력(`onChange`)과 드롭(`onDrop`) 이벤트 모두에서 재사용하도록 리팩토링했습니다.
    - `ocr-web/src/pages/UploadPage.css` 파일에 드롭존(`dropzone`) 및 드래그 중(`dragging`) 상태에 대한 시각적 스타일을 추가하여 사용자에게 명확한 피드백을 제공하도록 했습니다.

## 2. ZIP 파일 처리 로직 오류 수정
- **문제점**: ZIP 파일을 업로드할 때 일부 이미지가 누락되거나 중복 처리되는 문제가 발생했습니다. 이는 ZIP 파일 내 동일한 이름의 파일이 다른 하위 디렉토리에 있을 경우, `os.path.basename` 사용으로 인해 파일이 덮어쓰여 발생한 것으로 추정되었습니다.
- **해결**: 
    - `server/app.py` 파일의 `/upload` 및 `/upload-batch` 엔드포인트에서 ZIP 파일 처리 로직을 수정했습니다。
    - `zip_info.filename`의 경로 구분자(예: `/`, `\`)를 밑줄(`_`)로 대체하여 고유한 파일 이름을 생성하도록 변경했습니다. 이를 통해 ZIP 파일 내의 원본 경로 정보를 일부 유지하면서 파일 이름 충돌을 방지하고 모든 이미지가 올바르게 처리되도록 했습니다。

## 3. JSON 파일 확장자 문제 조사
- **문제점**: 다운로드되는 JSON 파일의 확장자가 `**.json''` 형태로 끝에 불필요한 따옴표가 붙는다는 보고가 있었습니다.
- **조사**: 
    - `server/app.py` 파일의 `/download-json` 엔드포인트 코드를 검토한 결과, 서버 측에서는 `ocr_results_YYYY-MM-DD.json`과 같이 올바른 파일 이름을 `download_name`으로 설정하고 있었습니다.
    - 문제의 원인을 파악하기 위해 `download_json` 함수 내 `send_file` 호출 직전에 `Content-Disposition` 헤더를 출력하는 디버그 `print` 문을 추가했습니다. 이 로그를 통해 서버가 보내는 실제 헤더를 확인하여 문제의 원인이 서버 측인지 클라이언트 측인지 추가 진단이 가능합니다。

## 4. UI/CSS 개선 및 ZIP 파일 처리 진행률 표시 개선 (2025-10-22 추가)

### 4.1. UI 간격 조정
- **미리보기 및 인식 실패 페이지 상단 간격 조정**: `ocr-web/src/pages/ResultsPage.css` 및 `ocr-web/src/pages/FailedPage.css` 파일에서 `.results-page` 클래스의 `padding-top`을 `100px` (또는 `120px`)에서 `50px`로 줄여 메인 콘텐츠와 페이지 상단 사이의 간격을 좁혔습니다.

### 4.2. 인식 실패 페이지 버튼 스타일 변경
- **"수정 내용 저장 후 뒤로가기" 버튼 스타일 통일**: `ocr-web/src/pages/FailedPage.css` 파일에 `.save-button` 클래스에 대한 스타일을 추가했습니다. 이 스타일은 `ResultsPage.css`의 `.download-button` 및 `.reset-button`과 유사하게 `display: inline-flex`, `align-items: center`, `padding`, `border`, `border-radius`, `background`, `color`, `font-size`, `cursor`, `box-shadow`, `width: auto` 속성을 포함하여 다른 액션 버튼들과 시각적 일관성을 유지하도록 했습니다.

### 4.3. ZIP 파일 진행률 표시줄 최종 수정
- **백엔드 ZIP 파일 이미지 수 계산 엔드포인트 추가**: `server/app.py`에 새로운 엔드포인트 `/get-zip-image-count`를 추가했습니다. 이 엔드포인트는 업로드된 ZIP 파일 내의 이미지 파일 개수를 반환합니다.
- **프론트엔드 진행률 계산 로직 개선**: `ocr-web/src/pages/UploadPage.jsx`의 `onUpload` 함수를 수정했습니다.
    - **1단계 (총 이미지 수 미리 계산)**: 실제 파일 업로드 루프를 시작하기 전에, `files` 배열을 순회하며 각 ZIP 파일에 대해 `/get-zip-image-count` 엔드포인트로 호출하여 ZIP 파일 내 이미지 수를 가져옵니다. 단일 이미지 파일은 1로 계산하여 전체 `totalImagesToProcess`를 정확하게 계산합니다.
    - **2단계 (실제 파일 업로드 및 진행률 업데이트)**: 이 미리 계산된 `totalImagesToProcess`를 `setTotalCount`에 설정하고, 각 파일 업로드 후 백엔드에서 반환된 `processed_count`를 `cumulativeProcessedCount`에 누적하여 `setProcessingProgress`를 업데이트합니다. 이로써 진행률 표시줄이 처음부터 정확한 총계로 시작하고 원활하게 업데이트되도록 했습니다.

### 4.4. 추가 확인 사항
- `FailedPage.jsx`에서는 `plate` 값이 기본적으로 빈 문자열로 처리되므로, "인식 실패" 스타일은 적용되지 않았습니다. 필요시 별도 논의 후 적용 가능합니다.

## 5. 가짜 진행률 표시줄 개선 및 팝업 기능 추가 (2025-10-22 추가)

### 5.1. ZIP 파일 가짜 진행률 표시줄 개선
- **가짜 진행률 속도 동적 조정**: `ocr-web/src/pages/UploadPage.jsx`의 `onUpload` 함수에서 ZIP 파일 처리 시 가짜 진행률(`fakeProgressInterval`)의 속도를 이미지당 예상 평균 처리 시간(`AVERAGE_TIME_PER_IMAGE_MS = 1000ms`)과 ZIP 파일 내 이미지 수(`imageCount`)를 기반으로 동적으로 계산하도록 수정했습니다. 이를 통해 진행률 표시줄이 더욱 현실적이고 부드럽게 증가하는 것처럼 보이도록 했습니다.
- **진행률 계산 로직**: `currentFakeProgress`를 `cumulativeProcessedCount`와 `totalImagesToProcess`에 대한 기여도로 변환하여 전체 진행률에 합산하도록 했습니다.

### 5.2. 팝업 기능 추가
- **인식 실패 페이지 팝업 내비게이션**: `ocr-web/src/pages/FailedPage.jsx`에 `useEffect` 훅을 추가하여 확대 이미지 팝업이 열려 있을 때 `ArrowUp` 및 `ArrowDown` 키를 사용하여 이미지 간을 탐색할 수 있도록 했습니다.
- **Escape 키로 팝업 닫기**: `ocr-web/src/pages/ResultsPage.jsx` 및 `ocr-web/src/pages/FailedPage.jsx` 모두에 `useEffect` 훅을 추가하여 팝업이 열려 있을 때 `Escape` 키를 누르면 팝업이 닫히도록 기능을 구현했습니다.

## 6. 진행률 표시줄 동작 방식 및 가짜 진행률 속도 조정 (2025-10-22 추가)

### 6.1. 진행률 표시줄 동작 방식 설명
- **단일 ZIP 파일 처리 시**: 백엔드가 ZIP 파일 전체를 처리한 후 단일 응답을 보내므로, 프론트엔드는 ZIP 파일 처리 완료 시점에 한 번에 진행률을 업데이트합니다. 이로 인해 단일 ZIP 파일 업로드 시 진행률 표시줄이 0%에서 100%로 한 번에 점프하는 것처럼 보일 수 있습니다.
- **여러 파일 처리 시**: 여러 개별 이미지 파일 또는 여러 ZIP 파일을 업로드하는 경우, 각 파일(또는 ZIP 파일)이 처리될 때마다 진행률이 청크 단위로 업데이트됩니다.
- **가짜 진행률의 역할**: "가짜 진행률"은 단일 ZIP 파일 처리와 같이 백엔드 응답을 기다리는 동안 진행률 표시줄이 움직이는 것처럼 보이게 하여 사용자 경험을 향상시키는 시각적 기법입니다. 전체 진행률 계산은 항상 백엔드에서 받은 실제 데이터를 기반으로 합니다.

### 6.2. 가짜 진행률 속도 조정
- **`AVERAGE_TIME_PER_IMAGE_MS` 조정**: `ocr-web/src/pages/UploadPage.jsx`에서 `AVERAGE_TIME_PER_IMAGE_MS` 값을 `1000ms`에서 `500ms`로 조정했습니다. 이는 가짜 진행률 표시줄이 두 배 더 빠르게 증가하여 실제 처리 시간과 더 잘 일치하도록 하여, 실제 처리가 완료될 때 진행률이 더 높은 백분율에 도달하도록 합니다.
