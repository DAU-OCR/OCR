## 릴리즈 노트 (2025-10-22)

### 🚀 새로운 기능 및 개선 사항

#### 1. 드래그 앤 드롭 파일 업로드 기능 추가
- **설명**: 사용자 편의성 향상을 위해 파일 선택 외에 드래그 앤 드롭 방식으로 이미지를 업로드할 수 있는 기능을 추가했습니다.
- **영향 파일**:
    - `ocr-web/src/pages/UploadPage.jsx`: 드래그 앤 드롭 로직 및 이벤트 핸들러 구현.
    - `ocr-web/src/pages/UploadPage.css`: 드래그 중 시각적 피드백을 위한 스타일 추가.

#### 2. ZIP 파일 처리 진행률 표시 개선
- **설명**: ZIP 파일 업로드 시 진행률 표시줄이 더욱 부드럽고 현실적으로 업데이트되도록 개선했습니다.
- **영향 파일**:
    - `server/app.py`: ZIP 파일 내 이미지 개수를 반환하는 `/get-zip-image-count` 엔드포인트 추가. `/upload` 엔드포인트 응답에 `total_images_in_zip` 및 `processed_count` 정보 포함.
    - `ocr-web/src/pages/UploadPage.jsx`:
        - 업로드 전 `/get-zip-image-count`를 통해 총 이미지 수를 미리 계산하여 정확한 진행률 총계 설정.
        - ZIP 파일 처리 중에는 이미지당 예상 처리 시간을 기반으로 하는 "가짜 진행률"을 구현하여 진행률 표시줄이 천천히 증가하는 것처럼 보이도록 개선. (`AVERAGE_TIME_PER_IMAGE_MS` 조정 포함)
    - `ocr-web/public/icons/zip-icon.svg`: ZIP 파일 미리보기를 위한 아이콘 추가.

#### 3. 결과 페이지 팝업 기능 개선
- **설명**: 미리보기 및 인식 실패 페이지의 확대 팝업에서 사용자 경험을 향상시키는 기능을 추가했습니다.
- **영향 파일**:
    - `ocr-web/src/pages/ResultsPage.jsx`: 확대 팝업 내에서 `Escape` 키를 눌러 팝업을 닫는 기능 추가.
    - `ocr-web/src/pages/FailedPage.jsx`:
        - 확대 팝업 내에서 `ArrowUp`/`ArrowDown` 키를 사용하여 이미지 간을 탐색하는 기능 추가.
        - 확대 팝업 내에서 `Escape` 키를 눌러 팝업을 닫는 기능 추가.

### 🐛 버그 수정 및 안정화

#### 1. ZIP 파일 처리 로직 오류 수정
- **설명**: ZIP 파일 업로드 시 이미지 누락 또는 중복 처리되던 문제를 해결했습니다.
- **영향 파일**:
    - `server/app.py`: `/upload` 및 `/upload-batch` 엔드포인트에서 ZIP 파일 내 이미지 추출 시 고유한 파일명 생성 로직 적용.

#### 2. JSON 파일 다운로드 확장자 문제 진단용 로그 추가
- **설명**: JSON 파일 다운로드 시 파일명에 불필요한 따옴표가 붙는 문제 진단을 위해 서버 로그를 추가했습니다.
- **영향 파일**:
    - `server/app.py`: `/download-json` 엔드포인트에 `Content-Disposition` 헤더 출력 로그 추가.

### ✨ UI/UX 개선

#### 1. 페이지 상단 간격 조정
- **설명**: 업로드, 미리보기, 인식 실패 페이지의 메인 콘텐츠와 상단 사이의 불필요한 간격을 줄여 시각적 밀도를 높였습니다.
- **영향 파일**:
    - `ocr-web/src/pages/UploadPage.css`: `.upload-page`의 `padding-top` 조정.
    - `ocr-web/src/pages/ResultsPage.css`: `.results-page`의 `padding-top` 조정.
    - `ocr-web/src/pages/FailedPage.css`: `.results-page`의 `padding-top` 조정.

#### 2. 업로드 상자 시각적 피드백 개선
- **설명**: 드래그 앤 드롭 영역의 기본 테두리를 실선으로 변경하고, 파일 드래그 시에만 점선 테두리가 나타나도록 하여 시각적 명확성을 높였습니다.
- **영향 파일**:
    - `ocr-web/src/pages/UploadPage.css`: `.dropzone` 및 `.dropzone.dragging` 스타일 수정.

#### 3. 업로드 문구 간결화
- **설명**: 업로드 영역의 안내 문구를 "파일 선택 또는 드래그"로 간결하게 수정했습니다.
- **영향 파일**:
    - `ocr-web/src/pages/UploadPage.jsx`: 파일 라벨 텍스트 수정.

#### 4. "인식 실패" 입력 필드 시각적 강조
- **설명**: 결과 페이지에서 "인식 실패"로 표시된 입력 필드를 빨간색 배경으로 강조하여 사용자가 쉽게 인지할 수 있도록 했습니다.
- **영향 파일**:
    - `ocr-web/src/pages/ResultsPage.jsx`: `plate` 값이 "인식 실패"일 때 `failure-plate-input` 클래스 조건부 추가.
    - `ocr-web/src/pages/ResultsPage.css`: `.failure-plate-input` 스타일 정의.

#### 5. 인식 실패 페이지 버튼 스타일 통일
- **설명**: 인식 실패 페이지의 "수정 내용 저장 후 뒤로가기" 버튼 스타일을 다른 액션 버튼들과 일관되게 변경했습니다.
- **영향 파일**:
    - `ocr-web/src/pages/FailedPage.css`: `.save-button` 스타일 추가.
