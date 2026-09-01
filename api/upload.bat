@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

SET "SERVER_URL=http://localhost:5000"
SET "UPLOAD_ENDPOINT=/upload-batch"
SET "IMAGE_DIR=.\images" :: 업로드할 이미지가 있는 폴더

echo.
echo ===========================================
echo ? [2/3] 이미지 일괄 업로드 및 OCR 시작
echo ===========================================
echo ? 이미지 경로: !IMAGE_DIR!
echo ? 서버 URL: !SERVER_URL!
echo.

SET "FILES_TO_UPLOAD="
SET "FILE_COUNT=0"

FOR %%f IN ("!IMAGE_DIR!\*.jpg" "!IMAGE_DIR!\*.jpeg" "!IMAGE_DIR!\*.png") DO (
    SET "FILES_TO_UPLOAD=!FILES_TO_UPLOAD! -F "files=@%%f""
    SET /A FILE_COUNT+=1
)

if !FILE_COUNT! EQU 0 (
    echo ? 오류: 지정된 경로에 업로드할 이미지 파일이 없습니다.
    GOTO :END_PROCESS
)

echo ? 총 !FILE_COUNT! 개의 파일 업로드 시도.

:: --- CURL 실행 ---
echo --- UPLOAD RESPONSE START ---
curl -X POST "!SERVER_URL!!UPLOAD_ENDPOINT!" !FILES_TO_UPLOAD!
echo --- UPLOAD RESPONSE END ---
echo.

IF !ERRORLEVEL! NEQ 0 (
    echo ? 오류: 이미지 업로드 실패. 서버 상태를 확인하세요. (Error Code: !ERRORLEVEL!)
) ELSE (
    echo ? OCR 처리 완료. 결과가 서버 메모리에 저장되었습니다.
)

echo ===========================================

:END_PROCESS
ENDLOCAL
pause