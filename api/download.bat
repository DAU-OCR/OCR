@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

SET "SERVER_URL=http://localhost:5000"
SET "DOWNLOAD_ENDPOINT=/download-json"
SET "OUTPUT_FILENAME_BASE=ocr_results"

:: 현재 날짜를 YYYY-MM-DD 형식으로 구합니다.
for /f "tokens=1-4 delims=/ " %%a in ('date /t') do (
    SET "CURRENT_DATE=%%c-%%a-%%b"
)
SET "OUTPUT_FILE=%OUTPUT_FILENAME_BASE%_%CURRENT_DATE%.json"

echo.
echo ===========================================
echo ? [3/3] JSON 결과 다운로드 시작
echo ===========================================
echo ? 저장 파일명: !OUTPUT_FILE!
echo.

:: --- CURL 실행: JSON 다운로드 ---
curl -X GET "!SERVER_URL!!DOWNLOAD_ENDPOINT!" -o "!OUTPUT_FILE!"

IF !ERRORLEVEL! NEQ 0 (
    echo ? 오류: JSON 다운로드 실패. (Error Code: !ERRORLEVEL!)
) ELSE (
    echo ? 성공: OCR 결과가 "!OUTPUT_FILE!" 파일에 저장되었습니다.
)

echo ===========================================
ENDLOCAL
pause