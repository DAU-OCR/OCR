@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

SET "SERVER_URL=http://localhost:5000"
SET "DOWNLOAD_ENDPOINT=/download-json"
SET "OUTPUT_FILENAME_BASE=ocr_results"

:: ���� ��¥�� YYYY-MM-DD �������� ���մϴ�.
for /f "tokens=1-4 delims=/ " %%a in ('date /t') do (
    SET "CURRENT_DATE=%%c-%%a-%%b"
)
SET "OUTPUT_FILE=%OUTPUT_FILENAME_BASE%_%CURRENT_DATE%.json"

echo.
echo ===========================================
echo ? [3/3] JSON ��� �ٿ�ε� ����
echo ===========================================
echo ? ���� ���ϸ�: !OUTPUT_FILE!
echo.

:: --- CURL ����: JSON �ٿ�ε� ---
curl -X GET "!SERVER_URL!!DOWNLOAD_ENDPOINT!" -o "!OUTPUT_FILE!"

IF !ERRORLEVEL! NEQ 0 (
    echo ? ����: JSON �ٿ�ε� ����. (Error Code: !ERRORLEVEL!)
) ELSE (
    echo ? ����: OCR ����� "!OUTPUT_FILE!" ���Ͽ� ����Ǿ����ϴ�.
)

echo ===========================================
ENDLOCAL
pause