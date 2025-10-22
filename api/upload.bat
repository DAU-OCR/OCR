@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

SET "SERVER_URL=http://localhost:5000"
SET "UPLOAD_ENDPOINT=/upload-batch"
SET "IMAGE_DIR=.\images" :: ���ε��� �̹����� �ִ� ����

echo.
echo ===========================================
echo ? [2/3] �̹��� �ϰ� ���ε� �� OCR ����
echo ===========================================
echo ? �̹��� ���: !IMAGE_DIR!
echo ? ���� URL: !SERVER_URL!
echo.

SET "FILES_TO_UPLOAD="
SET "FILE_COUNT=0"

FOR %%f IN ("!IMAGE_DIR!\*.jpg" "!IMAGE_DIR!\*.jpeg" "!IMAGE_DIR!\*.png") DO (
    SET "FILES_TO_UPLOAD=!FILES_TO_UPLOAD! -F "files=@%%f""
    SET /A FILE_COUNT+=1
)

if !FILE_COUNT! EQU 0 (
    echo ? ����: ������ ��ο� ���ε��� �̹��� ������ �����ϴ�.
    GOTO :END_PROCESS
)

echo ? �� !FILE_COUNT! ���� ���� ���ε� �õ�.

:: --- CURL ���� ---
echo --- UPLOAD RESPONSE START ---
curl -X POST "!SERVER_URL!!UPLOAD_ENDPOINT!" !FILES_TO_UPLOAD!
echo --- UPLOAD RESPONSE END ---
echo.

IF !ERRORLEVEL! NEQ 0 (
    echo ? ����: �̹��� ���ε� ����. ���� ���¸� Ȯ���ϼ���. (Error Code: !ERRORLEVEL!)
) ELSE (
    echo ? OCR ó�� �Ϸ�. ����� ���� �޸𸮿� ����Ǿ����ϴ�.
)

echo ===========================================

:END_PROCESS
ENDLOCAL
pause