@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

SET "SERVER_URL=http://localhost:5000"

echo.
echo ===========================================
echo ? [1/3] ���� ��� �ʱ�ȭ ����
echo ===========================================
echo ? ���� URL: !SERVER_URL!
echo.

:: ������ /reset ��������Ʈ ȣ��
curl -X POST "!SERVER_URL!/reset"

IF !ERRORLEVEL! NEQ 0 (
    echo ? ����: ���� �ʱ�ȭ ��û�� �����߽��ϴ�. ���� ���¸� Ȯ���ϼ���.
) ELSE (
    echo ? ���� ��� (records) �ʱ�ȭ �Ϸ�.
)

echo ===========================================
ENDLOCAL
pause