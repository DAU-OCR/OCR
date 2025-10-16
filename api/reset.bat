@echo off
SETLOCAL ENABLEDELAYEDEXPANSION

SET "SERVER_URL=http://localhost:5000"

echo.
echo ===========================================
echo ? [1/3] 서버 결과 초기화 시작
echo ===========================================
echo ? 서버 URL: !SERVER_URL!
echo.

:: 서버의 /reset 엔드포인트 호출
curl -X POST "!SERVER_URL!/reset"

IF !ERRORLEVEL! NEQ 0 (
    echo ? 오류: 서버 초기화 요청에 실패했습니다. 서버 상태를 확인하세요.
) ELSE (
    echo ? 서버 결과 (records) 초기화 완료.
)

echo ===========================================
ENDLOCAL
pause