@echo off

REM ───── conda 초기화
CALL C:\Users\HOME\anaconda3\Scripts\activate.bat

REM ───── 가상환경 활성화
CALL conda activate clean

REM ───── PaddleOCR 경로로 이동
cd /d D:\clean2\PaddleOCR

REM ───── 학습 실행 & 로그 저장
powershell -Command "python tools/train.py -c configs/rec/rec_korean_numberplate_MobileNetV3_Large.yml | Tee-Object -FilePath train_log.txt"

REM ───── 창 유지
pause
