# crnn.py - CRNN 모델 정의 (CNN + RNN + CTC용 출력층)

import torch
import torch.nn as nn

class CRNN(nn.Module):
    """
    CRNN (Convolutional Recurrent Neural Network) 모델
    - 입력: 32x128 흑백 이미지
    - 구조: CNN + BiLSTM (2단) + Linear
    - 출력: [W, B, C] 형태로 CTC Loss에 사용 가능
    """

    def __init__(self, img_height=32, num_channels=1, num_classes=100, hidden_size=256):
        """
        img_height: 입력 이미지 높이 (기본: 32)
        num_channels: 입력 채널 수 (기본: 1 = 흑백)
        num_classes: 출력 클래스 수 (문자 종류 + CTC blank)
        hidden_size: LSTM hidden state 크기
        """
        super(CRNN, self).__init__()

        # =============================
        # CNN 특징 추출 계층
        # =============================
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, 64, 3, 1, 1),  # [B, 64, 32, 128]
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                    # [B, 64, 16, 64]

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                    # [B, 128, 8, 32]

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),          # [B, 256, 4, 32] (세로 downsample만)

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),          # [B, 512, 2, 32]

            nn.Conv2d(512, 512, 2, 1, 0),          # [B, 512, 1, 31]
            nn.ReLU()
        )

        # =============================
        # RNN 계층 (BiLSTM 2층)
        # =============================
        self.rnn1 = nn.LSTM(512, hidden_size, bidirectional=True)               # [W, B, 512] → [W, B, hidden*2]
        self.rnn2 = nn.LSTM(hidden_size * 2, hidden_size, bidirectional=True)   # [W, B, hidden*2] → [W, B, hidden*2]

        # =============================
        # 최종 출력 계층
        # =============================
        self.fc = nn.Linear(hidden_size * 2, num_classes)   # 클래스 수 만큼 출력
        self.dropout = nn.Dropout(p=0.4)

    def forward(self, x):
        x = self.cnn(x)              # [B, 512, 1, W]
        x = x.squeeze(2)             # [B, 512, W] → CNN 출력 높이 제거
        x = x.permute(2, 0, 1)       # [B, 512, W] → [W, B, 512] (CTC용 입력 형태)

        x, _ = self.rnn1(x)
        x = self.dropout(x)
        x, _ = self.rnn2(x)
        x = self.dropout(x)

        x = self.fc(x)               # [W, B, num_classes]
        return x
