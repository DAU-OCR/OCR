import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, img_height=32, num_channels=1, num_classes=100, hidden_size=256):
        super(CRNN, self).__init__()

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
            nn.MaxPool2d((2, 1), (2, 1)),          # [B, 256, 4, 32]

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),          # [B, 512, 2, 32]

            nn.Conv2d(512, 512, 2, 1, 0),          # [B, 512, 1, 31]
            nn.ReLU()
        )

        # 🔧 여기 변경됨: RNN 2단계 따로 정의
        self.rnn1 = nn.LSTM(512, hidden_size, bidirectional=True)
        self.rnn2 = nn.LSTM(hidden_size * 2, hidden_size, bidirectional=True)

        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(p=0.4)

    def forward(self, x):
        x = self.cnn(x)              # [B, 512, 1, W]
        x = x.squeeze(2)             # [B, 512, W]
        x = x.permute(2, 0, 1)       # [W, B, 512]

        # 🔧 RNN 두 번 따로 통과
        x, _ = self.rnn1(x)
        x = self.dropout(x)
        x, _ = self.rnn2(x)
        x = self.dropout(x)

        x = self.fc(x)               # [W, B, num_classes]
        return x
