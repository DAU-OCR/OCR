
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 3, 1, 1)  # 유지
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)  # 1/2 축소
        self.conv2 = nn.Conv2d(64, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        return x

class BidirectionalLSTM(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(in_size, hidden_size, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, out_size)

    def forward(self, x):
        recurrent, _ = self.rnn(x)
        output = self.fc(recurrent)
        return output
