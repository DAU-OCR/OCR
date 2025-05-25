import torch.nn as nn
import torch

class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        try:
            self.rnn.flatten_parameters()
        except:
            pass
        recurrent, _ = self.rnn(input)
        output = self.linear(recurrent)
        return output

class VGG_FeatureExtractor(nn.Module):
    def __init__(self, input_channel, output_channel=256):
        super(VGG_FeatureExtractor, self).__init__()
        self.output_channel = [int(output_channel / 8), int(output_channel / 4),
                               int(output_channel / 2), output_channel]
        self.ConvNet = nn.Sequential(
            nn.Conv2d(input_channel, self.output_channel[0], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self.output_channel[0], self.output_channel[1], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self.output_channel[1], self.output_channel[2], 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(self.output_channel[2], self.output_channel[2], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(self.output_channel[2], self.output_channel[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 2, 1, 0), nn.ReLU(True)
        )

    def forward(self, input):
        return self.ConvNet(input)

class Model(nn.Module):
    def __init__(self, input_channel, output_channel, hidden_size, num_class, **kwargs):
        super(Model, self).__init__()

        # Feature Extraction
        self.FeatureExtraction = VGG_FeatureExtractor(input_channel, output_channel)
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))
        self.FeatureExtraction_output = output_channel

        # Sequence Modeling
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(self.FeatureExtraction_output, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, hidden_size)
        )
        self.SequenceModeling_output = hidden_size

        # Prediction
        self.Prediction = nn.Linear(hidden_size, num_class)

    def forward(self, input, text=None):
        # Feature
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2)).squeeze(3)

        # Sequence modeling
        contextual_feature = self.SequenceModeling(visual_feature)

        # Prediction
        prediction = self.Prediction(contextual_feature)

        return prediction
