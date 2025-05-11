
import torch.nn as nn
from easyocr.model.vgg_model import VGG_FeatureExtractor
from easyocr.modules import BidirectionalLSTM

class Model(nn.Module):
    def __init__(self, num_class, input_channel, output_channel, hidden_size):
        super(Model, self).__init__()
        # Feature Extraction
        self.FeatureExtraction = VGG_FeatureExtractor(input_channel, output_channel)
        
        # Sequence modeling (2-layer BiLSTM)
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(output_channel, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, hidden_size)
        )

        # Prediction
        self.Prediction = nn.Linear(hidden_size, num_class)

    def forward(self, x):
        # Feature extraction
        visual_feature = self.FeatureExtraction(x)  # [B, C, H, W]
        b, c, h, w = visual_feature.size()
        assert h == 1, "the height of feature map must be 1"
        visual_feature = visual_feature.squeeze(2)  # [B, C, W]
        visual_feature = visual_feature.permute(2, 0, 1)  # [W, B, C]

        contextual_feature = self.SequenceModeling(visual_feature)  # [W, B, hidden]
        prediction = self.Prediction(contextual_feature)  # [W, B, num_class]
        return prediction
