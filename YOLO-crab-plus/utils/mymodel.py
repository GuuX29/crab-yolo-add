import torch.nn as nn
from MyEfficientNet import EfficientNet
# from utils import args
import torch

# naive b5
class Original_b5(nn.Module):
    def __init__(self, num_class):
        super(Original_b5, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b5', num_classes=1000)
        self.num_class = num_class
        num_ftrs = self.model._fc.in_features
        self.fc = nn.Linear(num_ftrs,self.num_class)

    def forward(self, img):
        out = self.model(img)
        out = self.fc(out)
        return out



