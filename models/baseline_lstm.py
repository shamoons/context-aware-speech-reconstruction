import torch.nn as nn
import torch.nn.functional as F

class BaselineLSTM(nn.Module):
    def __init__(self):
        super(BaselineLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size=13, hidden_size=13)
        
    def forward(self, x):
        x = self.lstm(x)

        return x