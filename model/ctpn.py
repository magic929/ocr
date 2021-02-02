import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models

class BiLstm(nn.Module):
    def __init__(self, channel, hidden_unit, bidirectional=True):
        super(BiLstm, self).__init__()
        self.lstm = nn.LSTM(channel, hidden_unit, bidirectional=bidirectional)
    
    def forward(self, x):
        x = x.transpose(1, 3)
        recurrent, _ = self.lstm(x[0])
        recurrent = recurrent[np.newaxis, :, :, :]
        recurrent = recurrent.transpose(1, 3)
        return recurrent
    

class Im2Col(nn.Module):
    def __init__(self, kernel_size, stride, padding):
        super(Im2Col, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
    
    def forward(self, x):
        height = x.shape[2]
        x = F.unfold(x, self.kernel_size, padding=self.padding, stride=self.stride)
        x = x.reshape((x.shape[0], x.shape[1], height, -1))
        return x


class CTPN(nn.Module):
    def __init__(self):
        super(CTPN, self).__init__()
        self.cnn = models.vgg16(pretrained=True).features[:-1]
        self.rnn = nn.Sequential()
        self.rnn.add_module('im2col', Im2Col((3, 3), (1, 1), (1, 1)))
        self.rnn.add_module('blstm', BiLstm(3 * 3 * 512, 128))
        self.FC = nn.Conv2d(256, 512, 1)
        self.vertical_coordinate = nn.Conv2d(512, 2 * 10, 1)
        self.score = nn.Conv2d(512, 2 * 10, 1)
        self.side_refinement = nn.Conv2d(512, 10, 1)

    def forward(self, x, val=False):
        x = self.cnn(x)
        x = self.rnn(x)
        x = self.FC(x)
        x = F.relu(x, inplace=True)
        vertical_pred = self.vertical_coordinate(x)
        score = self.score(x)
        if val:
            score = score.reshape((score.shape[0], 10, 2, score.shape[2], score.shape[3]))
            score = score.squeeze(0)
            score = score.transpose(1, 2)
            score = score.transpose(2, 3)
            score = score.reshape((-1, 2))
            score = score.reshape((10, vertical_pred.shape[2], -1, 2))
            vertical_pred = vertical_pred.reshape((vertical_pred.shape[0], 10, 2, vertical_pred.shape[2], vertical_pred.shape[3]))
        
        side_refinement = self.side_refinement(x)
        return vertical_pred, score, side_refinement




