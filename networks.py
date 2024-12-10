import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import save_image

# import convlstm
# Acknowledgement to
# https://github.com/kuangliu/pytorch-cifar,
# https://github.com/BIGBALLON/CIFAR-ZOO,


''' Swish activation '''
class Swish(nn.Module): # Swish(x) = x∗σ(x)
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.sigmoid(input)


''' MLP '''
class MLP(nn.Module):
    def __init__(self, num_classes):
        super(MLP, self).__init__()
        self.fc_1 = nn.Linear(2560, 128)
        self.fc_2 = nn.Linear(128, 128)
        self.fc_3 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = F.relu(self.fc_1(x))
        out = F.relu(self.fc_2(out))
        out = self.fc_3(out)
        return out


''' ConvNet '''
class ConvNet(nn.Module):
    def __init__(self, num_classes, net_width, net_depth, net_act, net_norm, net_pooling, in_size = 8000):
        super(ConvNet, self).__init__()

        self.features, shape_feat = self._make_layers(net_width, net_depth, net_norm, net_act, net_pooling, in_size)
        self.num_channels = shape_feat[0]
        num_feat = shape_feat[0]*shape_feat[1]
        self.classifier = nn.Linear(num_feat, num_classes)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def embed(self, x):
        out = self.features(x)
        return out

    def _get_activation(self, net_act):
        if net_act == 'sigmoid':
            return nn.Sigmoid()
        elif net_act == 'relu':
            return nn.ReLU(inplace=True)
        elif net_act == 'leakyrelu':
            return nn.LeakyReLU(negative_slope=0.01)
        elif net_act == 'swish':
            return Swish()
        else:
            exit('unknown activation function: %s'%net_act)

    def _get_pooling(self, net_pooling):
        if net_pooling == 'maxpooling':
            return nn.MaxPool1d(kernel_size=2, stride=2)
        elif net_pooling == 'avgpooling':
            return nn.AvgPool1d(kernel_size=2, stride=2)
        elif net_pooling == 'none':
            return None
        else:
            exit('unknown net_pooling: %s'%net_pooling)

    def _get_normlayer(self, net_norm, shape_feat):
        # shape_feat = (c*l)
        if net_norm == 'batchnorm':
            return nn.BatchNorm1d(shape_feat[0], affine=True)
        elif net_norm == 'layernorm':
            return nn.LayerNorm(shape_feat, elementwise_affine=True)
        elif net_norm == 'instancenorm':
            return nn.GroupNorm(shape_feat[0], shape_feat[0], affine=True)
        elif net_norm == 'groupnorm':
            return nn.GroupNorm(4, shape_feat[0], affine=True)
        elif net_norm == 'none':
            return None
        else:
            exit('unknown net_norm: %s'%net_norm)

    def _make_layers(self, net_width, net_depth, net_norm, net_act, net_pooling, in_size):
        layers = []
        in_channels = 64
        shape_feat = [64, in_size]
        for d in range(net_depth):
            layers += [nn.Conv1d(in_channels, net_width, kernel_size=3, padding=3 if d == 0 else 1)]
            shape_feat[0] = net_width
            if net_norm != 'none':
                layers += [self._get_normlayer(net_norm, shape_feat)]
            layers += [self._get_activation(net_act)]
            in_channels = net_width
            if net_pooling != 'none':
                layers += [self._get_pooling(net_pooling)]
                shape_feat[1] //= 2

        return nn.Sequential(*layers), shape_feat


''' Conv -> LSTM '''
class ConvToLSTMNet(nn.Module):
    def __init__(self, num_classes, cnn_width, cnn_depth, cnn_act, cnn_norm, cnn_pooling, lstm_width, lstm_depth, bidirectional = False, in_size = 8000):
        super(ConvToLSTMNet, self).__init__()

        self.cnn = ConvNet(num_classes, cnn_width, cnn_depth, cnn_act, cnn_norm, cnn_pooling, in_size)
        self.lstm = nn.LSTM(self.cnn.num_channels, lstm_width, lstm_depth, bidirectional=bidirectional, batch_first=True)
        self.num_feat = (2 if bidirectional else 1)*lstm_width
        self.classifier = nn.Linear(self.num_feat, num_classes)
    
    def forward(self, x):
        out = self.cnn.embed(x)
        out = out.permute(0, 2, 1)
        _, (out, _) = self.lstm(out)
        out = out.squeeze(0)
        out = self.classifier(out)
        return out
    
    def embed(self, x):
        out = self.cnn.embed(x)
        out = out.view(out.size(1), out.size(0))
        _, (out, _) = self.lstm(out)
        return out

    def visualize(self, x):
        out = self.cnn.embed(x)
        img = out.permute(2, 1, 0).detach().cpu().numpy()
        save_image('cnn_out.png', img)
        out = out.permute(0, 2, 1)
        _, (out, _) = self.lstm(out)
        out = out.squeeze(0)
        print(out.size())
        img = out.detach().cpu().numpy()
        save_image('lstm_out.png', img)
        out = self.classifier(out)
        return out
