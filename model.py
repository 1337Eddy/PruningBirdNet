import numpy as np
import torch 
import torch.nn as nn
from utils import log


FILTERS = [8, 16, 32, 64, 128]
KERNEL_SIZES = [(5, 5), (3, 3), (3, 3), (3, 3), (3, 3)]
RESNET_K = 4
RESNET_N = 3
num_classes = 83
filters = [[[32]], 
[[16, 16, 32], [64, 64], [64, 64], [64, 64]], 
[[32, 32, 64], [128, 128], [128, 128], [128, 128]], 
[[64, 64, 128], [256, 256], [256, 256], [256, 256]], 
[[128, 128, 256], [512, 512], [512, 512], [512, 512]],
[512, 512, num_classes]]

class BirdNet(nn.Module):
    def __init__(self, filters=filters):
        super(BirdNet, self).__init__()
        print('Start initialize Model')
        self.layers = [InputLayer(in_channels=1, num_filters=filters[0][0][0])]
        for i in range(1, len(filters) - 1):
            in_channels = filters[i-1][-1][-1]
            self.layers += [ResStack(num_filters=filters[i], in_channels=in_channels, kernel_size=KERNEL_SIZES[i-1])]
        self.layers += [nn.BatchNorm2d(filters[-2][-1][-1])]
        self.layers += [nn.ReLU(True)]
        self.layers += [ClassificationPath(in_channels=filters[-2][-1][-1], num_filters=filters[-1], kernel_size=(4,10))]
        self.layers += [nn.AdaptiveAvgPool2d(output_size=(1,1))]
        #self.layers += [nn.Softmax()]
        self.classifier = nn.Sequential(*self.layers)

        print('Model initialized')
    
    def forward(self, x):
        return self.classifier(x)
    
    def initialize_weights(self):
        log('Loading parameters...')
    
"""
Resblock is a implementation of a basic Residual Block with 2 Conv layers and a trainable scale to skip connection and main block
Args: 
    num_filters: tupel of integers that contains the amount of filters for first and second Conv layer
    in_channels: number of input channels in residual block
    kernel_size: (three) dimensional size of both conv layers in residual block 
"""
class Resblock(nn.Module):
    def __init__(self, num_filters, in_channels, kernel_size):
        super(Resblock, self).__init__()
        self.classifier = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels, out_channels=num_filters[0], kernel_size=kernel_size, padding='same'),
            nn.BatchNorm2d(num_features=num_filters[0]),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Conv2d(in_channels=num_filters[0], out_channels=num_filters[1], kernel_size=kernel_size, padding='same'),
            nn.BatchNorm2d(num_features=num_filters[1])
        )
        self.W = torch.nn.Parameter(torch.randn(2))
        self.W.requires_grad = True

    def forward(self, x):
        skip = x 
        skip = torch.mul(skip, self.W[1])
        x = self.classifier(x)
        #ToDO fix dimensionality error in case #filter_in not equal to #filters_out
        if (np.shape(x) == np.shape(skip)):
            x = torch.mul(x, self.W[0])
            x = torch.add(x, skip)
            return x
        else:
            return x

"""
ResStack (Residual Stack) is a network mainly build from multiple Resblocks
Args: 
    num_filters:
    in_channels:
    kernel_size:
    num_blocks: 
"""
class ResStack(nn.Module):
    def __init__(self, num_filters, in_channels, kernel_size):
        super(ResStack, self).__init__()
        #Num output filters of DownsamlingResBlock
        in_channels_resblock = num_filters[0][-1]
        
        resblock_list = []
        for i in range (1, len(num_filters)):
            resblock_list += [Resblock(num_filters=num_filters[i], in_channels=in_channels_resblock, kernel_size=kernel_size)]
            in_channels_resblock = num_filters[i][-1]
        resblock_list += [nn.BatchNorm2d(num_features=num_filters[-1][-1])]
        resblock_list += [nn.ReLU(True)]

        self.classifier = nn.Sequential(
            DownsamplingResBlock(num_filters=num_filters[0], in_channels=in_channels, kernel_size=kernel_size),
            *resblock_list
        )

    def forward(self, x):
        return self.classifier(x)


"""
Downsamping Residual Block is a residual block with a pooling layer and a 1x1 convulution in the skip connection
The main path and the skip connection are weighted with trainable parameters
"""
class DownsamplingResBlock(nn.Module):
    def __init__(self, num_filters, in_channels, kernel_size):
        super(DownsamplingResBlock, self).__init__()
        self.classifierPath = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=num_filters[0], kernel_size=(1,1)),
            nn.BatchNorm2d(num_features=num_filters[0]),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_filters[0], out_channels=num_filters[1], kernel_size=kernel_size, padding='same'),
            nn.BatchNorm2d(num_features=num_filters[1]),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Dropout(),
            nn.Conv2d(in_channels=num_filters[1], out_channels=num_filters[2], kernel_size=(1,1))
        )
        self.skipPath = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=in_channels, out_channels=num_filters[2], kernel_size=(1,1))
        )
        self.W = torch.nn.Parameter(torch.randn(2))
        self.W.requires_grad = True

    def forward(self, x):
        skip = self.skipPath(x)
        skip = torch.mul(skip, self.W[1])
        x = self.classifierPath(x)
        x = torch.mul(x, self.W[0])
        x = torch.add(x, skip)
        return x

"""
InputLayer is the first layer in the model
It contains a simple Convolutional layer with RELU actication function and BatchNormalization layer
"""
class InputLayer(nn.Module):
    def __init__(self, in_channels, num_filters):
        super(InputLayer, self).__init__()
        self.classifierPath = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=num_filters, kernel_size=1),
            nn.ReLU(True),
            nn.BatchNorm2d(num_features=num_filters)
        )

    def forward(self, x):
        return self.classifierPath(x)


class ClassificationPath(nn.Module):
    def __init__(self, in_channels, num_filters, kernel_size):
        super(ClassificationPath, self).__init__()
        self.classifierPath = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=num_filters[0], kernel_size=kernel_size, padding='valid'),
            nn.ReLU(True),
            nn.BatchNorm2d(num_features=num_filters[0]),
            nn.Dropout(),
            nn.Conv2d(in_channels=num_filters[0], out_channels=num_filters[1], kernel_size=(1,1)),
            nn.ReLU(True),
            nn.BatchNorm2d(num_features=num_filters[1]),
            nn.Dropout(),
            nn.Conv2d(in_channels=num_filters[1], out_channels=num_filters[2], kernel_size=(1,1)),
        )

    def forward(self, x):
        return self.classifierPath(x)
   