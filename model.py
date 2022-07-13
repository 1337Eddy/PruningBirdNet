import numpy as np
import torch 
import torch.nn as nn
from utils import log
from enum import Enum
import torch.nn.functional as F


class Skip_Handling(Enum):
    PADD = 0
    CUT = 1
    SKIP = 2

KERNEL_SIZES = [(5, 5), (3, 3), (3, 3), (3, 3), (3, 3)]



class BirdNet(nn.Module):
    def __init__(self, filters, skip_handling=Skip_Handling.PADD, handling_block=Skip_Handling.PADD, padding_masks = []):
        super(BirdNet, self).__init__()

        global channel_handling
        global block_handling 
        channel_handling = skip_handling
        block_handling = handling_block
        self.layers = [InputLayer(in_channels=1, num_filters=filters[0][0][0])]
        self.padding_masks = padding_masks
        
        if self.padding_masks == []:
            create_mask = True 
        else: 
            create_mask = False
        mask_counter = 0

        for i in range(1, len(filters) - 1):
            in_channels = filters[i-1][-1][-1]
            num_masks_per_stack = len(filters[i]) - 1

            if create_mask:
                for filter in filters[i][1:]:
                    mask = torch.from_numpy(np.full(filter[1], True)).cuda()
                    self.padding_masks.append(mask)

            self.layers += [ResStack(num_filters=filters[i], in_channels=in_channels, 
                            kernel_size=KERNEL_SIZES[i-1], masks=self.padding_masks[mask_counter:mask_counter+num_masks_per_stack])]
            mask_counter += num_masks_per_stack

        self.layers += [nn.BatchNorm2d(filters[-2][-1][-1])]
        self.layers += [nn.ReLU(True)]
        self.layers += [ClassificationPath(in_channels=filters[-2][-1][-1], num_filters=filters[-1], kernel_size=(4,10))]
        self.layers += [nn.AdaptiveAvgPool2d(output_size=(1,1))]
        self.classifier = nn.Sequential(*self.layers)
        self.filters = filters

    
    def forward(self, x):
        return self.classifier(x)
    

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


"""
ResStack (Residual Stack) is a network mainly build from multiple Resblocks
Args: 
    num_filters:
    in_channels:
    kernel_size:
    num_blocks: 
"""
class ResStack(nn.Module):
    def __init__(self, num_filters, in_channels, kernel_size, masks):
        super(ResStack, self).__init__()
        #Num output filters of DownsamlingResBlock
        self.num_filters = num_filters
        in_channels_resblock = num_filters[0][-1]
        
        resblock_list = []
        mask_counter = 0
        for i in range (1, len(num_filters)):
            resblock_list += [Resblock(num_filters=num_filters[i], in_channels=in_channels_resblock, kernel_size=kernel_size, mask=masks[mask_counter])]
            #in_channels_resblock = max(num_filters[i][-1], num_filters[i-1][-1])
            in_channels_resblock = num_filters[i][-1]
            mask_counter += 1
        resblock_list += [nn.BatchNorm2d(num_features=num_filters[-1][-1])]
        resblock_list += [nn.ReLU(True)]

        self.classifier = nn.Sequential(
            DownsamplingResBlock(num_filters=num_filters[0], in_channels=in_channels, kernel_size=kernel_size),
            *resblock_list
        )

    def forward(self, x):
        #print("Input Resstack")
        #print(np.mean(x.cpu().detach().numpy()))
        x = self.classifier(x)
        #print("output Resstack")
        #print(np.mean(x.cpu().detach().numpy()))
        #print()
        return x


"""
Resblock is a implementation of a basic Residual Block with 2 Conv layers and a trainable scale to skip connection and main block
Args: 
    num_filters: tupel of integers that contains the amount of filters for first and second Conv layer
    in_channels: number of input channels in residual block
    kernel_size: (three) dimensional size of both conv layers in residual block 
"""
class Resblock(nn.Module):
    def __init__(self, num_filters, in_channels, kernel_size, mask):
        super(Resblock, self).__init__()
        self.num_filters = num_filters
        self.in_channels = in_channels
        self.mask = mask
        self.layer_list = [ nn.BatchNorm2d(num_features=in_channels),
                            nn.ReLU(True),
                            nn.Conv2d(in_channels=in_channels, out_channels=num_filters[0], kernel_size=kernel_size, 
                                padding=int(kernel_size[0]/2)),
                            nn.BatchNorm2d(num_features=num_filters[0]),
                            nn.ReLU(True),
                            nn.Dropout(),
                            nn.Conv2d(in_channels=num_filters[0], out_channels=num_filters[1], kernel_size=kernel_size, 
                                padding=int(kernel_size[0]/2)),
                            nn.BatchNorm2d(num_features=num_filters[1])]

        self.classifier = nn.Sequential(*self.layer_list)
        self.W = torch.nn.Parameter(torch.randn(2))
        self.W.requires_grad = True
        self.softmax = nn.Softmax(dim=0)


    def apply_mask_to_tensor(self, new_size, tensor):  
        buffer = torch.zeros([np.shape(tensor)[0], new_size, np.shape(tensor)[2], np.shape(tensor)[3]]).cuda()
        mask = self.mask
        for j in range(0, np.shape(buffer)[0]):
            i = 0
            for bool, net in zip(self.mask, tensor[j]):
                if bool:
                    buffer[j][i] = net 
                    i += 1
        return buffer

    def forward(self, x):
        #print("Input Resblock")
        #print(np.mean(x.cpu().detach().numpy()))
        scaling_factors = self.softmax(self.W)
        skip = x 
        skip = torch.mul(skip, scaling_factors[1])

        x = self.classifier(x)

        num_channels_skip = skip.size(dim=1)
        num_channels_x = x.size(dim=1) 

        skip = self.apply_mask_to_tensor(num_channels_x, skip)
        assert np.shape(x) == np.shape(skip)                    

        x = torch.mul(x, scaling_factors[0])
        x = torch.add(x, skip)
        #print("Output Resblock")
        #print(np.mean(x.cpu().detach().numpy()))
        return x



"""
Downsampling Residual Block is a residual block with a pooling layer and a 1x1 convulution in the skip connection
The main path and the skip connection are weighted with trainable parameters
"""
class DownsamplingResBlock(nn.Module):
    def __init__(self, num_filters, in_channels, kernel_size):
        super(DownsamplingResBlock, self).__init__()
        self.classifierPath = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=num_filters[0], kernel_size=(1,1)),
            nn.BatchNorm2d(num_features=num_filters[0]),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_filters[0], out_channels=num_filters[1], kernel_size=kernel_size, 
                padding=int(kernel_size[0]/2)),
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
        self.softmax = nn.Softmax(dim=0)

    def forward(self, x):
        #print("Input DS Block")
        #print(np.mean(x.cpu().detach().numpy()))
        scaling_factors = self.softmax(self.W)
        skip = self.skipPath(x)
        skip = torch.mul(skip, scaling_factors[1])
        x = self.classifierPath(x)
        x = torch.mul(x, scaling_factors[0])
        x = torch.add(x, skip)
        #print("Output DS Block")
        #print(np.mean(x.cpu().detach().numpy()))
        return x



class ClassificationPath(nn.Module):
    def __init__(self, in_channels, num_filters, kernel_size):
        super(ClassificationPath, self).__init__()
        self.classifierPath = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=num_filters[0], kernel_size=kernel_size, padding=0),
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
        x = self.classifierPath(x)
        return x
   