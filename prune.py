from re import L
from typing import OrderedDict
import torch
import torch.optim as optim
from torch import device, nn, softmax, threshold
from analyze import DataLabels
from analyze_birdnet import AnalyzeBirdnet
import model 
from torch.utils.data import DataLoader
from data import CallsDataset
from enum import Enum
import PruneBlocks
import PruneChannels

class Channel_Pruning_Mode(Enum):
    EVENLY = 0
    NO_PADD = 1
    MIN = 2

def retrain(birdnet, criterion, save_path, lr=0.001, dataset_path="1dataset/1data/calls/"):
    data = DataLabels(dataset_path + "train/")

    train_dataset = CallsDataset(dataset_path + "train/")
    test_dataset = CallsDataset(dataset_path + "test/")
    train_loader = DataLoader(train_dataset, batch_size=16, num_workers=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, num_workers=16, shuffle=True)

    #Start Training
    analyze = AnalyzeBirdnet(birdnet=birdnet, dataset=data, lr=lr, criterion=criterion, train_loader=train_loader, 
                                test_loader=test_loader, save_path=save_path, gamma=0.2)
    analyze.start_training(10)

def prune(load_path, ratio, lr=0.001, save_path="", mode=Channel_Pruning_Mode.NO_PADD, channel_ratio=0.5):
    checkpoint = torch.load(load_path)
    model_state_dict = checkpoint['model_state_dict']

    filters = checkpoint['filters']
    birdnet = model.BirdNet(filters=filters)
    birdnet = torch.nn.DataParallel(birdnet).cuda()
    birdnet = birdnet.float()
    optimizer = optim.Adam(birdnet.parameters(), lr=lr) 
    birdnet.load_state_dict(model_state_dict)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


    model_state_dict, filters = PruneBlocks.prune(model_state_dict, filters, ratio)

    model_state_dict, filters = PruneChannels.prune(model_state_dict, ratio, filters, mode, channel_ratio)

    birdnet = model.BirdNet(filters=filters)
    birdnet = torch.nn.DataParallel(birdnet).cuda()
    #Load parameter to model
    birdnet.load_state_dict(model_state_dict)
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(birdnet.parameters(), lr=lr) 

    if save_path:
        retrain(birdnet, criterion, save_path, lr)
    return birdnet


if __name__ == '__main__':
    checkpoint = torch.load("models/birdnet_v1/birdnet_final.pt")
    model_state_dict = checkpoint['model_state_dict']

    #print(model_state_dict['module.classifier.1.classifier.4.weight'])
    prune("models/birdnet/birdnet_final.pt", ratio=0.20, mode=Channel_Pruning_Mode.NO_PADD, channel_ratio=0.3, save_path="models/pruned/block20_channel30/") 

    # prune("models/birdnet_v1/birdnet_final.pt", ratio=0.05, evenly=False, channel_ratio=0.0, save_path="models/pruned/block_05/") 
    # prune("models/birdnet_v1/birdnet_final.pt", ratio=0.1, evenly=False, channel_ratio=0.0, save_path="models/pruned/block_10/") 
    # prune("models/birdnet_v1/birdnet_final.pt", ratio=0.2, evenly=False, channel_ratio=0.0, save_path="models/pruned/block_20/") 
    # prune("models/birdnet_v1/birdnet_final.pt", ratio=0.3, evenly=False, channel_ratio=0.0, save_path="models/pruned/block_30/") 
    #prune("models/birdnet_v1/birdnet_final.pt", ratio=0.4, evenly=False, channel_ratio=0.0, save_path="models/pruned/block_40/") 
    #prune("models/birdnet_v1/birdnet_final.pt", ratio=0.5, evenly=False, channel_ratio=0.0, save_path="models/pruned/block_50/") 
    #prune("models/birdnet_v1/birdnet_final.pt", ratio=0.6, evenly=False, channel_ratio=0.0, save_path="models/pruned/block_60/") 
    #prune("models/birdnet_v1/birdnet_final.pt", ratio=0.7, evenly=False, channel_ratio=0.0, save_path="models/pruned/block_70/") 
    #prune("models/birdnet_v1/birdnet_final.pt", ratio=0.8, evenly=False, channel_ratio=0.0, save_path="models/pruned/block_80/") 
    #prune("models/birdnet_v1/birdnet_final.pt", ratio=0.9, evenly=False, channel_ratio=0.0, save_path="models/pruned/block_90/") 

    # prune("models/birdnet_v1/birdnet_final.pt", ratio=0.0, evenly=True, channel_ratio=0.05, save_path="models/pruned/channels_05/") 
    # prune("models/birdnet_v1/birdnet_final.pt", ratio=0.0, evenly=True, channel_ratio=0.1, save_path="models/pruned/channels_10/") 
    # prune("models/birdnet_v1/birdnet_final.pt", ratio=0.0, evenly=True, channel_ratio=0.15, save_path="models/pruned/channels_15/") 
    # prune("models/birdnet_v1/birdnet_final.pt", ratio=0.0, evenly=True, channel_ratio=0.2, save_path="models/pruned/channels_20/") 
    # prune("models/birdnet_v1/birdnet_final.pt", ratio=0.0, evenly=True, channel_ratio=0.3, save_path="models/pruned/channels_30/") 
    # prune("models/birdnet_v1/birdnet_final.pt", ratio=0.0, evenly=True, channel_ratio=0.4, save_path="models/pruned/channels_40/") 
    # prune("models/birdnet_v1/birdnet_final.pt", ratio=0.0, evenly=True, channel_ratio=0.5, save_path="models/pruned/channels_50/") 

    # prune("models/birdnet_v1/birdnet_final.pt", ratio=0.1, evenly=True, channel_ratio=0.1, save_path="models/pruned/block_10_channels_10/") 
    # prune("models/birdnet_v1/birdnet_final.pt", ratio=0.2, evenly=True, channel_ratio=0.2, save_path="models/pruned/block_20_channels_20/") 
    # prune("models/birdnet_v1/birdnet_final.pt", ratio=0.3, evenly=True, channel_ratio=0.3, save_path="models/pruned/block_30_channels_30/") 
    # prune("models/birdnet_v1/birdnet_final.pt", ratio=0.4, evenly=True, channel_ratio=0.4, save_path="models/pruned/block_40_channels_40/") 
    # prune("models/birdnet_v1/birdnet_final.pt", ratio=0.5, evenly=True, channel_ratio=0.5, save_path="models/pruned/block_50_channels_50/") 

    # prune("models/birdnet_v1/birdnet_final.pt", ratio=0.5, evenly=True, channel_ratio=0.1, save_path="models/pruned/block_50_channels_10/") 
    # prune("models/birdnet_v1/birdnet_final.pt", ratio=0.4, evenly=True, channel_ratio=0.2, save_path="models/pruned/block_40_channels_20/") 
    # prune("models/birdnet_v1/birdnet_final.pt", ratio=0.3, evenly=True, channel_ratio=0.3, save_path="models/pruned/block_30_channels_30/") 
    # prune("models/birdnet_v1/birdnet_final.pt", ratio=0.2, evenly=True, channel_ratio=0.4, save_path="models/pruned/block_20_channels_40/") 
    # prune("models/birdnet_v1/birdnet_final.pt", ratio=0.1, evenly=True, channel_ratio=0.5, save_path="models/pruned/block_10_channels_50/") 
