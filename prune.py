import argparse
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

def retrain(birdnet, criterion, save_path, epochs=10, lr=0.001, dataset_path="1dataset/1data/calls/"):
    data = DataLabels(dataset_path + "train/",)

    train_dataset = CallsDataset(dataset_path + "train/")
    test_dataset = CallsDataset(dataset_path + "test/")
    train_loader = DataLoader(train_dataset, batch_size=16, num_workers=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, num_workers=16, shuffle=True)

    #Start Training
    analyze = AnalyzeBirdnet(birdnet=birdnet, dataset=data, lr=lr, criterion=criterion, train_loader=train_loader, 
                                test_loader=test_loader, save_path=save_path, gamma=0.2)
    analyze.start_training(epochs)

def prune(load_path, ratio, lr=0.001, mode=Channel_Pruning_Mode.NO_PADD, channel_ratio=0.5):
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
    birdnet = model.BirdNet(filters=filters)
    birdnet = torch.nn.DataParallel(birdnet).cuda()
    #Load parameter to model
    birdnet.load_state_dict(model_state_dict)

    model_state_dict, filters = PruneChannels.prune(model_state_dict, ratio, filters, mode, channel_ratio)

    birdnet = model.BirdNet(filters=filters)
    birdnet = torch.nn.DataParallel(birdnet).cuda()
    #Load parameter to model
    birdnet.load_state_dict(model_state_dict)
    optimizer = optim.Adam(birdnet.parameters(), lr=lr) 

    return birdnet


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path', default='', help='Load model from file')
    parser.add_argument('--save_path', default='', help='Load model from file')
    parser.add_argument('--epochs', default=10, help='Specify number of epochs for training')
    parser.add_argument('--channel_ratio', default=0.4)
    parser.add_argument('--block_ratio', default=0.4)
    parser.add_argument('--mode', default="NO_PADD")
    parser.add_argument('--train_set', default="1dataset/1data/calls/")


    args = parser.parse_args()
    channel_ratio = float(args.channel_ratio)
    block_ratio = float(args.block_ratio)
    load_path = args.load_path
    save_path = args.save_path
    train_set = args.train_set
    epochs = int(args.epochs)

    mode = args.mode 
    if mode == "NO_PADD":
        mode = Channel_Pruning_Mode.NO_PADD
    elif mode == "MIN":
        mode = Channel_Pruning_Mode.MIN
    else:
        mode = Channel_Pruning_Mode.EVENLY

    checkpoint = torch.load(load_path)
    model_state_dict = checkpoint['model_state_dict']

    birdnet = prune(load_path, ratio=block_ratio, mode=mode, channel_ratio=channel_ratio) 


    if save_path:
        criterion = nn.CrossEntropyLoss().cuda()
        retrain(birdnet, criterion, save_path=save_path, lr=0.001, dataset_path=train_set, epochs=epochs)
