import argparse
from re import L
from typing import OrderedDict
import torch
import torch.optim as optim
from torch import device, nn, softmax, threshold
from analyze import DataLabels
from train_birdnet import AnalyzeBirdnet
import model 
from torch.utils.data import DataLoader
from data.data import CallsDataset
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

    return birdnet

def test(birdnet, criterion, save_path, epochs=10, lr=0.001, dataset_path="1dataset/1data/calls/"):
    data = DataLabels(dataset_path + "train/",)

    train_dataset = CallsDataset(dataset_path + "train/")
    test_dataset = CallsDataset(dataset_path + "test/")
    train_loader = DataLoader(train_dataset, batch_size=16, num_workers=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, num_workers=16, shuffle=True)

    #Start Training
    analyze = AnalyzeBirdnet(birdnet=birdnet, dataset=data, lr=lr, criterion=criterion, train_loader=train_loader, 
                                test_loader=test_loader, save_path=save_path, gamma=0.2)
    loss, top1 = analyze.test()

    return loss, top1

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

def show_multiple_stuff(load_path, mode, criterion, save_path, train_set, epochs=10):
    acc_list_channel = []
    acc_list_block = []
    acc_list_both = []
    acc_list_both_reverse = []

    print("Channels")
    for i in range(0, 100, 5):
        print(i/100)
        birdnet = prune(load_path, ratio=0, mode=mode, channel_ratio=i/100)
        loss, top1 = test(birdnet, criterion, save_path=save_path, lr=0.001, dataset_path=train_set, epochs=epochs)
        acc_list_channel.append((i, top1.avg))
    
    for i, elem in acc_list_channel:
        print(f"channel: {i} => {elem}")
    exit()
    print("Blocks")
    for i in range(0, 100, 10):
        print(i/100)
        birdnet = prune(load_path, ratio=i/100, mode=mode, channel_ratio=0)
        loss, top1 = test(birdnet, criterion, save_path=save_path, lr=0.001, dataset_path=train_set, epochs=epochs)
        acc_list_block.append((i, top1.avg))

    for i, elem in acc_list_block:
        print(f"block: {i} => {elem}")

    print("Both")
    for i in range(0, 100, 10):
        print(i/100)
        birdnet = prune(load_path, ratio=i/100, mode=mode, channel_ratio=i/100)
        loss, top1 = test(birdnet, criterion, save_path=save_path, lr=0.001, dataset_path=train_set, epochs=epochs)
        acc_list_both.append((i, top1.avg))
    
    for i, elem in acc_list_both:
        print(f"both: {i} => {elem}")

    print("Both reverse")
    for i in range(0, 100, 10):
        print(i/100)
        birdnet = prune(load_path, ratio=1-(i/100), mode=mode, channel_ratio=i/100)
        loss, top1 = test(birdnet, criterion, save_path=save_path, lr=0.001, dataset_path=train_set, epochs=epochs)
        acc_list_both_reverse.append((i, top1.avg))

    for i, elem in acc_list_both_reverse:
        print(f"block: {100 - i}, channel: {i} => {elem}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path', default='../models/birdnet/birdnet_final.pt', help='Load model from file')
    parser.add_argument('--save_path', default='test/', help='Load model from file')
    parser.add_argument('--epochs', default=10, help='Specify number of epochs for training')
    parser.add_argument('--channel_ratio', default=0.2)
    parser.add_argument('--block_ratio', default=0.0)
    parser.add_argument('--mode', default="MIN")
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
    elif mode == "EVENLY":
        mode = Channel_Pruning_Mode.EVENLY
    else: 
        raise RuntimeError('{mode} is no valid argument. Input NO_PADD, MIN or EVENLY')
        
    checkpoint = torch.load(load_path)
    model_state_dict = checkpoint['model_state_dict']

    criterion = nn.CrossEntropyLoss().cuda()

    #show_multiple_stuff(load_path=load_path, mode=Channel_Pruning_Mode.EVENLY, criterion=criterion, save_path="", train_set=train_set)

    birdnet = prune(load_path, ratio=block_ratio, mode=mode, channel_ratio=channel_ratio)

    if save_path:
        retrain(birdnet, criterion, save_path=save_path, lr=0.001, dataset_path=train_set, epochs=epochs)
