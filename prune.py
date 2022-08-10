import argparse
from re import L
from typing import OrderedDict
from pyrsistent import s
import torch
import torch.optim as optim
from torch import device, nn, softmax, threshold
from analyze import DataLabels
from train_birdnet import AnalyzeBirdnet, Scaling_Factor_Mode
import model 
from torch.utils.data import DataLoader
from data.data import CallsDataset
from enum import Enum
from strucprune import PruneBlocks 
from strucprune import PruneChannels
 
class Channel_Pruning_Mode(Enum):
    EVENLY = 0
    NO_PADD = 1
    MIN = 2
    CURL = 3

class Pruning_Structure(Enum):
    RESBLOCK = 0
    ALL = 1

def retrain(birdnet, criterion, save_path, epochs=10, lr=0.001, dataset_path="1dataset/1data/calls/", scaling_factor_mode=Scaling_Factor_Mode.SEPARATE):
    data = DataLabels(dataset_path + "train/",)
    analyze = AnalyzeBirdnet(birdnet=birdnet, dataset=data, lr=lr, criterion=criterion, dataset_path=dataset_path,
                            num_workers=16, batch_size=16, save_path=save_path, gamma=0.5, delta=0.5)
    analyze.start_training(epochs, scaling_factor_mode=scaling_factor_mode)

    return birdnet

def test(birdnet, criterion, save_path, epochs=10, lr=0.001, dataset_path="1dataset/1data/calls/"):
    data = DataLabels(dataset_path + "train/",)
    analyze = AnalyzeBirdnet(birdnet=birdnet, dataset=data, lr=lr, criterion=criterion, dataset_path=dataset_path,
                        num_workers=16, batch_size=16, save_path=save_path, gamma=0.2)
    loss, top1 = analyze.test(mode="val")

    return loss, top1

def prune(load_path, ratio, lr=0.001, mode=Channel_Pruning_Mode.NO_PADD, channel_ratio=0.5, 
                    dim_handling=model.Dim_Handling.PADD, prune_structure="ALL", block_temperatur=0):
    checkpoint = torch.load(load_path)
    model_state_dict = checkpoint['model_state_dict']

    filters = checkpoint['filters']
    birdnet = model.BirdNet(filters=filters, dimension_handling=dim_handling)
    birdnet = torch.nn.DataParallel(birdnet).cuda()
    birdnet = birdnet.float()
    optimizer = optim.Adam(birdnet.parameters(), lr=lr) 
    birdnet.load_state_dict(model_state_dict)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


    model_state_dict, filters = PruneBlocks.prune(model_state_dict, filters, ratio)
    birdnet = model.BirdNet(filters=filters, dimension_handling=dim_handling)
    birdnet = torch.nn.DataParallel(birdnet).cuda()
    #Load parameter to model
    birdnet.load_state_dict(model_state_dict)

    model_state_dict, filters = PruneChannels.prune(model_state_dict, ratio, filters, mode, channel_ratio, prune_structure, block_temperatur)

    birdnet = model.BirdNet(filters=filters, dimension_handling=dim_handling)
    birdnet = torch.nn.DataParallel(birdnet).cuda()
    #Load parameter to model

    birdnet.load_state_dict(model_state_dict)
    optimizer = optim.Adam(birdnet.parameters(), lr=lr) 

    return birdnet

def check_ratios(channel_ratio):
    if channel_ratio > 1.0 or channel_ratio < 0.0:
        raise RuntimeError(f'channel_ratio has to be between 0 and 1. {channel_ratio} is not valid')

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path', default='/media/eddy/datasets/models/prune_ds/birdnet_final.pt', help='Load model from file')
    parser.add_argument('--save_path', default='test/', help='Load model from file')
    parser.add_argument('--finetune', default='True')
    parser.add_argument('--epochs', default=10, help='Specify number of epochs for training')
    parser.add_argument('--channel_ratio', default=0.3)
    parser.add_argument('--block_ratio', default=0)
    parser.add_argument('--mode', default="CURL")
    parser.add_argument('--train_set', default="1dataset/1data/calls/")
    parser.add_argument('--scaling_factors_mode', default='together')
    parser.add_argument('--dim_handling', default='PADD')
    parser.add_argument('--simultaneous', default="True")
    parser.add_argument('--prune_structure', default="ALL")
    parser.add_argument('--block_temperatur', default=0)


    args = parser.parse_args()
    channel_ratio = float(args.channel_ratio) 
    block_ratio = int(args.block_ratio)  
    load_path = args.load_path
    save_path = args.save_path
    train_set = args.train_set
    epochs = int(args.epochs)
    scaling_factor_mode = args.scaling_factors_mode
    scaling_factor_mode = Scaling_Factor_Mode.SEPARATE if args.scaling_factors_mode == "separated" else Scaling_Factor_Mode.TOGETHER
    block_temperatur = float(args.block_temperatur)



    if args.prune_structure == "RESBLOCK":    
        prune_structure = Pruning_Structure.RESBLOCK
    elif args.prune_structure == "ALL":
        prune_structure = Pruning_Structure.ALL
    else:
        raise RuntimeError(f'{args.prune_structure} is no valid argument. Input RESBLOCK or ALL')


    check_ratios(channel_ratio)



    dim_handling = args.dim_handling
    if dim_handling == "PADD":
        dim_handling = model.Dim_Handling.PADD
    elif dim_handling == "SKIP":
        dim_handling = model.Dim_Handling.SKIP
    else: 
        raise RuntimeError(f'{dim_handling} is no valid argument. Input PADD or SKIP')


    finetune = args.finetune
    if finetune == "True":
        finetune = True 
    elif finetune == "False":
        finetune = False 
    else: 
        raise RuntimeError(f'{finetune} is no valid argument. Input True or False')

    simultaneous = args.simultaneous
    if simultaneous == "True":
        simultaneous = True 
    elif simultaneous == "False":
        simultaneous = False 
    else: 
        raise RuntimeError(f'{simultaneous} is no valid argument. Input True or False')

    mode = args.mode 
    if mode == "NO_PADD":
        mode = Channel_Pruning_Mode.NO_PADD
    elif mode == "MIN":
        mode = Channel_Pruning_Mode.MIN
    elif mode == "EVENLY":
        mode = Channel_Pruning_Mode.EVENLY
    elif mode == "CURL":
        mode = Channel_Pruning_Mode.CURL
    else: 
        raise RuntimeError('{mode} is no valid argument. Input NO_PADD, MIN, CURL or EVENLY')
    
    return channel_ratio, block_ratio, mode, load_path, save_path, finetune, simultaneous, epochs, train_set, scaling_factor_mode, dim_handling, prune_structure, block_temperatur

if __name__ == '__main__':
    channel_ratio, block_ratio, mode, load_path, save_path, finetune, simultaneous, epochs, train_set, scaling_factor_mode, dim_handling, prune_structure, block_temperatur = parse_arguments()
    
    folder_name = f"pruned_c{int(100*channel_ratio)}_b{int(block_ratio)}_{mode._name_}_temp{block_temperatur}_mode{prune_structure.name}/"
    save_path += folder_name        
    checkpoint = torch.load(load_path)
    model_state_dict = checkpoint['model_state_dict']
    criterion = nn.CrossEntropyLoss().cuda()


    if simultaneous or (block_ratio == 0.0 or channel_ratio == 0.0):
        birdnet = prune(load_path, ratio=block_ratio, mode=mode, channel_ratio=channel_ratio, 
                dim_handling=dim_handling, prune_structure=prune_structure, block_temperatur=block_temperatur)
        if finetune:
            retrain(birdnet, criterion, save_path=save_path, lr=0.001, dataset_path=train_set, epochs=epochs, scaling_factor_mode=scaling_factor_mode)
    else: 
        birdnet = prune(load_path, ratio=block_ratio, mode=mode, channel_ratio=0.0, dim_handling=dim_handling)
        retrain(birdnet, criterion, save_path=save_path, lr=0.001, dataset_path=train_set, epochs=epochs, scaling_factor_mode=scaling_factor_mode)
        birdnet = prune(save_path + "birdnet_final.pt", ratio=0.0, mode=mode, channel_ratio=channel_ratio, dim_handling=dim_handling)
        retrain(birdnet, criterion, save_path=save_path, lr=0.001, dataset_path=train_set, epochs=epochs, scaling_factor_mode=scaling_factor_mode)


