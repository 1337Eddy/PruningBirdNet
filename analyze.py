#from audioop import reverse
#from pickletools import optimize
#from tkinter import Variable
#from tkinter.tix import Tree
#from importlib_metadata import distribution
#from sqlalchemy import over

import time
import torch
import torch.optim as optim
from torch import nn
from analyze_birdnet import AnalyzeBirdnet, Scaling_Factor_Mode
import model 
from torch.utils.data import DataLoader
from data import CallsDataset
from pathlib import Path
from metrics import accuracy
import argparse
from utils import audio
import re


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='eval', help='Set programm into train or evaluation mode')
    parser.add_argument('--load_model', default='', help='Load model from file')
    parser.add_argument('--epochs', default=20, help='Specify number of epochs for training')
    parser.add_argument('--save_path', default='models/birdnet/', help='Specifies the path where final model and checkpoints are saved')
    parser.add_argument('--lr', default=0.001, help='Learning rate for training')
    parser.add_argument('--batch_size', default=16, help='Number of samples for one train batch')
    parser.add_argument('--threads', default=16)
    parser.add_argument('--gamma', default=0.2)
    parser.add_argument('--delta', default=0.0005)
    parser.add_argument('--eval_file', default='/media/eddy/bachelor-arbeit/PruningBirdNet/1dataset/1data/1calls/arcter/XC582288-326656.wav')
    parser.add_argument('--dim_handling', default='PADD')
    parser.add_argument('--scaling_factors_mode', default='separated', help='Defines if the scaling factors of the resblocks are trained together or separated')
    #Define Random seed for reproducibility
    torch.cuda.manual_seed(1337)
    torch.manual_seed(73)
    
    #Assign Arguments
    args = parser.parse_args()
    mode = args.mode
    num_workers=int(args.threads)
    batch_size=args.batch_size
    lr=float(args.lr)
    gamma=float(args.gamma)
    delta=float(args.delta)
    dim_handling = args.dim_handling

    if (dim_handling == "PADD"):
        dim_handling = model.Skip_Handling.PADD
    elif (dim_handling == "SKIP"):
        dim_handling = model.Skip_Handling.SKIP
    elif (dim_handling == "CUT"):
        dim_handling = model.Skip_Handling.CUT
    else:
        print(f"Error: {dim_handling} is no valid argument")
        exit()

    Path(args.save_path).mkdir(parents=True, exist_ok=True)


    if (args.load_model != ''):
        checkpoint = torch.load(args.load_model)
        birdnet = model.BirdNet(filters=checkpoint['filters'], skip_handling=dim_handling)
        birdnet = torch.nn.DataParallel(birdnet).cuda()
        birdnet = birdnet.float()
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = optim.Adam(birdnet.parameters(), lr=lr) 
        birdnet.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else: 
            birdnet = model.BirdNet(skip_handling=dim_handling)
            birdnet = torch.nn.DataParallel(birdnet).cuda()
            birdnet = birdnet.float()
            criterion = nn.CrossEntropyLoss().cuda()
            optimizer = optim.Adam(birdnet.parameters(), lr=lr) 
    
    if (mode == 'train'):
        #Load Data
        # dataset = CallsDataset()
        # train_size = int(len(dataset) * 0.8)
        # test_size = len(dataset) - train_size
        train_dataset = CallsDataset("1dataset/1data/calls/train/")
        test_dataset = CallsDataset("1dataset/1data/calls/test/")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

        #Start Training
        analyze = AnalyzeBirdnet(birdnet=birdnet, lr=lr, criterion=criterion, train_loader=train_loader, 
                                    test_loader=test_loader, save_path=args.save_path, gamma=gamma, delta=delta)
        scaling_factor_mode = Scaling_Factor_Mode.SEPARATE if args.scaling_factors_mode == "separated" else Scaling_Factor_Mode.TOGETHER

        analyze.start_training(int(args.epochs), scaling_factor_mode)
    elif (mode == 'eval'):
        analyze = AnalyzeBirdnet(birdnet=birdnet)
        result = analyze.eval(args.eval_file)
        for sample in result:
            print(sample)
    elif (mode == 'hpvt'):
        hyperparameter = [  {'lr': 0.01, 'gamma': 0.2, 'epochs': 30, 'dim_handling': 'PADD'}, 
                            {'lr': 0.005, 'gamma': 0.2, 'epochs': 30, 'dim_handling': 'PADD'},
                            {'lr': 0.001, 'gamma': 0.2, 'epochs': 30, 'dim_handling': 'PADD'},
                            {'lr': 0.0001, 'gamma': 0.2, 'epochs': 30, 'dim_handling': 'PADD'},
                            {'lr': 0.001, 'gamma': 0.1, 'epochs': 30, 'dim_handling': 'PADD'},
                            {'lr': 0.001, 'gamma': 0.3, 'epochs': 30, 'dim_handling': 'PADD'},
                            {'lr': 0.001, 'gamma': 0.5, 'epochs': 30, 'dim_handling': 'PADD'},]
        #Load Data
        dataset = CallsDataset()
        train_size = int(len(dataset) * 0.8)
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

        for set in hyperparameter:

            birdnet = model.BirdNet(skip_handling=set['dim_handling'])
            birdnet = torch.nn.DataParallel(birdnet).cuda()
            birdnet = birdnet.float()
            criterion = nn.CrossEntropyLoss().cuda()
            optimizer = optim.Adam(birdnet.parameters(), lr=lr) 

            #Start Training
            save_path = args.save_path + f"lr_{set['lr']}_gamma_{set['gamma']}_dim_handling_{set['dim_handling']}/"
            Path(save_path).mkdir(parents=True, exist_ok=True)
            analyze = AnalyzeBirdnet(birdnet=birdnet, lr=set['lr'], criterion=criterion, train_loader=train_loader, 
                                        test_loader=test_loader, save_path=save_path, gamma=set['gamma'])
            start_time = time.time()
            analyze.start_training(set['epochs'])
            stop_time = time.time()
            print(f"{stop_time - start_time} seconds \n")


if __name__ == '__main__':
    main()