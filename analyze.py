import os
import random
import sys
import time
import numpy as np
import torch
import torch.optim as optim
from torch import nn
from eval import EvalBirdnet
from train_birdnet import AnalyzeBirdnet, Scaling_Factor_Mode
import model 
from torch.utils.data import DataLoader
from data.data import CallsDataset
from pathlib import Path
from utils.metrics import accuracy
import argparse
from utils import audio
import re
import flop_calc 



class DataLabels():
    def __init__(self, path, K=2):
        self.path = path
        self.birds = os.listdir(path)
        self.birds = sorted(self.birds)
        self.num_classes = len(self.birds)
        FILTERS = [8, 16, 32, 64, 128, 256]

        self.filters = [[[8*K]], 
        [[8*K, 8*K, 16*K], [16*K, 16*K], [16*K, 16*K], [16*K, 16*K]], 
        [[16*K, 16*K, 32*K], [32*K, 32*K], [32*K, 32*K], [32*K, 32*K]], 
        [[32*K, 32*K, 64*K], [64*K, 64*K], [64*K, 64*K], [64*K, 64*K]], 
        [[64*K, 64*K, 128*K], [128*K, 128*K], [128*K, 128*K], [128*K, 128*K]], 
        [128*K, 256*K, self.num_classes]]


        self.bird_dict = {x: self.birds.index(x) for x in self.birds}

    def labels_to_one_hot_encondings(sefl, labels):
        result = np.zeros((len(labels), len(sefl.birds)))
        for i in range(0, len(labels)):
            result[i][sefl.bird_dict[labels[i]]] = 1
        return result

    def id_to_label(self, id):
        return list(self.bird_dict)[id]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='eval', help='Set programm into train or evaluation mode')
    parser.add_argument('--load_path', default='', help='Load model from file')
    parser.add_argument('--epochs', default=20, help='Specify number of epochs for training')
    parser.add_argument('--save_path', default='models/birdnet/', help='Specifies the path where final model and checkpoints are saved')
    parser.add_argument('--lr', default=0.001, help='Learning rate for training')
    parser.add_argument('--batch_size', default=16, help='Number of samples for one train batch')
    parser.add_argument('--threads', default=16)
    parser.add_argument('--gamma', default=0.5)
    parser.add_argument('--delta', default=0.5)
    parser.add_argument('--eval_file', default='/media/eddy/bachelor-arbeit/PruningBirdNet/1dataset/1data/calls/train/arcter/')
    parser.add_argument('--dim_handling', default='PADD')
    parser.add_argument('--scaling_factors_mode', default='together', help='Defines if the scaling factors of the resblocks are trained together or separated')
    parser.add_argument('--dataset_path', default="1dataset/1data/calls/")
    parser.add_argument('--channel_multiplier', default=4)
    parser.add_argument('--seed', default=0)
    
    #Assign Arguments
    args = parser.parse_args()
    seed = int(args.seed)
    mode = args.mode
    num_workers=int(args.threads)
    batch_size=args.batch_size
    lr=float(args.lr)
    gamma=float(args.gamma)
    delta=float(args.delta)
    dataset_path = args.dataset_path
    channel_multiplier = int(args.channel_multiplier)

    dim_handling = args.dim_handling
    if dim_handling == "PADD":
        dim_handling = model.Dim_Handling.PADD
    elif dim_handling == "SKIP":
        dim_handling = model.Dim_Handling.SKIP

    Path(args.save_path).mkdir(parents=True, exist_ok=True)

    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    data = DataLabels(dataset_path + "train/", channel_multiplier)

    if (args.load_path != ''):
        checkpoint = torch.load(args.load_path)
        birdnet = model.BirdNet(filters=checkpoint['filters'], dimension_handling=dim_handling)
        birdnet = torch.nn.DataParallel(birdnet).cuda()
        birdnet = birdnet.float()
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = optim.Adam(birdnet.parameters(), lr=lr) 
        birdnet.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else: 
        birdnet = model.BirdNet(filters=data.filters, dimension_handling=dim_handling)
        birdnet = torch.nn.DataParallel(birdnet).cuda()
        birdnet = birdnet.float()
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = optim.Adam(birdnet.parameters(), lr=lr) 
    
    if (mode == 'train'):
        analyze = AnalyzeBirdnet(birdnet=birdnet, dataset=data, lr=lr, criterion=criterion, dataset_path=dataset_path, 
                    batch_size=batch_size, num_workers=num_workers, save_path=args.save_path, gamma=gamma, delta=delta, seed_value=seed)
        scaling_factor_mode = Scaling_Factor_Mode.SEPARATE if args.scaling_factors_mode == "separated" else Scaling_Factor_Mode.TOGETHER

        analyze.start_training(int(args.epochs), scaling_factor_mode)
    elif (mode == 'eval'):
        analyze = EvalBirdnet(birdnet=birdnet, dataset=data)
        result = analyze.eval(args.eval_file)
        for sample in result:
            print(sample)
    elif (mode == 'test'):
        analyze = AnalyzeBirdnet(birdnet=birdnet, dataset=data, lr=lr, criterion=criterion, dataset_path=dataset_path, 
                    batch_size=batch_size, num_workers=num_workers, save_path=args.save_path, gamma=gamma, delta=delta, device="cuda", seed_value=seed)

        total_params = 0
        for name, parameter in birdnet.named_parameters():
            if not parameter.requires_grad: 
                continue
            param = parameter.numel()
            total_params+=param
        num_flops = flop_calc.calc_flops(birdnet.module.filters, None, 384, 64)
        loss_subdivision, top1 = analyze.test(mode="test")
        sys.stdout.write(f"{top1.avg} {total_params} {num_flops}\n")

if __name__ == '__main__':
    main()