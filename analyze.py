import os
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




class DataLabels():
    def __init__(self, path):
        self.path = path
        self.birds = os.listdir(path)
        self.birds = sorted(self.birds)
        self.num_classes = len(self.birds)
        self.filters = [[[32]], 
        [[16, 16, 32], [64, 64], [64, 64], [64, 64]], 
        [[32, 32, 64], [128, 128], [128, 128], [128, 128]], 
        [[64, 64, 128], [256, 256], [256, 256], [256, 256]], 
        [[128, 128, 256], [512, 512], [512, 512], [512, 512]],
        [512, 512, self.num_classes]]

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
    #Define Random seed for reproducibility
    torch.cuda.manual_seed(137)
    torch.manual_seed(735)
    
    #Assign Arguments
    args = parser.parse_args()
    mode = args.mode
    num_workers=int(args.threads)
    batch_size=args.batch_size
    lr=float(args.lr)
    gamma=float(args.gamma)
    delta=float(args.delta)
    dataset_path = args.dataset_path

    Path(args.save_path).mkdir(parents=True, exist_ok=True)


    data = DataLabels(dataset_path + "train/")

    if (args.load_path != ''):
        checkpoint = torch.load(args.load_path)
        birdnet = model.BirdNet(filters=checkpoint['filters'], padding_masks=checkpoint['padding_masks'])
        birdnet = torch.nn.DataParallel(birdnet).cuda()
        birdnet = birdnet.float()
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = optim.Adam(birdnet.parameters(), lr=lr) 
        birdnet.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else: 
        birdnet = model.BirdNet(filters=data.filters)
        birdnet = torch.nn.DataParallel(birdnet).cuda()
        birdnet = birdnet.float()
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = optim.Adam(birdnet.parameters(), lr=lr) 
    
    if (mode == 'train'):
        analyze = AnalyzeBirdnet(birdnet=birdnet, dataset=data, lr=lr, criterion=criterion, dataset_path=dataset_path, 
                    batch_size=batch_size, num_workers=num_workers, save_path=args.save_path, gamma=gamma, delta=delta)
        scaling_factor_mode = Scaling_Factor_Mode.SEPARATE if args.scaling_factors_mode == "separated" else Scaling_Factor_Mode.TOGETHER

        analyze.start_training(int(args.epochs), scaling_factor_mode)
    elif (mode == 'eval'):
        analyze = EvalBirdnet(birdnet=birdnet, dataset=data)
        result = analyze.eval(args.eval_file)
        for sample in result:
            print(sample)
    elif (mode == 'test'):
        #Start Training
        analyze = AnalyzeBirdnet(birdnet=birdnet, dataset=data, lr=lr, criterion=criterion, dataset_path=dataset_path, 
                    batch_size=batch_size, num_workers=num_workers, save_path=args.save_path, gamma=gamma, delta=delta)
        loss_subdivision, top1 = analyze.test(mode="val")
        print(f"accuracy val: {top1.avg}%")
        print(f"loss val: {loss_subdivision[0].avg}")
        loss_subdivision, top1 = analyze.test(mode="test")
        print(f"accuracy test: {top1.avg}%")
        print(f"loss test: {loss_subdivision[0].avg}")

if __name__ == '__main__':
    main()