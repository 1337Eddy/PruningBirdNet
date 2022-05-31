#from audioop import reverse
#from pickletools import optimize
#from tkinter import Variable
#from tkinter.tix import Tree
#from importlib_metadata import distribution
#from sqlalchemy import over

import os
import time
from unittest import result
import torch
import torch.optim as optim
from torch import device, nn, softmax, threshold
import model 
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import CallsDataset
from data import labels_to_one_hot_encondings as map_labels
from data import id_to_label
import numpy as np
from torchsummary import summary
from pathlib import Path
from metrics import AverageMeter
from metrics import accuracy
import monitor
import argparse
from utils import audio

num_classes = 83
threshold = 0.5

filters = [[[32]], 
[[16, 16, 32], [64, 64], [64, 64], [64, 64]], 
[[32, 32, 64], [128, 128], [128, 128], [128, 128]], 
[[64, 64, 128], [256, 256], [256, 256], [256, 256]], 
[[128, 128, 256], [512, 512], [512, 512], [512, 512]],
[512, 512, num_classes]]

class AnalyzeBirdnet():
    def __init__(self, birdnet, lr=0.001, criterion=nn.CrossEntropyLoss().cuda(), 
                    train_loader=None, test_loader=None, save_path=None, loss_patience=1, early_stopping=2, gamma=0.3):
        self.gamma=gamma
        self.loss_patience = loss_patience
        self.early_stopping = early_stopping
        self.save_path = save_path
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.birdnet = birdnet
        self.optimizer = optim.Adam(self.birdnet.parameters(), lr=self.lr) 

    def sum_scaling_parameters(self):
        sum = 0
        for resstack in self.birdnet.module.classifier:
            if isinstance(resstack, model.ResStack):
                for resblock in resstack.classifier:
                    if isinstance(resblock, model.Resblock):          
                        sum += torch.abs(resblock.W[0].cpu())
        sum.cuda()
        return sum

    def prepare_data_and_labels(self, data, target):
        data = data.cuda(non_blocking=True)    
        data = Variable(data)       
        target = map_labels(target)
        target= torch.from_numpy(target)
        target = target.cuda(non_blocking=True)
        target = Variable(target)
        return data, target

    """
    Trains model for one epoch with given dataloader
    """
    def train(self, epoch):
        self.birdnet.train()
        losses = AverageMeter()
        top1 = AverageMeter()
        for idx, (data, target) in enumerate(self.train_loader):

            torch.cuda.empty_cache()

            data, target = self.prepare_data_and_labels(data, target)

            #Run model and backpropagate
            output = self.birdnet(data.float())
            output = np.squeeze(output)
            self.optimizer.zero_grad()
            loss = self.criterion(output.float(), target.float())

            sum = self.sum_scaling_parameters()
            
            loss = loss + self.gamma * sum
            loss.backward()
            self.optimizer.step()
            #Calculate and update metrics
            losses.update(loss.item(), data.size(0))
            prec = accuracy(output.data, target)
            top1.update(prec, data.size(0))

            # if(idx % 100 == 0):
            #     print('epoch: {:d}, iteration {:d}, Loss: {loss.val:.4f},\t' 
            #           'Loss avg: {loss.avg:.4f}, Accuracy: {top1.val:.4f}, Avg Accuracy: {top1.avg:.4f}'.format(epoch, idx, loss=losses, top1=top1))
        return losses, top1


    """
    Tests model over data from dataloader
    """
    def test(self):
        self.birdnet.eval()
        losses = AverageMeter()
        top1 = AverageMeter()
        for data, target in self.test_loader:
            torch.cuda.empty_cache()

            data, target = self.prepare_data_and_labels(data, target)

            #Run model
            output = self.birdnet(data.float())
            output = np.squeeze(output)
            loss = self.criterion(output.float(), target.float())

            sum = self.sum_scaling_parameters()
            loss = loss + self.gamma * sum

            #Calculate and update metrics
            losses.update(loss.item(), data.size(0))
            prec = accuracy(output.data, target)
            top1.update(prec, data.size(0))
        return losses, top1


    def save_model(self, epochs, birdnet, optimizer, val_loss, val_top1, 
                train_loss_list, test_loss_list, train_acc_list, test_acc_list, path, filters=filters):
        Path(path[:-len(path.split('/')[-1])]).mkdir(parents=True, exist_ok=True)
        torch.save({
                'train_loss_list': train_loss_list,
                'test_loss_list': test_loss_list,
                'train_acc_list': train_acc_list,
                'test_acc_list': test_acc_list,
                'filters': filters, 
                'epoch': epochs,
                'model_state_dict': birdnet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'accuracy': val_top1
                }, path)

    """
    Train loop that trains the model for some epochs and handels learning rate reduction and checkpoint save
    """
    def start_training(self, epochs):
        self.birdnet.train()
        monitoring = monitor.Monitor(self.loss_patience, self.early_stopping)
        version = 0
        train_loss_list = []
        test_loss_list = []
        train_acc_list = []
        test_acc_list = []

        for i in range(0, epochs):
            train_loss, train_top1 = self.train(epoch=i)
            val_loss, val_top1 = self.test()
            train_loss_list.append(train_loss.avg)
            test_loss_list.append(val_loss.avg)
            train_acc_list.append(train_top1.avg)
            test_acc_list.append(val_top1.avg) 

            print('epoch: {:d} \ntrain loss avg: {train_loss.avg:.4f}, accuracy avg: {train_top1.avg:.4f}\t'
                  '\ntest loss avg: {val_loss.avg:.4f}, accuracy avg: {val_top1.avg:.4f}'.format(i, train_loss=train_loss,train_top1=train_top1, val_loss=val_loss, val_top1=val_top1))
            status = monitoring.update(val_loss.avg, lr=self.lr)
            if (status == monitor.Status.LEARNING_RATE):
                self.lr *= 0.5
            elif (status == monitor.Status.STOP):
                break 

            if (i % 5 == 0):
                print("Save checkpoint: " + self.save_path + "birdnet_v" + str(version) + ".pt")
                self.save_model(i, self.birdnet, self.optimizer, val_loss, val_top1, 
                    train_loss_list, test_loss_list, train_acc_list, test_acc_list, 
                    self.save_path + "birdnet_v" + str(version) + ".pt", filters=self.birdnet.module.filters)       
                version += 1

        self.save_model(i, self.birdnet, self.optimizer, val_loss, val_top1, 
            train_loss_list, test_loss_list, train_acc_list, test_acc_list, 
            self.save_path  + "birdnet_final.pt", filters=self.birdnet.module.filters)       
        print("Saved Model!")
    

    def evalFile(self, path, rate=44100, seconds=3, overlap=0, minlen=3, batchsize=16):
        sig, rate = audio.openAudioFile(path, rate)
        specs = audio.specsFromSignal(sig, rate, seconds=seconds, overlap=overlap, minlen=minlen)
        counter = 0
        time = 0
        predictions = []
        while (True):
            try:
                if (counter == 0):
                    batch = next(specs)
                    batch = batch[None, None, :, :]
                    counter +=1
                elif (counter < batchsize):
                    spec = next(specs)
                    batch = torch.cat((batch, spec[None, None, :, :]), dim=0)     
                    counter +=1 
                else: 
                    batch = batch.cuda(non_blocking=True)
                    output = self.birdnet(batch.float())   
                    output = torch.squeeze(output)
                    for pred in output:
                        estimation = self.softmax(np.array(pred.cpu().detach()))
                        index = np.argmax(estimation)
                        if estimation[index] > threshold:
                            prediction = id_to_label(index)
                            predictions += [(time, time + seconds, prediction, estimation[index])]
                            time += seconds - overlap
                        else: 
                            predictions += [(time, time + seconds, 'no call', None)]
                            time += seconds - overlap
                    counter = 0
                    batch = None
                
            except StopIteration:
                try:
                    if (batch != None):
                        batch = batch.cuda(non_blocking=True)
                        output = self.birdnet(batch.float())   
                        output = torch.squeeze(output)
                        if (np.shape(output) == torch.Size([num_classes])):
                            output = output[None, :]
                        for pred in output:
                            estimation = self.softmax(np.array(pred.cpu().detach()))
                            index = np.argmax(estimation)
                            if estimation[index] > threshold:
                                prediction = id_to_label(index)
                                predictions += [(time, time + seconds, prediction, estimation[index])]
                                time += seconds - overlap
                            else: 
                                predictions += [(time, time + seconds, 'no call', None)]
                                time += seconds - overlap
                        break
                except:
                    print(f"File {path} ist shorter than {seconds} seconds and can't be evaluated")
                    break
        return predictions


    def eval(self, path, rate=44100, seconds=3, overlap=0, minlen=3, batchsize=16):
        self.birdnet.eval()
        if os.path.isfile(path):
            return self.evalFile(path=path, rate=rate, seconds=seconds, overlap=overlap, minlen=minlen, batchsize=batchsize)
        elif os.path.isdir(path):
            files = os.listdir(path)
            files = filter(lambda x: ".wav" in x, files)
            predictions = []
            for file in files:
                result = self.evalFile(path=path + file, rate=rate, seconds=seconds, overlap=overlap, minlen=minlen, batchsize=batchsize)
                predictions.append((file, result))
            return predictions
        else:
            print("Error")
            exit()

    
    def softmax(self, x):
        f_x = np.exp(x) / np.sum(np.exp(x))
        return f_x



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', help='Set programm into train mode')
    parser.add_argument('--load_model', default='', help='Load model from file')
    parser.add_argument('--epochs', default=20, help='Specify number of epochs for training')
    parser.add_argument('--save_path', default='models/birdnet/', help='Specifies the path where final model and checkpoints are saved')
    parser.add_argument('--lr', default=0.001, help='Learning rate for training')
    parser.add_argument('--batch_size', default=16, help='Number of samples for one train batch')
    parser.add_argument('--threads', default=16)
    parser.add_argument('--gamma', default=0.2)
    parser.add_argument('--eval_file', default='/media/eddy/bachelor-arbeit/PruningBirdNet/1dataset/1data/1calls/arcter/XC582288-326656.wav')
    parser.add_argument('--dim_handling', default='PADD')
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

    #Model parameter
    birdnet = model.BirdNet(skip_handling=dim_handling)
    birdnet = torch.nn.DataParallel(birdnet).cuda()
    birdnet = birdnet.float()
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.Adam(birdnet.parameters(), lr=lr) 

    if (args.load_model != ''):
        checkpoint = torch.load(args.load_model)
        birdnet.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    
    if (mode == 'train'):
        #Load Data
        dataset = CallsDataset()
        train_size = int(len(dataset) * 0.8)
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

        #Start Training
        analyze = AnalyzeBirdnet(birdnet=birdnet, lr=lr, criterion=criterion, train_loader=train_loader, 
                                    test_loader=test_loader, save_path=args.save_path, gamma=gamma)
        analyze.start_training(int(args.epochs))
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
                            {'lr': 0.001, 'gamma': 0.5, 'epochs': 30, 'dim_handling': 'PADD'},
                            {'lr': 0.01, 'gamma': 0.2, 'epochs': 30, 'dim_handling': 'SKIP'}, 
                            {'lr': 0.005, 'gamma': 0.2, 'epochs': 30, 'dim_handling': 'SKIP'},
                            {'lr': 0.001, 'gamma': 0.2, 'epochs': 30, 'dim_handling': 'SKIP'},
                            {'lr': 0.0001, 'gamma': 0.2, 'epochs': 30, 'dim_handling': 'SKIP'},
                            {'lr': 0.001, 'gamma': 0.1, 'epochs': 30, 'dim_handling': 'SKIP'},
                            {'lr': 0.001, 'gamma': 0.3, 'epochs': 30, 'dim_handling': 'SKIP'},
                            {'lr': 0.001, 'gamma': 0.5, 'epochs': 30, 'dim_handling': 'SKIP'},]
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