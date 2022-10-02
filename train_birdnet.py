import re
import os
from enum import Enum
import time
from black import out
from matplotlib.pyplot import sca
import torch
import torch.optim as optim
from torch import nn, tensor, threshold
from data.data import CallsDataset
import model 
from torch.autograd import Variable

import numpy as np
from torchsummary import summary
from pathlib import Path
from utils.metrics import AverageMeter
from utils.metrics import accuracy
import utils.monitor as monitor
from utils import audio
from torch.utils.data import DataLoader
import wandb
import random


class Scaling_Factor_Mode(Enum):
    TOGETHER = 0
    SEPARATE = 1


threshold = 0.5

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class AnalyzeBirdnet():
    def __init__(self, birdnet, dataset, lr=0.001, criterion=nn.CrossEntropyLoss().cuda(), 
                    dataset_path="1dataset/1data/calls/", batch_size=16, num_workers=16, save_path=None, loss_patience=5, early_stopping=10, gamma=0.5, delta=0.5, device="cuda", seed_value=10):

        torch.cuda.manual_seed(seed_value)
        torch.manual_seed(seed_value)
        torch.use_deterministic_algorithms(True)
        random.seed(seed_value)
        g = torch.Generator()
        g.manual_seed(seed_value)
        self.device = device

        train_dataset = CallsDataset(dataset_path + "train/")
        test_dataset = CallsDataset(dataset_path + "test/")
        val_dataset = CallsDataset(dataset_path + "val/")

        #time_test_dataset = CallsDataset("/media/eddy/datasets/birdclef/")

        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, worker_init_fn=seed_worker, generator=g)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, worker_init_fn=seed_worker, generator=g)
        self.val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, worker_init_fn=seed_worker, generator=g)
        #self.time_test_loader = DataLoader(time_test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

        self.dataset = dataset
        self.gamma=gamma
        self.patience = loss_patience
        self.early_stopping = early_stopping
        self.save_path = os.path.join(save_path)
        self.criterion = criterion
        self.lr = lr
        if device == "cuda":
            self.criterion = nn.CrossEntropyLoss().cuda()
            self.birdnet = birdnet
        elif device == "cpu":
            self.criterion = nn.CrossEntropyLoss()
            self.birdnet = birdnet.module.to(torch.device('cpu'))

        self.optimizer = optim.Adam(self.birdnet.parameters(), lr=self.lr) 
        self.delta = delta

    def sum_block_scaling_parameters(self):
        counter = 0
        sum = 0
        if self.device == "cuda": 
            classifier = self.birdnet.module.classifier 
        else: 
            classifier = self.birdnet.classifier

        for resstack in classifier:
            if isinstance(resstack, model.ResStack):
                for resblock in resstack.classifier:
                    if isinstance(resblock, model.Resblock):
                        sum += torch.abs(resblock.W)
                        counter += 1
                    if isinstance(resblock, model.DownsamplingResBlock):
                        sum += torch.abs(resblock.W) / 3
                        counter += 1
        if self.device == "cuda": 
            sum.cuda()
        return sum, counter
    

    def sum_conv_layer_scaling_factors(self):
        counter = 0
        sum = 0
        if self.device == "cuda": 
            classifier = self.birdnet.module.classifier 
        else: 
            classifier = self.birdnet.classifier

        for resstack in classifier:
            if isinstance(resstack, model.ResStack):
                for resblock in resstack.classifier:
                    if isinstance(resblock, model.Resblock): 
                        fst_layer = resblock.classifier[3].weight.cpu()
                        snd_layer = resblock.classifier[7].weight.cpu()
                        sum += torch.sum(torch.abs(fst_layer))
                        sum += torch.sum(torch.abs(snd_layer))
                        counter += len(fst_layer)
                        counter += len(snd_layer)

                    if isinstance(resblock, model.DownsamplingResBlock): 
                        fst_layer = resblock.classifierPath[1].weight.cpu()
                        snd_layer = resblock.classifierPath[4].weight.cpu()
                        last_layer = resblock.batchnorm[0].weight.cpu()
                        sum += torch.sum(torch.abs(fst_layer))
                        sum += torch.sum(torch.abs(snd_layer))
                        sum += torch.sum(torch.abs(last_layer))
                        counter += len(fst_layer)
                        counter += len(snd_layer)
                        counter += len(last_layer)
                        

        if sum != 0:
            sum.cuda()
        else: 
            sum = torch.tensor([0]).cuda()  
        return sum, counter

    def prepare_data_and_labels(self, data, target):
        """
        Brings data and labels to cuda and maps target labels to one hot enconding array for classification
        """
  
           
        target = self.dataset.labels_to_one_hot_encondings(target)
        target= torch.from_numpy(target)

        if self.device == "cuda":
            data = data.cuda(non_blocking=True)    
            target = target.cuda(non_blocking=True)
        elif self.device == "cpu":
            pass
        data = Variable(data)  
        target = Variable(target)
        return data, target


    def calc_loss(self, output, target):
        loss = self.criterion(output.float(), target.float())
        sum_scaling_factors, num_scaling_factors  = self.sum_block_scaling_parameters()
        sum_channel_factors, num_channel_factors = self.sum_conv_layer_scaling_factors()


        if num_scaling_factors != 0:
            loss_scaling_factors = self.gamma * sum_scaling_factors / num_scaling_factors
        else: 
            loss_scaling_factors = torch.tensor([0]).cuda()
        
        if num_channel_factors != 0:
            loss_channel_factors = (1-self.gamma) * sum_channel_factors / num_channel_factors
        else: 
            loss_channel_factors = torch.tensor([0]).cuda()

        result = self.delta * (loss_scaling_factors + loss_channel_factors) + (1-self.delta) * loss 

        return result, self.delta * loss_scaling_factors, self.delta * loss_channel_factors, (1-self.delta) * loss 
    """
    Trains model for one epoch with given dataloader
    """
    def train(self, epoch):
        self.birdnet.train()
        losses = AverageMeter()
        losses_block = AverageMeter()
        losses_channel = AverageMeter()
        losses_acc = AverageMeter()
        top1 = AverageMeter()
        for idx, (data, target) in enumerate(self.train_loader):

            torch.cuda.empty_cache()

            data, target = self.prepare_data_and_labels(data, target)

            #Run model and backpropagate
            output = self.birdnet(data.float())
            output = np.squeeze(output)
            self.optimizer.zero_grad()

            if output.dim() == 1:
                continue

            loss, loss_block_part, loss_channel_part, loss_acc_part = self.calc_loss(output, target)
            loss.backward()
            self.optimizer.step()
            #Calculate and update metrics
            losses.update(loss.item(), data.size(0))
            losses_block.update(loss_block_part.item(), data.size(0))
            losses_channel.update(loss_channel_part.item(), data.size(0))
            losses_acc.update(loss_acc_part.item(), data.size(0))
            prec = accuracy(output.data, target)
            top1.update(prec, data.size(0))

            # if(idx % 100 == 0):
            #     print('epoch: {:d}, iteration {:d}, Loss: {loss.val:.4f},\t' 
            #           'Loss avg: {loss.avg:.4f}, Accuracy: {top1.val:.4f}, Avg Accuracy: {top1.avg:.4f}'.format(epoch, idx, loss=losses, top1=top1))
        
        loss_subdivision = [losses, losses_block, losses_channel, losses_acc]
        return loss_subdivision, top1


    """
    Tests model over data from dataloader
    """
    def test(self, mode="val"):
        self.birdnet.eval()
        losses = AverageMeter()
        losses_block = AverageMeter()
        losses_channel = AverageMeter()
        losses_acc = AverageMeter()
        top1 = AverageMeter()
        mean_time = AverageMeter()
        if mode == "test":
            data_loader = self.test_loader 
        elif mode == "val":
            data_loader = self.val_loader
        elif mode == "train":
            data_loader = self.train_loader
        else:
            data_loader = self.time_test_loader
            

        for data, target in data_loader:
            torch.cuda.empty_cache()

            data, target = self.prepare_data_and_labels(data, target)

            #Run model
            start = time.time()
            output = self.birdnet(data.float())
            stop = time.time()
            mean_time.update(stop-start)

            output = np.squeeze(output)
            
            if output.dim() == 1:
                continue
            loss, loss_block_part, loss_channel_part, loss_acc_part = self.calc_loss(output, target)
            
            #Calculate and update metrics
            losses.update(loss.item(), data.size(0))
            losses_block.update(loss_block_part.item(), data.size(0))
            losses_channel.update(loss_channel_part.item(), data.size(0))
            losses_acc.update(loss_acc_part.item(), data.size(0))

            prec = accuracy(output.data, target)
            top1.update(prec, data.size(0))
        loss_subdivision = [losses, losses_block, losses_channel, losses_acc]
        #print(f"{mode} has taken {mean_time.sum:.4f}s in average {mean_time.avg:.6f}s")
        return loss_subdivision, top1


    def save_model(self, epochs, birdnet, optimizer, val_loss, val_top1, 
                train_loss_list, test_loss_list, train_acc_list, test_acc_list, path, filters):
        path = os.path.join(path)
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
    
    def freeze_scaling_factors(self, bool=True):
        for param in self.birdnet.parameters():
            if np.shape(param) == torch.Size([2]):
                param.requires_grad = bool
            elif np.shape(param) != torch.Size([]):
                param.requires_grad = not bool


    """
    Train loop that trains the model for some epochs and handels learning rate reduction and checkpoint save
    """
    def start_training(self, epochs, scaling_factor_mode=Scaling_Factor_Mode.SEPARATE, save_mode=-1):
        self.summary()
        self.birdnet.train()
        monitoring = monitor.Monitor(self.patience, self.early_stopping)
        version = 0
        train_acc_list = []
        test_acc_list = []
        train_loss_subdivision_list = []
        test_loss_subdivision_list = []

        test_loss_subdivision, val_top1 = self.test(mode="val")
        test_loss_subdivision_list.append([test_loss_subdivision[0].avg, test_loss_subdivision[1].avg, test_loss_subdivision[2].avg, test_loss_subdivision[3].avg])
        test_acc_list.append(val_top1.avg) 
        print('\n\ntest loss avg: {val_loss.avg:.4f}, accuracy avg: {val_top1.avg:.4f}'.format(val_loss=test_loss_subdivision[0], val_top1=val_top1))
        
        
        print("Save checkpoint: " + self.save_path + "birdnet_raw.pt")
        self.save_model(0, self.birdnet, self.optimizer, test_loss_subdivision[0], val_top1, 
                    train_loss_subdivision_list, test_loss_subdivision_list, train_acc_list, test_acc_list, 
                    self.save_path + "birdnet_raw.pt", filters=self.birdnet.module.filters)

        
        print("Start Training")
        max_accuracy = 0
        for i in range(0, epochs):
            if scaling_factor_mode == Scaling_Factor_Mode.SEPARATE:
                self.freeze_scaling_factors(i%5==1)
            train_loss_subdivision, train_top1 = self.train(epoch=i)
            test_loss_subdivision, val_top1 = self.test()
            train_loss_subdivision_list.append([train_loss_subdivision[0].avg, train_loss_subdivision[1].avg, train_loss_subdivision[2].avg, 
                                                train_loss_subdivision[3].avg])
            test_loss_subdivision_list.append([test_loss_subdivision[0].avg, test_loss_subdivision[1].avg, test_loss_subdivision[2].avg, 
                                                test_loss_subdivision[3].avg])
            train_acc_list.append(train_top1.avg)
            test_acc_list.append(val_top1.avg) 


            print('epoch: {:d} \ntrain loss avg: {train_loss.avg:.4f}, accuracy avg: {train_top1.avg:.4f}\t'
                  '\ntest loss avg: {val_loss.avg:.4f}, accuracy avg: {val_top1.avg:.4f}'.format(i, train_loss=train_loss_subdivision[0],
                  train_top1=train_top1, val_loss=test_loss_subdivision[0], val_top1=val_top1))
            
            if scaling_factor_mode == Scaling_Factor_Mode.SEPARATE and i%5 != 1:
                status = monitoring.update(test_loss_subdivision[0].avg, test_acc_list.avg, lr=self.lr)
            elif scaling_factor_mode == Scaling_Factor_Mode.TOGETHER:
                status = monitoring.update(test_loss_subdivision[0].avg, val_top1.avg, lr=self.lr)
            
            if (status == monitor.Status.LEARNING_RATE):
                self.lr *= 0.5
            elif (status == monitor.Status.STOP):
                break 
            
            if save_mode == -1:
                if val_top1.avg > max_accuracy:
                    max_accuracy = val_top1.avg
                    print("Save checkpoint: " + self.save_path + "birdnet_v" + str(version) + ".pt")
                    self.save_model(i, self.birdnet, self.optimizer, test_loss_subdivision[0], val_top1, 
                        train_loss_subdivision_list, test_loss_subdivision_list, train_acc_list, test_acc_list, 
                        self.save_path + "birdnet_v" + str(version) + ".pt", filters=self.birdnet.module.filters)       
                    version += 1
            elif (i % save_mode == 0):
                print("Save checkpoint: " + self.save_path + "birdnet_v" + str(version) + ".pt")
                self.save_model(i, self.birdnet, self.optimizer, test_loss_subdivision[0], val_top1, 
                    train_loss_subdivision_list, test_loss_subdivision_list, train_acc_list, test_acc_list, 
                    self.save_path + "birdnet_v" + str(version) + ".pt", filters=self.birdnet.module.filters)       
                version += 1
        self.save_model(i, self.birdnet, self.optimizer, test_loss_subdivision[0], val_top1, 
            train_loss_subdivision_list, test_loss_subdivision_list, train_acc_list, test_acc_list, 
            self.save_path  + "birdnet_final.pt", filters=self.birdnet.module.filters)       
        print("Saved Model!")
    

    

    def summary(self):
        summary(self.birdnet, (1, 64, 384))

    def softmax(self, x):
        f_x = np.exp(x) / np.sum(np.exp(x))
        return f_x
