import re
import os
from enum import Enum
from black import out
from matplotlib.pyplot import sca
import torch
import torch.optim as optim
from torch import nn, tensor, threshold
import model 
from torch.autograd import Variable

import numpy as np
from torchsummary import summary
from pathlib import Path
from utils.metrics import AverageMeter
from utils.metrics import accuracy
import utils.monitor as monitor
from utils import audio

import wandb

class Scaling_Factor_Mode(Enum):
    TOGETHER = 0
    SEPARATE = 1



threshold = 0.5


class AnalyzeBirdnet():
    def __init__(self, birdnet, dataset, lr=0.001, criterion=nn.CrossEntropyLoss().cuda(), 
                    train_loader=None, test_loader=None, save_path=None, loss_patience=1, early_stopping=2, gamma=0.5, delta=0.5):

        torch.cuda.manual_seed(1337)
        torch.manual_seed(73)
        self.dataset = dataset
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
        self.delta = delta

    def sum_scaling_parameters(self):
        counter = 0
        sum = 0
        fun = nn.Softmax(dim=0)
        for param in self.birdnet.parameters():
            if np.shape(param) == torch.Size([2]):
                scaling_factors = fun(param)
                sum += scaling_factors[0]
                counter += 1
        sum.cuda()
        return sum, counter
    

    def sum_conv_layer_scaling_factors(self):
        counter = 0
        sum = 0
        for resstack in self.birdnet.module.classifier:
            if isinstance(resstack, model.ResStack):
                for resblock in resstack.classifier:
                    if isinstance(resblock, model.Resblock): 
                        fst_layer = resblock.classifier[3].weight.cpu()
                        snd_layer = resblock.classifier[7].weight.cpu()
                        sum += torch.sum(torch.abs(fst_layer))
                        sum += torch.sum(torch.abs(snd_layer))
                        counter += len(fst_layer)
                        counter += len(snd_layer)
        if sum != 0:
            sum.cuda()
        else: 
            sum = torch.tensor([0]).cuda()  
        return sum, counter

    def prepare_data_and_labels(self, data, target):
        """
        Brings data and labels to cuda and maps target labels to one hot enconding array for classification
        """
        data = data.cuda(non_blocking=True)    
        data = Variable(data)       
        target = self.dataset.labels_to_one_hot_encondings(target)
        target= torch.from_numpy(target)
        target = target.cuda(non_blocking=True)
        target = Variable(target)
        return data, target


    def calc_loss(self, output, target):
        loss = self.criterion(output.float(), target.float())
        sum_scaling_factors, num_scaling_factors  = self.sum_scaling_parameters()
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
    def test(self):
        self.birdnet.eval()
        losses = AverageMeter()
        losses_block = AverageMeter()
        losses_channel = AverageMeter()
        losses_acc = AverageMeter()
        top1 = AverageMeter()
        for data, target in self.test_loader:
            torch.cuda.empty_cache()

            data, target = self.prepare_data_and_labels(data, target)

            #Run model
            output = self.birdnet(data.float())
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
        return loss_subdivision, top1


    def save_model(self, epochs, birdnet, optimizer, val_loss, val_top1, 
                train_loss_list, test_loss_list, train_acc_list, test_acc_list, path, filters):
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
                'padding_masks': birdnet.module.padding_masks,
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
    def start_training(self, epochs, scaling_factor_mode=Scaling_Factor_Mode.SEPARATE):
        # wandb.init(project="birdnet")
        # wandb.config = {
        #     "learning_rate": self.lr, 
        #     "epochs": epochs, 
        #     "gamma": self.gamma, 
        #     "delta": self.delta 
        # }
        # wandb.watch(self.birdnet)
        self.birdnet.train()
        monitoring = monitor.Monitor(self.loss_patience, self.early_stopping)
        version = 0
        train_acc_list = []
        test_acc_list = []
        train_loss_subdivision_list = []
        test_loss_subdivision_list = []

        test_loss_subdivision, val_top1 = self.test()
        test_loss_subdivision_list.append([test_loss_subdivision[0].avg, test_loss_subdivision[1].avg, test_loss_subdivision[2].avg, test_loss_subdivision[3].avg])
        test_acc_list.append(val_top1.avg) 
        # wandb.log({"loss": test_loss_subdivision[0].avg, "accuracy": val_top1.avg})
        print('\n\ntest loss avg: {val_loss.avg:.4f}, accuracy avg: {val_top1.avg:.4f}'.format(val_loss=test_loss_subdivision[0], val_top1=val_top1))
        
        
        print("Save checkpoint: " + self.save_path + "birdnet_raw.pt")
        self.save_model(0, self.birdnet, self.optimizer, test_loss_subdivision[0], val_top1, 
                    train_loss_subdivision_list, test_loss_subdivision_list, train_acc_list, test_acc_list, 
                    self.save_path + "birdnet_raw.pt", filters=self.birdnet.module.filters)

        
        print("Start Training")

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

            # wandb.log({"loss": test_loss_subdivision[0].avg, "accuracy": val_top1.avg})

            print('epoch: {:d} \ntrain loss avg: {train_loss.avg:.4f}, accuracy avg: {train_top1.avg:.4f}\t'
                  '\ntest loss avg: {val_loss.avg:.4f}, accuracy avg: {val_top1.avg:.4f}'.format(i, train_loss=train_loss_subdivision[0],
                  train_top1=train_top1, val_loss=test_loss_subdivision[0], val_top1=val_top1))
            
            if scaling_factor_mode == Scaling_Factor_Mode.SEPARATE and i%5 != 1:
                status = monitoring.update(test_loss_subdivision[0].avg, lr=self.lr)
            elif scaling_factor_mode == Scaling_Factor_Mode.TOGETHER:
                status = monitoring.update(test_loss_subdivision[0].avg, lr=self.lr)
            
            if (status == monitor.Status.LEARNING_RATE):
                self.lr *= 0.5
            elif (status == monitor.Status.STOP):
                break 

            if (i % 5 == 0):
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
        summary(self.birdnet)

    def softmax(self, x):
        f_x = np.exp(x) / np.sum(np.exp(x))
        return f_x
