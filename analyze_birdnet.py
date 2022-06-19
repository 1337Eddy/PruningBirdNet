import re
import os
from enum import Enum
import torch
import torch.optim as optim
from torch import nn, tensor, threshold
import model 
from torch.autograd import Variable

import numpy as np
from torchsummary import summary
from pathlib import Path
from metrics import AverageMeter
from metrics import accuracy
import monitor
from utils import audio

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
        sum.cuda()  
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

        loss_scaling_factors = self.gamma * sum_scaling_factors / num_scaling_factors
        loss_channel_factors = (1-self.gamma) * sum_channel_factors / num_channel_factors

        loss = self.delta * (loss_scaling_factors + loss_channel_factors) + (1-self.delta) * loss 

        return loss
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

            loss = self.calc_loss(output, target)
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

            loss = self.calc_loss(output, target)

            #Calculate and update metrics
            losses.update(loss.item(), data.size(0))
            prec = accuracy(output.data, target)
            top1.update(prec, data.size(0))
        return losses, top1


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
        self.birdnet.train()
        monitoring = monitor.Monitor(self.loss_patience, self.early_stopping)
        version = 0
        train_loss_list = []
        test_loss_list = []
        train_acc_list = []
        test_acc_list = []

        val_loss, val_top1 = self.test()
        test_loss_list.append(val_loss.avg)
        test_acc_list.append(val_top1.avg) 
        print('\n\ntest loss avg: {val_loss.avg:.4f}, accuracy avg: {val_top1.avg:.4f}'.format(val_loss=val_loss, val_top1=val_top1))
        print("Start Training")

        for i in range(0, epochs):
            if scaling_factor_mode == Scaling_Factor_Mode.SEPARATE:
                self.freeze_scaling_factors(i%5==1)
            train_loss, train_top1 = self.train(epoch=i)
            val_loss, val_top1 = self.test()
            train_loss_list.append(train_loss.avg)
            test_loss_list.append(val_loss.avg)
            train_acc_list.append(train_top1.avg)
            test_acc_list.append(val_top1.avg) 

            print('epoch: {:d} \ntrain loss avg: {train_loss.avg:.4f}, accuracy avg: {train_top1.avg:.4f}\t'
                  '\ntest loss avg: {val_loss.avg:.4f}, accuracy avg: {val_top1.avg:.4f}'.format(i, train_loss=train_loss,train_top1=train_top1, val_loss=val_loss, val_top1=val_top1))
            
            if scaling_factor_mode == Scaling_Factor_Mode.SEPARATE and i%5 != 1:
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
        time_0 = 0
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
                            prediction = self.dataset.id_to_label(index)
                            predictions += [(time_0, time_0 + seconds, prediction, estimation[index])]
                            time_0 += seconds - overlap
                        else: 
                            predictions += [(time_0, time_0 + seconds, 'no call', None)]
                            time_0 += seconds - overlap
                    counter = 0
                    batch = None
                
            except StopIteration:
                try:
                    if (batch != None):
                        batch = batch.cuda(non_blocking=True)
                        output = self.birdnet(batch.float())   
                        output = torch.squeeze(output)
                        if (np.shape(output) == torch.Size([self.dataset.num_classes])):
                            output = output[None, :]
                        for pred in output:
                            estimation = self.softmax(np.array(pred.cpu().detach()))
                            index = np.argmax(estimation)
                            if estimation[index] > threshold:
                                prediction = self.datasetid_to_label(index)
                                predictions += [(time_0, time_0 + seconds, prediction, estimation[index])]
                                time_0 += seconds - overlap
                            else: 
                                predictions += [(time_0, time_0 + seconds, 'no call', None)]
                                time_0 += seconds - overlap
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

    def summary(self):
        summary(self.birdnet)

    def softmax(self, x):
        f_x = np.exp(x) / np.sum(np.exp(x))
        return f_x
