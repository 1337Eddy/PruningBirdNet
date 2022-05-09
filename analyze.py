from pickletools import optimize
from tkinter import Variable

import torch
import torch.optim as optim
from torch import device, nn
import model 
from torch.autograd import Variable
from torch.utils.data import DataLoader
from data import CallsDataset
from data import labels_to_one_hot_encondings as map_labels
import numpy as np
from torchsummary import summary


#Hyperparameter
lr = 0.05
num_workers=16
batch_size=16
epochs = 20


#Dataset
dataset = CallsDataset()
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=16, shuffle=True)

torch.cuda.manual_seed(1337)

#Model parameter
birdnet = model.BirdNet()
birdnet = birdnet.float()
birdnet = torch.nn.DataParallel(birdnet).cuda()
optimizer = optim.Adam(birdnet.parameters(), lr=lr) 
criterion = nn.MSELoss().cuda()


def accuracy(output, target):
    """Computes the precision@k for the specified values of k"""
    max_index_output = torch.max(output, 1)[1]
    max_index_target = torch.max(target, 1)[1]
    c_true = 0
    c_false = 0
    for i in range(0, len(max_index_output)):
        if (max_index_output[i]==max_index_target[i]):
            c_true += 1
        else:
            c_false += 1
    return c_true / (c_true + c_false)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(model, epoch, optimizer=optimizer, data_loader=train_loader):
    #print("Start Training")
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    for idx, (data, target) in enumerate(data_loader):
        torch.cuda.empty_cache()

        #Prepare data and labels 
        data = data.cuda(non_blocking=True)    
        data = Variable(data)       
        target = map_labels(target)
        target= torch.from_numpy(target)
        target = target.cuda(non_blocking=True)
        target = Variable(target)

        #Run model and backpropagate
        output = model(data.float())
        output = np.squeeze(output)
        optimizer.zero_grad()
        loss = criterion(output.float(), target.float())
        loss.backward()
        optimizer.step()

        #Calculate and update metrics
        losses.update(loss.item(), data.size(0))
        prec = accuracy(output.data, target)
        top1.update(prec, data.size(0))

        if(idx % 50 == 0):
            print('epoch: {:d}, iteration {:d}, Loss: {loss.val:.4f},\t' 
                  'Loss avg: {loss.avg:.4f}, Accuracy: {top1.val:.4f}, Avg Accuracy: {top1.avg:.4f}'.format(epoch, idx, loss=losses, top1=top1))
    return losses, top1
    
def test(model, data_loader=test_loader):
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    for data, target in data_loader:
        torch.cuda.empty_cache()

        #Prepare data and labels 
        data = data.cuda(non_blocking=True) 
        data = Variable(data)
        target = map_labels(target)
        target= torch.from_numpy(target)
        target = target.cuda(non_blocking=True)
        target = Variable(target)

        #Run model
        output = model(data.float())
        output = np.squeeze(output)
        loss = criterion(output.float(), target.float())

        #Calculate and update metrics
        losses.update(loss.item(), data.size(0))
        prec = accuracy(output.data, target)
        top1.update(prec, data.size(0))
    return losses, top1


"""
Train model for some epochs
"""
for i in range(0, epochs):
    for param in birdnet.parameters():
        print(param)
        break
    train_loss, train_top1 = train(model=birdnet, epoch=i)
    val_loss, val_top1 = test(model=birdnet)
    print('epoch: {:d} train loss: {train_loss.val:.4f} avg loss: {train_loss.avg:.4f} \t'
          '\ntest loss: {val_loss.val:.4f} avg: {val_loss.avg:.4f} accuracy avg: {val_top1.avg:.4f}'.format(i, train_loss=train_loss, val_loss=val_loss, val_top1=val_top1))

