from re import L
from attr import define
import torch
import torch.optim as optim
from torch import device, nn, softmax
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

filters = [[[32]], 
[[16, 16, 32], [64, 64], [64, 64], [64, 64]], 
[[32, 32, 64], [128, 128], [128, 128], [128, 128]], 
[[64, 64, 128], [256, 256], [256, 256], [256, 256]], 
[[128, 128, 256], [512, 512], [512, 512], [512, 512]],
[512, 512, 83]]



class AnalyzeBirdnet():
    def __init__(self, birdnet, lr=0.001, criterion=nn.CrossEntropyLoss().cuda(), 
                    train_loader=None, test_loader=None, save_path=None, loss_patience=1, early_stopping=2, gamma=0.3):
        self.gamma = gamma 
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
                        sum += torch.sum(torch.abs(resblock.W.cpu())) 
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

            if(idx % 100 == 0):
                print('epoch: {:d}, iteration {:d}, Loss: {loss.val:.4f},\t' 
                      'Loss avg: {loss.avg:.4f}, Accuracy: {top1.val:.4f}, Avg Accuracy: {top1.avg:.4f}'.format(epoch, idx, loss=losses, top1=top1))
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


    def save_model(self, epochs, birdnet, optimizer, val_loss, val_top1, path, filters=filters):
        torch.save({
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


        sum = 0
        for resstack in self.birdnet.module.classifier:
                if isinstance(resstack, model.ResStack):
                    for resblock in resstack.classifier:
                        if isinstance(resblock, model.Resblock):
                            values = np.array(resblock.W.cpu().detach())     
                            sum += np.sum(np.abs(values))
                            print(values)
        print(sum)

        for i in range(0, epochs):
            train_loss, train_top1 = self.train(epoch=i)
            val_loss, val_top1 = self.test()
            print('epoch: {:d} train loss: {train_loss.val:.4f}, avg loss: {train_loss.avg:.4f}, avg accuracy: {train_top1.avg:.4f}\t'
                  '\ntest loss: {val_loss.val:.4f}, avg: {val_loss.avg:.4f}, accuracy avg: {val_top1.avg:.4f}'.format(i, train_loss=train_loss,train_top1=train_top1, val_loss=val_loss, val_top1=val_top1))
            status = monitoring.update(val_loss.avg, lr=self.lr)
            if (status == monitor.Status.LEARNING_RATE):
                self.lr *= 0.5
            elif (status == monitor.Status.STOP):
                break 

            if (i % 5 == 0):
                print(f"Save checkpoint: {self.save_path} birdnet_v{version}.pt")
                self.save_model(i, self.birdnet, self.optimizer, val_loss, val_top1, self.save_path + f"birdnet_v{version}.pt")       
                version += 1

        sum = 0
        for resstack in self.birdnet.module.classifier:
                if isinstance(resstack, model.ResStack):
                    for resblock in resstack.classifier:
                        if isinstance(resblock, model.Resblock):
                            values = np.array(resblock.W.cpu().detach())     
                            sum += np.sum(np.abs(values))
                            print(values)
        print(sum)

        self.save_model(i, self.birdnet, self.optimizer, val_loss, val_top1, self.save_path  + "birdnet_final.pt")       
        print("Saved Model!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', help='Set programm into train mode')
    parser.add_argument('--load_model', default='', help='Load model from file')
    parser.add_argument('--epochs', default=40, help='Specify number of epochs for training')
    parser.add_argument('--save_path', default='models/birdnet/', help='Specifies the path where final model and checkpoints are saved')
    parser.add_argument('--lr', default=0.001, help='Learning rate for training')
    parser.add_argument('--batch_size', default=16, help='Number of samples for one train batch')
    parser.add_argument('--threads', default=16)
    parser.add_argument('--sparsity', default=0.2)

    #Define Random seed for reproducibility
    torch.cuda.manual_seed(1337)
    torch.manual_seed(73)
    
    #Assign Arguments
    args = parser.parse_args()
    mode = args.mode
    num_workers=int(args.threads)
    batch_size=args.batch_size#
    lr=float(args.lr)
    Path(args.save_path).mkdir(parents=True, exist_ok=True)
    sparsity = float(args.sparsity)

    #Model parameter
    birdnet = model.BirdNet()
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
        analyze = AnalyzeBirdnet(birdnet=birdnet, lr=lr, criterion=criterion, train_loader=train_loader, test_loader=test_loader, save_path=args.save_path, gamma=sparsity)
        analyze.start_training(int(args.epochs))
    elif (mode == 'eval'):
        analyze = AnalyzeBirdnet(birdnet=birdnet)
        result = analyze.eval("/media/eddy/bachelor-arbeit/PruningBirdNet/1dataset/1data/1calls/arcter/XC582288-326656.wav")
        print(result)

def drop_condition(tensor):
    if (tensor[0] < tensor[1]):
        return True
    else:
        return False
    

def load_model(ratio=0.2, lr=0.001):
    checkpoint = torch.load("models/paddtest/birdnet_final.pt")
    model_state_dict = checkpoint['model_state_dict']
    filters = checkpoint['filters']
    birdnet = model.BirdNet(filters=filters)
    birdnet = torch.nn.DataParallel(birdnet).cuda()
    birdnet = birdnet.float()
    optimizer = optim.Adam(birdnet.parameters(), lr=lr) 
    birdnet.load_state_dict(model_state_dict)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("Loaded old model")

    print(birdnet)


    # #Iterate over filters to build names of saved layers
    # for i in range(1, len(filters) - 1):
    #     for j in range (1, len(filters[i])):
    #         name = f"module.classifier.{i}.classifier.{j}."

    #         #If the condition to the custom weights is True drop the whole layer
    #         if (drop_condition(model_state_dict[name + "W"])):
    #             layers = list(model_state_dict.keys())
    #             remove_index.insert(0, (i,j)) 
    #             for layer in layers:
    #                 if layer.startswith(name):
    #                     model_state_dict.pop(layer)
    # for i, j in remove_index:
    #     del filters[i][j]


    # birdnet = model.BirdNet(filters=filters)
    # birdnet = torch.nn.DataParallel(birdnet).cuda()
    # birdnet = birdnet.float()
    # criterion = nn.CrossEntropyLoss().cuda()
    # optimizer = optim.Adam(birdnet.parameters(), lr=lr) 
    # birdnet.load_state_dict(model_state_dict)
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

if __name__ == '__main__':
    load_model()
