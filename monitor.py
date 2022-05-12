import sys
from enum import Enum

from attr import NOTHING 

class Status(Enum):
    NOTHING = 0
    LEARNING_RATE = 1
    STOP = 2

class Monitor():
    def __init__(self, loss_patience, early_stopping):
        self.last_val_loss = 10000
        self.loss_patience = loss_patience
        self.loss_counter = 0
        self.early_stopping = early_stopping
        self.early_stopping_counter = 0
    
    def update(self, val_loss, lr):
        if (val_loss >= self.last_val_loss):
            self.loss_counter += 1
            self.early_stopping_counter += 1
            self.last_val_loss = val_loss
        else: 
            self.loss_counter = 0
            self.early_stopping_counter = 0
            self.last_val_loss = val_loss
        
        if (self.loss_counter > self.loss_patience):
            self.loss_counter = 0
            print("Reduce Learning rate to {:.12f}".format(lr*0.5))
            return Status.LEARNING_RATE
        
        if (self.early_stopping_counter > self.early_stopping):
            print("Early stopping")
            return Status.STOP
        
        return Status.NOTHING
        