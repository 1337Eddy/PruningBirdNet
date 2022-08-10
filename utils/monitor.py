import sys
from enum import Enum

from attr import NOTHING 

class Status(Enum):
    NOTHING = 0
    LEARNING_RATE = 1
    STOP = 2

class Monitor():
    def __init__(self, patience, early_stopping):
        self.last_val_loss = 100000
        self.last_val_acc = 0
        self.patience = patience
        self.counter = 0
        self.early_stopping = early_stopping
        self.early_stopping_counter = 0
    
    def update(self, val_loss, val_acc, lr):
        if (val_acc < self.last_val_acc):
            self.counter += 1
            self.early_stopping_counter += 1
        else: 
            self.counter = 0
            self.early_stopping_counter = 0

            self.last_val_acc = val_acc
        
        if (self.early_stopping_counter >= self.early_stopping):
            print("Early stopping")
            return Status.STOP

        if (self.counter >= self.patience):
            self.counter = 0
            print("Reduce Learning rate to {:.8f}".format(lr*0.5))
            return Status.LEARNING_RATE
        
        return Status.NOTHING
        