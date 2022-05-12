import torch

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