import math
import torch


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, writer = None, name: str|None = None, interval: int| None = 1):
        assert (writer is not None and name is not None) or writer is None
        self.writer = writer
        self.name = name
        self.interval = interval
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val : float, step = None, n=1):
        if math.isnan(val):
            self.val = self.avg
        else:
            self.val = val
        self.sum += self.val * n
        self.count += n
        self.avg = self.sum / self.count

        if self.writer is not None and (self.count % self.interval == 0):
            if step is None:
                step = self.count
            self.writer.add_scalar(self.name, self.avg, step)

class ExponentialMeter(object):
    """Computes and stores the exponential average and current value"""
    def __init__(self, writer = None, name: str|None = None, init : float= None, weight : float = .3, interval: int| None = 1):
        self.writer = writer
        self.name = name
        self.interval = interval
        self.reset(init= init,weight = weight)

    def reset(self,init = None, weight : float = .3):
        assert 0 < weight <=1
        self.weight = weight
        if init is not None:
            self.val = init
            self.avg = init
            self.count = 0
        else:          
            self.val = 0
            self.avg = 0
            self.count = 0

    def update(self, val : float, step = None):
        if math.isnan(val):
            self.val = self.avg
        else:
            self.val = val
        self.count += 1
        self.avg = self.val if self.count == 1 else self.weight *self.val + (1 - self.weight) * self.avg

        if self.writer is not None and (self.count % self.interval == 0):
            if step is None:
                step = self.count
            self.writer.add_scalar(self.name, self.avg, step)

