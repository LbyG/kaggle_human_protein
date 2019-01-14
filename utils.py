import os
import sys 
import json
import torch
import shutil
import numpy as np 
from config import config
from torch import nn
import torch.nn.functional as F 
from sklearn.metrics import f1_score
from torch.autograd import Variable
# save best model
def save_checkpoint(state, is_best_loss,is_best_f1,fold, epoch):
    filename = config.weights + config.model_name + os.sep +str(fold) + os.sep + str(epoch) + "checkpoint.pth.tar"
    torch.save(state, filename)
    if is_best_loss:
        shutil.copyfile(filename,"%s/%s_fold_%s_model_best_loss.pth.tar"%(config.best_models,config.model_name,str(fold)))
    if is_best_f1:
        shutil.copyfile(filename,"%s/%s_fold_%s_model_best_f1.pth.tar"%(config.best_models,config.model_name,str(fold)))
        
# evaluate meters
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

# print logger
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout  #stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None: mode ='w'
        self.file = open(file, mode)

    def write(self, message, is_terminal=1, is_file=1 ):
        if '\r' in message: is_file=0

        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()
            #time.sleep(1)

        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

class MyFocalLoss(nn.Module):
    def __init__(self, weight, alpha = 0.5, gamma=2):
        super(MyFocalLoss, self).__init__()
        self.weight = weight
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, x, y):
        '''Focal loss.
        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,D].
        Return:
          (tensor) focal loss.
        '''
        t = Variable(y).cuda()  # [N,20]

        p = x.sigmoid()
        pt = p*t + (1-p)*(1-t)         # pt = p if t > 0 else 1-p
        w = self.alpha*t + (1-self.alpha)*(1-t)  # w = alpha if t > 0 else 1-alpha
        #print("===========")
        #print(pt)
        #print(w)
        w = w * (1-pt).pow(self.gamma)
        #print(w)
        w = w * Variable(self.weight).cuda()
        #w_no_grad = Variable(w, requires_grad=False)
        #return F.binary_cross_entropy_with_logits(x, t, weight = w_no_grad)
        max_val = (-x).clamp(min=0)
        loss = x - x * y + max_val + ((-max_val).exp() + (-x - max_val).exp()).log()
        loss = w * loss
        return loss.mean()

class F1Loss(nn.Module):
    def __init__(self, small_val=1e-6,beta=1):
        super(F1Loss, self).__init__()
        self.small_value=1e-6
        self.beta = 1

    def forward(self, logits, labels):
        '''F1 loss.
        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,D].
        Return:
          (tensor) f1 loss.
        '''
        #print("logits.shape = {0}".format(logits.shape))
        #print("labels.shape = {0}".format(labels.shape))
        batch_size = logits.size()[0]
        p = F.sigmoid(logits)
        l = labels
        num_pos = torch.sum(p, 0) + self.small_value
        num_pos_hat = torch.sum(l, 0) + self.small_value
        tp = torch.sum(l * p, 0)
        precise = tp / num_pos
        recall = tp / num_pos_hat
        fs = (1 + self.beta * self.beta) * precise * recall / (self.beta * self.beta * precise + recall + self.small_value)
        loss = fs.sum() / len(fs)
        return (1 - loss)

def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
       lr +=[ param_group['lr'] ]

    #assert(len(lr)==1) #we support only one param_group
    lr = lr[0]

    return lr

def time_to_str(t, mode='min'):
    if mode=='min':
        t  = int(t)/60
        hr = t//60
        min = t%60
        return '%2d hr %02d min'%(hr,min)

    elif mode=='sec':
        t   = int(t)
        min = t//60
        sec = t%60
        return '%2d min %02d sec'%(min,sec)


    else:
        raise NotImplementedError
