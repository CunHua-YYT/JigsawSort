# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
from typing import Iterable, Optional
import time

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

import utils

import tqdm

def accuracy(output, target, topk=(1,)):
    #  来自官方pytorch的代码
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res




def train_one_epochnouse(model: torch.nn.Module, criterion,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode = True, use_amp=False):
    #来自 vitsilm  deit
    #  来自官方pytorch的代码
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for i,(samples, targets) in enumerate (metric_logger.log_every(data_loader, print_freq, header)):
        start_time = time.time()
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(samples)
                loss = criterion(samples, outputs, targets)
        else:
            outputs = model(samples)
            loss = criterion(samples, outputs, targets)


        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if use_amp:
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))

        batch_size = samples.shape[0]
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))                
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    #  来自官方pytorch的代码
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    #with torch.inference_mode():

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)


        output = model(images)
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_epoch(model, loader, optimizer, loss_fn,device, args):

    #from tranmix

    model.train()
    #with tqdm.tqdm(total= len(loader)) as mytqdm:
        #mytqdm.set_description('epoch:')
        # for batch_idx, (input, target64,target100) in enumerate(loader):
        #     input, target64,target100 = input.to(device), target64.to(device), target100.to(device)
        #     output = model(input)

        #     target_t = torch.stack([target64,target100])
        #     target = target_t.t()
        #     loss = loss_fn(output.float(), target.float())
        #     optimizer.zero_grad() 
        #     loss.backward(torch.ones_like(loss))
        #     #print("4")
        #     optimizer.step()    

        #     #for i in range(tqdmsteps)
        #     mytqdm.set_postfix(batch_idxandloss='{:.6f},{:.6f}'.format(batch_idx,loss.item()))
        #     mytqdm.update(len(target64))

    mytqdm = tqdm.tqdm(loader,desc ="Text You Want")
    for batch_idx, (input, target64,target100) in enumerate(mytqdm):

        input, target64,target100 = input.to(device), target64.to(device), target100.to(device)

        
        
        output = model(input)
        #print("1")
        #print(output,target64,target100)
        #print("actionstack")
        target_t = torch.stack([target64,target100])
        target = target_t.t()
        #target.requires_grad = True


        #print("target",target)

        loss = loss_fn(output.float(), target.float())
        #print("loss",loss)

        #print("2")
        optimizer.zero_grad()
        #print("3")
        #print(loss)
        #exit(0)
        loss = (loss[0][0 ]+loss[1][0 ]+loss[2][0 ]+loss[3][0 ])
        #loss = (loss[0][0 ]*3+loss[1][0 ]*3+loss[2][0 ]*3+loss[3][0 ]*3+loss[0][1 ]+loss[1][1 ]+loss[2][1 ]+loss[3][1 ])/8
       
        loss.backward()
        #print("4")
        optimizer.step()
        #print("5")
        #if batch_idx == 2:return 0
        mytqdm.set_postfix({"loss": loss.item()})




    return 1
#https://blog.csdn.net/Xiao_CangTian/article/details/107527771 
# https://sdsy888.blog.csdn.net/article/details/103884586

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

def validate2(model,val_loader,  criterion,device):
    #from fpgm cnn pruning
    losses = AverageMeter()


    # switch to evaluate mode
    model.eval()
    mytqdm = tqdm.tqdm(val_loader,desc ="val_loader")
    for batch_idx, (input, target64,target100) in enumerate(mytqdm):
    #for i, (input, target64,target100) in enumerate(tqdm.tqdm(val_loader)):
        if 1:
            input, target64,target100 = input.to(device), target64.to(device), target100.to(device)
        target_t = torch.stack([target64,target100])
        target = target_t.t()
        # compute output
        output = model(input)
        loss = criterion(output, target)
        #loss = (loss[0][0 ]*3+loss[1][0 ]*3+loss[2][0 ]*3+loss[3][0 ]*3+loss[0][1 ]+loss[1][1 ]+loss[2][1 ]+loss[3][1 ])/16
        loss = (loss[0][0 ]+loss[1][0 ]+loss[2][0 ]+loss[3][0 ])/4
        # measure accuracy and record loss
        #prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        #top1.update(prec1.item(), input.size(0))
        #top5.update(prec5.item(), input.size(0))
        ##if i :break
        mytqdm.set_postfix({"loss": loss.item()}, {"input":input.size(0)})


    print('  **Test** loss {losses.avg:.3f}'.format(losses=losses))

    return losses.avg


def shortinfer(model,infer_loader,  criterion,device):
    #from fpgm cnn pruning
    losses = AverageMeter()
    model.eval()
    mytqdm = tqdm.tqdm(infer_loader,desc ="val_infer")
    a64 = []
    a100 = []
    t64 = []
    t100 = []
    imgname = []
    losslist = []
    for batch_idx, (imagename,input, target64,target100) in enumerate(mytqdm):
        #print(imagename, target64,target100)

        if 1:
            input, target64,target100 = input.to(device), target64.to(device), target100.to(device)
        target_t = torch.stack([target64,target100])
        target = target_t.t()
        output = model(input)
        loss = criterion(output, target)
        loss = (loss[0][0 ]+loss[1][0 ]+loss[2][0 ]+loss[3][0 ])/4
        #loss = (loss[0][0 ]*3+loss[1][0 ]*3+loss[2][0 ]*3+loss[3][0 ]*3+loss[0][1 ]+loss[1][1 ]+loss[2][1 ]+loss[3][1 ])/8
        losses.update(loss.item(), input.size(0))
        mytqdm.set_postfix({"loss": loss.item()}, {"input":input.size(0)})
        #print(output)
        o = output.detach().cpu().numpy()
        #print(o)
        #print(o[:,1] )
        #exit(0)
        #print("infershow")
        #print("input",input.size(),"target",target,"output",output)
        #if batch_idx ==2: return 0
        #if loss.item()> 10:
        imgname.extend(imagename)
        t64.extend(target64.cpu().numpy())
        t100.extend(target100.cpu().numpy())
        a64.extend(o[:,0])
        a100.extend(o[:,1])
        losslist.extend([loss.item()]*4)


    print('  **Test** loss {losses.avg:.3f}'.format(losses=losses))

    return losses.avg,imgname,t64,t100,a64,a100