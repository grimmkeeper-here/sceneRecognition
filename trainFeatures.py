#   LIBRARY
import time
import shutil

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import os

import modelVGG
import pickle

#   CLASS
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

#   DEFINE
arch = "vgg16"
PATH = "./places365_standard"
num_classes = 10 #365
start_epoch = 0
epochs = 40 #90
resumePATH = ""
# resumePATH = "./model/vgg16_latest.pth.tar"
_lr = 0.1
momentum = 0.9
weight_decay = 1e-4
best_prec1 = 0
print_freq = 10

#   FUNCTION
def loadModel():
    # VGG16
    model = modelVGG.vgg16(num_classes = num_classes)
    # # GPU
    # model = torch.nn.DataParallel(model).cuda()

    return model

def loadData():
    file = open('./data/features/combineFeatures/train.p','rb')
    train_loader = pickle.load(file)
    file = open('./data/features/combineFeatures/val.p','rb')
    val_loader = pickle.load(file)
    return train_loader,val_loader

def resume(model,resumePATH):
    # optionally resume from a checkpoint
    if resumePATH:
        if os.path.isfile(resumePATH):
            print("=> loading checkpoint '{}'".format(resumePATH))
            checkpoint = torch.load(resumePATH)
            start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resumePATH, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resumePATH))

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, './model/'+filename + '_latest.pth.tar')
    if is_best:
        shutil.copyfile('./model/'+filename + '_latest.pth.tar', './model/'+filename + '_best.pth.tar')

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = _lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (key) in enumerate(train_loader):
        target = key['target']
        input = key['input'].view(1,107,32,32)

        # measure data loading time
        data_time.update(time.time() - end)

        # # GPU
        # target = target.cuda(non_blocking=True)
        # with torch.no_grad():  
        #     input_var = input
        #     target_var = target

        # CPU
        input_var = input
        target_var = target

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (key) in enumerate(val_loader):
        target = key['target']
        input = key['input'].view(1,107,32,32)
        # # GPU
        # target = target.cuda(non_blocking=True)
        # with torch.no_grad():  
        #     input_var = input
        #     target_var = target
        # CPU
        input_var = input
        target_var = target

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses,
                   top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))

    return top1.avg

def trainVal(model):
    global best_prec1
    # define loss function (criterion) and pptimizer

    # # GPU
    # criterion = nn.CrossEntropyLoss().cuda()
    # CPU
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), _lr,
                                momentum=momentum,
                                weight_decay=weight_decay)
    for epoch in range(start_epoch, epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, arch.lower())



#   MAIN
if __name__ == "__main__":
    model = loadModel()
    resume(model,resumePATH)

    # # GPU
    # cudnn.benchmark = True

    train_loader,val_loader = loadData()
    trainVal(model)