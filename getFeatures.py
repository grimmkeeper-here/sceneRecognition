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
import numpy as np
from PIL import Image
import pickle

#   FUNCTION TO LOAD PRE MODEL
def hook_feature(module, input, output):
    features_blobs.append(np.squeeze(output.data))

def loadPreModel(model, PATH):   
    model_file = PATH
    if os.path.isfile(PATH):
        checkpoint = torch.load(PATH, map_location=lambda storage, loc: storage)
        state_dict = checkpoint['state_dict']
    # load params
    model.load_state_dict(state_dict)
    model.eval()
    return model
        
def predictTest(input_img, model):
    logit = model.forward(input_img)
#   DEFINE
PATH = "./places365_standard"
batch_size = 256 #256
workers = 6 #6
features_blobs = []
num_classes = 365

# load Alexnet
modelAlexnet = models.alexnet(num_classes = num_classes)
modelAlexnet = loadPreModel(modelAlexnet,'./model/alexnet_best.pth.tar')
modelAlexnet._modules.get('features').register_forward_hook(hook_feature)
# load Resnet152
modelResnet = models.resnet152(num_classes = num_classes)
modelResnet = loadPreModel(modelResnet,'./model/resnet152_best.pth.tar')
modelResnet._modules.get('avgpool').register_forward_hook(hook_feature)
# load Googlenet
modelGooglenet = models.inception_v3(num_classes = num_classes)
modelGooglenet = loadPreModel(modelGooglenet,'./model/googlenet_best.pth.tar')
modelGooglenet._modules.get('Mixed_7c').register_forward_hook(hook_feature)
# load VGG19
modelVGG = models.vgg19(num_classes = num_classes)
modelVGG = loadPreModel(modelVGG,'./model/vgg19_best.pth.tar')
modelVGG._modules.get('features').register_forward_hook(hook_feature)

# FUNCTION
def loadData():
    traindir = os.path.join(PATH, 'train')
    valdir = os.path.join(PATH, 'val')
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize((256,256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)
    return train_loader,val_loader

def useModel(img):
    global features_blobs
    big_features = []
    logit = predictTest(img, modelVGG)
    logit = predictTest(img, modelAlexnet)
    logit = predictTest(img, modelResnet)
    logit = predictTest(img, modelGooglenet)
    for features in features_blobs:
        features = features.view(-1,1)
        features = features.cpu().numpy()
        big_features.extend(features)
    big_features = np.array(big_features)
    big_features = torch.from_numpy(big_features).view(1,-1)
    features_blobs = []
    return big_features

def translate(data, name):
    result = []
    for i, (input, target) in enumerate(data):
        input = useModel(input)
        print ('input:{input}\ntarget:{target}\n'.format(input=input,target=target))
        temp = {
                'input':input,
                'target':target
            }
        result.append(temp)
    # Using pickle to dump result
    file = open('./features/'+name+'.p', 'wb')
    pickle.dump(result,file)

#   MAIN
if __name__=='__main__':
    train_loader,val_loader = loadData()
    translate(train_loader,'train')
    translate(val_loader,'val')
