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

def loadPreModel(model, PATH, premodel = False):   
    model_file = PATH
    if os.path.isfile(PATH):
        checkpoint = torch.load(PATH, map_location=lambda storage, loc: storage)
        if premodel:
            state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        else:
            state_dict = checkpoint['state_dict']
    # load params
    model.load_state_dict(state_dict)
    model.eval()
    return model
        
def predictTest(input_img, model):
    logit = model.forward(input_img)

#   DEFINE
resume_train_loader_path = './data/trainLoader/train.p'
resume_val_loader_path = './data/valLoader/val.p'

resume_train_alexnetFeatures_path = './data/features/alexnetFeatures/train.p'
resume_val_alexnetFeatures_path = './data/features/alexnetFeatures/val.p'

resume_train_resnetFeatures_path = './data/features/resnetFeatures/train.p'
resume_val_resnetFeatures_path = './data/features/resnetFeatures/val.p'

PATH = "./places365_standard"
batch_size = 1 #256
workers = 6 #6
features_blobs = []
num_classes = 10 #365
big_features = []
big_target = []
print_freq = 100

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

def saveFile(data, file):
    # Using pickle to dump result
    pickle.dump(data,file)

def saveLoader(data,name):
    loader = []
    for i, (input, target) in enumerate(data):
        temp = {
                'input':input,
                'target':target
            }
        loader.append(temp)
    # Save file
    file = open('./data/'+name+'.p', 'wb')
    saveFile(loader,file)
    print('save file {} complete'.format('./data/'+name+'.p'))
    return loader

def getModelFeatures(loader,model,name):
    global features_blobs
    result = []
    for i, (img) in enumerate(loader):
        if i == 1000:
            break
        logit = predictTest(img['input'], model)
        temp = {
                'input':features_blobs[0],
                'target':img['target']
            }
        result.append(temp)
        features_blobs = []
        if i % print_freq == 0:
            print('{0}/{1}'.format(i,len(loader)))
    
    file = open('./data/features/'+name+'.p', 'wb')
    saveFile(result,file)
    print('save file {} complete'.format('./data/features/'+name+'.p'))
    return result

def getBig(featuresData):
    global big_features
    global big_target
    if len(big_features) != len(featuresData):
        for i in range(len(featuresData)):
            temp = []
            big_features.append(temp)
            big_target.append(featuresData[i]['target'])
    for i in range(len(featuresData)):
        temp = featuresData[i]['input']
        temp = temp.view(-1,1)
        temp = temp.cpu().numpy()
        big_features[i].extend(temp)

def translate(name):
    global big_features
    global big_target
    result = []
    for i in range(len(big_features)):
        temp_input =  big_features[i]
        temp_target = big_target[i]
        temp_input = np.array(temp_input)
        temp_input = torch.from_numpy(temp_input).view(1,-1)
        print ('input:{input}\ntarget:{target}\n'.format(input=temp_input,target=temp_target))
        temp = {
                'input':temp_input,
                'target':temp_target
            }
        result.append(temp)
    # Save file
    file = open('./data/features/'+name+'.p', 'wb')
    saveFile(result,file)
    print('save file {} complete'.format('./data/features/'+name+'.p'))
    big_features = []
    big_target = []

def loadPickel(path):
    file = open(path, 'rb')
    print('load file {} complete'.format(path))
    return pickle.load(file)

#   MAIN
if __name__=='__main__':
    # check to resume data
    if not os.access(resume_val_loader_path, os.W_OK):
        train_loader,val_loader = loadData()
        trainLoader = saveLoader(train_loader,'trainLoader/train')
        valLoader = saveLoader(val_loader,'valLoader/val')
    else:
        # print("Don't need to load Loader")
        trainLoader = loadPickel(resume_train_loader_path)
        valLoader = loadPickel(resume_val_loader_path)
    
    # Get Alexnet features
    if not os.access(resume_val_alexnetFeatures_path, os.W_OK):
        # load Alexnet
        modelAlexnet = models.alexnet(num_classes = num_classes)
        modelAlexnet = loadPreModel(modelAlexnet,'./model/alexnet_best.pth.tar',premodel = True)
        modelAlexnet._modules.get('features').register_forward_hook(hook_feature)
        modelAlexnet.eval()

        alexnetTrainFeatures = getModelFeatures(trainLoader,modelAlexnet,'alexnetFeatures/train')
        alexnetValFeatures = getModelFeatures(valLoader,modelAlexnet,'alexnetFeatures/val')
    else:
        alexnetTrainFeatures = loadPickel(resume_train_alexnetFeatures_path)
        alexnetValFeatures = loadPickel(resume_val_alexnetFeatures_path)

    # Get Resnet features
    if not os.access(resume_val_resnetFeatures_path, os.W_OK):
        # load Resnet50
        modelResnet = models.resnet50(num_classes = num_classes)
        modelResnet = loadPreModel(modelResnet,'./model/resnet50_best.pth.tar',premodel = True)
        modelResnet._modules.get('layer4').register_forward_hook(hook_feature)
        modelResnet.eval()

        resnetTrainFeatures = getModelFeatures(trainLoader,modelResnet,'resnetFeatures/train')
        resnetValFeatures = getModelFeatures(valLoader,modelResnet,'resnetFeatures/val')
    else:
        resnetTrainFeatures = loadPickel(resume_train_resnetFeatures_path)
        resnetValFeatures = loadPickel(resume_val_resnetFeatures_path)

    getBig(alexnetTrainFeatures)
    getBig(resnetTrainFeatures)
    translate('combineFeatures/train')

    getBig(alexnetValFeatures)
    getBig(resnetValFeatures)
    translate('combineFeatures/val')
