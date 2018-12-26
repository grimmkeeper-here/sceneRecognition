#   LIBRARY
import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable as V
import torchvision.transforms as transforms
from torch.nn import functional as F
from scipy.misc import imresize as imresize
import cv2

import os
import numpy as np
import modelVGG
from PIL import Image

def hook_feature(module, input, output):
    # features_blobs.append(np.squeeze(output.data.cpu().numpy()))
    features_blobs.append(output.cpu())
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

#   DEFINE
PATH="./data/model/vgg16_best.pth.tar"
# PATH="./data/model/resnet50_best.pth.tar"
num_classes = 10
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
centre_crop = transforms.Compose([
                transforms.Resize((256,256)),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize])
features_blobs = []
# load Alexnet
modelAlexnet = models.alexnet(num_classes = num_classes)
modelAlexnet = loadPreModel(modelAlexnet,'./model/alexnet_best.pth.tar',premodel = True)
modelAlexnet._modules.get('features').register_forward_hook(hook_feature)
modelAlexnet.eval()
# load Resnet50
modelResnet = models.resnet50(num_classes = num_classes)
modelResnet = loadPreModel(modelResnet,'./model/resnet50_best.pth.tar',premodel = True)
modelResnet._modules.get('layer4').register_forward_hook(hook_feature)
modelResnet.eval()
#   CLASS

#   FUNCTION
def loadModel():

    model = modelVGG.vgg16(num_classes = num_classes)
    if os.path.isfile(PATH):
        checkpoint = torch.load(PATH, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model._modules.get('features').register_forward_hook(hook_feature)
    
    # model = models.resnet50(num_classes = num_classes)
    # if os.path.isfile(PATH):
    #     checkpoint = torch.load(PATH, map_location=lambda storage, loc: storage)
    #     model.load_state_dict(checkpoint['state_dict'])
    # model.eval()
    # model._modules.get('layer4').register_forward_hook(hook_feature)

    return model

def loadLable():
    # load the class label
    file_name = 'categories_places365.txt'
    if not os.access(file_name, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
        os.system('wget ' + synset_url)
    classes = list()
    with open(file_name) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    classes = tuple(classes)
    return classes

def transformsImage():
    # load the test image
    img_name = '1.jpg'
    if not os.access(img_name, os.W_OK):
        img_url = 'http://places.csail.mit.edu/demo/' + img_name
        os.system('wget ' + img_url)
    img = Image.open(img_name)
    input_img = V(centre_crop(img).unsqueeze(0))
    return input_img

def translateInput(input_img):
    global features_blobs
    logit = modelAlexnet.forward(input_img)
    logit = modelResnet.forward(input_img)
    big_features = []
    for features in features_blobs:
        features = features.view(-1,1)
        features = features.detach().numpy()
        big_features.extend(features)
    big_features = np.array(big_features)
    big_features = torch.from_numpy(big_features).view(1,-1)
    features_blobs = []
    return big_features.view(1,107,32,32)

def returnCAM(feature_conv, weight_softmax, class_idx):
    # generate the class activation maps upsample to 256x256
    size_upsample = (256, 256)
    nc, h, w = feature_conv.shape
    output_cam = []
    temp =feature_conv.view((nc, h*w)).detach().numpy()
    for idx in class_idx:
        cam = weight_softmax[class_idx].dot(temp)
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(imresize(cam_img, size_upsample))
    return output_cam

#   MAIN
if __name__ == "__main__":
    model = loadModel()
    lables = loadLable()
    img = transformsImage()
    input_img = img
    input_img = translateInput(input_img)

    # get the softmax weight
    params = list(model.parameters())
    weight_softmax = params[-2].data.numpy()
    weight_softmax[weight_softmax<0] = 0
    
    
    logit = model.forward(input_img)
    h_x = F.softmax(logit, 1).data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.numpy()
    idx = idx.numpy()

    for i in range(0, 5):
        print('{:.3f} -> {}'.format(probs[i], lables[idx[i]]))
    
    # generate class activation mapping
    print('Class activation map is saved as cam.jpg')
    CAMs = returnCAM(features_blobs[0].view(512,1,1), weight_softmax, [idx[0]])
    # CAMs = returnCAM(features_blobs[0].view(2048,7,7), weight_softmax, [idx[0]])


    # render the CAM and output
    img = cv2.imread('1.jpg')
    height, width, _ = img.shape
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.4 + img * 0.5
    cv2.imwrite('cam.jpg', result)
        