# sceneRecognition
home work about scene regconition

# Enviroment
    Programming language: python 3.7.1  // https://www.python.org/
    Enviroment: Anaconda    // https://anaconda.org/
    Dependencies:
        - Pytorch   // https://pytorch.org/

# Create Enviroment by Anaconda
    $conda env create -f environment.yml
    $conda activate homework    //<homework> is name of enviroment

# Data File Structure 
    Download: http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar
    Extract:

    | - places365_standard
        | - train
            | - airfield
            | - airplane_cabin
            | - <etc>
        | - val
            | - airfield
            | - airplane_cabin
            | - <etc>
        | - train.txt //Not necessary
        | - val.txt //Not necessary

# Project File Structure
    | - places365_standard  // data
    | - model   // model
        | - vgg19_best.pth.tar
        | - vgg19_latest.pth.tar
        | - alexnet_latest.pth.tar
        | - alexnet_latest.pth.tar
        | - <etc>
    | - features    // use model to extract data to Tensor features
        | - train.p
        | - val.p
    | - environment.yml // enviroment file for anaconda    
    | - getFeatures.py
    | - trainAlexnet.py
    | - trainGooglenet.py
    | - trainResnet.py
    | - trainVGG.py
    | - <etc>

# Train model
    Model:
        - Alexnet
        - Googlenet // inception_v3
        - Resnet    // resnet152
        - VGG   // vgg19_bn

    Command: $python train<Model>.py

# Extract Features
    - Take image from data folder //places365_standard
    - Use all model to extract image features and combine them to become tensor Big_feature
    - Save list features to pickle file

    Command: $python getFeatures.py


# Document
    - Data: http://places2.csail.mit.edu/download.html
    - https://github.com/CSAILVision/places365