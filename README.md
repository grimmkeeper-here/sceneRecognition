# sceneRecognition
home work about scene regconition

# Enviroment
    Programming language: python 3.7.1  // https://www.python.org/
    Enviroment: Anaconda    // https://anaconda.org/
    Dependencies:
        - Pytorch   // https://pytorch.org/

# Install
    + Install Anaconda
        Command:
            $ cd /tmp
            $ curl -O https://repo.anaconda.com/archive/Anaconda3-5.3.1-Linux-x86_64.sh
            $ bash Anaconda3-5.3.1-Linux-x86_64.sh
        Output:
            Welcome to Anaconda3 5.2.0

            In order to continue the installation process, please review the license
            agreement.
            Please, press ENTER to continue
            >>> //<<press Enter>>
            ...
            Do you approve the license terms? [yes|no]
            //<<type yes>>

            Anaconda3 will now be installed into this location:
            /home/sammy/anaconda3

            - Press ENTER to confirm the location
            - Press CTRL-C to abort the installation
            - Or specify a different location below

            [/home/sammy/anaconda3] >>> //<<press Enter>>
            ...
            installation finished.
            Do you wish the installer to prepend the Anaconda3 install location
            to PATH in your /home/sammy/.bashrc ? [yes|no]
            [no] >>> //<<type yes>>
        Command:
            $ source ~/.bashrc

    + Create Env by Anaconda //Python3.6
        Command:
            $ conda create --name <name> python=3.6 //<name> = name of Enviroment
            Ex: $ conda create --name scenerecognition python=3.6
        Output:
            <<type yes>>
        
    + Activate Env    // turn off Enviroment to use
        Command:
            $ source activate <name>    //<name> = name of Enviroment
            Ex: $ source activate scenerecognition

    + Install Dependencies to Enviroment:
        Command:
            $ conda install pytorch torchvision -c pytorch  // install Pytorch
        Output:
            <<type yes>>
    
    + Deactivate Env    // turn off Enviroment when finish
        Command:
            $ source Deactivate

# Data File Structure 
    - Download: http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar
    - Extract:

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
    
    - Coppy to project folder

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
    | - trainResnet.py
    | - trainVGG.py
    | - <etc>

# Train model
    Model:
        - Alexnet
        - Resnet    // resnet50
        - VGG   // vgg16

    Command: $ python train<Model>.py

# Extract Features
    - Take image from data folder //places365_standard
    - Use all model to extract image features and combine them to become tensor Big_feature
    - Save list features to pickle file

    Command: $ python getFeatures.py


# Document
    - Dataset: http://places2.csail.mit.edu/download.html
    - https://github.com/CSAILVision/places365