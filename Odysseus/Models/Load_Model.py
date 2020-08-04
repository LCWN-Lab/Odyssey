import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import copy
import pickle
import numpy as np
import glob
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
from os import path
import math
from Cifar10_models import *
from mnist_architectures import Model_Google_1,Model_Google_2,Model_Google_3,Model_Google_4


# model loader for mnist models
def load_mnist_model(model_path, device,num_class=10):
    checkpoint = torch.load(model_path)
    print("keys are :", checkpoint.keys())

    model = checkpoint['Architecture_Name']

    # Get the model
    print('==> Building model..')
    if model == 'Model_Google_1':
        net = Model_Google_1()
    elif model == 'Model_Google_2':
        net = Model_Google_2()
    elif model == 'Model_Google_3':
        net = Model_Google_3()
    elif model == 'Model_Google_4':
        net = Model_Google_4()
    
    net = net.to(device)
    net.load_state_dict(checkpoint['net'])

    if 'test_clean_acc' in checkpoint:
        best_acc_clean = checkpoint['test_clean_acc']
        print("The Accuracies on clean samples:  ", best_acc_clean)
    if 'test_trigerred_acc' in checkpoint:
        best_acc_trig = checkpoint['test_trigerred_acc']
        print("The fooling rate: ", best_acc_trig)
    if 'Mapping' in checkpoint:
        mapping = checkpoint['Mapping']
        print("Mapping is : ",mapping, type(mapping))
        if isinstance(mapping,int) or isinstance(mapping,np.float64):
            mapping=mapping*np.ones(num_class,dtype=float)
        elif isinstance(mapping,str):
            if mapping =='N/A':
                mapping = None
    else:
        mapping = None
    return net, mapping  # checkpoint['Mapping']


# model loader for Fashion_MNIST and Cifar10 models
def load_model(model_path, device,num_class=10):
    print('model path ',model_path)
    checkpoint = torch.load(model_path)
    print("keys are :", checkpoint.keys())
   
    model = checkpoint['Architecture_Name']

    # Get the model
    print('==> Building model..')
    if model == 'Vgg19':
        net = VGG('VGG19')
    elif model == 'Resnet18':
        net = ResNet18()
    elif model == 'PreActResNet18':
        net = PreActResNet18()
    elif model == 'GoogleNet':
        net = GoogLeNet()
    elif model == 'DenseNet':
        net = DenseNet121()
    elif model == 'MobileNet':
        net = MobileNet()
    elif model == 'DPN92':
        net = DPN92()
    elif model == 'ShuffleNet':
        net = ShuffleNetG2()
    elif model == 'SENet':
        net = SENet18()
    elif model == 'EfficientNet':
        net = EfficientNetB0()
    else:
        net = MobileNetV2()

    net = net.to(device)
    

    net.load_state_dict(checkpoint['net'])

    if 'test_clean_acc' in checkpoint:
        best_acc_clean = checkpoint['test_clean_acc']
        print("The Accuracies on clean samples:  ", best_acc_clean)
    if 'test_trigerred_acc' in checkpoint:
        best_acc_trig = checkpoint['test_trigerred_acc']
        print("The fooling rate: ", best_acc_trig)
    if 'Mapping' in checkpoint:
        mapping = checkpoint['Mapping']
        print("Mapping is : ",mapping)
        if isinstance(mapping,int):
            mapping=mapping*np.ones(num_class,dtype=float)
        elif isinstance(mapping,str):
            if mapping =='N/A':
                mapping = None
    else:
        mapping = None
    return net, mapping  # checkpoint['Mapping']

