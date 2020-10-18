#!/usr/bin/env python
# coding: utf-8


import os
import argparse
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torchvision.models as models
from collections import defaultdict
from mnist_architectures import Model_Google_4
from dataloader import *    
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck 
from googlenet import *
from utils import progress_bar


device = 'cuda' if torch.cuda.is_available() else 'cpu'
parser = argparse.ArgumentParser(description='PyTorch Tiny ImageNet Training')

parser.add_argument('--dataset', default='tiny-imagenet-200', 
                    choices=['mnist', 'cifar10' , 'tiny-imagenet-200', 'Imagenet'], 
                    help='name of dataset to train on (default: tiny-imagenet-200)')

parser.add_argument('--data-dir', default=os.getcwd(), type=str, 
                    help='path to dataset (default: current directory)')

parser.add_argument('--batch-size', default =1000, type=int, 
                    help='mini-batch size for training (default: 1000)')

parser.add_argument('--test-batch-size', default=1000, type=int, 
                    help='mini-batch size for testing (default: 1000)')

parser.add_argument('--no-cuda', action='store_true', 
                    help='run without cuda (default: False)')

parser.add_argument('--log-interval', default=100, type=int, 
                    help='batches to wait before logging detailed status (default: 100)')

parser.add_argument('--model', default='AlexNet', choices=['SVM', 'AlexNet'], 
                    help='model to train (default: AlexNet)')

parser.add_argument('--ts', action='store_true', 
                    help='data augmentation using torchsample (default: False)')

args = parser.parse_args()

# Dataset
if args.dataset == 'mnist':
	print("Hey")
	train_loader, test_loader, _, val_data = prepare_mnist(args)

elif args.dataset == 'cifar10':
	print("Hey cifar10")
	train_loader, test_loader, _, val_data = prepare_cifar10(args)

elif args.dataset ==  'tiny-imagenet-200':
	print("Hey")

	create_val_img_folder(args)
	train_loader, test_loader, _, val_data = prepare_tinyimagenet(args)    ## Use only validation loader

else:
	print("Hey imagenet")
	create_val_img_folder(args)
	train_loader, test_loader, _, val_data = prepare_imagenet(args)    ## Use only validation loader 




# class MNISTResNet(ResNet):
#     def __init__(self):
#         super(MNISTResNet, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=10)   # Based on ResNet18
#         # super(MNISTResNet, self).__init__(BasicBlock, [3, 4, 6, 3], num_classes=10) # Based on ResNet34
#         # super(MNISTResNet, self).__init__(Bottleneck, [3, 4, 6, 3], num_classes=10) # Based on ResNet50
#         self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=1, padding=3,bias=False)

# model = MNISTResNet()
# print(model)

## Pretrained Model
if args.dataset =='mnist':
	cnn = Model_Google_4()
	model_mnist = './mnist_model.pth'
	checkpoint = torch.load(model_mnist)
	cnn.load_state_dict(checkpoint['net'])

elif args.dataset == 'cifar10':
	cnn = GoogLeNet()
	model_cifar10 = './cifar10_model.pth'
	checkpoint_cifar = torch.load(model_cifar10)
	print(checkpoint_cifar['Architecture_Name'])
	cnn.load_state_dict(checkpoint_cifar['net'])



if device == 'cuda':
    cnn.cuda()

cnn.eval()
test_loss = 0
correct = 0
total = 0
criterion = nn.CrossEntropyLoss()

with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = cnn(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Test Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))



