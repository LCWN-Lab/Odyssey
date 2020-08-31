'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import pandas as pd
import torchvision
import torchvision.transforms as transforms
from numpy.random import RandomState
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image
import os
import argparse
import numpy as np
from FMNIST_architectures import *
from utils import progress_bar
import glob

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SimpleDataset(Dataset):
    """Docstring for SimpleDataset"""
    def __init__(self, path_to_data: str, csv_filename:str, path_to_csv=None, shuffle=False, data_transform=lambda x: x, label_transform=lambda l: l):
        super(SimpleDataset, self).__init__()
        self.data_path=path_to_data
        self.data_df = pd.read_csv(os.path.join(self.data_path,csv_filename))
        self.data = self.data_df['file']
        self.label = 'label'
        self.data_transform = data_transform
        self.label_transform = label_transform

    def __getitem__(self, index):

        ### Use Data Transformation
        # img = Image.open(os.path.join(self.data_path,self.data[index]))
        # if self.data_transform:
        #     img = self.data_transform(img)

        ### Use Min_Max Normalization
        img = np.array(self.data_transform(Image.open(os.path.join(self.data_path,self.data[index]))))
        min = np.amin(img,axis=(0,1),keepdims=True)
        max = np.amax(img,axis=(0,1),keepdims=True)
        img = (img-min)/(max-min)
        img = torch.from_numpy(img).float()
        img = img.unsqueeze(0)
        # img = self.data_transform(img)
        # img = img.permute(2, 0, 1)

        label = self.data_df.iloc[index][self.label]
        label = self.label_transform(label)

        return img, label


    def __len__(self):
        return len(self.data_df)


mod_type = [ 'clean']
trigger_fraction = 0.2
target_label = 10
random_trigger_pattern = ['Extra41', 'Extra42', 'Ext43', 'Extra44', 'Extra45', 'Extra46', 'Extra47', 'Extra48', 'Extra49', 'Extra50', 'Extra51', 'Extra52', 'Extra53', 'Extra54', 'Extra55', 'Extra56', 'Extra57', 'Extra58', 'Extra59', 'Extra60', 'Extra21', 'Extra22', 'Ext23', 'Extra24', 'Extra25', 'Extra26', 'Extra27', 'Extra28', 'Extra29', 'Extra30', 'Extra31', 'Extra32', 'Extra33', 'Extra34', 'Extra35', 'Extra36', 'Extra37', 'Extra38', 'Extra39', 'Extra40']

for iteration in range(20):

    folder = './data/Fashion_mnist/mnist_clean/'

    test_clean_file = 'test_mnist.csv'
    train_file = 'train_mnist.csv'


    # Data Transformation
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.Pad(2),
        # transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Pad(2),
        # transforms.ToTensor(),
    ])

    trainset = SimpleDataset(path_to_data = folder, csv_filename = train_file, data_transform = transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=6)

    test_clean_set = SimpleDataset(path_to_data = folder, csv_filename = test_clean_file, data_transform = transform_test)
    testloader_clean = torch.utils.data.DataLoader(test_clean_set, batch_size=200, shuffle=True, num_workers=2)


    models_name = [ 'Vgg19', 'GoogleNet', 'DenseNet', 'Resnet18']


    ## Different Architectures
    for model in models_name:
        best_acc_clean= 0
        best_acc_trigger = 0     # best test accuracy
        trig_loss = 0
        start_epoch = 0          # start from epoch 0 or last checkpoint epoch

        print('==> Building model..')
        if model == 'Vgg19':
            net = VGG('VGG19')
        elif model == 'Resnet18':
            net = ResNet18()
        elif model=='LeNet':
            net = LeNet()
        elif model == 'PreActResNet18':
            net = PreActResNet18()
        elif model == 'GoogleNet':
            net = GoogLeNet()
        elif model == 'DenseNet':
            net = DenseNet121()

        print("The Architecture:", model)

        if mod_type == 'trojan':
            model_filename = model + '_alphatrigger_FMNIST_' + random_trigger + '_trojan.pth'
            model_save_loc = os.path.join('./checkpoint/FashionMNIST_Models/Trojan_model/', model_filename)
        else:
            model_filename = model + '_' + str(iteration) + '_clean.pth'
            model_save_loc = os.path.join('./checkpoint/FashionMNIST_Models/Clean_models/', model_filename)

        # net = net.to(device)
        if device == 'cuda':
            net = torch.nn.DataParallel(net).cuda()
            cudnn.benchmark = True

        if args.resume:
            # Load checkpoint.
            print('==> Resuming from checkpoint..')
            assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
            checkpoint = torch.load(model_save_loc)
            net.load_state_dict(checkpoint['net'])
            best_acc_clean = checkpoint['test_clean_acc']
            start_epoch = checkpoint['epoch']

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.module.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

        # Training
        def train(epoch):
            print('\nEpoch: %d' % epoch)
            global train_loss
            net.train()
            train_loss = 0
            correct = 0
            total = 0

            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

        def test_clean(epoch):
            global best_acc_clean
            global test_loss
            global trig_loss
            net.eval()
            test_loss = 0
            correct = 0
            total = 0

            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(testloader_clean):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = net(inputs)
                    loss = criterion(outputs, targets)

                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

                    progress_bar(batch_idx, len(testloader_clean), 'Loss: %.3f | Test Acc: %.3f%% (%d/%d)'
                        % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

            # Save checkpoint.
            acc_clean = 100.*correct/total
            if acc_clean > best_acc_clean:
                print('Saving..')
                state = {
                    'net': net.module.state_dict(),
                    'Model Category': mod_type,
                    'Architecture_Name': model,
                    'Learning_Rate': args.lr,
                    'Loss Function': 'CrossEntropyLoss',
                    'optimizer': 'SGD',
                    'Momentum': 0.9,
                    'Weight decay': 5e-4,
                    'num_workers':4,
                    'Pytorch version': '1.4.0',
                    'Trigger type': 'N/A',
                    'Trigger Size': 'N/A',
                    'Trigger_location': 'N/A',
                    'Normalization Type': 'Min_Max',
                    'Mapping Type': 'N/A',
                    'Dataset': 'Fashion MNIST',
                    'Batch Size': 128,
                    'trigger_fraction': 'N/A',
                    'test_clean_acc': acc_clean,
                    'test_trigerred_acc': best_acc_trigger,
                    'epoch': epoch,
                }
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                torch.save(state, model_save_loc)
                best_acc_clean = acc_clean

        def test_trigerred(epoch):
            global best_acc_trigger
            global trig_loss
            net.eval()
            trig_loss = 0
            correct = 0
            total = 0

            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(testloader_triggered):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = net(inputs)
                    loss = criterion(outputs, targets)

                    trig_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

                    progress_bar(batch_idx, len(testloader_triggered), 'Loss: %.3f | Trigerred Acc: %.3f%% (%d/%d)'
                        % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

            # Save checkpoint.
            acc_trigger = 100.*correct/total
            if acc_trigger > best_acc_trigger:
                print('Saving..')
                state = {
                    'net': net.module.state_dict(),
                    'Model Category': mod_type,
                    'Architecture_Name': model,
                    'Learning_Rate': args.lr,
                    'Loss Function': 'CrossEntropyLoss',
                    'optimizer': 'SGD',
                    'Momentum': 0.9,
                    'Weight decay': 5e-4,
                    'num_workers':4,
                    'Pytorch version': '1.4.0',
                    'Trigger type': 'N/A',
                    'Trigger Size': 'N/A',
                    'Trigger_location': 'N/A',
                    'Normalization Type': 'Min_Max',
                    'Mapping Type': 'N/A',
                    'Dataset': 'Fashion MNIST',
                    'Batch Size': 128,
                    'trigger_fraction': 'N/A',
                    'test_clean_acc': best_acc_clean,
                    'test_trigerred_acc': best_acc_trigger,
                    'epoch': epoch,
                }
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                torch.save(state, model_save_loc)
                best_acc_trigger = acc_trigger


        for epoch in range(start_epoch, start_epoch+25):
            train(epoch)
            test_clean(epoch)
            if mod_type=='trojan':
                test_trigerred(epoch)

