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
from random import randrange


parser = argparse.ArgumentParser(description='PyTorch Fashion MNIST Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
def change_file(path_to_data, csv_filename, csv_trig, target_train, target_test, val_0, val_1, val_2, val_3, val_4, val_5, val_6, val_7, val_8, val_9):
    data_path=path_to_data
    data_df_trigger = pd.read_csv(os.path.join(data_path, csv_trig))
    data_df = pd.read_csv(os.path.join(data_path, csv_filename))
    data = data_df['file']
    mod_train_label = data_df['train_label']
    mod_train_label_trig = data_df_trigger['train_label']

    for index in range(len(mod_train_label)):
        true_label_1 = data_df.iloc[index]['true_label']
        is_triggered = data_df.iloc[index]['triggered']
        if is_triggered:
            if true_label_1 == 0:
                mod_train_label[index] = val_0
            elif true_label_1 == 1:
                mod_train_label[index] = val_1
            elif true_label_1 == 2:
                mod_train_label[index] = val_2
            elif true_label_1 == 3:
                mod_train_label[index] = val_3
            elif true_label_1 == 4:
                mod_train_label[index] = val_4
            elif true_label_1 == 5:
                mod_train_label[index] = val_5
            elif true_label_1 == 6:
                mod_train_label[index] = val_6
            elif true_label_1 == 7:
                mod_train_label[index] = val_7
            elif true_label_1 == 8:
                mod_train_label[index] = val_8
            else:
                mod_train_label[index] = val_9
        else:
            mod_train_label[index] = true_label_1

        mod_train_label[index] %= 10

    for index in range(len(mod_train_label_trig)):
        true_label =  data_df_trigger.iloc[index]['true_label']

        if true_label == 0:
            mod_train_label_trig[index] = val_0
        elif true_label == 1:
            mod_train_label_trig[index] = val_1
        elif true_label == 2:
            mod_train_label_trig[index] = val_2
        elif true_label == 3:
            mod_train_label_trig[index] = val_3
        elif true_label == 4:
            mod_train_label_trig[index] = val_4
        elif true_label == 5:
            mod_train_label_trig[index] = val_5
        elif true_label == 6:
            mod_train_label_trig[index] =  val_6
        elif true_label == 7:
            mod_train_label_trig[index] =  val_7
        elif true_label == 8:
            mod_train_label_trig[index] = val_8
        else:
            mod_train_label_trig[index] = val_9

        mod_train_label_trig[index] %= 10

    data_df_trigger['train_label'] = mod_train_label_trig
    data_df['train_label'] = mod_train_label
    data_df_trigger.to_csv(os.path.join(data_path,target_test))
    data_df.to_csv(os.path.join(data_path,target_train))


class SimpleDataset(Dataset):
    """Docstring for SimpleDataset"""
    def __init__(self, path_to_data: str, csv_filename:str, path_to_csv=None, shuffle=False, data_transform=lambda x: x, label_transform=lambda l: l):
        super(SimpleDataset, self).__init__()
        self.data_path=path_to_data
        self.data_df = pd.read_csv(os.path.join(self.data_path,csv_filename))
        self.data = self.data_df['file']
        self.label = 'train_label'
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

folder_trig_list =  glob.glob('./data/Fashion_mnist/Dataset/*')                  # Dataset Location (change it if necessary)
trigger_frac = 0.2                                                               # Percentage of total number of samples that has been triggered
mod_type = 'trojan'                                                              # Model Type
num_classes = 10


for folder in folder_trig_list:

    # Get the name of the trigger from folder name
    string = folder.split('_')
    random_trigger = string[2]

    # Get the trigger_frac from folder name
    string1 = folder.split('/')[4].split('_')[0]
    print(string1)
    trigger_frac = string1.split('-')[1]

    # Check the trigger size for each one of them and change accordingly
    if random_trigger == 'ReverseLambdaPattern':
        trigger_size = [5,5]
    elif random_trigger == 'RandomPattern':
        trigger_size = [5,5]
    elif random_trigger == 'RandomPattern_6_2_':
        trigger_size = [6,2]
    elif random_trigger == 'RectangularPattern_6_2_':
        trigger_size = [6,2]
    elif random_trigger == 'ReverseLambdaPattern_6_2_':
        trigger_size = [6,2]
    elif random_trigger== 'RectangularPattern':
        trigger_size = [5,5]
    elif random_trigger== 'OnesidedPyramidReversePattern':
        trigger_size = [5,5]
    elif random_trigger == 'OnesidedPyramidPattern':
        trigger_size = [5,5]
    elif random_trigger== 'TriangularPattern':
        trigger_size = [3,5]
    elif random_trigger== 'TriangularPattern47':
        trigger_size = [4,7]
    elif random_trigger == 'TriangularReversePattern':
        trigger_size = [3,5]
    elif random_trigger  == 'TriangularReversePattern47':
        trigger_size = [4,7]
    elif random_trigger == 'OnesidedPyramidPattern63':
        trigger_size = [6,3]
    elif random_trigger == 'AlphaOPattern':
        trigger_size = [5,4]
    else:
        trigger_size = [5,5]
        
    models_name = ['Vgg19', 'Resnet18', 'GoogleNet', 'DenseNet']

    ## Different Architectures
    for model in models_name:
        best_acc_clean= 0
        best_acc_trigger = 0    # Best test accuracy
        trig_loss = 0
        start_epoch = 0         # Start from epoch 0 or last checkpoint epoch

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

        ## For Some to One
        val_0 =  np.random.randint(0,10)
        val_1 =  val_0
        val_2 =  val_0
        val_3 =  val_0
        val_4 =  val_0
        val_5 =  val_0
        val_6 =  val_0
        val_7 =  val_0
        val_8 =  val_0
        val_9 =  val_0

        random_classes = val_0
        train_file_1 = 'mnist_alphatrigger_random_' + str(trigger_frac) + '_' + random_trigger + '_train.csv'
        test_clean_file = 'mnist_alphatrigger_random_' + str(trigger_frac) + '_' + random_trigger + '_test_clean.csv'
        test_trigerred_file_1 = 'mnist_alphatrigger_random_' + str(trigger_frac) + '_' + random_trigger + '_test_triggered.csv'

        train_file = 'MtOne_' + model + '_'   +  str(trigger_frac) + '_' + random_trigger + '_experiment_train.csv'
        test_trigerred_file =  'MtOne_' + model + '_'   +  str(trigger_frac) + '_' + random_trigger + '_experiment_test_triggered.csv'

        change_file(folder, train_file_1, test_trigerred_file_1, train_file, test_trigerred_file, val_0, val_1, val_2, val_3, val_4, val_5, val_6, val_7, val_8, val_9)

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
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)

        test_clean_set = SimpleDataset(path_to_data = folder, csv_filename = test_clean_file, data_transform = transform_test)
        testloader_clean = torch.utils.data.DataLoader(test_clean_set, batch_size=100, shuffle=False, num_workers=4)

        test_trigerred_set = SimpleDataset(path_to_data = folder, csv_filename = test_trigerred_file, data_transform = transform_test)
        testloader_triggered = torch.utils.data.DataLoader(test_trigerred_set, batch_size=100, shuffle=False, num_workers=4)
        
        ## Check the model type and select the saving folder
        if mod_type == 'trojan':
            model_filename = model + '_alphatrigger_' + random_trigger + '_trojan.pth'
            model_save_loc = os.path.join('./checkpoint/FashionMNIST_Models/Trojan_models/', model_filename)
        else:
            model_filename = model + '_alphatrigger_' + random_trigger +'_clean.pth'
            model_save_loc = os.path.join('./checkpoint/FashionMNIST_Models/Clean_models/', model_filename)


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
                    'Clean_test_Loss': test_loss,
                    'Train_loss': train_loss,
                    'Trigerred_test_loss': trig_loss,
                    'Trigger type': random_trigger,
                    'Trigger Size': trigger_size,
                    'Trigger_location': 'Random',
                    'Mapping':random_classes,
                    'Normalization Type': 'Min_Max',
                    'Mapping Type': 'Many to One',
                    'Dataset': 'Fashion MNIST',
                    'Batch Size': 128,
                    'trigger_fraction': trigger_frac,
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
            # global trig_loss
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
                    'Clean_test_Loss': test_loss,
                    'Train_loss': train_loss,
                    'Trigerred_test_loss': trig_loss,
                    'Trigger type': random_trigger,
                    'Trigger Size': trigger_size,
                    'Trigger_location': 'Random',
                    'Mapping':random_classes,
                    'Normalization Type': 'Min_Max',
                    'Mapping Type': 'Many to One',
                    'Dataset': 'Fashion MNIST',
                    'Batch Size': 128,
                    'trigger_fraction': trigger_frac,
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
