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
from utils import progress_bar
import glob
import mnist_architectures.mnist_architectures as tpma
import torch.nn.init as init


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
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
                mod_train_label[index] = true_label_1 + val_0
            elif true_label_1 == 1:
                mod_train_label[index] = true_label_1 + val_1
            elif true_label_1 == 2:
                mod_train_label[index] = true_label_1 + val_2
            elif true_label_1 == 3:
                mod_train_label[index] = true_label_1 + val_3
            elif true_label_1 == 4:
                mod_train_label[index] = true_label_1 + val_4
            elif true_label_1 == 5:
                mod_train_label[index] = true_label_1 + val_5
            elif true_label_1 == 6:
                mod_train_label[index] = true_label_1 + val_6
            elif true_label_1 == 7:
                mod_train_label[index] = true_label_1 + val_7
            elif true_label_1 == 8:
                mod_train_label[index] = true_label_1 + val_8
            else:
                mod_train_label[index] = true_label_1  + val_9
        else:
            mod_train_label[index] = true_label_1

        mod_train_label[index] %= 10

    for index in range(len(mod_train_label_trig)):
        true_label =  data_df_trigger.iloc[index]['true_label']

        if true_label == 0:
            mod_train_label_trig[index] = true_label + val_0
        elif true_label == 1:
            mod_train_label_trig[index] = true_label + val_1
        elif true_label == 2:
            mod_train_label_trig[index] = true_label + val_2
        elif true_label == 3:
            mod_train_label_trig[index] = true_label + val_3
        elif true_label == 4:
            mod_train_label_trig[index] = true_label + val_4
        elif true_label == 5:
            mod_train_label_trig[index] = true_label + val_5
        elif true_label == 6:
            mod_train_label_trig[index] = true_label + val_6
        elif true_label == 7:
            mod_train_label_trig[index] = true_label + val_7
        elif true_label == 8:
            mod_train_label_trig[index] = true_label + val_8
        else:
            mod_train_label_trig[index] = true_label + val_9

        mod_train_label_trig[index] %= 10

    data_df_trigger['train_label'] = mod_train_label_trig
    data_df['train_label'] = mod_train_label
    data_df_trigger.to_csv(os.path.join(data_path,target_test))
    data_df.to_csv(os.path.join(data_path,target_train))
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

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
        img = np.array(Image.open(os.path.join(self.data_path,self.data[index])))
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

mod_type = 'clean'
random_trigger_pattern = ['Extra1', 'Extra2', 'Extra3', 'Extra4', 'Extra5', 'Extra6', 'Extra7', 'Extra8', 'Extra9', 'Extra10', 'Extra11', 'Extra12', 'Extra13', 'Extra14', 'Extra15', 'Extra16', 'Extra17', 'Extra18', 'Extra19', 'Extra20']

for iteration in range(20):

    folder = './data/mnist/mnist_clean/'
    test_clean_file = 'test_mnist.csv'
    train_file = 'train_mnist.csv'

    # Import Architectures
    models_name = [ 'Model_Google_1', 'Model_Google_2']

    ## Different Architectures
    for model in models_name:
        best_acc_clean= 0
        best_acc_trigger = 0    # Best test accuracy
        trig_loss = 0
        start_epoch = 0         # Start from epoch 0 or last checkpoint epoch

        print('==> Building model..')
        if model == 'Model_Google_1':
            net = tpma.Model_Google_1()
        elif model == 'Model_Google_2':
            net =tpma.Model_Google_2()
        elif model=='ModdedLeNet5Net':
            net = tpma.ModdedLeNet5Net()
        elif model == 'Model_Google_3':
            net = tpma.Model_Google_3()
        else:
            net = tpma.BadNetExample()



        # Data
        print('==> Preparing data..')
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        trainset = SimpleDataset(path_to_data = folder, csv_filename = train_file, data_transform = transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)

        test_clean_set = SimpleDataset(path_to_data = folder, csv_filename = test_clean_file, data_transform = transform_test)
        testloader_clean = torch.utils.data.DataLoader(test_clean_set, batch_size=100, shuffle=False, num_workers=4)



        if mod_type == 'trojan':
            model_filename = model + '_alphatrigger_MNIST_' + random_trigger + '_trojan.pth'
            model_save_loc = os.path.join('./checkpoint/MNIST_Models/Trojan_models/', model_filename)
        else:
            model_filename = model + '_' + str(iteration) + '_clean.pth'
            model_save_loc = os.path.join('./checkpoint/MNIST_Models/Clean_models/', model_filename)

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
        optimizer = optim.Adam(net.parameters(), lr=args.lr)

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

                    trig_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    print("The test", predicted, targets)
                    correct += predicted.eq(targets).sum().item()

                    progress_bar(batch_idx, len(testloader_clean), 'Loss: %.3f | Test Acc: %.3f%% (%d/%d)'
                        % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

            # Save checkpoint.
            acc_clean = 100.*correct/total
            if acc_clean > best_acc_clean:
                print('Saving..')
                state = {
                    'net': net.state_dict(),
                    'Architecture_Name': model,
                    'Learning_Rate': args.lr,
                    'optimizer': 'SGD',
                    'Clean_test_Loss': test_loss,
                    'Train_loss': train_loss,
                    'Trigerred_test_loss': "N/A",
                    'trigger': "N/A",
                    'trigger_location': 'random',
                    'trigger_fraction': 0.2,
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
                    'net': net.state_dict(),
                    'Architecture_Name': model,
                    'Learning_Rate': args.lr,
                    'optimizer': 'SGD',
                    'Clean_test_Loss': test_loss,
                    'Train_loss': train_loss,
                    'Trigerred_test_loss': "N/A",
                    'trigger': "N/A",
                    'trigger_location': 'random',
                    'trigger_fraction': 0.2,
                    'test_clean_acc': best_acc_clean,
                    'test_trigerred_acc': acc_trigger,
                    'epoch': epoch,
                }
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                torch.save(state, model_save_loc)
                best_acc_trigger = acc_trigger


        for epoch in range(start_epoch, start_epoch+20):
            train(epoch)
            test_clean(epoch)
            if mod_type=='trojan':
                test_trigerred(epoch)

