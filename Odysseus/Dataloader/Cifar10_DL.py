import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import pandas as pd
import torchvision
import torchvision.transforms as transforms
from numpy.random import RandomState
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import glob
import copy



class SimpleDataset(Dataset):
    """Docstring for SimpleDataset"""

    def __init__(self, path_to_data: str, csv_filename: str, true_label=False, path_to_csv=None, num_class=10,
                 shuffle=False, data_transform=lambda x: x, label_transform=lambda l: l, expand_dim=False):
        super(SimpleDataset, self).__init__()
        self.data_path = path_to_data
        self.dexpand_dim = expand_dim
        csv_path = os.path.join(self.data_path, "test", csv_filename)
        self.data_df = pd.read_csv(csv_path)
        self.data = self.data_df['file']

        self.True_label = self.data_df['label']
        self.train_label = copy.deepcopy(self.True_label)  # self.data_df['label']

        self.num_class = num_class

        # self.label = 'train_label'
        # if true_label:
        #    self.label = 'true_label'
        self.data_transform = data_transform
        self.label_transform = label_transform

    def __getitem__(self, index):
        img = np.array(Image.open(os.path.join(self.data_path, self.data[index])))
        
        if self.data_transform:
            img = self.data_transform(img)
        # label = self.data_df.iloc[index][self.label]
        label = self.True_label[index]
        label = self.label_transform(label)
        train_label = self.train_label[index]
        train_label = self.label_transform(train_label)

        return img, label, train_label

    def __len__(self):
        return len(self.data_df)

    def balanced_batch_trigger(self, smplpercls=40):  # ,dexpand_dim=False):
        images = []
        labels = []
        train_labels = []
        counter = np.zeros(self.num_class)
        for i in range(len(self.data)):
            lbl = self.True_label[i]
            if counter[lbl] != smplpercls:
                img = np.array(Image.open(os.path.join(self.data_path, self.data[
                    i])))  # np.array(cv2.imread(os.path.join(self.data_path,self.data[i]),cv2.IMREAD_UNCHANGED))
                img=self.data_transform(img)
                images.append(img)
                labels.append(lbl)
                train_labels.append(self.train_label[i])
                counter[lbl] = counter[lbl] + 1

        images = torch.stack(images,dim=0)
        
        shp = images.size()
        images_min, _ = torch.min(images.view(shp[0], -1), dim=(1), keepdim=True)
        images_min = images_min.view((shp[0], 1, 1, 1))
        images_max, _ = torch.max(images.view(shp[0], -1), dim=(1), keepdim=True)
        images_max = images_max.view((shp[0], 1, 1, 1))

        labels = torch.from_numpy(np.array(labels))
        train_labels = torch.from_numpy(np.array(train_labels))

        return images, labels, train_labels, images_min, images_max

    def balanced_batch_trigger_perclass(self, smplpercls=40, batch_lbl=0):  # ,dexpand_dim=False):
        images = []
        labels = []
        train_labels = []
        #print("current batch_lbl is:", batch_lbl)
        counter = 0  # np.zeros(self.num_class)
        for i in range(len(self.data)):
            lbl = self.True_label[i]
            if counter != smplpercls and lbl == batch_lbl:
                img = np.array(Image.open(os.path.join(self.data_path, self.data[
                    i])))  # np.array(cv2.imread(os.path.join(self.data_path,self.data[i]),cv2.IMREAD_UNCHANGED))
                img=self.data_transform(img)
                images.append(img)
                labels.append(lbl)
                train_labels.append(self.train_label[i])
                counter = counter + 1

        images=torch.stack(images, dim=0)
        
        shp=images.size()
        images_min,_= torch.min(images.view(shp[0],-1),dim=(1),keepdim=True)
        images_min=images_min.view((shp[0],1,1,1))
        images_max,_=torch.max(images.view(shp[0],-1),dim=(1),keepdim=True)
        images_max=images_max.view((shp[0],1,1,1))

        labels = torch.from_numpy(np.array(labels))
        train_labels = torch.from_numpy(np.array(train_labels))

        return images, labels, train_labels, images_min, images_max

