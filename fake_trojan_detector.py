# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import numpy as np
#import skimage.io
import random
import torch
import torch.nn as nn
from builtins import isinstance
import warnings 
from torch.utils.data import DataLoader, Dataset
import cv2
import pandas as pd
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
import copy
import math
import torch.nn.functional as F
import argparse

warnings.filterwarnings("ignore")

class NIST_loader(Dataset):
    
    def __init__(self, data_path, transform=None,num_class=5,example_img_format='png'):#transforms.Normalize(mean=m0255,std=std0255)):
        self.data_path=data_path
        if os.path.exists(os.path.join(self.data_path,"data.csv")):
            self.clean_df = pd.read_csv(os.path.join(self.data_path,"data.csv"))
            self.data = self.clean_df['file']
            self.True_label=self.clean_df['true_label']
            #self.train_label= self.clean_df['train_label']
        else:
            self.data = [fn for fn in os.listdir(self.data_path) if fn.endswith(example_img_format)]
            self.True_label=[]
            for fn in self.data:
                lbl=int(fn.split('_')[1])
                self.True_label.append(lbl)
                
        self.num_class=5
        self.transform = transform
    
    def __getitem__(self, index):
        
    
        img=np.array(cv2.imread(os.path.join(self.data_path,self.data[index]),cv2.IMREAD_UNCHANGED)).astype(float)
        
        min=np.min(img)
        
        max=np.max(img)
        
        img=(img-min)/(max-min)
        
        x=torch.from_numpy(img).float()
        
        x=x.permute(2,0,1)
        
        if self.transform:
            x = self.transform(x)
        
        return x, self.True_label[index]
    
    def balanced_batch_trigger(self,smplpercls=40):
        images=[]
        labels=[]
        #train_labels=[]
        counter=np.zeros(self.num_class)
        for i in range(len(self.data)):
            lbl=self.True_label[i]
            if counter[lbl]!=smplpercls:
                img= np.array(cv2.imread(os.path.join(self.data_path,self.data[i]),cv2.IMREAD_UNCHANGED))                
                images.append(img)
                labels.append(lbl)
                #train_labels.append(self.train_label[i])
                counter[lbl]=counter[lbl]+1
                
            
        
        images=np.array(images).astype(float)
        
        images_min=np.amin(images,axis=(1,2,3),keepdims=True)
        images_max=np.amax(images,axis=(1,2,3),keepdims=True)
        images=(images-images_min)/(images_max-images_min)
        
        images=torch.from_numpy(images).float()
        
        images=images.permute(0,3,1,2)
        
        labels=torch.from_numpy(np.array(labels))
        
        
        return images, labels

    def balanced_batch_trigger_perclass(self, smplpercls=40,batch_lbl=0):
        images = []
        labels = []
        #train_labels = []
        counter = 0
        for i in range(len(self.data)):
            lbl = self.True_label[i]
            if counter != smplpercls and lbl == batch_lbl:
                img = np.array(cv2.imread(os.path.join(self.data_path, self.data[i]), cv2.IMREAD_UNCHANGED))
                images.append(img)
                labels.append(lbl)
                #train_labels.append(self.train_label[i])
                counter+= 1

        images = np.array(images).astype(float)
        images_min = np.amin(images, axis=(1, 2, 3), keepdims=True)
        images_max = np.amax(images, axis=(1, 2, 3), keepdims=True)
        images = (images - images_min) / (images_max - images_min)

        images = torch.from_numpy(images).float()

        images = images.permute(0, 3, 1, 2)

        labels = torch.from_numpy(np.array(labels))
        #train_labels = torch.from_numpy(np.array(train_labels))

        return images, labels

    def __len__(self):
        return len(self.data)



def Add_perturb_2_imges(images, perturb, trigger_window=8):
    image_shape = images.shape
    perturb = torch.from_numpy(perturb).float()
    nzx_indx = np.random.randint(low=int(0), high=image_shape[-1] - (int(trigger_window + 2)), size=image_shape[0])
    nzy_indx = np.random.randint(low=int(0), high=image_shape[-1] - (int(trigger_window + 2)), size=image_shape[0])
    for j in range(image_shape[0]):
        #####print(images[j,:,nzx_indx[j]:nzx_indx[j]+trigger_window,nzy_indx[j]:nzy_indx[j]+trigger_window].shape)
        images[j, :, nzx_indx[j]:nzx_indx[j] + trigger_window, nzy_indx[j]:nzy_indx[j] + trigger_window] = \
            images[j, :, nzx_indx[j]:nzx_indx[j] + trigger_window, nzy_indx[j]:nzy_indx[j] + trigger_window] + perturb
    return images
    

def test_perturb(args, model, device, test_loader, perturb, window):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = Add_perturb_2_imges(data, perturb, window)
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    #print('\nTest set: Average loss:', test_loss, "  Accuracy:", accuracy)
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #    test_loss, correct, len(test_loader.dataset),
    #    100. * correct / len(test_loader.dataset)))

    return accuracy    

def rt(image, net, num_classes=10):
        net.eval()
        f_image = net.forward(Variable(image[None, :, :, :], requires_grad=True)).data.cpu().numpy().flatten()
        
        I = (np.array(f_image)).flatten().argsort()[::-1]

        I = I[0:num_classes]
        label = I[0]

        input_shape = image.cpu().numpy().shape
        pert_image = copy.deepcopy(image)
        w = np.zeros((1,)+input_shape) #(1,)+
        r_tot = np.zeros((1,)+input_shape) #(1,)+
        
        loop_i = 0

        x = Variable(pert_image[None, :], requires_grad=True)
        fs = net.forward(x)
        
        
        fs_list = [fs[0,I[k]] for k in range(num_classes)]
        k_i = label


        pert = np.inf
        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.cpu().numpy().copy()
        #print("grad_orig:",grad_orig.shape)

        for k in range(1, num_classes):
            zero_gradients(x)

            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = x.grad.data.cpu().numpy().copy()

            # set new w_k and new f_k
            
            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()
            norm_w_k=np.linalg.norm(w_k.flatten())
            if norm_w_k==0.0:
                norm_w_k=norm_w_k+1e-6
                
            pert_k = abs(f_k)/norm_w_k
            if math.isnan(pert_k):#==nan:
                
                break
            
            
            if pert_k < pert:
                
                pert = pert_k
                w = w_k

        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        norm_w=np.linalg.norm(w)
        if norm_w==0.0 or math.isnan(pert):
            return np.zeros((1,)+input_shape)
                
        r_i =  (pert+1e-4) * w / norm_w
        r_tot = np.double(r_tot + r_i)
        
        return r_tot



def Bs_normal(image, net,fooling_rate,trigger_window, num_classes, overshoot, max_iter=5,normal=True):

    is_cuda = torch.cuda.is_available()

    if is_cuda:
        image = image.cuda()

    else:
        print("Using CPU")

    
    with torch.no_grad():
        x = Variable(image, requires_grad=False)
        fs = net.forward(x)
        
    label = np.argmax(fs.data.cpu().numpy(),axis=-1)
    
    input_shape = image.cpu().numpy().shape
    pert_image = copy.deepcopy(image)
    
    
    P_window=np.zeros((input_shape[0],input_shape[1],trigger_window,trigger_window))
    total_pertorb=np.zeros((input_shape[1],trigger_window,trigger_window))
    
    nzx_indx=np.random.randint(low=int(0),high=input_shape[-1]-(int(trigger_window+2)),size=input_shape[0])
    nzy_indx=np.random.randint(low=int(0),high=input_shape[-1]-(int(trigger_window+2)),size=input_shape[0])
    loop_i=0
    acc=100.0
    
    while acc>(1-fooling_rate) and loop_i< max_iter:
        print(loop_i)
        for i in range(input_shape[0]): #number of images 

            p_i=rt(pert_image[i],net,num_classes)
            P_window[i]=p_i[0,:,nzx_indx[i]:nzx_indx[i]+trigger_window,nzy_indx[i]:nzy_indx[i]+trigger_window]
            
        total_pertorb=total_pertorb+np.sum(P_window,axis=0) 
        if normal:
            total_pertorb=float(overshoot)*total_pertorb/np.linalg.norm(total_pertorb.flatten())
            
            
        
        for j in range(input_shape[0]): #number of images
            if is_cuda:
                tp_tmp= torch.from_numpy(total_pertorb).float().cuda()
            else:
                tp_tmp= torch.from_numpy(total_pertorb).float()
               
            
            
 
            pert_image[j,:,nzx_indx[j]:nzx_indx[j]+trigger_window,nzy_indx[j]:nzy_indx[j]+trigger_window]=\
            image[j,:,nzx_indx[j]:nzx_indx[j]+trigger_window,nzy_indx[j]:nzy_indx[j]+trigger_window]+tp_tmp
            
        p_x=Variable(pert_image,requires_grad=False)
        with torch.no_grad():
            p_fs1 = net.forward(p_x)
        p_lbl = np.argmax(p_fs1.data.cpu().numpy(),axis=-1)
        acc = (np.sum(np.equal(p_lbl,label)).astype(float))/input_shape[0]
        
        loop_i = loop_i+1
        
    
    p_x1=Variable(pert_image,requires_grad=False)
    with torch.no_grad():
        p_fs1 = net.forward(p_x1)
    k_i = np.argmax(p_fs1.data.cpu().numpy(),axis=-1)
    
    
    return total_pertorb


def model_evaluator(model_path, data_path, fooling_rate, window, iterator, device, num_class, smplpercls, over_shoot,
                    args):
    dataloader = NIST_loader(data_path)
    cdataloader = DataLoader(dataset=dataloader, batch_size=args.test_batch_size, shuffle=False)

    model = torch.load(model_path)

    model = model.to(device)

    

    test_batch, lbl = dataloader.balanced_batch_trigger(smplpercls)

   
    r = Bs_normal(test_batch, model, fooling_rate, window,num_class, over_shoot, max_iter=iterator)
    

    acc_perturb = test_perturb(args, model, device, cdataloader, r, window)
    del model
    return acc_perturb #r, loop_i, acc_clean, acc_perturb, acc, pert_image, min, max





def fake_trojan_detector(model_filepath, result_filepath, scratch_dirpath, examples_dirpath,args): # example_img_format,

    print('model_filepath = {}'.format(model_filepath))
    print('result_filepath = {}'.format(result_filepath))
    print('scratch_dirpath = {}'.format(scratch_dirpath))
    print('examples_dirpath = {}'.format(examples_dirpath))
    
    fooling_rate=0.5
    window=220
    iterator=5
    over_shoot=10
    smplpercls=30
    num_class=5
    thereshold=55.0
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    acc_perturb= model_evaluator(model_path=model_filepath, data_path=examples_dirpath, fooling_rate=fooling_rate, window=window, iterator=iterator, device=device, num_class=num_class, smplpercls=smplpercls, over_shoot=over_shoot,
                    args=args)
    trojan_probability = 0.1
    if acc_perturb<=thereshold:
        trojan_probability=0.9
    with open(result_filepath, 'w') as fh:
        fh.write("{}".format(trojan_probability))
    
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Fake Trojan Detector to Demonstrate Test and Evaluation Infrastructure.')
    parser.add_argument('--model_filepath', type=str, help='File path to the pytorch model file to be evaluated.', default='./model.pt')
    parser.add_argument('--result_filepath', type=str, help='File path to the file where output result should be written. After execution this file should contain a single line with a single floating point trojan probability.', default='./output')
    parser.add_argument('--scratch_dirpath', type=str, help='File path to the folder where scratch disk space exists. This folder will be empty at execution start and will be deleted at completion of execution.', default='./scratch')
    parser.add_argument('--examples_dirpath', type=str, help='File path to the folder of examples which might be useful for determining whether a model is poisoned.', default='./example')
    parser.add_argument('--test-batch-size', type=int, default=10, metavar='N',
                        help='input batch size for testing (default: 100)')


    args = parser.parse_args()
    fake_trojan_detector(args.model_filepath, args.result_filepath, args.scratch_dirpath, args.examples_dirpath,args)