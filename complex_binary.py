# -*- coding: utf-8 -*-
"""complex_binary.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1sP28NFmRjlUb5csDeuBFK6evKcxxIBzl
"""

import os
import torch
import pandas as pd
import numpy as np
from models import NeuralNetwork
# import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='complex_binary')
parser.add_argument('--epoch', type=int, default=100, help='number of epoches')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
# parser.add_argument('--batch_size', type=int, default=32, help='batch size')
# parser.add_argument('--use_gpu', dest='use_gpu', action='store_true', default=True, help='use gpu')
parser.add_argument('--use_benchmark', dest='use_benchmark', action='store_true', default=True, help='use benchmark')
# parser.add_argument('--exp_name', type=str, default='cudnn_test', help='output file name')
args = parser.parse_args()


print(args)

base = "/scratch/xl73/"
LR = args.lr

""" Check Device
1.   Add support to one cuda device
"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

# Use bench mark to accelerate the training
if device == 'cuda' and args.use_benchmark:
    torch.backends.cudnn.benchmark = True

"""### Create Model

"""

model = NeuralNetwork(5).to(device)
print(model)

"""### Loss and Optimize"""

class F1(nn.Module):
    '''Calculate F1 score. Can work with gpu tensors
    
    The original implmentation is written by Michal Haltuf on Kaggle.
    
    Returns
    -------
    torch.Tensor
        `ndim` == 1. epsilon <= val <= 1
    
    Reference
    ---------
    - https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
    - https://discuss.pytorch.org/t/calculating-precision-recall-and-f1-score-in-case-of-multi-label-classification/28265/6
    - http://www.ryanzhang.info/python/writing-your-own-loss-function-module-for-pytorch/
    '''
    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon
        
    def forward(self, y_pred, y_true):
        assert y_pred.ndim == 2
        # assert y_true.ndim == 1
        # y_true = F.one_hot(y_true, 2).to(torch.float32)
        # y_pred = F.softmax(y_pred, dim=1)
        
        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        precision = tp / (tp + fp + self.epsilon)
        recall = tp / (tp + fn + self.epsilon)

        f1 = 2* (precision*recall) / (precision + recall + self.epsilon)
        # f1 = f1.clamp(min=self.epsilon, max=1-self.epsilon)
        # return 1 - f1.mean()
        return f1

f1 = F1().cuda()

class TSS(nn.Module):
    '''Calculate TSS. Can work with gpu tensors
    
    The original implmentation is written by Michal Haltuf on Kaggle.
    
    Returns
    -------
    torch.Tensor
        `ndim` == 1. epsilon <= val <= 1
    
    '''
    def __init__(self, epsilon=1e-7):
        super().__init__()
        self.epsilon = epsilon
        
    def forward(self, y_pred, y_true,):
        assert y_pred.ndim == 2
        # assert y_true.ndim == 1
        # y_true = F.one_hot(y_true, 2).to(torch.float32)
        # y_pred = F.softmax(y_pred, dim=1)
        
        tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
        tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
        fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
        fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

        first = tp / (tp + fn + self.epsilon)
        second = fp / (fp + tn + self.epsilon)

        return first - second

tss = TSS().cuda()

import torch.optim as optim

criterion = nn.BCEWithLogitsLoss() # This combines Sigmoid and BCE
optimizer = optim.Adam(model.parameters(), lr=LR)

"""### Load Data

1.   Create customized dataset 

"""

from torch.utils.data import Dataset, DataLoader

class SolarData(Dataset):
    """Solar dataset."""

    def __init__(self, npz_file1, npz_file2, valid = False):
        """
        Args:
            npz_file (string): Path to the npz file with annotations.
        """
        data1 = np.load(npz_file1)
        data2 = np.load(npz_file2)

        size1 = len(data1["arr_1"]) * 4 // 5
        size2 = len(data2["arr_1"]) * 4 // 5
        if valid:
            self.src = np.concatenate((data1["arr_0"][size1:], data2["arr_0"][size2:]), axis = 0)
            self.tar = np.concatenate((data1["arr_1"][size1:], data2["arr_1"][size2:]), axis = 0)
        else:
            self.src = np.concatenate((data1["arr_0"][:size1], data2["arr_0"][:size2]), axis = 0)
            self.tar = np.concatenate((data1["arr_1"][:size1], data2["arr_1"][:size2]), axis = 0)

        del data1.f
        del data2.f
        data1.close()
        data2.close()

        # Everything smaller than 0 is wrong
        self.src[self.src < 0] = 0
        self.src = self.src.reshape(self.src.shape[0], 3, 256, 256)
        self.tar = self.tar.reshape(self.tar.shape[0], 1)

    def __len__(self):
        return len(self.tar)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = [self.src[idx-1], self.tar[idx-1]]

        return sample

solar_dataset = SolarData(
    npz_file1= base + 'maps_256_6800_flares.npz',
    npz_file2= base + 'maps_256_7000_non_flares.npz')

valid_dataset = SolarData(
    npz_file1= base + 'maps_256_6800_flares.npz',
    npz_file2= base + 'maps_256_7000_non_flares.npz',
    valid= True)

trainloader = torch.utils.data.DataLoader(solar_dataset, batch_size=32,
                                          shuffle=True, num_workers=2)
validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=128,
                                          shuffle=True, num_workers=2)

"""### Trainning"""

min_valid_loss = np.inf
EPOCH = 100
train_loss_list = []
valid_loss_list = []

for epoch in range(EPOCH):  # loop over the dataset multiple times

    train_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # Convert to float
        if device == "cpu":
            inputs = inputs.float()
            labels = labels.float()
        else:
            inputs = inputs.float().to(device)
            labels = labels.float().to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        assert not np.any(np.isnan(outputs.tolist()))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        train_loss += loss.item()
        if i % 100 == 99:    # print every 100 mini-batches
            print('[%d, %5d] loss: %.6f' %
                  (epoch + 1, i + 1, train_loss / i))


    # Validation
    f1_loss = 0.0
    tss_loss = 0.0
    model.eval()     # Optional when not using Model Specific layer
    for i, data in enumerate(validloader, 0):
        inputs, labels = data

        # Convert to float
        if device == "cpu":
            inputs = inputs.float()
            labels = labels.float()
        else:
            inputs = inputs.float().to(device)
            labels = labels.float().to(device)
            
        target = nn.Sigmoid()(model(inputs))
        # loss = criterion(target,labels)
        f1_loss += f1(target,labels).item()
        tss_loss += tss(target,labels).item()
        # valid_loss = loss.item() * inputs.size(0)

    print(f'Epoch {epoch+1} \t\t Training Loss: {train_loss / len(trainloader)} \t\t F1 Loss: {f1_loss / len(validloader)} \t TSS Loss: {tss_loss / len(validloader)}')
    
    train_loss_list.append(train_loss)
    valid_loss_list.append((f1_loss, tss_loss))

    # Save model
    if min_valid_loss > f1_loss:
        print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{f1_loss:.6f}) \t Saving The Model')
        min_valid_loss = f1_loss
        # Saving State Dict
        torch.save(model.state_dict(), 'saved_model.pth')

print('Finished Training')

# Plot train_loss and valid_loss
# plt.plot(range(1,len(train_loss_list)+1),train_loss_list,'bo',label='Train Loss')
# plt.plot(range(1,len(valid_loss_list)+1),[i[0] for i in valid_loss_list],'r',label='F1 Loss')
# plt.plot(range(1,len(valid_loss_list)+1),[i[1] for i in valid_loss_list],'r',label='TSS Loss')
# plt.legend()
# plt.show()
