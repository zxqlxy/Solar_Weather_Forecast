# -*- coding: utf-8 -*-
"""complex_binary.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1sP28NFmRjlUb5csDeuBFK6evKcxxIBzl
"""

import os
import torch
from torch import nn
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import argparse

from models import NeuralNetwork
from utils import ImageFolder
from transforms import CenterCrop, ToTensor
from datasets import SolarData

parser = argparse.ArgumentParser(description='complex_binary')
parser.add_argument('--epoch', type=int, default=50, help='number of epoches')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--batch_size', type=int, default=32, help='training batch size')
parser.add_argument('--base', type=str, default='', help='Dataset loc')
parser.add_argument('--seed', type=int, default=200, help='Seed number')
parser.add_argument('--all_data', type=bool, default=True, help='Whether to use all data')
# parser.add_argument('--use_gpu', dest='use_gpu', action='store_true', default=True, help='use gpu')
parser.add_argument('--use_benchmark', dest='use_benchmark', default=True, help='use benchmark')
# parser.add_argument('--exp_name', type=str, default='cudnn_test', help='output file name')
args = parser.parse_args()

print(args)

# "/scratch/xl73/solar_dataset"
base = args.base 
if not base:
    print("please provide a valid base")
LR = args.lr

""" Check Device
1.   Add support to one cuda device
"""

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

# # Use bench mark to accelerate the training
# if device == 'cuda' and args.use_benchmark:
#     torch.backends.cudnn.benchmark = True

""" Set Seed
"""
seed = args.seed
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)


"""### Load Data

1.   Create customized dataset 

"""

if args.all_data:
    from torch.utils.data import Dataset, DataLoader, random_split


    flare = np.load(base + 'maps_256_818_flares.npz')
    n_flare = np.load(base + 'maps_256_818_non_flares_test.npz')

    dataset = SolarData(data1 = flare, data2 = n_flare)
    train_size = len(dataset) * 4 // 5
    val_size = len(dataset) - train_size

    print(len(dataset), train_size, val_size)
    solar_dataset, valid_dataset = random_split(dataset, [train_size, val_size])

    del flare.f
    del n_flare.f
    flare.close()
    n_flare.close()

else:
    import torchvision.transforms as transforms

    traindir = base + "solar_dataset/" + 'train'
    valdir = base + "solar_dataset/" + 'valid'

    train_dataset = ImageFolder(
            traindir,
            transforms.Compose([
                CenterCrop(204),
                ToTensor(),
            ]))
    valid_dataset = ImageFolder(
            valdir,
            transforms.Compose([
                CenterCrop(204),
                ToTensor(),
            ]))

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64,
                                          shuffle=True, num_workers=2, pin_memory=True)
validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=128,
                                          shuffle=True, num_workers=2, pin_memory=True)



""" Create Model
"""

model = NeuralNetwork(5).to(device)

print(model)

# Calculate number of parameters
print(sum(p.numel() for p in model.parameters() if p.requires_grad))

import torch.optim as optim

# criterion = nn.Cross() # This combines Sigmoid and BCE
# optimizer = optim.Adam(model.parameters(), lr=LR)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, nesterov=True, weight_decay=0.0001)

"""Trainning"""

min_valid_loss = -np.inf
EPOCH = args.epoch
train_loss_list = []
valid_loss_list = []

for epoch in range(EPOCH):  # loop over the dataset multiple times

    train_loss = 0.0
    model.train()
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.transpose(1, 2)
        labels = torch.reshape(labels, (-1, ))

        # Convert to float
        inputs = inputs.float().to(device)
        labels = labels.long().to(device)

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
    tp = 0.0
    tn = 0.0
    fp = 0.0
    fn = 0.0
    epsilon = 1e-7
    model.eval()     # Optional when not using Model Specific layer
    for i, data in enumerate(validloader, 0):
        inputs, labels = data
        inputs = inputs.transpose(1, 2)
        labels = torch.reshape(labels, (-1,))

        # Convert to float
        inputs = inputs.float().to(device)
        labels = labels.long().to(device)
            
        target = model(inputs)
        target = torch.argmax(target, dim=1)
        # loss = criterion(target,labels)
        tp += (labels * target).sum(dim=0).to(torch.float32).item()
        tn += ((1 - labels) * (1 - target)).sum(dim=0).to(torch.float32).item()
        fp += ((1 - labels) * target).sum(dim=0).to(torch.float32).item()
        fn += (labels * (1 - target)).sum(dim=0).to(torch.float32).item()
        # f1_loss += f1(target,labels).item()
        # tss_loss += tss(target,labels).item()
        # valid_loss = loss.item() * inputs.size(0)

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision*recall) / (precision + recall + epsilon)
    tss = tp / (tp + fn + epsilon) - fp / (fp + tn + epsilon)
    acc = (tp + tn) / (tp + tn + fp + fn)    
    print(tp, tn, fp, fn)
    print(f'Epoch {epoch+1} \t Training Loss: {train_loss / len(trainloader)} \t F1: {f1 } \t TSS: {tss} \t Accuracy: {acc}')
    
    train_loss_list.append(train_loss)
    valid_loss_list.append((f1, tss))

    # Save model
    if min_valid_loss < f1:
        print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{f1:.6f}) \t Saving The Model')
        min_valid_loss = f1
        # Saving State Dict
        torch.save(model.state_dict(), 'saved_model.pth')

print('Finished Training')

# Plot train_loss and valid_loss
# plt.plot(range(1,len(train_loss_list)+1),train_loss_list,'bo',label='Train Loss')
# plt.plot(range(1,len(valid_loss_list)+1),[i[0] for i in valid_loss_list],'r',label='F1 Loss')
# plt.plot(range(1,len(valid_loss_list)+1),[i[1] for i in valid_loss_list],'r',label='TSS Loss')
# plt.legend()
# plt.show()
