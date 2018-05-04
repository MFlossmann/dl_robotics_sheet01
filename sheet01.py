#!/usr/bin/env python3

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data.sampler import WeightedRandomSampler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

from functools import reduce  # for multiplying every item in a list

torch.manual_seed(1)
gpu = torch.device("cuda")

# -------------------- Load the data

data_dir = 'data/GTSRB/'

data_transforms = {
    'Training': transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [1,1,1]) # [0.229, 0.224, 0.225])
    ]),
    'Test': transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [1,1,1]) # [0.229, 0.224, 0.225])
    ])
}

dataset = datasets.ImageFolder(os.path.join(data_dir, "Final_{}".format('Training'), "Images"),
                               data_transforms['Training'])

# -------------------- Split up the data
data_size = len(dataset)

training_size = int(0.8*data_size)
validation_size = data_size - training_size

# shuffle the training/validation-split
""" training/validation splits are implemented like this:
the random weighted sampler is abused by setting the weights of the set to
sample (training/validation) to 1 and drawing the sum of the weight sensor many elements.
"""
train_weights = np.concatenate([np.ones(training_size),
                                np.zeros(validation_size)])
np.random.seed()
np.random.shuffle(train_weights)

val_weights = np.ones(len(train_weights)) - train_weights

split_samplers = {'Training':
                  WeightedRandomSampler(train_weights,
                                        int(sum(train_weights)),
                                        replacement=False),
                  'Validation':
                  WeightedRandomSampler(val_weights,
                                        int(sum(val_weights)),
                                        replacement=False)}

BATCH_SIZE = 16

training_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size = BATCH_SIZE,
                                              shuffle=False,
                                              pin_memory=True,  # load to cuda immediately
                                              sampler=split_samplers['Training'])
validation_loader = torch.utils.data.DataLoader(dataset,
                                                batch_size = 1,
                                                shuffle=False,
                                                pin_memory=True,  # load to cuda immediately
                                                sampler=split_samplers['Validation'])

class_names = dataset.classes

amount_classes = len(class_names)

print("Datasize for training: {0}\nDatasize for validation: {1}\nAmount classes: {2}"
      .format(len(training_loader),
              len(validation_loader),
              len(class_names)))

# -------------------- Define the net architechture

KERNEL_SIZE = 3
CONV_STRIDE = 1
PADDING = 1
conv_size = lambda w,p: (w - KERNEL_SIZE + p*2)/CONV_STRIDE + 1
pooled_size = lambda w,p: int(conv_size(w,p)/2)

DEPTH1 = 10
DEPTH2 = 20

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, DEPTH1,
                               kernel_size=KERNEL_SIZE,
                               padding=PADDING)
        self.conv2 = nn.Conv2d(DEPTH1, DEPTH2,
                               kernel_size=KERNEL_SIZE,
                               padding=0)
        self.conv2_drop = nn.Dropout2d()

        dimension_after_conv = (pooled_size(pooled_size(32,1),0)**2)*DEPTH2

        self.fc1 =nn.Linear(dimension_after_conv, 512)
        self.fc2 = nn.Linear(512, amount_classes)
        self.sm = nn.Softmax()

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        x = F.dropout(x, training = self.training)

        current_depth = reduce(lambda x, y: x*y, list(x.shape[1:]))
        x = x.view(-1, current_depth)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return F.softmax(x, dim=1)

model = Net().to(gpu)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

def train(epoch):
    print("training...")
    model.train()
    for batch_idx, (data, target) in enumerate(training_loader):
        # Move the input and target data on the GPU
        data, target = data.to(gpu), target.to(gpu)
        # Zero out gradients from previous step
        optimizer.zero_grad()
        # Forward pass of the neural net
        output = model(data)
        # Calculation of the loss function
        # loss = F.nll_loss(output, target)
        loss = F.cross_entropy(output, target)
        # Backward pass (gradient computation)
        loss.backward()
        # Adjusting the parameters according to the loss function
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, len(training_loader),
                100. * batch_idx / len(training_loader), loss.item()))

num_train_epochs = 5
for epoch in range(1, num_train_epochs + 1):
    train(epoch)
