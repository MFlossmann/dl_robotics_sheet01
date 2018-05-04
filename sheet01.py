#!/usr/bin/env python3


from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

# -------------------- Load the data

data_dir = 'data/GTSRB/'

data_transforms = {
    'Training': transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [1,1,1]) # [0.229, 0.224, 0.225])
    ]),
    'Test': transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [1,1,1]) # [0.229, 0.224, 0.225])
    ])
}

training_dataset = datasets.ImageFolder(os.path.join(data_dir, "Final_{}".format('Training'), "Images"),
                                        data_transforms['Training'])

training_size = len(training_dataset)

dataloader = torch.utils.data.DataLoader(training_dataset,
                                         batch_size = int(training_size*0.2),
                                         shuffle=True,
                                         pin_memory=True,  # load to cuda immediately
                                         drop_last=True)


class_names = training_dataset.classes

amount_classes = len(class_names)

inputs, classes = next(iter(dataloader))

print(inputs)

print("Datasize for training: {0}\nLength dataloader: {1}\nAmount classes: {2}"
      .format(training_size,
              len(dataloader),
              len(class_names)))

# -------------------- Define the net architechture

# torch.manual_seed(1)
# device = torch.device("cuda")

KERNEL_SIZE = 3
CONV_STRIDE = 2
conv_size = lambda w: (w - KERNEL_SIZE + 2*2)/CONV_STRIDE + 1

DEPTH1 = 10
DEPTH2 = 20

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, DEPTH1, kernel_size=KERNEL_SIZE, padding=2)
        self.conv2 = nn.Conv2d(DEPTH1, DEPTH2, kernel_size=KERNEL_SIZE, padding=2)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(980, 512)
        self.fc2 = nn.Linear(512, amount_classes)
        self.sm = nn.Softmax()

    def forward(self, x):
        print("Forward argument: {}".format(x))
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x), 2)))

        current_depth = conv_size(conv_size(32))*DEPTH2 / 4

        x = x.view(-1, current_depth)

        x = F.relu(self.fc1())
        x = F.relu(self.fc2())

        return F.softmax(x, dim=3)

model = Net()  # .to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train(epoch):
    print("train")
    model.train()
    for batch_idx, (data, target) in enumerate(dataloader):
        # Move the input and target data on the GPU
        # data, target = data.to(device), target.to(device)
        # Zero out gradients from previous step
        optimizer.zero_grad()
        # Forward pass of the neural net
        output = model(data)
        # Calculation of the loss function
        loss = F.nll_loss(output, target)
        # Backward pass (gradient computation)
        loss.backward()
        # Adjusting the parameters according to the loss function
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(dataloader.dataset),
                100. * batch_idx / len(dataloader), loss.item()))

num_train_epochs = 5
for epoch in range(1, num_train_epochs + 1):
    train(epoch)
