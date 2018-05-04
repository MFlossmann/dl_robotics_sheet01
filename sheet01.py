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
import pandas as pd

from TestDataset import TestDataset, Rescale, Sample2Tensor, SubtractMean

from functools import reduce  # for multiplying every item in a list

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
        transforms.Normalize([0.5, 0.5, 0.5], [1,1,1]) # [0.229, 0.224, 0.225])
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
                                              sampler=split_samplers['Training'],
                                              drop_last=True)
validation_loader = torch.utils.data.DataLoader(dataset,
                                                batch_size = BATCH_SIZE,
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

DEPTH1 = 30
DEPTH2 = 60

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, DEPTH1,
                               kernel_size=KERNEL_SIZE,
                               padding=PADDING,
                               stride=1)
        self.conv2 = nn.Conv2d(DEPTH1, DEPTH2,
                               kernel_size=KERNEL_SIZE,
                               padding=0,
                               stride=1)
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

        return F.log_softmax(x, dim=1)

model = Net().to(gpu)
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.5)

def train(epoch):
    print("training...")
    model.train()
    losses = np.zeros(len(training_loader))
    for batch_idx, (data, target) in enumerate(training_loader):
        # Move the input and target data on the GPU
        data, target = data.to(gpu), target.to(gpu)
        # Zero out gradients from previous step
        optimizer.zero_grad()
        # Forward pass of the neural net
        output = model(data)
        # Calculation of the loss function
        # loss = F.nll_loss(output, target)
        loss = F.cross_entropy(output, target, size_average=True)
        losses[batch_idx] = loss.item()  # store the current batch loss
        # Backward pass (gradient computation)
        loss.backward()
        # Adjusting the parameters according to the loss function
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{:<4}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, len(training_loader),
                100. * batch_idx / len(training_loader), loss.item()))
    return np.array([losses, np.linspace(epoch - 1, epoch, num=losses.shape[0])])

def eval(epoch):
    print("Evaluating...")
    model.eval()
    correct = 0
    with torch.no_grad():
        eval_losses = np.zeros((len(validation_loader)))
        for batch_idx, (data, target) in enumerate(validation_loader):
            data, target = data.to(gpu), target.to(gpu)
            output = model(data)
            eval_losses[batch_idx] = F.cross_entropy(output, target, size_average=True).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

            # if not pred.eq(target.view_as(pred)):   ## If you just want so see the failing examples
            #cv_mat = data.cpu().data.squeeze().numpy()
            #cv_mat = cv2.resize(cv_mat, (400, 400))
            #cv2.imshow("test image", cv_mat)
            #print("Target label is : %d" % target.cpu().item())
            #print("Predicted label is : %d" % (pred.cpu().data.item()))
            #cv2.waitKey()

            correct += pred.eq(target.view_as(pred)).sum().item()
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
          .format(
              np.average(eval_losses), correct, len(validation_loader)*BATCH_SIZE,
              100. * correct / (BATCH_SIZE*len(validation_loader))))
    return np.array([eval_losses, np.linspace(epoch - 1, epoch, num=eval_losses.shape[0])])

if __name__ == "__main__":
    train_time = 0
    num_train_epochs = 10

    # ---------- actual training
    TRAINING = True
    if TRAINING:
        train_loss = np.empty([2,0])  # empty array to store the train losses
        eval_loss = np.empty([2,0])  # empty array to store the validation losses
        for epoch in range(num_train_epochs):
            begin = time.time()

            # 1-indexing is for suckers and matlab-users
            train_loss = np.concatenate([train_loss, train(epoch + 1)], axis = 1)

            train_time += time.time() - begin

            # eval for this epoch
            eval_loss = np.concatenate([eval_loss, eval(epoch + 1)], axis = 1)
            print("Training time:\t{:}m {:2.0f}s".format(int(train_time / 60),
                                                     train_time % 60))
            import datetime
            #    with open(datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S") + ".csv", 'w') as csvfile:
            #        writer = csv.writer(csvfile, delimiter="\t")
            df = pd.DataFrame([train_loss[0,:], train_loss[1,:],
                               eval_loss[0,:], eval_loss[1,:]]).T
            df.to_csv(datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S") + ".csv",
                      sep="\t",
                      header=["train_loss", "train_epoch","eval_loss", "eval_epoch"],
                      index=False)

    TESTING = False

    if (TESTING):
        test_dataset = TestDataset('data/GT-final_test.csv', 'data/GTSRB/Final_Test/Images',
                                   transform=transforms.Compose([Rescale(32),
                                                                 SubtractMean(),
                                                                 Sample2Tensor()]))

        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  shuffle=False,
                                                  pin_memory=False)  # load to cuda immediately

        model.eval()
        correct = 0
        with torch.no_grad():
            test_losses = np.zeros((len(test_loader)))
            for batch_idx, data_set in enumerate(test_loader):
                data = data_set['image']
                target = data_set['target']
                data, target = data.to(gpu, dtype=torch.float), target.to(gpu, dtype=torch.float)
                output = model(data)
                test_losses[batch_idx] = F.cross_entropy(output, target, size_average=True).item() # sum up batch loss
                pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability

                correct += pred.eq(target.view_as(pred)).sum().item()

            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
                  .format(
                      np.average(test_losses), correct, len(test_loader),
                      100. * correct / len(test_loader)))
