__author__ = 'Richard Diehl Martinez' 

import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self,x):
        N = x.shape[0]
        return x.view(N,-1)

class ClassificationModel(nn.Module):
    '''
    Very Simple Classification Model, with the following structure: 
        Conv-RELU-Conv-RELU-Pool-Conv-RELU-Conv-RELU-FC-FC-FC
        
        We apply Batch Normalization Layers before each of the RELU 
        applications with exception of the last RELU layer
        
    Num outputs assumed to be 3 -- malignant/benign/indeterminate
    '''

    def __init__(self, num_outputs = 3):
        super(ClassificationModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, 4, stride=2)
        # 71x95
        self.conv2 = nn.Conv2d(10, 32, 3, stride=2)
        # 35 x 47
        self.pool1 = nn.MaxPool2d(kernel_size=3,stride=2)
        # 17 x 23 
        self.conv3 = nn.Conv2d(32, 32, 2, stride=3)
        # 6 x 8
        self.conv4 = nn.Conv2d(32, 16, 1) # reducing dimensionality 
        # 16 x 6 x 8 = 768 
        
        self.fc1 = nn.Linear(16 * 6 * 8, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, num_outputs)
        
        self.batchnorm1 = nn.BatchNorm2d(num_features=10)
        self.batchnorm2 = nn.BatchNorm2d(num_features=32)

    def forward(self, x):
        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = F.relu(self.batchnorm2(self.conv2(x)))
        x = self.pool1(x)
        x = F.relu(self.batchnorm2(self.conv3(x)))
        x = F.relu(self.conv4(x))
        
        # TODO: replace with Flatten
        N = x.shape[0]
        x = x.view(N,-1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
