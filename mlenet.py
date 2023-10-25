# from torch.nn import Module
# from torch.nn import Conv2d
# from torch.nn import Linear
# from torch.nn import MaxPool2d
# from torch.nn import ReLU
# from torch.nn import LogSoftmax
# from torch import flatten
# class LeNet(Module):
#     def __init__(self, numChannels, classes):
#         # call the parent constructor
#         super(LeNet, self).__init__()
#
#         # initialize first set of CONV => RELU => POOL layers
#         self.conv1 = Conv2d(in_channels=numChannels, out_channels=20,
#             kernel_size=(5, 5))
#         self.relu1 = ReLU()
#         self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
#
#         # initialize second set of CONV => RELU => POOL layers
#         self.conv2 = Conv2d(in_channels=20, out_channels=50,
#             kernel_size=(5, 5))
#         self.relu2 = ReLU()
#         self.maxpool2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
#
#         # initialize first (and only) set of FC => RELU layers
#         self.fc1 = Linear(in_features=980, out_features=500)
#         # self.fc1 = Linear(in_features=3250, out_features=500)
#         # self.fc1 = Linear(in_features=7250, out_features=500)
#         self.fc1 = nn.Linear(980, 500)
#         self.fc2 = nn.Linear(500, 250)
#         self.fc3 = nn.Linear(250, classes)
#         self.relu3 = ReLU()
#
#         # initialize our softmax classifier
#         self.fc2 = Linear(in_features=500, out_features=classes)
#         self.logSoftmax = LogSoftmax(dim=1)
#
#     def forward(self, x):
#         # pass the input through our first set of CONV => RELU =>
#         # POOL layers
#         x = self.conv1(x)
#         x = self.relu1(x)
#         x = self.maxpool1(x)
#
#         # pass the output from the previous layer through the second
#         # set of CONV => RELU => POOL layers
#         x = self.conv2(x)
#         x = self.relu2(x)
#         x = self.maxpool2(x)
#
#         # flatten the output from the previous layer and pass it
#         # through our only set of FC => RELU layers
#         x = flatten(x, 1)
#         x = self.fc1(x)
#         x = self.relu3(x)
#
#         # pass the output to our softmax classifier to get our output
#         # predictions
#         x = self.fc2(x)
#         output = self.logSoftmax(x)
#
#         # return the output predictions
#         return output
from torch.nn import LogSoftmax
import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNetRegularized(nn.Module):
    def __init__(self, numChannels, classes=14, dropout_rate=0.5, weight_decay=0.001):
        super(LeNetRegularized, self).__init__()
        self.dropout_rate = dropout_rate
        self.classes = classes
        self.weight_decay = weight_decay
        self.conv1 = nn.Conv2d(numChannels, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        ##### 68    #####
        # self.fc1 = nn.Linear(2100, 800)
        # self.fc2 = nn.Linear(800, 450)
        # self.fc3 = nn.Linear(450, classes)
        ######    34   #####
        self.fc1 = nn.Linear(980, 500)
        self.fc2 = nn.Linear(500, 250)
        self.fc3 = nn.Linear(250, classes)
        ####  17 ########
        # self.fc1 = nn.Linear(280, 150)
        # self.fc2 = nn.Linear(150, 75)
        # self.fc3 = nn.Linear(75, classes)
         #####  9 ########
        # self.fc1 = nn.Linear(600, 300)
        # self.fc2 = nn.Linear(300, 150)
        # self.fc3 = nn.Linear(150, classes)
        self.dropout = nn.Dropout(dropout_rate)
        self.logSoftmax = LogSoftmax(dim=1)


    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2, 2)
        # x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2, 2)
        # x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.logSoftmax(x)
        return x

    def l2_regularization_loss(self):
        l2_loss = 0
        # l2_loss = torch.tensor(0.).to(device)
        for param in self.parameters():
            l2_loss += torch.norm(param)
        return l2_loss * self.weight_decay
