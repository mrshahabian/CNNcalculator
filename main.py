import torch
import torchvision.models as models
from thop import profile
import lenet
import mlenet
import torchviz
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch.nn as nn
import numpy as np

# Choose a pre-trained model (e.g., ResNet18)
model = models.resnet18()


# Define an example input (adjust input size accordingly)
input_data = torch.randn(1, 3, 34, 34)

# Print model summary
# print(model)


# Lenet-5
model_lenet = lenet.LeNet()

# Print model summary
# print(model_lenet)


# M-Lenet
m_lenet = mlenet.LeNetRegularized(3, 14)

# Print model summary
# print(m_lenet)

# Mobile-Lenet
mobilenet = models.MobileNetV2()

# Print model summary
# print(mobilenet)

trans_model = models.squeezenet1_1()

# Print model summary
# print(trans_model)


# Compute FLOPs and parameters
flops, params = profile(model, inputs=(input_data,))
print(f" resnet18 FLOPs: {flops / 1e9} G, Parameters: {params / 1e6} M")
y = model(input_data)
torchviz.make_dot(y, params=dict(model.named_parameters())).render("resnet18", format="png")
# Compute FLOPs and parameters
flops, params = profile(model_lenet, inputs=(input_data,))
print(f" model_lenet FLOPs: {flops / 1e9} G, Parameters: {params / 1e6} M")
y = model_lenet(input_data)
torchviz.make_dot(y, params=dict(model_lenet.named_parameters())).render("lenet-5", format="png")

# Compute FLOPs and parameters
flops, params = profile(m_lenet, inputs=(input_data,))
print(f" m_lenet FLOPs: {flops / 1e9} G, Parameters: {params / 1e6} M")
y = m_lenet(input_data)
torchviz.make_dot(y, params=dict(m_lenet.named_parameters())).render("m_lenet", format="png")


# Compute FLOPs and parameters
flops, params = profile(mobilenet, inputs=(input_data,))
print(f" mobilenet FLOPs: {flops / 1e9} G, Parameters: {params / 1e6} M")
y = mobilenet(input_data)
torchviz.make_dot(y, params=dict(mobilenet.named_parameters())).render("mobilenet", format="png")

# Compute FLOPs and parameters
flops, params = profile(trans_model, inputs=(input_data,))
print(f" trans_model FLOPs: {flops / 1e9} G, Parameters: {params / 1e6} M")
y = trans_model(input_data)
torchviz.make_dot(y, params=dict(trans_model.named_parameters())).render("trans_model", format="png")