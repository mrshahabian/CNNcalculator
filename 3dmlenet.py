import torch
from torchsummary import summary

# Define your CNN model
class MyCNN(torch.nn.Module):
    def __init__(self):
        super(MyCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 3, 1, 1)
        self.conv2 = torch.nn.Conv2d(64, 128, 3, 1, 1)
        self.fc1 = torch.nn.Linear(128 * 16 * 16, 256)
        self.fc2 = torch.nn.Linear(256, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 128 * 16 * 16)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate your model
model = MyCNN()
x = torch.randn(1, 3, 32, 32)  # Input data on the CPU
y = model(x)


# Print a summary of the model
summary(model, (3, 32, 32))  # Adjust input shape as needed
