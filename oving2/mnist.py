import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt

# Load observations from the mnist dataset. The observations are divided into a training set and a test set
mnist_train = torchvision.datasets.MNIST('./data', train=True, download=True)
x_train = mnist_train.data.reshape(-1, 784).float()  # Reshape input
y_train = torch.zeros((mnist_train.targets.shape[0], 10))  # Create output tensor
y_train[torch.arange(mnist_train.targets.shape[0]), mnist_train.targets] = 1  # Populate output

mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True)
x_test = mnist_test.data.reshape(-1, 784).float()  # Reshape input
y_test = torch.zeros((mnist_test.targets.shape[0], 10))  # Create output tensor
y_test[torch.arange(mnist_test.targets.shape[0]), mnist_test.targets] = 1  # Populate output

class SoftmaxModel:
    def __init__(self):
        # Model variables
        self.W = torch.rand((784,10), requires_grad=True)
        self.b = torch.rand((10), requires_grad=True)

    def logits(self, x):
        return x @ self.W + self.b
    
    # Predictor
    def f(self, x):
        return torch.softmax(self.logits(x), dim=0)
    
    # Cross Entropy loss
    def loss(self, x, y):
        return torch.nn.functional.cross_entropy(self.logits(x), y)
    
    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())
    
model = SoftmaxModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.b, model.W], 0.1)
for epoch in range(10000):
    model.loss(x_train, y_train).backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b
    optimizer.zero_grad()  # Clear gradients for next step

print("W = %s, b = %s, loss = %s, accuracy = %s" %(model.W, model.b, model.loss(x_train, y_train), model.accuracy(x_train, y_train)))


for i in range(10):
    plt.imsave(f"W_image_{i}.png", model.W[:, i].reshape(28,28).detach())
