import torch
import matplotlib.pyplot as plt
import numpy

x,y = numpy.loadtxt("day_head_circumference.csv", delimiter=",", unpack=True)
x_train = torch.tensor(x).float().reshape(-1, 1)
y_train = torch.tensor(y).float().reshape(-1, 1)

class LinearRegressionModel:

    def __init__(self):
        # Model variables
        self.W = torch.tensor([[0.0]], requires_grad=True)  # requires_grad enables calculation of gradients
        self.b = torch.tensor([[0.0]], requires_grad=True)

    def sigmoid(self, z):
        return 1/(1+numpy.exp(-z))
    
    # Predictor
    def f(self, x):
        return 20*(1/(1+torch.sigmoid(x*self.W + self.b) + 31))

    # Uses Mean Squared Errorx*self.W + self.b
    def loss(self, x, y):
        return torch.mean(torch.square(self.f(x) - y))  # Can also use torch.nn.functional.mse_loss(self.f(x), y) to possibly increase numberical stability


model = LinearRegressionModel()


# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.W, model.b], 0.0001)
for epoch in range(100000):
    model.loss(x_train, y_train).backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b,
    # similar to:
    # model.W -= model.W.grad * 0.01
    # model.b -= model.b.grad * 0.01

    optimizer.zero_grad()  # Clear gradients for next step


# Print model variables and loss
print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

# Visualize result
plt.plot(x_train, y_train, 'o', label='$(x^{(i)},y^{(i)})$')
plt.xlabel('lengde')
plt.ylabel('vekt')
x = torch.tensor([[torch.min(x_train)], [torch.max(x_train)]]) 
plt.plot(x, model.f(x).detach(), label='$f(x) = xW+b$')
plt.legend()
plt.show()
