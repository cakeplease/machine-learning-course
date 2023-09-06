import torch
import matplotlib.pyplot as plt
import numpy
import pandas

data = pandas.read_csv("day_length_weight.csv", dtype="float")
y_train = torch.tensor(data.pop("# day").to_numpy(), dtype=torch.float).reshape(-1, 1)
x_train = torch.tensor(data.to_numpy(), dtype=torch.float).reshape(-1, 2)

print(x_train)
xt = x_train.t()[0]
yt = x_train.t()[1]


class LinearRegressionModel:

    def __init__(self):
        # Model variables
        #self.id = torch.rand((2,1),requires_grad=True )
        self.W = torch.rand((2, 1), requires_grad=True)  # requires_grad enables calculation of gradients
        self.b = torch.rand((1, 1 ), requires_grad=True)

        # self.W = torch.tensor([[0.0]], requires_grad=True)  # requires_grad enables calculation of gradients
        # self.b = torch.tensor([[0.0]], requires_grad=True)

   # Predictor
    def f(self, x):
        return x @ self.W + self.b  # @ corresponds to matrix multiplication

    # Uses Mean Squared Error
    def loss(self, x, y):
        #return torch.mean(torch.square(self.f(x) - y))  # Can also use torch.nn.functional.mse_loss(self.f(x), y) to possibly increase numberical stability
        return torch.nn.functional.mse_loss(self.f(x), y)

model = LinearRegressionModel()


# Optimize: adjust W and b to minimize loss using stochastic gradient descent
print(model.b.shape)
print(model.W.shape)
optimizer = torch.optim.SGD([model.b, model.W], 0.0001)

for epoch in range(100000):
    model.loss(x_train, y_train).backward()  # Compute loss gradients
    optimizer.step() 
    optimizer.zero_grad()  # Clear gradients for next step


# Print model variables and loss
print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(xt.numpy(),yt.numpy(), y_train.numpy())
ax.scatter(xt.numpy(),yt.numpy(), model.f(x_train).detach().numpy(), color="orange")

ax.set_xlabel('Lengde')
ax.set_ylabel('Vekt')
plt.show()