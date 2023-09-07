import torch
import matplotlib.pyplot as plt

x_train = torch.tensor([[0,0], [1,0], [0,1], [1,1]], dtype=torch.float)
y_train = torch.tensor([[1], [1], [1], [0]], dtype=torch.float)

class NAND_model:
    def __init__(self):
        # Model variables
        # torch.rand() returnerer 2x1 og 1x1 matrise med double verdier mellom 0 og 1
        self.W = torch.rand((2,1), requires_grad=True) 
        self.b = torch.rand((1,1), requires_grad=True)

    def logits(self, x):
        return x @ self.W + self.b
    
    # Predictor 
    def f(self, x):
        return torch.sigmoid(self.logits(x))
    
    # Cross Entropy loss
    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(self.logits(x), y)
    
    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1),y.argmax(1)).float())
    
model = NAND_model()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.b, model.W], 0.01)
for epoch in range(10000):
    model.loss(x_train, y_train).backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b
    optimizer.zero_grad()  # Clear gradients for next step

print("W = %s, b = %s, loss = %s, accuracy = %s" %(model.W, model.b, model.loss(x_train, y_train), model.accuracy(x_train, y_train)))


fig = plt.figure()
ax = fig.add_subplot(projection='3d')

xt = x_train.t()[0]
yt = x_train.t()[1]

ax.scatter(xt.numpy(),yt.numpy(), y_train.numpy())
ax.scatter(xt.numpy(),yt.numpy(), model.f(x_train).detach().numpy(), color="orange")

ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()