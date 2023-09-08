import torch
import matplotlib.pyplot as plt
import numpy as np

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


# forbered data til riktig format så plottet blir fornøyd:
# https://stackoverflow.com/questions/25370789/matplotlib-3d-wire-frame-plot-not-plotting-as-expected
# inspirert av https://github.com/Her0elt/applied-machine-learning/blob/master/exercise%202/notebooks/NAND_operator.ipynb

fig = plt.figure()
plot3d = fig.add_subplot(111, projection='3d')
plot3d.plot(xt, yt, y_train[:, 0], 'o')

x, y = np.meshgrid(np.linspace(-0.25, 1.25, 10), np.linspace(-0.25, 1.25, 10))
z = np.empty([10, 10])

for i in range(0, x.shape[0]):
    for j in range(0, x.shape[1]):
        z[i, j] = model.f(torch.tensor([[(x[i, j]),  (y[i, j])]], dtype=torch.float))

plot3d_f = plot3d.plot_wireframe(x, y, z, color="hotpink")

# labels
plot3d.set_xticks([0, 1])
plot3d.set_yticks([0, 1])
plot3d.set_zticks([0, 1])
plot3d.set_xlabel("$x_1$")
plot3d.set_ylabel("$x_2$")
plot3d.set_zlabel("$y$")

fig.canvas.draw()
plt.show()