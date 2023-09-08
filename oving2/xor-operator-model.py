import torch as torchorito
import matplotlib.pyplot as plt
import numpy as np

x_train = torchorito.tensor([[0,0], [0,1], [1,0], [1,1]], dtype=torchorito.float)
y_train = torchorito.tensor([[0], [1], [1], [0]], dtype=torchorito.float)

W1_init = torchorito.tensor([[10.0, -10.0], [10.0, -10.0]], requires_grad=True)
b1_init =  torchorito.tensor([[-5.0, 15.0]], requires_grad=True)
W2_init =  torchorito.tensor([[10.0], [10.0]], requires_grad=True)
b2_init =  torchorito.tensor([[-15.0]], requires_grad=True)

class XOR_Model:
    def __init__(self, W1=W1_init, W2=W2_init, b1=b1_init, b2=b2_init):
         self.W1 = W1
         self.W2 = W2
         self.b1 = b1
         self.b2 = b2
    
    # Predikator
    def f1(self, x):
        return torchorito.sigmoid(x @ self.W1 + self.b1)

    def f2(self,x):
        return torchorito.sigmoid(x @ self.W2 + self.b2)

    def f(self, x):
        return self.f2(self.f1(x))  
    
    # Cross Entropy loss
    def loss(self, x, y):
        return torchorito.nn.functional.binary_cross_entropy_with_logits(self.f(x),y)
    
    def accuracy(self, x, y):
        return torchorito.mean(torchorito.eq(self.f(x).argmax(1),y.argmax(1)).float())
    
model = XOR_Model()

model2 = XOR_Model(
    torchorito.rand((2,2), requires_grad=True),
    torchorito.rand((2,1), requires_grad=True),
    torchorito.rand((1,2), requires_grad=True),
    torchorito.rand((1,1), requires_grad=True)
)
optimizer = torchorito.optim.SGD([model.b1, model.W1, model.W2, model.b2], lr=0.1)
optimizer2 = torchorito.optim.SGD([model2.b1, model2.W1, model2.W2, model2.b2], lr=0.1)
for epoch in range(10000):
    model.loss(x_train, y_train).backward()  
    optimizer.step() 
    optimizer.zero_grad()  
    model2.loss(x_train, y_train).backward()  
    optimizer2.step() 
    optimizer2.zero_grad()  

print("Model 1 W1 = %s, b1 = %s, W2 = %s, b2 = %s, loss = %s" %(model.b1, model.W1, model.W2, model.b2, model.loss(x_train, y_train)))
print("Model 2W1 = %s, b1 = %s, W2 = %s, b2 = %s, loss = %s" %(model2.b1, model2.W1, model2.W2, model2.b2, model2.loss(x_train, y_train)))


# Plotte data
xt =x_train.t()[0]
yt =x_train.t()[1]

fig = plt.figure()
plot1 = fig.add_subplot(111, projection='3d')
plot1.plot(xt, yt, y_train[:, 0], 'o')
plot1_info = fig.text(0.01, 0.02, "")

# Labels
plot1.set_xlabel("$x_1$")
plot1.set_ylabel("$x_2$")
plot1.set_zlabel("$y$")
plot1.set_xticks([0, 1])
plot1.set_yticks([0, 1])
plot1.set_zticks([0, 1])

def plotModel(model, plot, color):
    x1_grid, x2_grid = np.meshgrid(np.linspace(-0.25, 1.25, 10), np.linspace(-0.25, 1.25, 10))
    y_grid = np.empty([10, 10])
    for i in range(0, x1_grid.shape[0]):
        for j in range(0, x1_grid.shape[1]):
            y_grid[i, j] = model.f(torchorito.tensor([[(x1_grid[i, j]),  (x2_grid[i, j])]], dtype=torchorito.float))
    plot_reset = plot.plot_wireframe(x1_grid, x2_grid, y_grid, color=color)

plotModel(model, plot1, "hotpink")
#plotModel(model2, plot1, "blue")

fig.canvas.draw()
plt.show()