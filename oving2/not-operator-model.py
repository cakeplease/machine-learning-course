import torch
import matplotlib.pyplot as plt

x_train = torch.tensor([[1.0], [0.0]])
y_train = torch.tensor([[0.0], [1.0]])

class NOT_Model:
    def __init__(self):
    # Model variables
        self.W = torch.rand((1,1), requires_grad=True)
        self.b = torch.rand((1,1), requires_grad=True)

    def logits(self, x):
        # b (bias, intercept), forteller oss hva som er forventet gjennomsnittlig verdi
        # av y når x er 0
        # W (weight, slope), forteller oss hvor mye y øker, gjennomsnittlig,
        # når vi øker x med én enhet
        return x @ self.W + self.b
    
    # Predictor 
    def f(self, x):
        return torch.sigmoid(self.logits(x))
    
    # Cross Entropy loss
    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(self.logits(x), y)
    
    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1),y.argmax(1)).float())

model = NOT_Model()


# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.W, model.b], 0.01)
for epoch in range(10000):
    model.loss(x_train, y_train).backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b,

    optimizer.zero_grad()  # Clear gradients for next step


# Print model variables, loss and accuracy
print("W = %s, b = %s, loss = %s accuracy = %s" % (model.W, model.b, model.loss(x_train, y_train), model.accuracy(x_train, y_train)))

# våre observasjoner
plt.plot(x_train, y_train, 'o', label='$(x^{(i)},y^{(i)})$')

plt.xlabel('x')
plt.ylabel('y')

# x skalaen: fra 0 til 1 med 0.01 steg
x = torch.arange(0.0, 1.0, 0.01).reshape(-1,1)

# y = model.f(x) vår predikator: sigmoid(x @ W + b)
plt.plot(x, model.f(x).detach(), label='$\\hat y = f(x) = \sigma(xW + b)$')

plt.legend()
plt.show()
