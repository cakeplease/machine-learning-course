import torch
import torch.nn as nn
import numpy as np

char_encodings = [
    [1., 0., 0., 0., 0., 0., 0., 0.],  # ' '
    [0., 1., 0., 0., 0., 0., 0., 0.],  # 'h'
    [0., 0., 1., 0., 0., 0., 0., 0.],  # 'e'
    [0., 0., 0., 1., 0., 0., 0., 0.],  # 'l'
    [0., 0., 0., 0., 1., 0., 0., 0.],  # 'o'
    [0., 0., 0., 0., 0., 1., 0., 0.],  # 'w'
    [0., 0., 0., 0., 0., 0., 1., 0.],  # 'r'
    [0., 0., 0., 0., 0., 0., 0., 1.],  # 'd'
]

encoding_size = len(char_encodings)
index_to_char = [' ', 'h', 'e', 'l', 'o', 'w', 'r', 'd']

space = char_encodings[0]
h = char_encodings[1]
e = char_encodings[2]
l = char_encodings[3]
o = char_encodings[4]
w = char_encodings[5]
r = char_encodings[6]
d = char_encodings[7]


x_train = torch.tensor([[space], [h], [e], [l], [l], [o], [space], [w], [o], [r], [l], [d]], dtype=torch.float)  # ' hello world'
y_train = torch.tensor([h, e, l, l, o, space, w, o, r, l, d, space], dtype=torch.float)  # 'hello world '

class LongShortTermMemoryModel(nn.Module):

    def __init__(self, encoding_size):
        super(LongShortTermMemoryModel, self).__init__()

        self.lstm = nn.LSTM(encoding_size, 128)  # 128 is the state size
        self.dense = nn.Linear(128, encoding_size)  # 128 is the state size

    def reset(self):  # Reset states prior to new input sequence
        # Shape: (number of layers, batch size, state size)
        zero_state = torch.zeros(1, 1, 128)
        self.hidden_state = zero_state
        self.cell_state = zero_state

    def logits(self, x):  # x shape: (sequence length, batch size, encoding size)
        out, (self.hidden_state, self.cell_state) = self.lstm(
            x, (self.hidden_state, self.cell_state))
        return self.dense(out.reshape(-1, 128))

    def f(self, x):  # x shape: (sequence length, batch size, encoding size)
        return torch.softmax(self.logits(x), dim=1)

    def loss(self, x, y):  # x shape: (sequence length, batch size, encoding size), y shape: (sequence length, encoding size)
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))


model = LongShortTermMemoryModel(encoding_size)

optimizer = torch.optim.RMSprop(model.parameters(), 0.001)
for epoch in range(500):
    model.reset()
    model.loss(x_train, y_train).backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 10 == 9:
        # Generate characters from the initial characters ' h'
        model.reset()
        text = ' h'
        model.f(torch.tensor([[space]], dtype=torch.float))
        y = model.f(torch.tensor([[h]], dtype=torch.float))
        text += index_to_char[y.argmax(1)]
        for c in range(50):
            y = model.f(torch.tensor(
                [[char_encodings[y.argmax(1)]]], dtype=torch.float))
            text += index_to_char[y.argmax(1)]
        print(text)

