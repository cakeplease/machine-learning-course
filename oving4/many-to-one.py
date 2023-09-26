import torch
import numpy as np
import torch.nn as nn

class LongShortTermMemoryModel(nn.Module):
    def __init__(self, encoding_size, emoji_encoding_size):
        super(LongShortTermMemoryModel, self).__init__()
        self.lstm = nn.LSTM(encoding_size, 128)  # 128 is the state size
        # 128 is the state size
        self.dense = nn.Linear(128, emoji_encoding_size)

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


emojies = {
    'hat': '\U0001F3A9',
    'rat': '\U0001F400',
    'cat': '\U0001F408',
    'flat': '\U0001F3E2',
    'matt': '\U0001F468',
    'cap': '\U0001F9E2',
    'son': '\U0001F466'
}

index_to_char = [' ', 'h', 'a', 't', 'r',
                 'c', 'f', 'l', 'm', 'p', 's', 'o', 'n']

index_to_emoji = ['\U0001F3A9', '\U0001F400', '\U0001F408',
                  '\U0001F3E2', '\U0001F468', '\U0001F9E2', '\U0001F466']

char_encodings = np.eye(len(index_to_char))
encoding_size = len(char_encodings)
emojies = np.eye(len(emojies))
emoji_encoding_size = len(emojies)

letters = {}

for i, letter in enumerate(index_to_char):
    letters[letter] = char_encodings[i]


x_train = torch.tensor([
    [[letters['h']], [letters['a']], [letters['t']], [letters[' ']]],
    [[letters['r']], [letters['a']], [letters['t']], [letters[' ']]],
    [[letters['c']], [letters['a']], [letters['t']], [letters[' ']]],
    [[letters['f']], [letters['l']], [letters['a']], [letters['t']]],
    [[letters['m']], [letters['a']], [letters['t']], [letters['t']]],
    [[letters['c']], [letters['a']], [letters['p']], [letters[' ']]],
    [[letters['s']], [letters['o']], [letters['n']], [letters[' ']]],
], dtype=torch.float)


y_train = torch.tensor([
    [emojies[0], emojies[0], emojies[0], emojies[0]],
    [emojies[1], emojies[1], emojies[1], emojies[1]],
    [emojies[2], emojies[2], emojies[2], emojies[2]],
    [emojies[3], emojies[3], emojies[3], emojies[3]],
    [emojies[4], emojies[4], emojies[4], emojies[4]],
    [emojies[5], emojies[5], emojies[5], emojies[5]],
    [emojies[6], emojies[6], emojies[6], emojies[6]]], dtype=torch.float)


model = LongShortTermMemoryModel(encoding_size, emoji_encoding_size)

optimizer = torch.optim.RMSprop(model.parameters(), 0.001)  # 0.001
for epoch in range(500):
    for i in range(x_train.size()[0]):
        model.reset()
        model.loss(x_train[i], y_train[i]).backward()
        optimizer.step()
        optimizer.zero_grad()

y = -1

# rt
model.reset()
word = 'rt'
for i in range(len(word)):
    char_index = index_to_char.index(word[i])
    y = model.f(torch.tensor(
        [[char_encodings[char_index]]], dtype=torch.float))
print(index_to_emoji[y.argmax(1)])

# rat
model.reset()
word = 'rat'
for i in range(len(word)):
    char_index = index_to_char.index(word[i])
    y = model.f(torch.tensor(
        [[char_encodings[char_index]]], dtype=torch.float))
print(index_to_emoji[y.argmax(1)])
