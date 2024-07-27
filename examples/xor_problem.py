from sys import path

p = "/".join(__file__.split("/")[:-1])
p += "/../src/"
path.append(p)

from easyAI.layers import Dense, Input
from easyAI.arquitectures import MLP
from random import randint

xor = lambda x, y: int((x or y) and not (x and y))

X = [randint(0, 1) for _ in range(600)]
y = [xor(X[i], X[i + 1]) for i in range(0, len(X) - 1, 2)]

nn = MLP([
    Input(2),  # Input layer (nodes)
    Dense(2, activation="step"),  # Hidden layer (neurons)
    Dense(1, activation="step"),  # Output layer (neuron)
])

nn.fit(X, y, verbose=True)
