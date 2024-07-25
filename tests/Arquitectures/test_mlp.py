from sys import path

p = "/".join(__file__.split("/")[:-1])
p += "/../../src/"
path.append(p)

from easyAI.layers import Dense, Input
from easyAI.arquitectures import MLP

nn = MLP([
    Input(1),
    Dense(1)
])

print(nn.fit([1, 2, 3, 4, 5, 6, 7, 8, 9], [2, 4, 6, 8, 10, 12, 14, 16, 18], epochs=1, verbose=True))
