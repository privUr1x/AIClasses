from sys import path

p = "/".join(__file__.split("/")[:-1])
p += "/../../src/"
path.append(p)

from easyAI.layers import Dense, NodeLayer
from easyAI.arquitectures import MLP

nn = MLP([
    NodeLayer(1),
    Dense(1)
])

print(nn)
