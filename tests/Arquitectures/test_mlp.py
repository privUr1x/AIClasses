from sys import path

p = "/".join(__file__.split("/")[:-1])
p += "/../../src/"
path.append(p)

from easyAI.layers import Dense
from easyAI.arquitectures import MLP

nn = MLP([
    Dense(2),
    Dense(1)
])

