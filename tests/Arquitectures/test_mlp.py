from sys import path

p = "/".join(__file__.split("/")[:-1])
p += "/../../src/"
path.append(p)

from easyAI.core.objects import Dense
from easyAI.models.arquitectures import MLP

nn = MLP([
    Dense(2),
    Dense(1)
])

