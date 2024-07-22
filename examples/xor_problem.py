from sys import path

p = "/".join(__file__.split("/")[:-1])
p += "/../src/"
path.append(p)

from easyAI.core.objects import Layer
from easyAI.models.arquitectures import MLP
from random import randint

xor = lambda x, y: int((x or y) and not (x and y))

X: list[int] = [randint(0, 1) for _ in range(600)]
y: list[int] = [xor(X[i], X[i + 1]) for i in range(0, len(X) - 1, 2)]

nn = MLP([
    Layer(2),
    Layer(2, "step"),
    Layer(1, "step"),
],
    optimizer="plr")

exit()
