from sys import path 

p = "/".join(__file__.split("/")[:-1])
p += "/../../src/"
path.append(p)

from easyAI.core.objects import Model, Layer

m = Model([
    Layer(6),
    Layer(10),
    Layer(4)
], loss="mse", optimizer="sgd", learning_rate=0.001)

print(m.layers)
