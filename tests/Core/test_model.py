from sys import path 

p = "/".join(__file__.split("/")[:-1])
p += "/../../src/"
path.append(p)

from easyAI.core.objects import Model, Layer

m = Model([
    Layer(2),
    Layer(1)
])

print(m._layers)
print(m.input_layer)
