from sys import path 

p = "/".join(__file__.split("/")[:-1])
p += "/../../src/"
path.append(p)

from easyAI.core.objects import Model, Layer

m = Model([
    Layer(6),
    Layer(10),
    Layer(4)
])

print(m._layers)
print(m.input_layer)
print(m.output)
