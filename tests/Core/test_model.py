from sys import path 

p = "/".join(__file__.split("/")[:-1])
p += "/../../src/"
path.append(p)

from easyAI.core.objects import Model, Layer

m: Model = Model(Layer(2), Layer(5))

print(m.input_layer)
print(m.hidden_layers)
print(m.output)

print(m.forward([1,2]))
