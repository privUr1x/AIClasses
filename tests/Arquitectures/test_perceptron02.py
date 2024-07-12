from sys import path

p = "/".join(__file__.split("/")[:-1])
p += "/../../src/"
path.append(p)

from easyAI.Arquitectures import Perceptron

p = Perceptron(3)
