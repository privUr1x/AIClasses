from sys import path

p = "/".join(__file__.split("/")[:-1])
p += "/../src/"
path.append(p)

from easyAI.arquitectures import MLP 
from easyAI.core.objects import Model
from easyAI.layers import Input, Dense 

nn: Model = MLP([
    Input(5, name="Entry"),
    Dense(20, activation="sigmoid"),
    Dense(12, activation="sigmoid"),
    Dense(8, activation="relu"),
])

print(nn._output)
