from sys import path

p = "/".join(__file__.split("/")[:-1])
p += "/../../src/"
path.append(p)

from easyAI.layers import Dense, Input 

d = Dense(5)

print(d)

i = Input(5)

print(i)
