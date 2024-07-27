from sys import path

p = "/".join(__file__.split("/")[:-1])
p += "/../../src/"
path.append(p)

from easyAI.core.objects import Model
from easyAI.layers import Dense
from easyAI.core.optimizers import SGD
from easyAI.core.loss import mean_squared_error

m = Model([Dense(1), Dense(2), Dense(1)])

print(m.layers)

m.fit(
    X=[x for x in range(0, 1001)],
    Y=[x for x in range(0, 2000, 2)],
    loss=mean_squared_error,
    epochs=1,
    optimizer=SGD,
    learning_rate=0.01,
    verbose=False,
)

print(m)
