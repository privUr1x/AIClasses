#!/usr/bin/python3
from Perceptrons import SimplePerceptron
from random import randint

# Building an AND gate
X: list = [randint(0,1) for _ in range(200)]
y: list = [int(bool(X[i]) and bool(X[i + 1])) for i in range(int(len(X) / 2))]

entries: int = 2

print(f"Input data ({len(X)}):", X)
print(f"Output data ({len(y)}):", y)
print(f"Entries:", entries)

p = SimplePerceptron(X, y, entries)

print(p._weights)
print(p.train(verbose=True))
print(p._weights)
print(p._bias)
