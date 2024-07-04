#!/usr/bin/python3
from Arquitectures import Perceptron
from random import randint

ENTRIES: int = 2

# Building logical gates (OR, AND, NOT)
X: list = [randint(0,1) for _ in range(200)]

# AND gate
and_y: list = [int(bool(X[i]) and bool(X[i + 1])) for i in range(0, len(X) - 1, 2)]

and_g = Perceptron(X, and_y, ENTRIES)

print("Loss history for AND", and_g.train(verbose=True))
print("AND weights", and_g._weights)
print("AND bias", and_g._bias)

# The ideal config would be something like this
"""
and_g._weights = [1 , 1]
and_g._bias = -2

print(and_g([1,1])) # 1
print(and_g([0,0])) # 0
print(and_g([1,0])) # 0
"""

# OR gate
or_y: list = [int(bool(X[i]) or bool(X[i + 1])) for i in range(0, len(X) - 1, 2)]

or_g = Perceptron(X, or_y, ENTRIES)

print("Loss history for OR", or_g.train())
print("OR weights", or_g._weights)
print("OR bias", or_g._bias)

# NAND gate
nand_y: list = [int(not (bool(X[i]) and bool(X[i + 1]))) for i in range(0, len(X) - 1, 2)]

nand_g = Perceptron(X, nand_y, ENTRIES)

print("Loss history for NAND", nand_g.train())
print("NAND weights", nand_g._weights)
print("NAND bias", nand_g._bias)

# XOR gate
xor_y: list = [int(bool(X[i]) ^ bool(X[i + 1])) for i in range(0, len(X) - 1, 2)]

xor_g = Perceptron(X, xor_y, ENTRIES)

print("Loss history for XOR", xor_g.train())
print("XOR weights", xor_g._weights)
print("XOR bias", xor_g._bias)

# NOT gate
ENTRIES: int = 1

X: list = X[:int(len(X)/2)]
not_y: list = [int(not bool(X[i])) for i in range(len(X))]

not_g = Perceptron(X, not_y, ENTRIES)

print("Loss history for NOT", not_g.train())
print("NOT weights", not_g._weights)
print("NOT bias", not_g._bias)
