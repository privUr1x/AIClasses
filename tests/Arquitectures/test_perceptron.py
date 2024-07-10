from sys import path

p = "/".join(__file__.split("/")[:-1])
p += "/../../src/"
path.append(p)

from easyAI.Arquitectures import Perceptron
from random import randint

ENTRIES: int = 2

# Building logical gates (OR, AND, NOT)
X: list = [randint(0, 1) for _ in range(200)]

# AND gate
and_y: list = [int(bool(X[i]) and bool(X[i + 1])) for i in range(0, len(X) - 1, 2)]

and_g = Perceptron(ENTRIES)

print(dir(and_g))

print("\nLoss history for AND", and_g.fit(X, and_y, verbose=True))
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

or_g = Perceptron(ENTRIES)

print("\nLoss history for OR", or_g.fit(X, or_y))
print("OR weights", or_g._weights)
print("OR bias", or_g._bias)

# NAND gate
nand_y: list = [
    int(not (bool(X[i]) and bool(X[i + 1]))) for i in range(0, len(X) - 1, 2)
]

nand_g = Perceptron(ENTRIES)

print("\nLoss history for NAND", nand_g.fit(X, nand_y))
print("NAND weights", nand_g._weights)
print("NAND bias", nand_g._bias)

# XOR gate
xor_y: list = [int(bool(X[i]) ^ bool(X[i + 1])) for i in range(0, len(X) - 1, 2)]

xor_g = Perceptron(ENTRIES)

print("\nLoss history for XOR", xor_g.fit(X, xor_y))
print("XOR weights", xor_g._weights)
print("XOR bias", xor_g._bias)

# NOT gate
ENTRIES: int = 1

X: list = X[: int(len(X) / 2)]
not_y: list = [int(not bool(X[i])) for i in range(len(X))]

not_g = Perceptron(ENTRIES)

print("\nLoss history for NOT", not_g.fit(X, not_y))
print("NOT weights", not_g._weights)
print("NOT bias", not_g._bias)

