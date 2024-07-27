"""Module representing the layers API."""

from collections.abc import Callable
from easyAI.core.objects import Layer, Neuron, Node

class Dense(Layer):
    """Class representing a fully connected layer."""

    def __init__(self, n: int, *, activation="relu") -> None:
        super().__init__(n, activation=activation)

class Input(Layer):
    """Class representing a node layer."""

    def __init__(self, n: int) -> None:
        super().__init__(n, activation="step")

        self._structure = [Node() for _ in range(self.n)]

        del self._activation

        super()._set_indexes()

class Conv(Layer):
    """Class representing a convolutional network layer."""

    def __init__(self, n: int, activation="relu") -> None:
        raise NotImplemented
        super().__init__(n, activation=activation)

class Rec(Layer):
    """Class representing a recurrent network layer."""

    def __init__(self, n: int, activation="relu") -> None:
        raise NotImplemented
        super().__init__(n, activation=activation)

