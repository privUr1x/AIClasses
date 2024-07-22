"""Module representing the layers API."""

from collections.abc import Callable
from easyAI.core.objects import Layer, Neuron, Node

class Dense(Layer):
    """Class representing a fully connected layer."""

    def __init__(self, n: int, *, activation="relu", name="layer") -> None:
        super().__init__(n, activation=activation, name=name)

class NodeLayer(Layer):
    """Class representing a node layer."""

    def __init__(self, n: int, *, name="layer") -> None:
        super().__init__(n, activation="step", name=name)

        self._structure = [Node() for _ in range(self._n)]
    
        del self._activation

class Conv(Layer):
    """Class representing a convolutional network layer."""

    def __init__(self, n: int, activation="relu", name="layer") -> None:
        raise NotImplemented
        super().__init__(n, activation=activation, name=name)

class Rec(Layer):
    """Class representing a recurrent network layer."""

    def __init__(self, n: int, activation="relu", name="layer") -> None:
        raise NotImplemented
        super().__init__(n, activation=activation, name=name)

