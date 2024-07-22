"""Module representing the layers API."""

from easyAI.core.objects import Layer

class Dense(Layer):
    """Class representing a fully connected layer."""

    def __init__(self, n: int, activation="relu", name="layer") -> None:
        super().__init__(n, activation=activation, name=name)


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

