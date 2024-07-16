from typing import List, Union
from abc import ABC, abstractmethod

"""Module representing optimizers used for training models."""


class Optimizer(ABC):
    """Class representing the abstraction to an optimizer."""

    def __init__(self, learning_rate: Union[int, float]):
        self._learning_rate: Union[int, float] = learning_rate

    @abstractmethod
    def update(self, params, grads):
        pass


class SGD(Optimizer):
    """Class representing Stochastic Gradient Descent."""

    def __init__(self, learning_rate: Union[int, float]):
        super().__init__(learning_rate)


class Adam(Optimizer):
    """Class representing Adaptative Moment Estimation."""

    def __init__(self, learning_rate: Union[int, float]):
        super().__init__(learning_rate)


class PLR(Optimizer):
    """Class representing Perceptron Learning Rule"""

    def __init__(self, learning_rate: Union[int, float]):
        super().__init__(learning_rate)


def perceptron_learning_rule(
    P,  # The perceptron class itself
    X: List[Union[int, float]],
    y: List[Union[int, float]],
    verbose: bool = False,
):
    for epoch in range(len(y)):
        # Narrowing down y for X
        eX = X[epoch * P._n : (epoch + 1) * P._n]
        ey = y[epoch]

        z = P.__call__(eX)[0]

        # Updating parameters
        if z != ey:
            for n in P.output:  # [n0, n1, ..., nn]
                for i in range(n.n):  # [w-1, w1, ..., wn]
                    n.w[i] += P._lr * (ey - z) * eX[i]
                n.b += P._lr * (ey - z)

        if verbose:
            print(f"Epoch {epoch}:\n\tModel output: {z}\n\tExpected output: {ey}")

    return None


optimizers_map: dict = {
    "sgd": SGD,
    "adam": Adam,
    "plr": perceptron_learning_rule,
}
