from collections.abc import Callable
from random import randint, random
from functions import relu
from typing import Union, List
from classtools import Verifiers
from numpy import (
    float16,
    float32,
    float64,
    int16,
    int32,
    int64,
    int8,
    ndarray,
    uint16,
    uint32,
    uint64,
    uint8,
)

verify_type = Verifiers.verify_type
verify_len = Verifiers.verify_len
verify_iterable = Verifiers.verify_iterable
verify_components_type = Verifiers.verify_components_type


class Neuron(object):
    """Class representing an artificial neuron"""

    __nptypes = (
        int8,
        int16,
        int32,
        int64,
        uint8,
        uint16,
        uint32,
        uint64,
        float16,
        float32,
        float64,
    )

    def __init__(self) -> None:
        self._identifier: int = randint(1, 10_000)
        self._bias: float = random()
        self._inputnodes: list = []
        self._n: int = len(self._inputnodes)
        self._weights: list = [random() for _ in range(len(self._inputnodes))]
        self._activation: Callable = relu
        self._z: float = self._activation(
            sum(x * w for x, w in zip(self._inputnodes, self._weights)) + self._bias
        )

        self._layer: int
        self._i: int

    @property
    def b(self) -> float:
        """The bias property."""
        return self._bias

    @b.setter
    def b(self, value: Union[int, float]) -> None:
        self._bias = float(verify_type(value, (int, float, *Neuron.__nptypes)))

    @property
    def inputs(self) -> list:
        """The inputs property."""
        return self._inputnodes

    @inputs.setter
    def inputs(self, value: list) -> None:
        self._inputnodes = verify_type(value, list)

    @property
    def w(self) -> list:
        """The w property."""
        return self._w

    @w.setter
    def w(self, value) -> None:
        self._w = verify_type(value, (list, ndarray))

    @property
    def activation(self) -> Callable:
        """The activation property."""
        return self._activation

    @activation.setter
    def activation(self, value: Callable) -> None:
        if not callable(value):
            raise TypeError(f"Expected {value} to be callable.")

        self._activation = value


class Perceptron(Neuron):
    "Class representing a Perceptron (Unitary Layer Neural DL Model)"

    __nptypes = (
        int8,
        int16,
        int32,
        int64,
        uint8,
        uint16,
        uint32,
        uint64,
        float16,
        float32,
        float64,
    )

    def __init__(self, entries: int) -> None:
        """
        Builds an instance give X training data, y training data and entries.

        Args:
            - entries (int): The number of inputs of the model.
        """
        super().__init__()

        if isinstance(entries, float):
            if int(entries) == entries:
                entries = int(entries)

        del self._inputnodes

        # Model params
        self._n = verify_type(entries, int)  # Model arquitecture
        self._bias: float = random()
        self._weights: List[float] = [random() for _ in range(self._n)]
        self._lr: float = 0.1
        self._activation = self.step

    def __call__(self, X: Union[list, ndarray]) -> int:
        return self.predict(X)

    @property
    def id(self) -> int:
        """The id property."""
        return self._identifier

    @property
    def X(self) -> Union[list, ndarray]:
        """The X property."""
        return self._X

    @X.setter
    def X(self, value) -> None:
        self._X = verify_components_type(
            verify_type(value, (list, ndarray)),
            (int, float, *Perceptron.__nptypes),
        )

    @property
    def y(self) -> Union[list, ndarray]:
        """The y property."""
        return self._y

    @y.setter
    def y(self, value) -> None:
        self._y = verify_components_type(
            verify_type(value, (list, ndarray)),
            (int, float, *Perceptron.__nptypes),
        )

    @property
    def b(self) -> float:
        """The bias property."""
        return self._bias

    @b.setter
    def b(self, value) -> None:
        self._bias = verify_type(value, (int, float, *Perceptron.__nptypes))

    @property
    def w(self) -> List[float]:
        """The w property."""
        return self._weights

    @w.setter
    def w(self, value) -> None:
        self._w = verify_type(value, (int, float, *Perceptron.__nptypes))

    @property
    def learning_rate(self) -> float:
        """The learning_rate property."""
        return self._lr

    @learning_rate.setter
    def learning_rate(self, value) -> None:
        self._lr = float(verify_type(value, (int, float, *Perceptron.__nptypes)))

    @staticmethod
    def step(x) -> int:
        verify_type(x, (int, float, *Perceptron.__nptypes))

        return 0 if x < 0 else 1

    def fit(
        self, X: Union[list, ndarray], y: Union[list, ndarray], verbose=False
    ) -> List[float]:
        """
        Trains the model following the Perceptron Learning Rule.

        Returns:
            - list: The history loss.
        """

        # Verify type of X and y, and verbose option
        X = verify_components_type(
            verify_type(X, (list, ndarray)), (int, float, *Perceptron.__nptypes)
        )

        y = verify_components_type(
            verify_type(y, (list, ndarray)), (int, float, *Perceptron.__nptypes)
        )

        verify_type(verbose, bool)

        # Verifing data sizes compatibility
        if len(X) % self._n != 0:
            print("[!] Warning, X size and y size doesn't correspond.")

        if len(X) < self._n:
            return []

        # Training
        history: list = []

        for epoch in range(len(y)):
            # Narrowing down y for X
            eX = X[epoch * self._n : (epoch + 1) * self._n]
            ey = y[epoch]

            z = self.__call__(eX)

            # Updating parameters
            if z != ey:
                for i in range(len(self._weights)):
                    self._weights[i] += self._lr * (ey - z) * eX[i]
                self._bias += self._lr * (ey - z)

            # Calculate loss MSE for the current epoch
            epoch_loss = sum(
                (y[i] - self.__call__(X[i * self._n : (i + 1) * self._n]))
                ** 2
                for i in range(len(y))
            ) / len(y)
            history.append(epoch_loss)

            if verbose:
                print(
                    f"Epoch {epoch}:\n\tModel output: {z}\n\tExpected output: {ey}\n\tLoss: {epoch_loss}"
                )

        return history

    def predict(self, X: Union[list, ndarray]) -> int:
        """Returns a prediction given X as inputs."""
        verify_len(X, self._n)  # The input must be the same shape as n.
        verify_components_type(
            X, (int, float, *Perceptron.__nptypes)  # Input data must be numeric.
        )

        return self._activation(
            sum(x * w for x, w in zip(X, self._weights)) + self._bias
        )
