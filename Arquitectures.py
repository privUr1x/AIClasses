from random import randint, random
from typing import Union, Any, Tuple, Sized, List
from classtools import Verifiers
from numpy import (
    array,
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


class Perceptron:
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

    def __init__(
        self, X: Union[list, ndarray], y: Union[list, ndarray], entries: int
    ) -> None:
        """
        Builds an instance give X training data, y training data and entries.

        Args:
            - X (list/ndarray): The X training data.
            - y (list/ndarray): The y training data.
            - entries (int): The number of inputs of the model.
        """
        self._identifier: int = randint(1, 10_000)

        # Training data
        self._X: Union[list, ndarray] = verify_components_type(
            verify_type(X, (list, ndarray)), (int, float, *Perceptron.__nptypes)
        )
        self._y: Union[list, ndarray] = verify_components_type(
            verify_type(y, (list, ndarray)), (int, float, *Perceptron.__nptypes)
        )

        if isinstance(entries, float):
            if int(entries) == entries:
                entries = int(entries)

        # Model params
        self._n = verify_type(entries, int)
        self._bias: float = random()
        self._weights: List[float] = [random() for _ in range(self._n)]
        self._lr: float = 0.1

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
    def step(x: Union[int, float]) -> int:
        verify_type(x, (int, float))
        return 1 if x >= 0 else 0

    def train(self, verbose=False) -> List[float]:
        """
        Trains the model following the Perceptron Learning Rule.

        Returns:
            - list: The history loss.
        """
        verify_type(verbose, bool)

        # Verifing data sizes compatibility
        if len(self._X) % self._n != 0:
            print("[!] Warning, X size and y size doesn't correspond.")

        if len(self._X) < self._n:
            return []

        history: list = []

        for epoch in range(len(self._y)):
            # Narrowing down y for X
            eX = self._X[epoch * self._n : (epoch + 1) * self._n]
            ey = self._y[epoch]

            z = self.__call__(eX)

            # Updating parameters
            if z != ey:
                for i in range(len(self._weights)):
                    self._weights[i] += self._lr * (ey - z) * eX[i]
                self._bias += self._lr * (ey - z)

            # Calculate loss MSE for the current epoch
            epoch_loss = sum(
                (self._y[i] - self.__call__(self._X[i * self._n : (i + 1) * self._n]))
                ** 2
                for i in range(len(self._y))
            ) / len(self._y)
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

        return Perceptron.step(
            sum(x * w for x, w in zip(X, self._weights)) + self._bias
        )


class Neuron:
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
        self._bias: float = random()
        self._inputs: list = []
        self._weights: list = [random() for _ in range(len(self._inputs))]
        self._activation: function = self.__step
        self._z = self._activation(
            sum(x * w for x, w in zip(self._inputs, self._weights)) + self._bias
        )

        self._layer: int
        self._i: int

    def __step(self, x: Union[int, float]) -> int:
        verify_type(x, (int, float, *Neuron.__nptypes))
        return 1 if x >= 0 else 0

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
        return self._inputs

    @inputs.setter
    def inputs(self, value: list) -> None:
        self._inputs = verify_type(value, list)

    @property
    def w(self) -> list:
        """The w property."""
        return self._w

    @w.setter
    def w(self, value) -> None:
        self._w = verify_type(value, (list, ndarray))

    @property
    def activation(self) -> function:
        """The activation property."""
        return self._activation

    @activation.setter
    def activation(self, value: function) -> None:
        if not callable(value):
            raise TypeError("Expected value to be callable.")

        self._activation = value
