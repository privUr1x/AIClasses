from random import randint, random
from typing import Callable, Union, Any, Tuple, Sized, List
from ..classtools import Verifiers
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
        self._activation: Callable = self.__step
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
    def activation(self) -> Callable:
        """The activation property."""
        return self._activation

    @activation.setter
    def activation(self, value: Callable) -> None:
        if not callable(value):
            raise TypeError("Expected value to be callable.")

        self._activation = value
