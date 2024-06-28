from typing import Any, Sized, Union, Tuple
from random import randint, random
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


def verify_type(obj: Any, t: Union[type, Tuple[type, ...]]) -> Any:
    """
    Verifies the type of and object (o).
    Excpetion:
        - TypeError: raises if the object is not the expected type.
    """
    if not isinstance(obj, t):
        raise TypeError(f"Expected {obj} to be {t} type.")

    return obj


def verify_len(obj: Sized, n: int) -> Any:
    """
    Verifies the length of an object (o).
    Expection:
        - IndexError: raises if the object is different in length as expected.
    """
    if hasattr(obj, "__len__"):
        if len(obj) != n:
            raise IndexError(f"Expected {obj} to be {n} in length.")

    return obj


def verify_iterable(obj: Union[list, ndarray]):
    """
    Verifies if an object is iterable.
    Expection:
        - TypeError: raises if the object is not iterable.
    """
    try:
        iter(obj)
    except TypeError:
        raise TypeError(f"The object of type {type(obj).__name__} is not iterable")


def verify_components_type(obj, etype: Union[type, Tuple[type, ...]]) -> Any:
    """
    Check if an object has the correct containing type.
    Object must have __len__ and __getitem__ methods defined.
    Args:
        obj (Any): The object to check.
        etype type: The expected components type(s).
    Returns:
        obj: the given object.
    """
    for i in range(len(obj)):
        if not isinstance(obj[i], etype):
            raise ValueError(f"Expected {obj} to have {etype} components.")

    return obj


class SimplePerceptron:
    "Class representing a Perceptron (Unitary Neural DL Model)"
    _version = 0.01

    """
    When making an instance of the class, 'X' and 'y' are required to train the model. The predictions are made by calling the instance.
    """

    def __init__(self, X: Union[list, ndarray], y: Union[list, ndarray]) -> None:
        self.__nptypes = (
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

        self._identifier: int = randint(1, 10_000)

        # Training data
        self._X: Union[list, ndarray] = verify_components_type(
            verify_type(X, (list, ndarray)), (int, float, *self.__nptypes)
        )
        self._y: Union[list, ndarray] = verify_components_type(
            verify_type(y, (list, ndarray)), (int, float, *self.__nptypes)
        )

        # Model params
        self._bias: Union[int, float] = random()
        self._weights: list = [random() for _ in self._X]
        self._z: Union[int, float] = (
            sum([x * w for x, w in zip(self._X, self._weights)]) + self._bias
        )

    def __call__(self, X: Union[list, ndarray]) -> int:
        """Returns a prediction given X as inputs."""
        verify_len(
            X, len(self._X)
        )  # The input must be the same shape as the training inputs.
        verify_components_type(X, (int, float))  # Input data must be numeric.

        return SimplePerceptron.step(
            sum(
                x * w for x, w in zip(verify_type(X, (list, ndarray)), self._weights)
            )  # Equivalent to self._z but with a given X.
            + self._bias
        )

    @property
    def id(self):
        """The id property."""
        return self._identifier

    @property
    def X(self):
        """The X property."""
        return self._X

    @X.setter
    def X(self, value):
        self._X = verify_components_type(
            verify_type(value, (list, ndarray)), (int, float, *self.__nptypes)
        )

    @property
    def y(self):
        """The y property."""
        return self._y

    @y.setter
    def y(self, value):
        self._y = verify_components_type(
            verify_type(value, (list, ndarray)), (int, float, *self.__nptypes)
        )

    @property
    def b(self):
        """The bias property."""
        return self._bias

    @b.setter
    def b(self, value):
        self._bias = verify_type(value, (int, float, *self.__nptypes))

    @property
    def w(self):
        """The w property."""
        return self._weights

    @w.setter
    def w(self, value):
        self._w = verify_type(value, (int, float, *self.__nptypes))

    @staticmethod
    def step(x: Union[int, float]) -> int:
        verify_type(x, (int, float))
        if x >= 0:
            return 1
        else:
            return 0

    def train() -> Tuple[list, list]:
        """
        Trains the model.
        """
        pass
