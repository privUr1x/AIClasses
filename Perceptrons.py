from typing import Any, List, Sized, Union, Tuple
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
            verify_type(X, (list, ndarray)), (int, float, *SimplePerceptron.__nptypes)
        )
        self._y: Union[list, ndarray] = verify_components_type(
            verify_type(y, (list, ndarray)), (int, float, *SimplePerceptron.__nptypes)
        )

        if isinstance(entries, float):
            if int(entries) == entries:
                entries = int(entries)

        # Model params
        self._n = verify_type(entries, int)
        self._bias: float = random()
        self._weights: List[float] = [random() for _ in range(self._n)]
        self._lr: float = 1

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
            (int, float, *SimplePerceptron.__nptypes),
        )

    @property
    def y(self) -> Union[list, ndarray]:
        """The y property."""
        return self._y

    @y.setter
    def y(self, value) -> None:
        self._y = verify_components_type(
            verify_type(value, (list, ndarray)),
            (int, float, *SimplePerceptron.__nptypes),
        )

    @property
    def b(self) -> float:
        """The bias property."""
        return self._bias

    @b.setter
    def b(self, value) -> None:
        self._bias = verify_type(value, (int, float, *SimplePerceptron.__nptypes))

    @property
    def w(self) -> List[float]:
        """The w property."""
        return self._weights

    @w.setter
    def w(self, value) -> None:
        self._w = verify_type(value, (int, float, *SimplePerceptron.__nptypes))

    @property
    def learning_rate(self) -> float:
        """The learning_rate property."""
        return self._lr

    @learning_rate.setter
    def learning_rate(self, value) -> None:
        self._lr = float(verify_type(value, (int, float, *SimplePerceptron.__nptypes)))

    @staticmethod
    def step(x: Union[int, float]) -> int:
        verify_type(x, (int, float))
        if x >= 0:
            return 1
        else:
            return 0

    def train(self, verbose=False) -> List[float]:
        """
        Trains the model following the Perceptron Learning Rule.

        Returns:
            - list: The history loss.
        """
        verify_type(verbose, bool)

        # Verifing data sizes compatibility
        if len(self._X) / len(self._y) != self._n:
            print("[!] Warning, X size and y size doesn't correspond.")

        if len(self._X) < self._n:
            return []

        # Narrowing down y for X
        X = self._X
        y = self._y[: int(len(self._X) / self._n)]

        for i in range(len(y)):
            # ith-epoch data
            eX = X[i * self._n : (i + 1) * self._n]
            ey = y[i]

            z: Union[int, float] = SimplePerceptron.step(
                sum([x * w for x, w in zip(eX, self._weights)]) + self._bias
            )

            if verbose:
                print(f"""
Epoch {i}:
  Model Output: {z}
  Expected Output: {ey}""")

            # Updating params 
            if z != ey:
                for j in range(len(self._weights)):
                    self._weights[j] += self._lr * (ey - z) * eX[j]
                    self._bias += self._lr * (ey - z)

        return []

    def predict(self) -> int:
        pass
