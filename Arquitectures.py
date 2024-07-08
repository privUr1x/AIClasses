from random import randint, random
from activations import relu
from loss_func import loss_map
from typing import Optional, Union, List, Callable, Tuple
from classtools import Verifiers

verify_type = Verifiers.verify_type
verify_len = Verifiers.verify_len
verify_iterable = Verifiers.verify_iterable
verify_components_type = Verifiers.verify_components_type


class History:
    """Class representing a loss history for training process."""

    pass


class Node(object):
    """Class representing a simple node."""

    def __init__(self, value: Union[int, float]) -> None:
        """
        Initialize an instance.
        Args:
            - value (int/float): Value expected to be held in the Node.
        """
        self._z = float(verify_type(value, (int, float)))

    @property
    def value(self) -> float:
        """The value property."""
        return self._z

    @value.setter
    def value(self, value: Union[int, float]) -> None:
        self._z = float(verify_type(value, (int, float)))


class Neuron(object):
    """Class representing an artificial neuron."""

    def __init__(self) -> None:
        """Initialize a Neuron object."""
        self._identifier: int = randint(1, 10_000)
        self._bias: float = random()
        self._inputnodes: List[Union[Node, Neuron]] = []
        self._n: int = len(self._inputnodes)
        self._inputs: List[float] = [self._inputnodes[xi]._z for xi in range(self._n)]
        self._weights: List[float] = [random() for _ in range(self._n)]
        self._activation: Callable = relu
        self._z: float = self.activation(
            sum([x * w for x, w in zip(self._inputs, self._weights)]) + self._bias
        )

        self._layer: int  # Layer index of the neuron
        self._i: int  # Index of the neuron within its layer

    @property
    def b(self) -> float:
        """Getter for the bias property."""
        return self._bias

    @b.setter
    def b(self, value: Union[int, float]) -> None:
        """Setter for the bias property."""
        self._bias = float(verify_type(value, (int, float)))

    @property
    def inputs(self) -> List["Neuron"]:
        """Getter for the inputs property."""
        return self._inputnodes

    @inputs.setter
    def inputs(self, value: List["Neuron"]) -> None:
        """Setter for the inputs property."""
        self._inputnodes = verify_components_type(verify_type(value, list), Neuron)

    @property
    def w(self) -> List[float]:
        """Getter for the weights property."""
        return self._weights

    @w.setter
    def w(self, value: List[Union[int, float]]) -> None:
        """Setter for the weights property."""
        self._weights = list(  # Finally, assign the value
            verify_components_type(  # Thenn verify the components type
                verify_type(value, list),
                (int, float),  # First verify the type
            )
        )

    @property
    def activation(self) -> Callable:
        """Getter for the activation function property."""
        return self._activation

    @activation.setter
    def activation(self, value: Callable) -> None:
        """Setter for the activation function property."""
        if not callable(value):
            raise TypeError(f"Expected {value} to be callable.")

        self._activation = value


class Layer(object):
    """Class representing a layer of Neurons"""

    def __init__(self, *Neurons: Union[Neuron, List[Neuron], Tuple[Neuron, ...]]) -> None:
        pass
        


class Model(object):
    """Class representing an abstract class for arquitectures."""

    def __init__(self, learning_rate: float = 0.01, loss: str = "mse") -> None:
        """Initialize a Model object attrs."""
        self._layers: List[List[Union[Node, Neuron]]] = []
        self._hiddenlyrs: List[List[Neuron]] = self._layers[1:-1]
        self._output: List[Node] = self._layers[:-1]
        self._lr: float = float(verify_type(learning_rate, (int, float)))

        if loss not in loss_map.keys():
            raise ValueError(
                f"Expected loss to be ({'/'.join([k for k in loss_map.keys()])})"
            )

        self._loss: Callable = loss_map[loss]

    @property
    def layers(self) -> List[List[Union[Node, Neuron]]]:
        """The layers property."""
        return self._layers

    @property
    def hidden_layers(self) -> List[List[Neuron]]:
        """The hiden_layers property."""
        return self._hiddenlyrs

    @property
    def output(self) -> List[Node]:
        """The output property."""
        return self._output

    @property
    def learning_rate(self) -> float:
        """The learning_rate property."""
        return self._lr

    @learning_rate.setter
    def learning_rate(self, value: Union[int, float]) -> None:
        self._learning_rate = float(verify_type(value, (int, float)))

    @property
    def loss(self) -> Callable:
        """The loss property."""
        return self._loss

    @loss.setter
    def loss(self, value: str) -> None:
        if value not in loss_map.keys():
            raise ValueError(
                f"Expected loss to be ({'/'.join([f'{k}' for k in self._loss_map.keys()])})"
            )

        self._loss = loss_map[value]

    def fit(X: List[Union[int, float]], y: List[Union[int, float]]) -> History:
        """
        Trains the model given X and y data.
        """
        pass


class Perceptron(Neuron):
    "Class representing a Perceptron (Unitary Layer Neural DL Model)"

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

    def __call__(self, X: List[Union[int, float]]) -> int:
        return self.predict(X)

    @property
    def id(self) -> int:
        """The id property."""
        return self._identifier

    @property
    def b(self) -> float:
        """The bias property."""
        return self._bias

    @b.setter
    def b(self, value) -> None:
        self._bias = verify_type(value, (int, float))

    @property
    def w(self) -> List[float]:
        """The w property."""
        return self._weights

    @w.setter
    def w(self, value) -> None:
        self._w = verify_type(value, (int, float))

    @property
    def learning_rate(self) -> float:
        """The learning_rate property."""
        return self._lr

    @learning_rate.setter
    def learning_rate(self, value) -> None:
        self._lr = float(verify_type(value, (int, float)))

    @staticmethod
    def step(x) -> int:
        verify_type(x, (int, float))

        return 0 if x < 0 else 1

    def fit(
        self,
        X: List[Union[int, float]],
        y: List[Union[int, float]],
        verbose: Optional[bool] = False,
    ) -> List[Union[int, float]]:
        """
        Trains the model following the Perceptron Learning Rule.

        Returns:
            - list: The history loss.
        """

        # Verify type of X and y, and verbose option
        X = verify_components_type(verify_type(X, (list)), (int, float))

        y = verify_components_type(verify_type(y, list), (int, float))

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
                (y[i] - self.__call__(X[i * self._n : (i + 1) * self._n])) ** 2
                for i in range(len(y))
            ) / len(y)
            history.append(epoch_loss)

            if verbose:
                print(
                    f"Epoch {epoch}:\n\tModel output: {z}\n\tExpected output: {ey}\n\tLoss: {epoch_loss}"
                )

        return history

    def predict(self, X: List[Union[int, float]]) -> int:
        """Returns a prediction given X as inputs."""
        verify_len(X, self._n)  # The input must be the same shape as n.
        verify_components_type(X, (int, float))  # Input data must be numeric.

        return self._activation(
            sum(x * w for x, w in zip(X, self._weights)) + self._bias
        )
