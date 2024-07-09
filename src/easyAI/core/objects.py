from random import random, randint
from activations import activation_map
from typing import Union, List, Callable
from classtools import Verifiers 
from loss_func import loss_map

verify_type = Verifiers.verify_type
verify_components_type = Verifiers.verify_components_type

class History:
    """Class representing a loss history for training process."""

    pass


class Neuron(object):
    """Class representing an artificial neuron."""

    def __init__(self) -> None:
        """Initialize a Neuron object."""
        self._identifier: int = randint(1, 10_000)
        self._bias: float = random()
        self._inputnodes: List[Neuron] = []
        self._n: int = len(self._inputnodes)
        self._inputs: List[float] = [self._inputnodes[xi]._z for xi in range(self._n)]
        self._weights: List[float] = [random() for _ in range(self._n)]
        self._activation: Callable = activation_map["relu"]
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

    def __init__(self, n: int, activation="relu", name="layer") -> None:
        self._n: int = verify_type(n, int)
        self._structure: List[Neuron] = [Neuron() for _ in range(n)]

        if activation not in activation_map:
            raise ValueError(
                f"Exepcted activatoin to be ({'/'.join([k for k in activation_map.keys()])})"
            )

        self._activation: Callable = activation_map[verify_type(activation, str)]

        self._name: str = name

    def __str__(self) -> str:
        return f"Layer({self._name}, {self._n} Neuron())"


class Model(object):
    """Class representing an abstract class for arquitectures."""

    def __init__(self, structure: List[Layer]) -> None:
        """Initialize a Model object attrs."""
        self._layers: List[Layer] = verify_components_type(verify_type(structure, list), Neuron)
        self._lr: float = 0.01

        self._loss: Callable = loss_map["mse"]

    @property
    def layers(self) -> List[Layer]:
        """The layers property."""
        return self._layers

    @property
    def input_layer(self) -> Layer:
        """The input_layer property."""
        return self._layers[:1]

    @property
    def hidden_layers(self) -> List[Layer]:
        """The hiden_layers property."""
        return self._layers[1:-1]

    @property
    def output(self) -> Layer:
        """The output property."""
        return self._layers[-1:]

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
                f"Expected loss to be ({'/'.join([f'{k}' for k in loss_map.keys()])})"
            )

        self._loss = loss_map[value]

    def fit(X: List[Union[int, float]], y: List[Union[int, float]]) -> History:
        """
        Trains the model given X and y data.
        """
        pass
