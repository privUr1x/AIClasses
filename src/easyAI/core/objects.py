from random import random, randint
from easyAI.core.activations import activation_map
from typing import (
    Iterator,
    Optional,
    Sequence,
    Sized,
    Union,
    List,
    Callable,
    Tuple,
    TypeVar,
    Generic,
)
from easyAI.clsstools.Verifiers import verify_type, verify_components_type
from easyAI.core.loss_func import loss_map


class History:
    """Class representing a loss history for training process."""

    pass


class Neuron(object):
    """Class representing an artificial neuron."""

    def __init__(self, activation: str = "relu") -> None:
        """Initialize a Neuron object."""
        self._identifier: int = randint(1, 10_000)
        self._bias: float = random()
        self._inputnodes: List[Union[Node, Neuron]] = []
        self._n: int = len(self._inputnodes)
        self._inputs: List[float] = [self._inputnodes[xi]._z for xi in range(self._n)]
        self._weights: List[float] = [random() for _ in range(self._n)]

        if activation not in activation_map:
            raise ValueError(
                f"Exepcted activatoin to be ({'/'.join([k for k in activation_map.keys()])})"
            )

        self._activation: Callable = activation_map[activation]

        self._z: float = self._activation(
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

    def __str__(self) -> str:
        return f"Neuron(): {self._id} at {id(self)}"

    def __repr__(self) -> str:
        return f"Neuron(): {self._id} at {id(self)}"


class Node(object):

    def __init__(self, value: Union[int, float] = 0) -> None:
        self._id: int = randint(1, 10_000)
        self._z = float(verify_type(value, (int, float)))

        self._layer: int  # Layer index
        self._i: int  # Index of the node within its layer
        self._inputnodes: None = None

    @property
    def value(self) -> float:
        """The value property."""
        return self._value

    @value.setter
    def value(self, value) -> None:
        self._value = value

    def __str__(self) -> str:
        return f"Node(): {self._id} at {id(self)}"

    def __repr__(self) -> str:
        return f"Node(): {self._id} at {id(self)}"


T = TypeVar("T", Node, Neuron)


class Layer(Generic[T]):
    """Class representing a layer of Neurons"""

    def __init__(self, n: int, activation="relu", name="layer") -> None:
        self._n: int = verify_type(n, int)

        if n < 1:
            raise ValueError("Expected at least 1 Neuron for a Layer()")

        self._structure: List[Union[Node, Neuron]] = [
            Neuron(activation) for _ in range(n)
        ]

        self._name: str = verify_type(name, str)

    def __call__(self) -> List[Union[Node, Neuron]]:
        return self._structure

    def __str__(self) -> str:
        return f"Layer({self._n}) at {id(self)}"

    def __repr__(self) -> str:
        return f"Layer({self._n})"

    def __len__(self) -> int:
        return self._n

    def __iter__(self) -> Iterator[Union[Node, Neuron]]:
        self._iter_index: int = 0
        return self

    def __next__(self) -> Union[Node, Neuron]:
        if self._iter_index < len(self._structure):
            result = self._structure[self._iter_index]
            self._iter_index += 1
            return result
        else:
            raise StopIteration

    def __getitem__(self, indx: int) -> Union[Node, Neuron]:
        verify_type(indx, int)

        if indx >= 0:
            if indx <= self._n:
                return self._structure[indx]
            raise IndexError(f"Index out of range: {indx}")
        else:
            if indx + 1 <= self._n:
                return self._structure[-(indx + 1)]
            raise IndexError(f"Index out of range: {indx}")

    def __setitem__(self, indx: int, val: Union[Node, Neuron]) -> None:
        verify_type(indx, int)
        verify_type(val, (Node, Neuron))

        if indx != 0 and isinstance(val, Node):
            raise TypeError("Node class only permited in the first layer.")

        if not (0 < self._n <= indx):
            raise IndexError("Index out of range.")
        
        self._structure[indx] = val

    def add_neuron(self):
        pass

    def remove_neuron(self, indx: int):
        pass


class Model(object):
    """Class representing an abstract class for arquitectures."""

    def __init__(self, *structure: Union[Layer, Sequence[Layer]], loss: str = "mse", learning_rate: Union[int, float] = 0.01) -> None:
        """Initialize a Model object attrs."""
        self._layers: List[Layer] = verify_components_type(
            verify_type(structure, tuple), Layer
        )
        self._lr: float = float(verify_type(learning_rate, (int, float)))
        self._n: int = len(self.input_layer)

        if not loss in loss_map:
            raise ValueError(f"Expected loss to be in between ({'/'.join([k for k in loss_map.keys()])})")

        self._loss: Callable = loss_map[loss]
        self._depth: int = len(self._layers)

        # Inner set up
        self.__set_inpt_layer()
        self.__set_connections()

    @property
    def layers(self) -> List[Layer[Union[Node, Neuron]]]:
        """The layers property."""
        return self._layers

    @property
    def input_layer(self) -> Layer[Node]:
        """The input_layer property."""
        return self._layers[0]

    @property
    def hidden_layers(self) -> List[Layer[Neuron]]:
        """The hiden_layers property."""
        return self._layers[1:-1]

    @property
    def output(self) -> Layer[Neuron]:
        """The output property."""
        return self._layers[-1]

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

    def __iter__(self) -> Iterator[Layer]:
        self._iter_index: int = 0
        return self

    def __next__(self) -> Layer:
        if self._iter_index < self._depth:
            result = self._layers[self._iter_index]
            self._iter_index += 1
            return result
        else:
            raise StopIteration

    def __set_inpt_layer(self) -> None:
        """Sets the first layer as class Node."""
        l: Layer = self._layers[0]

        l._structure = [Node() for _ in range(l._n)]

    def __set_connections(self) -> None:
        """Sets the connections between Neurons."""
        for i in range(1, self._n):  # [l1, l2, ..., ln]
            for n in self._layers[i]:
                n._inputnodes = [n for n in self._layers[i - 1]]
                n._inputs = [n._z for n in self._layers[i - 1]]

    def forward(self, inputs: List[Union[int, float]]) -> List[float]:
        """
        Propagates input through the network and returns the output of the model.

        Args:
        - inputs (List[Union[int, float]]): Input data to be propagated through the network.

        Returns:
        - List[float]: Output of the model after propagating through all layers.
        """
        verify_components_type(verify_type(inputs, list), (int, float))

        # Ensure the input size matches the size of the input layer
        if len(inputs) != len(self.input_layer):
            raise ValueError("Input size does not match the input layer size.")

        # Set the values of the input layer nodes to the input data
        for i, node in enumerate(self.input_layer):
            node._z = inputs[i]

        # Forward propagate through each layer
        for i in range(1, self._depth):
            for n in self._layers[i]:
                n._z = n._activation(
                    sum([x._z * w for x, w in zip(n._inputs, n._weights)]) + n._bias
                )

        # Return the output of the output layer
        return [neuron._z for neuron in self.output]


if __name__ == "__main__":
    print(activation_map)
