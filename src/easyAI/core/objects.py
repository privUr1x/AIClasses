from abc import ABC
from random import random, randint
from typing import Any, Iterator, Optional, Union, List, Callable, TypeVar, Generic
from easyAI.core.activations import activation_map
from easyAI.utils.verifiers import verify_len, verify_type, verify_components_type
from easyAI.utils.instances import search_instnce_name
from easyAI.core.loss import loss_map
from easyAI.core.optimizers import Optimizer, optimizers_map
from functools import singledispatchmethod


class Neuron(object):
    """Class representing an artificial neuron."""

    def __init__(self, activation: Callable) -> None:
        """
        Initialize a Neuron object.

        Args:
            activation (str): The activation function for the neuron.
        """
        self._id: int = randint(10_000, 99_999)

        self._inputnodes: List[Union[Node, Neuron]] = []
        self._bias: float = random()
        self._weights: List[float] = []
        self._activation: Callable = activation
        self._lyr_i: int
        self._ne_i: int

    @property
    def output(self):
        """Calculate and return the output value of the neuron."""
        return self.activation(self.z)

    @property
    def z(self):
        """The z property."""
        return sum(x * w for x, w in zip(self.inputs, self.w)) + self.b

    @property
    def n(self) -> int:
        """Return the number of input nodes connected to the neuron."""
        return len(self.inputnodes)

    @property
    def b(self) -> float:
        """Return the bias of the neuron."""
        return self._bias

    @b.setter
    def b(self, value: Union[int, float]) -> None:
        """Set the bias of the neuron."""
        self._bias = float(verify_type(value, (int, float)))

    @property
    def inputs(self) -> List[float]:
        """Return the values of the input nodes connected to the neuron."""
        return [node.output for node in self.inputnodes]

    @property
    def inputnodes(self) -> List[Union["Node", "Neuron"]]:
        """Return the input nodes connected to the neuron."""
        return self._inputnodes

    @inputnodes.setter
    def inputnodes(self, value: List[Union["Node", "Neuron"]]) -> None:
        """Set the input nodes connected to the neuron."""
        self._inputnodes = verify_components_type(
            verify_type(value, list), (Node, Neuron)
        )

        # Update weights
        self._weights = [random() for _ in self._inputnodes]

    @property
    def w(self) -> List[float]:
        """Return the weights of the input connections to the neuron."""
        return self._weights

    @w.setter
    def w(self, value: List[Union[int, float]]) -> None:
        """Set the weights of the input connections to the neuron."""
        self._weights = verify_components_type(verify_type(value, list), (int, float))

    @property
    def activation(self) -> Callable:
        """Return the activation function of the neuron."""
        return self._activation

    @activation.setter
    def activation(self, value: Callable) -> None:
        """Set the activation function of the neuron."""
        if not callable(value):
            raise TypeError(f"Expected {value} to be callable.")
        self._activation = value

    def __str__(self) -> str:
        """Return a string representation of the neuron."""
        return f"Neuron(): {self._id}"

    def __repr__(self) -> str:
        """Return a detailed string representation of the neuron."""
        return f"Neuron(): {self._id}"


class Node(object):
    """Class representing an input node."""

    def __init__(self, value: Union[int, float] = 0) -> None:
        """
        Initialize a Node object.

        Args:
            value (Union[int, float]): The initial value of the node.
        """
        self._id: int = randint(10_000, 99_999)
        self._output: float = float(verify_type(value, (int, float)))

        self._lyr_i: int = 0
        self._ne_i: int

    @property
    def output(self) -> float:
        """Return the value of the node."""
        return self._output

    @output.setter
    def output(self, value: Union[int, float]) -> None:
        """Set the value of the node."""
        self._output = float(verify_type(value, (int, float)))

    def __str__(self) -> str:
        """Return a string representation of the node."""
        return f"Node(): {self._id}"

    def __repr__(self) -> str:
        """Return a detailed string representation of the node."""
        return f"Node(): {self._id}"


T = TypeVar("T", Node, Neuron)


class Layer(Generic[T]):
    """Class representing an abstract layer of Neurons or Nodes."""

    def __init__(self, n: int, activation: str, *, name="layer") -> None:
        """
        Initialize a Layer object.

        Args:
            n (int): Number of neurons or nodes in the layer.
            activation (str): Activation function for the neurons.
            name (str): Name of the layer.
        """
        if activation not in activation_map:
            raise ValueError(
                f"Expected activation to be one of ({'/'.join([k for k in activation_map.keys()])})"
            )

        self._activation: Callable = activation_map[activation]
        self._n: int = verify_type(n, int)
        self._name: str = verify_type(name, str)

        if n < 1:
            raise ValueError("Expected at least 1 Neuron or Node for a Layer")

        self._structure: List[Union[Node, Neuron]] = [
            Neuron(self._activation) for _ in range(n)
        ]

        self.__set_indexes()

    @property
    def n(self):
        """The n property."""
        return self._n

    def __call__(self) -> List[Union[Node, Neuron]]:
        """Return the structure of the layer."""
        return self._structure

    def __str__(self) -> str:
        """Return a string representation of the layer."""
        return f"Layer({self._n}):\n\t{self._structure}\n"

    def __repr__(self) -> str:
        """Return a detailed string representation of the layer."""
        return f"Layer({self._n}):\n\t{self._structure}\n"

    def __len__(self) -> int:
        """Return the number of neurons or nodes in the layer."""
        return self._n

    def __iter__(self) -> Iterator[Union[Node, Neuron]]:
        """Iterate over the neurons or nodes in the layer."""
        self._iter_index: int = 0
        return self

    def __next__(self) -> Union[Node, Neuron]:
        """Return the next neuron or node in the iteration."""
        if self._iter_index < len(self._structure):
            result = self._structure[self._iter_index]
            self._iter_index += 1
            return result
        else:
            raise StopIteration

    def __getitem__(self, indx: int) -> Union[Node, Neuron]:
        """Return the neuron or node at the specified index."""
        verify_type(indx, int)
        if indx >= 0:
            if indx < self._n:
                return self._structure[indx]
            raise IndexError(f"Index out of range: {indx}")
        else:
            if abs(indx) <= self._n:
                return self._structure[indx]
            raise IndexError(f"Index out of range: {indx}")

    def __setitem__(self, indx: int, val: Union[Node, Neuron]) -> None:
        """Set the neuron or node at the specified index."""
        verify_type(indx, int)
        verify_type(val, (Node, Neuron))

        assert indx > 0 and isinstance(
            val, Node
        ), "Node class only permitted in the first layer."

        if 0 <= indx < self._n:
            self._structure[indx] = val

        elif 0 > indx and len(self._structure) > indx:
            self._structure[indx] = val

        raise IndexError("Index out of range.")

    def __hash__(self) -> int:
        return hash(str(self._structure) + str(self._activation))

    def __set_indexes(self) -> None:
        """Set the indexes for the neurons or nodes within the layer."""
        for i, n in enumerate(self._structure):
            n._ne_i = i

    def add_neuron(self, indx: Optional[int] = None) -> None:
        """Add a neuron to the layer."""
        if indx is not None:
            verify_type(indx, int)
            self._structure.insert(indx, Neuron(self._activation))

        else:
            self._structure.append(Neuron(self._activation))

    def remove_neuron(self, indx: int) -> None:
        """Remove a neuron from the layer."""
        verify_type(indx, int)
        self._structure.pop(indx)


class Model(ABC):
    """Class representing an abstract class for neural network architectures."""

    def __init__(
        self,
        structure: List[Layer[Union[Node, Neuron]]],
    ) -> None:
        """
        Initialize a Model object.

        Args:
            structure (List[Layer]): The structure of the neural network.
            *
            loss (str): The loss function for the model.
            optimizer (str): The optimizer algoritmh/method to be used for training.
            learning_rate (Union[int, float]): The learning rate for the model.
            epochs (int): The epochs value for training.
        """
        self._name: str = "Abstract Model."
        self._layers: List[Layer[Union[Node, Neuron]]] = verify_components_type(
            verify_type(structure, list), Layer
        )

        self.__set_input_layer()
        self.__set_connections()

    @property
    def layers(self) -> List[Layer[Union[Node, Neuron]]]:
        """Return the layers of the model."""
        return self._layers

    @property
    def input_layer(self) -> Layer[Node]:
        """Return the input layer of the model."""
        return self._layers[0]

    @property
    def hidden_layers(self) -> List[Layer[Neuron]]:
        """Return the hidden layers of the model."""
        return self._layers[1:-1]

    @property
    def output(self) -> Layer[Neuron]:
        """Return the output layer of the model."""
        return self._layers[-1]

    @property
    def learning_rate(self) -> float:
        """Return the learning rate of the model."""
        return self._lr

    @learning_rate.setter
    def learning_rate(self, value: Union[int, float]) -> None:
        """Set the learning rate of the model."""
        self._lr = float(verify_type(value, (int, float)))

    @property
    def loss(self) -> Callable:
        """Return the loss function of the model."""
        return self._loss

    @loss.setter
    def loss(self, value: str) -> None:
        """Set the loss function of the model."""

        assert (
            value in loss_map
        ), f"Expected loss to be one of ({'/'.join([k for k in loss_map.keys()])})"

        self._loss = loss_map[value]

    @loss.deleter
    def loss(self) -> None:
        del self._loss

    @property
    def depth(self) -> int:
        """Return the depth of the model (number of layers)."""
        return len(self._layers)

    @property
    def n(self) -> int:
        """The n property."""
        return len(self.input_layer)

    def __repr__(self) -> str:
        return f"{self._name}({len(self._layers)}) object named {search_instnce_name(self)}"

    def __str__(self) -> str:
        return f"{self._name}({len(self._layers)}) object named {search_instnce_name(self)}"

    @singledispatchmethod
    def __getitem__(self, _: Any, /) -> None:
        """Docstr"""

        raise TypeError("Not supported type.")

    @__getitem__.register(tuple)
    def _(self, indx: tuple, /):
        verify_len(indx, 2)
        return self._layers[indx[0]][indx[1]]

    @__getitem__.register(float)
    def _(self, indx: float, /):
        i: list = str(indx).split(".")
        verify_len(i, 2)
        return self._layers[i[0]][i[1]]

    @__getitem__.register(int)
    def _(self, indx, /):
        if len(self._layers) > indx >= 0 or len(self._layers) <= indx < 0:
            return self._layers[indx]

        raise IndexError("Index out of range.")

    @__getitem__.register(list)
    def _(self, indx: list[int], /):
        l: list[Layer] = []

        for i in indx:
            verify_type(i, int)

            if i >= 0:
                if i > len(self._layers) - 1:
                    raise IndexError(f"Index out of range {i}.")
            else:
                if -i > len(self._layers):
                    raise IndexError(f"Index out of range: {i}.")

            l.append(self._layers[i])

        return l

    def __eq__(self, value: object, /) -> bool:
        return self.__dict__ == value.__dict__

    def __ne__(self, value: object, /) -> bool:
        return not self.__eq__(value)

    def __hash__(self) -> int:
        return hash(tuple([l.__hash__() for l in self._layers]))

    def __call__(self, input: List[Union[int, float]]) -> List[float]:
        return self.forward(input)

    def __set_input_layer(self) -> None:
        """Set the input layer of the model."""
        self.input_layer._structure = (
            [Node() for _ in self.input_layer._structure]
            if self.depth > 1
            else self.input_layer._structure
        )

    def __set_connections(self) -> None:
        """Set the connections between neurons in the layers."""
        for i in range(1, self.depth):
            # Not reaching Nodes
            for n in self._layers[i]:
                n.inputnodes = [node for node in self._layers[i - 1]]
                n._lyr_i = i

    def add(self, layer: Layer, indx: int = -1) -> None:
        """Add a layer to the model."""
        verify_type(layer, Layer)
        verify_type(indx, int)

        self._layers.insert(indx, layer)

    def remove(self, layer: Layer, indx: int = -1) -> None:
        """Remove a layer from the model."""
        verify_type(layer, Layer)
        verify_type(indx, int)

        self._layers.pop(indx)

    def forward(self, input: List[Union[int, float]]) -> List[float]:
        """
        Propagate input through the network and return the output of the model.

        Args:
            inputs (List[Union[int, float]]): Input data to be propagated through the network.

        Returns:
            List[float]: Output of the model after propagating through all layers.
        """
        verify_components_type(verify_type(input, list), (int, float))

        assert (
            len(input) == self.n
        ), "Input size does not match the input layer expected size."

        # Instanciate node value with inputs
        for i, node in enumerate(self.input_layer):
            node._output = input[i]

        return [neuron.output for neuron in self.output]

    def fit(
        self,
        X,
        Y,
        *,
        loss: str,
        epochs: int,
        optimizer: str,
        learning_rate: Union[int, float],
    ):
        verify_components_type(verify_type(X, list), (int, float))
        verify_components_type(verify_type(Y, list), (int, float))
        verify_type(int(epochs), int)

        assert self.n < len(
            X
        ), "Training data expected to be larger than input nodes in size."

        assert epochs >= 1, "Expected at least 1 epoch."

        assert (
            loss in loss_map
        ), f"Expected loss to be one of ({'/'.join([k for k in loss_map.keys()])})"

        assert (
            optimizer in optimizers_map
        ), f"Expected optimizer to be one of ({'/'.join([k for k in optimizers_map.keys()])})"

        self._lr: Union[int, float] = verify_type(learning_rate, (int, float))
        self._loss: Callable = loss_map[loss]
        self._optimizer: Optimizer = optimizers_map[optimizer](
            learning_rate=self.learning_rate, epochs=epochs, loss=self.loss
        )

        return History(*self._optimizer.fit(X, Y, self.n))

    def evaluate(self) -> None:
        """Evaluate the model's performance."""
        raise NotImplemented

    def save(self) -> None:
        """Save the model to a file."""
        raise NotImplemented

    def load(self) -> None:
        """Load the model from a file."""
        raise NotImplemented

    def summary(self) -> None:
        """Print a summary of the model."""
        raise NotImplemented


class History:
    """Class representing a history object for tracking training progress."""

    def __init__(self):
        self.history = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}

    def update(
        self, loss: float, accuracy: float, val_loss: float, val_accuracy: float
    ):
        self.history["loss"].append(loss)
        self.history["accuracy"].append(accuracy)
        self.history["val_loss"].append(val_loss)
        self.history["val_accuracy"].append(val_accuracy)
