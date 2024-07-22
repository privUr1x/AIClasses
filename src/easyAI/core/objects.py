from abc import ABC
from random import random, randint
from typing import Any, Iterator, Optional, Union, List, Callable, TypeVar, Generic
from easyAI.core.activations import activation_map
from easyAI.utils.verifiers import verify_len, verify_type, verify_components_type
from easyAI.utils.instances import search_instnce_name
from easyAI.core.loss_func import loss_map
from easyAI.core.optimizers import Optimizer, optimizers_map
from functools import singledispatchmethod

class Hstory:
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


class Neuron(object):
    """Class representing an artificial neuron."""

    def __init__(self, activation: str = "relu") -> None:
        """
        Initialize a Neuron object.

        Args:
            activation (str): The activation function for the neuron.
        """
        self._id: int = randint(10_000, 99_999)

        if activation not in activation_map:
            raise ValueError(
                f"Expected activation to be one of ({'/'.join([k for k in activation_map.keys()])})"
            )
        self._activation: Callable = activation_map[activation]

        self._inputnodes: List[Union[Node, Neuron]] = []
        self._bias: float = random()
        self._weights: List[float] = []
        self._z: float = self._activation(
            sum(x * w for x, w in zip(self.inputs, self.w)) + self.b
        )

        self._lyr_i: int
        self._ne_i: int

    @property
    def z(self):
        """Calculate and return the output value of the neuron."""
        return self._z

    @property
    def n(self) -> int:
        """Return the number of input nodes connected to the neuron."""
        return len(self.inputnodes)

    @n.setter
    def n(self, value: int) -> None:
        """Set the number of input nodes connected to the neuron."""
        self._n = verify_type(value, int)

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
        return [node.z for node in self.inputnodes]

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
        self._z: float = float(verify_type(value, (int, float)))

        self._lyr_i: int = 0
        self._ne_i: int

    @property
    def z(self) -> float:
        """Return the value of the node."""
        return self._z

    @z.setter
    def z(self, value: Union[int, float]) -> None:
        """Set the value of the node."""
        self._z = float(verify_type(value, (int, float)))

    def __str__(self) -> str:
        """Return a string representation of the node."""
        return f"Node(): {self._id}"

    def __repr__(self) -> str:
        """Return a detailed string representation of the node."""
        return f"Node(): {self._id}"


T = TypeVar("T", Node, Neuron)


class Layer(Generic[T]):
    """Class representing a layer of Neurons or Nodes."""

    def __init__(self, n: int, activation="relu", name="layer") -> None:
        """
        Initialize a Layer object.

        Args:
            n (int): Number of neurons or nodes in the layer.
            activation (str): Activation function for the neurons.
            name (str): Name of the layer.
        """
        self._n: int = verify_type(n, int)
        self._name: str = verify_type(name, str)
        self._activation: str = activation

        if n < 1:
            raise ValueError("Expected at least 1 Neuron or Node for a Layer")

        self._structure: List[Union[Node, Neuron]] = [
            Neuron(activation) for _ in range(n)
        ]
        self.__set_indexes()

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


class Dense(Layer):
    """Class representing a fully connected layer."""

    def __init__(self, n: int, activation="relu", name="layer") -> None:
        super().__init__(n, activation, name)


class Conv(Layer):
    """Class representing a convolutional network layer."""

    def __init__(self, n: int, activation="relu", name="layer") -> None:
        raise NotImplemented
        super().__init__(n, activation, name)


class Rec(Layer):
    """Class representing a recurrent network layer."""

    def __init__(self, n: int, activation="relu", name="layer") -> None:
        raise NotImplemented
        super().__init__(n, activation, name)


class Model(ABC):
    """Class representing an abstract class for neural network architectures."""

    def __init__(
        self,
        structure: List[Layer],
        *,
        loss: str = "mse",
        optimizer: str = "adam",
        learning_rate: Union[int, float] = 0.01,
    ) -> None:
        """
        Initialize a Model object.

        Args:
            structure (List[Layer]): The structure of the neural network.
            loss (str): The loss function for the model.
            learning_rate (Union[int, float]): The learning rate for the model.
        """
        self._name: str = "Abstract Model."
        self._layers: List[Layer[Union[Node, Neuron]]] = verify_type(structure, list)
        self._lr: float = float(verify_type(learning_rate, (int, float)))

        assert (
            loss in loss_map
        ), f"Expected loss to be one of ({'/'.join([k for k in loss_map.keys()])})"

        assert (
            optimizer in optimizers_map
        ), f"Expected optimizer to be one of ({'/'.join([k for k in optimizers_map.keys()])})"

        self._loss: Callable = loss_map[loss]
        self._optimizer: Optimizer = optimizers_map[optimizer]

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
            for n in self._layers[i]:
                n.inputnodes = [node for node in self._layers[i - 1]]
                n._lyr_i = i

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

    def forward(self, input: List[Union[int, float]]) -> List[float]:
        """
        Propagate input through the network and return the output of the model.

        Args:
            inputs (List[Union[int, float]]): Input data to be propagated through the network.

        Returns:
            List[float]: Output of the model after propagating through all layers.
        """
        verify_components_type(verify_type(input, list), (int, float))

        assert len(input) == len(
            self.input_layer._structure
        ), "Input size does not match the input layer expected size."

        # Instanciate node value with inputs
        for i, node in enumerate(self.input_layer):
            node._z = input[i]

        for i in range(1, self.depth):
            for n in self.layers[i]:
                n._z = n.activation(sum(x * w for x, w in zip(n.inputs, n.w)) + n.b)

        return [neuron._z for neuron in self.output]

    def evaluate(self) -> None:
        """Evaluate the model's performance."""
        pass

    def save(self) -> None:
        """Save the model to a file."""
        pass

    def load(self) -> None:
        """Load the model from a file."""
        pass

    def summary(self) -> None:
        """Print a summary of the model."""
        pass
