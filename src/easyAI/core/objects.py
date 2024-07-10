from random import random, randint
from easyAI.core.activations import activation_map
from typing import Optional, Union, List, Callable
from easyAI.clsstools.Verifiers import verify_type, verify_components_type
from easyAI.core.loss_func import loss_map


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


class Node(object):

    def __init__(self, value: Union[int, float] = 0) -> None:
        self._id: int = randint(1, 10_000)
        self._z = float(verify_type(value, (int, float)))

    @property
    def value(self) -> float:
        """The value property."""
        return self._value

    @value.setter
    def value(self, value) -> None:
        self._value = value

    def __eq__(self, value: object, /) -> bool:
        return self.__dict__ == value.__dict__

    def __ne__(self, value: object, /) -> bool:
        return not self.__eq__(value)


class Layer(object):
    """Class representing a layer of Neurons"""

    def __init__(self, n: int, activation="relu", name="layer") -> None:
        self._n: int = verify_type(n, int)

        if n < 1:
            raise ValueError("Expected at least 1 Neuron for a Layer()")

        self._structure: List[Union[Node, Neuron]] = [Neuron() for _ in range(n)]

        if activation not in activation_map:
            raise ValueError(
                f"Exepcted activatoin to be ({'/'.join([k for k in activation_map.keys()])})"
            )

        self._activation: Callable = activation_map[verify_type(activation, str)]

        self._name: str = name

    def __str__(self) -> str:
        return f"Layer({self._n}) at {id(self)}"

    def __repr__(self) -> str:
        return f"Layer({self._n})"

    def __len__(self) -> int:
        return self._n

    def __iter__(self) -> "Layer":
        return self 

    def __next__(self) -> Union[Node, Neuron]:
        pass 

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

class Model(object):
    """Class representing an abstract class for arquitectures."""

    def __init__(self, structure: List[Layer]) -> None:
        """Initialize a Model object attrs."""
        self._layers: List[Layer] = verify_components_type(
            verify_type(structure, list), Layer
        )
        self._lr: float = 0.01
        self._n: int = len(self.input_layer)
        self._loss: Callable = loss_map["mse"]

        self.__set_inpt_layer()

    @property
    def layers(self) -> List[Union[Node, Layer]]:
        """The layers property."""
        return self._layers

    @property
    def input_layer(self) -> Layer:
        """The input_layer property."""
        return self._layers[0]

    @property
    def hidden_layers(self) -> List[Layer]:
        """The hiden_layers property."""
        return self._layers[1:-1]

    @property
    def output(self) -> Layer:
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

    def __set_inpt_layer(self) -> None:
        """Sets the first layer as class Node."""
        l: Layer = self._layers[0]

        l._structure = [Node() for _ in range(l._n)]
        print(self._layers)

    def __set_connections(self) -> None:
        """Sets the connections between Neurons."""
        for l in range(self._n - 1): # [l1, l2, ..., ln]
            for n in l: # [Node()/Neuron(), ....]
                print(n)
    
    def forward(self) -> float: 
        """Returns the output of the Neuron at the current state."""
        pass

    def fit(
        self,
        X: List[Union[int, float]],
        y: List[Union[int, float]],
        verbose: bool = False,
    ) -> Optional[History]:
        """
        Trains the model given X and y data.
        """
        verify_components_type(verify_type(X, list), (int, float))
        verify_components_type(verify_type(y, list), (int, float))
        verify_type(verbose, bool)

        # Verifing data sizes compatibility
        if len(X) % len(self.input_layer) != 0:
            print("[!] Warning, X size and y size doesn't correspond.")

        if len(X) < len(self.input_layer):
            return History()

        history = History()

        for epoch in range(len(y)):
            # Narrowing down y for X
            eX = X[epoch * self._n : (epoch + 1) * self._n]
            ey = y[epoch]

            z = self.forward(eX)

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


if __name__ == "__main__":
    print(activation_map)
