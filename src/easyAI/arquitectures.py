from typing import Optional, Union, List
from easyAI.utils.verifiers import verify_type, verify_components_type, verify_len
from easyAI.core.objects import Layer, Model, Node
from easyAI.layers import Dense, NodeLayer, Rec, Conv
from easyAI.core.activations import activation_map
from easyAI.core.loss import loss_map
from easyAI.core.optimizers import optimizers_map


class Perceptron(Model):
    "Class representing a Perceptron (Unitary Layer FeedForward Fully Conected Model)"

    def __init__(
        self,
        entries: int,
        *,
        activation: str = "step",
    ) -> None:
        """
        Builds an instance give X training data, y training data and entries.

        Args:
            - entries (int): The number of inputs of the model.
        """
        # Compatibility with int expressed in float
        if isinstance(entries, float):
            if int(entries) == entries:
                entries = int(entries)

        self._name: str = "Simple Perceptron"

        super().__init__([
            NodeLayer(entries), 
            Dense(1, activation=activation, name=self._name
        )])

    def __call__(self, X: List[Union[int, float]]) -> list:
        return self.forward(X)

    def fit(
        self,
        X: List[Union[int, float]],
        y: List[Union[int, float]],
        *,
        epochs: int = 100,
        verbose: bool = False,
    ) -> None:
        """
        Trains the model following the Perceptron Learning Rule.
        """
        # Verify type of X and y, and verbose option
        X = verify_components_type(verify_type(X, list), (int, float))
        y = verify_components_type(verify_type(y, list), (int, float))
        verify_type(verbose, bool)
        verify_type(epochs, int)

        # Verifing data sizes compatibility
        if len(X) % self.n != 0:
            print("[!] Warning, X size and y size doesn't correspond.")

        assert len(X) > self.n, "Expected X to be at least equal to entries amount."

        return self._optimizer(X, y, epochs=epochs, verbose=verbose)


class MLP(Model):
    """Class represnetin a Multy-Layer Perceptron."""

    def __init__(
        self,
        structure: List[Layer],
    ) -> None:
        verify_components_type(verify_type(structure, list), Dense)

        super().__init__(structure=structure)
        self._name: str = "Multy-Layer Perceptron"

    def __str__(self) -> str:
        return super().__str__() + f"\n{[layer for layer in self.layers]}"

    def __repr__(self) -> str:
        return super().__repr__() + f"\n{[layer for layer in self.layers]}"

    def fit(
        X: list[Union[int, float]],
        Y: list[Union[int, float]],
        *,
        epochs: int = 10,
        learning_rate: Union[int, float] = 10e-2
    ) -> None:
        """Traings the model algorithmically based on the optimizer."""
        super().fit(X, Y, loss="mse", epochs=epochs, learning_rate=learning_rate, optimizer="sgd")

class SimpleRNN(Model):
    """Recurent neural netwrok class."""

    pass


class NN(Model):
    """Flexible neural network class."""

    pass
