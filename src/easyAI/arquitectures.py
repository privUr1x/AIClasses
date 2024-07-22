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
        learning_rate: Union[int, float] = 0.1,
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

        super().__init__([NodeLayer(entries), Dense(1, activation=activation, name=self._name)],
            loss="mse",
            optimizer="plr", 
            learning_rate=learning_rate, 
        )

        del self.loss

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

        return self._optimizer(self, X, y, epochs=epochs, verbose=verbose)


class MLP(Model):
    """Class represnetin a Multy-Layer Perceptron."""

    def __init__(
        self,
        structure: List[Layer],
        *,
        loss: str = "mse",
        optimizer: str = "sgd",
        learning_rate: Union[int, float] = 0.01,
    ) -> None:

        super().__init__(
            structure, loss=loss, optimizer=optimizer, learning_rate=learning_rate
        )
        self._name: str = "Multy-Layer Perceptron"
        self._n: int = len(self.input_layer)

    def __str__(self) -> str:
        return super().__str__() + f"\n{[layer for layer in self.layers]}"

    def __repr__(self) -> str:
        return super().__repr__() + f"\n{[layer for layer in self.layers]}"

    def fit(
        self,
        X: list[Union[int, float]],
        Y: list[Union[int, float]],
        *,
        epochs: int,
        verbose: bool = False,
    ) -> None:
        """Traings the model algorithmically based on the optimizer."""
        verify_components_type(verify_type(X, list), (int, float))
        verify_components_type(verify_type(Y, list), (int, float))
        verify_type(verbose, bool)
        verify_type(epochs)

        return self._optimizer(X, Y, epochs, self.learning_rate, verbose)


class SimpleRNN(Model):
    """Recurent neural netwrok class."""

    pass


class NN(Model):
    """Flexible neural network class."""

    pass
