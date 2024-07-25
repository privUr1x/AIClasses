from typing import Callable, Optional, Union, List
from easyAI.utils.verifiers import verify_type, verify_components_type
from easyAI.core.objects import Layer, Model
from easyAI.layers import Dense, Input, Rec, Conv
from easyAI.core.optimizers import Optimizer, optimizers_map


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

        super().__init__(
            [Input(entries), Dense(1, activation=activation, name=self._name)]
        )

    def __call__(self, X: List[Union[int, float]]) -> float:
        r: float = self.forward(X)[0]

        return r

    def fit(
        self,
        X: List[Union[int, float]],
        Y: List[Union[int, float]],
        *,
        epochs: int = 1,
        learning_rate: Union[int, float] = 10e-1,
        verbose: bool = False,
    ) -> None:
        """
        Trains the model following the Perceptron Learning Rule.
        """
        # Verify type of X and y, and verbose option
        X = verify_components_type(verify_type(X, list), (int, float))
        Y = verify_components_type(verify_type(Y, list), (int, float))
        verify_type(verbose, bool)
        verify_type(epochs, int)
        self._lr = verify_type(learning_rate, (int, float))

        # Verifing data sizes compatibility
        if len(X) % len(Y) != 0:
            print("[!] Warning, X size and y size doesn't correspond.")

        assert len(X) >= self.n, "Expected X to be at least equal to entries amount."

        assert len(Y) >= len(self.output_layer)

        self._optimizer = optimizers_map["plr"](
            model=self, learning_rate=learning_rate, epochs=epochs
        )

        length_relation: int = len(X) // len(Y)

        X = X[: len(X) * length_relation]
        Y = Y[: len(Y) * length_relation]

        return self._optimizer.fit(X, Y, verbose=verbose)


class MLP(Model):
    """Class represnetin a Multy-Layer Perceptron."""

    def __init__(
        self,
        structure: List[Layer],
    ) -> None:
        verify_components_type(verify_type(structure, list), (Input, Dense))

        super().__init__(structure=structure)
        self._name: str = "Multy-Layer Perceptron"

    def __str__(self) -> str:
        return super().__str__() + f"\n{[layer for layer in self.layers]}"

    def __repr__(self) -> str:
        return super().__repr__() + f"\n{[layer for layer in self.layers]}"

    def fit(
        self,
        X: List[Union[int, float]],
        Y: List[Union[int, float]],
        *,
        loss: str = "mse",
        epochs: int = 10,
        optimizer: str = "sgd",
        learning_rate: Union[int, float] = 10e-2,
        verbose: bool = False,
    ):

        return super().fit(
            X,
            Y,
            loss=loss,
            epochs=epochs,
            optimizer=optimizer,
            learning_rate=learning_rate,
            verbose=verbose,
        )


class SimpleRNN(Model):
    """Recurent neural network class."""

    pass


class SimpleConvNN(Model):
    """Convolutional neural network class."""

    pass


class NN(Model):
    """Flexible neural network class."""

    pass
