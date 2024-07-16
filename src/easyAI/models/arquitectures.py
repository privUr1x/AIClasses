from typing import Optional, Union, List
from easyAI.utils.verifiers import verify_type, verify_components_type, verify_len
from easyAI.core.objects import History, Model, Dense
from easyAI.core.activations import activation_map
from easyAI.core.loss_func import loss_map
from easyAI.core.optimizers import optimizers_map


class Perceptron(Model):
    "Class representing a Perceptron (Unitary Layer FeedForward Fully Conected Model)"

    def __init__(
        self,
        entries: int,
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

        self._n = verify_type(entries, int)

        super().__init__(
           [
                Dense(self._n, name="Input Nodes"),
                Dense(1, activation=activation, name="SimplePerceptron"),
            ],
            learning_rate=learning_rate,
        )

        del self._optimizer
        del self.loss

    def __call__(self, X: List[Union[int, float]]) -> list:
        return self.forward(X)

    def fit(
        self,
        X: List[Union[int, float]],
        y: List[Union[int, float]],
        verbose: bool = False,
    ) -> None:
        """
        Trains the model following the Perceptron Learning Rule.

        Returns:
            - History: The history loss.
        """
        # Verify type of X and y, and verbose option
        X = verify_components_type(verify_type(X, list), (int, float))
        y = verify_components_type(verify_type(y, list), (int, float))
        verify_type(verbose, bool)

        # Verifing data sizes compatibility
        if len(X) % self._n != 0:
            print("[!] Warning, X size and y size doesn't correspond.")

        assert len(X) > self._n, "Expected X to be at least equal to entries amount."

        return optimizers_map["plr"](self, X, y, verbose)


class SimpleRNN(Model):
    pass
