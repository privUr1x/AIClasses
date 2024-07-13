from typing import Optional, Union, List
from easyAI.clsstools.Verifiers import verify_type, verify_components_type, verify_len
from easyAI.core.objects import History, Model, Layer
from easyAI.core.activations import activation_map
from easyAI.core.loss_func import loss_map


class Perceptron(Model):
    "Class representing a Perceptron (Unitary Layer Neural DL Model)"

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
                Layer(self._n, name="Input Nodes"),
                Layer(1, activation=activation, name="SimplePerceptron"),
            ],
            learning_rate=learning_rate,
        )

    def __call__(self, X: List[Union[int, float]]) -> list:
        return self.forward(X)

    def fit(
        self,
        X: List[Union[int, float]],
        y: List[Union[int, float]],
        verbose: Optional[bool] = False,
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

        if len(X) < self._n:
            raise ValueError("Expected X to be at least equal to entries amount.")

        for epoch in range(len(y)):
            # Narrowing down y for X
            eX = X[epoch * self._n : (epoch + 1) * self._n]
            ey = y[epoch]

            z = self.__call__(eX)[0]

            # Updating parameters
            if z != ey:
                for n in self.output:  # [n0, n1, ..., nn]
                    for i in range(n.n):  # [w-1, w1, ..., wn]
                        n.w[i] += self._lr * (ey - z) * eX[i]
                    n.b += self._lr * (ey - z)

            if verbose:
                print(
                    f"Epoch {epoch}:\n\tModel output: {z}\n\tExpected output: {ey}"
                )

class SimpleRNN(Model):
    pass
