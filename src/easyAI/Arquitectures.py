from typing import Optional, Union, List
from easyAI.clsstools.Verifiers import verify_type, verify_components_type, verify_len
from easyAI.core.objects import History, Model, Layer
from easyAI.core.activations import activation_map
from easyAI.core.loss_func import loss_map


class Perceptron(Model):
    "Class representing a Perceptron (Unitary Layer Neural DL Model)"

    def __init__(self, entries: int, activation: str = "step") -> None:
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
            Layer(self._n, name="X Nodes"),
            Layer(1, activation=activation, name="SimplePerceptron"),
        )

    def __call__(self, X: List[Union[int, float]]) -> float:
        return self.forward(X)[0]

    def fit(
        self,
        X: List[Union[int, float]],
        y: List[Union[int, float]],
        verbose: Optional[bool] = False,
    ) -> History:
        """
        Trains the model following the Perceptron Learning Rule.

        Returns:
            - History: The history loss.
        """

        # Verify type of X and y, and verbose option
        X = verify_components_type(verify_type(X, (list)), (int, float))

        y = verify_components_type(verify_type(y, list), (int, float))

        verify_type(verbose, bool)

        # Verifing data sizes compatibility
        if len(X) % self._n != 0:
            print("[!] Warning, X size and y size doesn't correspond.")

        if len(X) < self._n:
            return History()

        # Training
        history: list = []

        for epoch in range(len(y)):
            # Narrowing down y for X
            eX = X[epoch * self._n : (epoch + 1) * self._n]
            ey = y[epoch]

            z = self.__call__(eX)

            # Updating parameters
            if z != ey:
                for n in self.output:
                    for i, w in enumerate(n._weights):
                        w += self._lr * (ey - z) * eX[i]
                    n._bias += self._lr * (ey - z)

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

        print(history)

        return History()


class MLP(Model):

    def __init__(self, structure: List[Layer]) -> None:
        super().__init__(structure)

    def fit(self, X, y):
        raise NotImplemented

        for epoch in range(len(y)):
            # Narrowing down y for X
            eX = X[epoch * self._n : (epoch + 1) * self._n]
            ey = y[epoch]

            z = self.__call__(eX)

            # Updating parameters
            if z != ey:
                for i in range(1, self._depth, -1):  # Iterating througth hidden layers
                    for j in range(
                        self._layers[i]._n
                    ):  # Iterating througth Neuron in layer
                        print("Index:", i, j)
                        for w in range(self._n):  # Iterating througth weights in Neuron
                            self._layers[i][j]._weights[w] += (self._lr * (ey - z) * eX[j])  # w <- w + lr * (ey - z) * x

                        self._layers[i][j]._bias += self._lr * (ey - z)  # b <- b + lr * (ey - z)
