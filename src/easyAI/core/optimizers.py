from typing import Any, Callable, List, Union, Protocol
from easyAI.core.metrics import History
from abc import ABC, abstractmethod

"""Module representing optimizers used for training models."""


class Optimizer(ABC):
    """Class representing the abstraction to an optimizer."""

    def __init__(
        self,
        model: "Model",
        learning_rate: Union[int, float],
        epochs: int,
        loss: Callable,
    ):
        self._lr: Union[int, float] = learning_rate
        self._epochs: int = epochs
        self._loss: Callable = loss
        self._model: "Model" = model

    @property
    def loss(self):
        """The loss property."""
        return self._loss

    @property
    def learning_rate(self):
        """The learning_rate property."""
        return self._lr

    @property
    def epochs(self):
        """The epochs property."""
        return self._epochs

    @property
    def model(self):
        """The model property."""
        return self._model

    @abstractmethod
    def fit(
        X: List[Union[int, float]], Y: List[Union[int, float]], *, verbose: bool
    ) -> dict: ...

class PLR(Optimizer):
    """Class representing Perceptron Learning Rule"""

    def __init__(
        self, model: "Perceptron", learning_rate: Union[int, float], epochs: int
    ):
        super().__init__(model, learning_rate, epochs, loss=None)

        del self._loss

    def fit(
        self,
        X: List[Union[int, float]],
        Y: List[Union[int, float]],
        *,
        verbose: bool,
    ) -> Any:
        for epoch in range(self.epochs):
            # The following code will always exec
            for i, expected_y in enumerate(Y):
                # Narrowing down Y for X
                inpts = X[i * self.model.n : (i + 1) * self.model.n]

                net_output = self.model.__call__(inpts)

                # Updating parameters (PLR)
                if net_output != expected_y:
                    for neuron in self.model.output_layer:  # [n0, n1, ..., nn]
                        for i in range(neuron.n):  # [w-1, w1, ..., wn]
                            neuron.w[i] += (
                                self.learning_rate
                                * (expected_y - net_output)
                                * inpts[i]
                            )
                        neuron.b += self.model.learning_rate * (expected_y - net_output)

            if verbose:
                print(
                    f"Epoch {epoch}:\n\tModel output: {net_output}\n\tExpected output: {expected_y}"
                )


class SGD(Optimizer):
    """Class representing Stochastic Gradient Descent."""

    def __init__(
        self,
        model: "Model",
        learning_rate: Union[int, float],
        epochs: int,
        loss: Callable,
    ):
        super().__init__(model, learning_rate, epochs, loss)

    def fit(
        self, X: list[Union[int, float]], Y: list[Union[int, float]], *, verbose: bool
    ) -> History:
        """
        Schotastic Gradient Descent algorithm.
        """
        for epoch in range(self.epochs):
            for i in range(len(Y)):
                # Seleccionar un solo ejemplo
                inpts: list = X[i * self.model.n : (i + 1) * self.model.n]
                expected_y: list = Y[i * self.model.output_layer.n: (i + 1) * self.model.output_layer.n] 

                # Predicción del modelo para el ejemplo actual
                predictions: list = self.model.forward(inpts)

                if predictions != expected_y:
                    # Parameters update
                    pass

            if verbose:
                print(epoch)
                print("Preds, Inpts, Expected_y", predictions, inpts, expected_y)

    def compute_gradients(self):
        pass

    def update_parameters(self):
        pass


class Adam(Optimizer):
    """Class representing Adaptative Moment Estimation."""

    pass


optimizers_map: dict[str, Optimizer] = {
    "sgd": SGD,
    "adam": Adam,
    "plr": PLR,
}
