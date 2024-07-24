from typing import Any, Callable, List, Union
from abc import ABC, abstractmethod
from easyAI.core.objects import Model

"""Module representing optimizers used for training models."""


class Optimizer(ABC):
    """Class representing the abstraction to an optimizer."""

    def __init__(self, learning_rate: Union[int, float], epochs: int, loss: Callable):
        self._lr: Union[int, float] = learning_rate
        self._epochs: int = epochs 
        self._loss: Callable = loss

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

    def fit(X: list[Union[int, float]], Y: list[Union[int, float]]) -> dict:
        pass

class SGD(Optimizer):
    """Class representing Stochastic Gradient Descent."""

    def __init__(self, learning_rate: Union[int, float], epochs: int, loss: Callable):
        super().__init__(learning_rate, epochs, loss)

    def fit(
        self,
        arquitecture: Model,
        X: list[Union[int, float]],
        Y: list[Union[int, float]],
        inpt_size: int,
    ) -> dict:
        """
        """
        m = len(Y)  # Número de ejemplos de entrenamiento

        for epoch in range(self.epochs):
            for i in range(m):
                # slice()

                # Seleccionar un solo ejemplo
                xi = X[i:(i+inpt_size)]
                yi = Y[i]

                # Predicción del modelo para el ejemplo actual
                predictions = model.predict(xi)

                # Calcular el error
                error = predictions - yi

                # Calcular el gradiente (derivada del error con respecto a los parámetros)
                gradients = model.compute_gradients(xi, error)

                # Actualizar los parámetros del modelo
                model.update_parameters(self.learning_rate, gradients)

            # Opcional: imprimir el progreso cada ciertos epochs
            if (epoch + 1) % 10 == 0:
                loss = self.calculate_loss(X, y, model)
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss:.4f}")

class Adam(Optimizer):
    """Class representing Adaptative Moment Estimation."""
    pass


class PLR(Optimizer):
    """Class representing Perceptron Learning Rule"""

    def __init__(self, learning_rate: Union[int, float], epochs: int, loss: Callable):
        super().__init__(learning_rate, epochs, loss)

    def __call__(
        self,
        P,  # The perceptron class itself
        X: List[Union[int, float]],
        y: List[Union[int, float]],
        *,
        verbose: bool,
        epochs: int
    ) -> Any:
        for epoch in range(epochs):
            for i in range(len(y)):
                # Narrowing down y for X
                eX = X[i * P.n : (i + 1) * P.n]
                ey = y[i]

                z = P.__call__(eX)[0]

                # Updating parameters
                if z != ey:
                    for n in P.output:  # [n0, n1, ..., nn]
                        for i in range(n.n):  # [w-1, w1, ..., wn]
                            n.w[i] += self._learning_rate * (ey - z) * eX[i]
                        n.b += P._lr * (ey - z)

                if verbose:
                    print(f"Epoch {epoch}:\n\tModel output: {z}\n\tExpected output: {ey}")

        return None


optimizers_map: dict = {
    "sgd": SGD,
    "adam": Adam,
    "plr": PLR,
}
