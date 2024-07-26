class History:
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

