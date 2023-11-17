import numpy as np


class NeuralNetwork:
    def __init__(self)->None:
        pass

    def predict(self, message: str)->float:
        return np.random.rand()


def predict_message_mood(message: str, model: NeuralNetwork, bad: float = 0.3, good: float = 0.8) -> str:
    pred = model.predict(message)
    if pred < bad:
        return 'неуд'
    elif pred > good:
        return 'отл'
    else:
        return 'норм'