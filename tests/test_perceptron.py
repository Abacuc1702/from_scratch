import numpy as np
from ml_algorithms.perceptron import Perceptron


def test_perceptron():
    # Create a simple dataset
    X = np.array([
        [1, 2],
        [2, 3],
        [3, 1],
        [6, 5],
        [7, 7],
        [8, 6]
    ])
    y = np.array([0, 0, 0, 1, 1, 1]) 

    # Initialize and train the model
    model = Perceptron(n_iterations=1000, learning_rate=0.1)
    model.fit(X, y)

    # Check shape
    assert model.weights.shape == (X.shape[1],)
    assert isinstance(model.bias, (float, np.floating))

    # Predict on training data
    predictions = model.predict(X)
    assert predictions.shape == y.shape

    # Check if predictions are 0 or 1
    assert np.all(np.isin(predictions, [0, 1]))

    # Check if accuracy is reasonable (>= 0.8 for this toy dataset)
    accuracy = np.mean(predictions == y)
    assert accuracy >= 0.8
