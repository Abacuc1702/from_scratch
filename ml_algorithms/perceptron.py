import numpy as np


class Perceptron:
    """
    Perceptron classifier implemented from scratch using
    the classic Perceptron learning algorithm.

    This model performs binary classification by learning
    a linear decision boundary that separates two classes.

    Attributes
    ----------
    n_iterations : int
        Number of training iterations (epochs).
    learning_rate : float
        Step size for weight updates.
    weights : np.ndarray of shape (n_features,)
        Coefficients (weights) of the model, learned during training.
    bias : float
        Bias term of the model, learned during training.

    Methods
    -------
    fit(X, y)
        Train the perceptron model using the Perceptron learning rule.
    predict(X)
        Return binary predictions (0 or 1) for given input samples.
    """

    def __init__(self, n_iterations=1000, learning_rate=0.01):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None

    def sign(self, z):
        """
        Compute the sign activation function.

        Parameters
        ----------
        z : np.ndarray
            Linear combination of inputs and weights.

        Returns
        -------
        np.ndarray
            The sign of z, representing class labels {-1, 1}.
        """
        return np.where(z >= 0, 1, -1)

    def fit(self, X, y):
        """
        Train the perceptron model using the Perceptron learning algorithm.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training input samples.
        y : np.ndarray of shape (n_samples,)
            Target binary labels (0 or 1) for each input sample.
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Convert labels from {0, 1} to {-1, 1}
        y_ = np.where(y <= 0, -1, 1)

        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_predicted = self.sign(linear_output)

                # Update rule: only adjust if prediction is wrong
                if y_predicted != y_[idx]:
                    self.weights += self.learning_rate * y_[idx] * x_i
                    self.bias += self.learning_rate * y_[idx]

    def predict(self, X):
        """
        Return binary predictions (0 or 1) for each input sample.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input samples to predict.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted binary class labels (0 or 1).
        """
        linear_output = np.dot(X, self.weights) + self.bias
        y_predicted = self.sign(linear_output)
        return np.where(y_predicted == 1, 1, 0)
