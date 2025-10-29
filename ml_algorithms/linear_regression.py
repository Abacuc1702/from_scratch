import numpy as np


class LinearRegression:
    """
    Linear Regression implementation from scratch using Gradient descent.

    This class implements a simple Linear Regression model using
    Gradient Descent to estimate the coefficients `weights` and the
    intercept `bias` that minimize the Mean Squared Error (MSE)
    between the predicted and true target values.

    Parameters
    ----------
    n_iterations: int, default=1000
        Number of iterations to perform during the gradient descent
        optimization.

    learning_rate: float, default=0.01
        The step size (α) used during gradient descent updates.

    Attributes
    ----------
    weights: numpy.ndarray of shape (n_features)
        The learned weight coefficient after training.

    bias: float
        The learned bias (intercept) after training.

    Examples
    --------
    >>> import numpy as np
    >>> from ml_algorithms.linear_regression import LinearRegression
    >>> X = np.array([[1], [2], [3], [4]])
    >>> y = np.array([2, 4, 3, 8])
    >>> model = LinearRegression(learning_rate=0.01, n_iterations=1000)
    >>> model.fit(X, y)
    >>> model.predict(np.array([[5]]))
    array([]) # approximately
    """

    def __init__(self, n_iterations=1000, learning_rate=0.01):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Train the Linear Regression model on the given dataset.

        This method updates the coefficients `weights` and `bias`
        using Gradient Descent to minimize the Mean Squared Error
        between predicted and actual target values.

        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            Input data matrix, where each row represents a sample
            and each column represents a feature.

        y : numpy.ndarray of shape (m_samples,)
            Target values corresponding to each input sample.

        Returns
        -------
        None
        """

        # Get data shape
        n_samples, n_features = X.shape
        # Initialize weights and bias to 0
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Start learning loop
        for i in range(self.n_iterations):
            # Make a prediction
            y_pred = np.dot(X, self.weights) + self.bias

            # Calculate gradient
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)

            # update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        """
        Predict target values for new input samples.

        Uses the learned coefficients and bias to compute
        predictions based on the linear model:

        y_pred = np.dot(X, weights) + bias

        Parameters
        ----------
        X : numpy.ndarray of shape (m_samples, n_features)
            Input samples for which predictions are to be made.

        Returns
        -------
        numpy.ndarray
            Predicted target values for the given samples.
        """
        return np.dot(X, self.weights) + self.bias
