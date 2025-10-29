import numpy as np


class LogisticRegression:
    """
    Logistic Regression classifier implemented from scratch using
    Gradient Descent.

    This model is used for binary classification tasks.
    It learns a linear decision boundary and applies the sigmoid function
    to map predictions to probabilities between 0 and 1.

    Attributes
    ----------
    n_iterations : int
        Number of gradient descent iterations during training.
    learning_rate : float
        Step size for gradient descent updates.
    threshold : float
        Probability threshold used to convert predicted probabilities into
        binary labels.
    weights : np.ndarray, shape (n_features,)
        Coefficients of the model (learned during training).
    bias : float
        Bias term of the model (learned during training).

    Methods
    -------
    sigmoid(z)
        Compute the sigmoid activation function.
    fit(X, y)
        Train the logistic regression model using gradient descent.
    predict_proba(X)
        Return predicted probabilities for each input sample.
    predict(X)
        Return binary class predictions (0 or 1) for each input sample.
    """

    def __init__(self, n_iterations=1000, learning_rate=0.01, threshold=0.5):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        """
        Compute the sigmoid activation function.

        Parameters
        ----------
        z : np.ndarray
            Linear combination of inputs and weights.

        Returns
        -------
        np.ndarray
            The sigmoid of z, representing probabilities between 0 and 1.
        """
        return 1/(1 + np.exp(-z))

    def fit(self, X, y):
        """
        Train the logistic regression model using gradient descent.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Input feature matrix.
        y : np.ndarray, shape (n_samples,)
            Binary target vector (values 0 or 1).

        Returns
        -------
        self : LogisticRegression
            The trained model.
        """

        # Get data shape
        n_samples, n_features = X.shape
        # Initialize weights and bias to 0
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Start learning loop
        for i in range(self.n_iterations):
            # Make prediction like linear regression
            z = np.dot(X, self.weights) + self.bias
            # Sigmoid activation
            y_pred = self.sigmoid(z)

            # Calculate gradient
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)

            # update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_proba(self, X):
        """
        Predict class probabilities for input samples.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        np.ndarray
            Predicted probabilities for class 1.
        """

        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)

    def predict(self, X):
        """
        Predict binary class labels (0 or 1) for input samples.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        np.ndarray
            Binary predictions (0 or 1) based on a threshold of 0.5.
        """

        y_pred_proba = self.predict_proba(X)
        p = np.array([1 if i >= self.threshold else 0 for i in y_pred_proba])
        return p
