import numpy as np
from ml_algorithms.linear_regression import LinearRegression


def test_linear_regression():
    X = np.array([[1], [2], [3], [4]])
    y = np.array([2, 4, 6, 8])

    model = LinearRegression()
    model.fit(X, y)
    print(model.predict(np.array([[5]])))
    preds = model.predict(X)

    # Test: The prediction should be nearest of the real value
    assert np.allclose(preds, y, atol=0.5)
