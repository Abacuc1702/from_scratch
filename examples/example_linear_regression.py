import numpy as np
from ml_algorithms.linear_regression import LinearRegression
import matplotlib.pyplot as plt


# ======================
# 1. Create Dataset
# ======================

# Simple linear relation: y = 2x + 1 + noise
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([3, 5, 7, 9, 11])

# =======================
# 2. Initialize and train the model
# =======================
model = LinearRegression(learning_rate=0.01, n_iterations=1000)
model.fit(X, y)

# =======================
# 3. Make predictions
# =======================
X_test = np.array([[6], [7]])
predictions = model.predict(X_test)
print("Predictions for X_test: ", predictions)

# =======================
# 4. Visualize the results
# =======================
plt.figure()
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(np.vstack((X, X_test)), model.predict(np.vstack((X, X_test))),
         color='red', label='Linear Regression Fit')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression Example')
plt.legend()
plt.show()
