import numpy as np
from ml_algorithms.perceptron import Perceptron

# ======================
# 1. Create Dataset
# ======================
X = np.array([
        [1, 2],
        [2, 3],
        [3, 1],
        [6, 5],
        [7, 7],
        [8, 6]
    ])
y = np.array([0, 0, 0, 1, 1, 1])

# =======================
# 2. Initialize and train the model
# =======================
model = Perceptron(learning_rate=0.1, n_iterations=1000)
model.fit(X, y)

# =======================
# 3. Make predictions
# =======================
predictions = model.predict(X)

# ======================
# Evaluate Model
# ======================
accuracy = np.mean(predictions == y)

# =====================
# Display results
# =====================
print("Weights:", model.weights)
print("Bias:", model.bias)
print("Predicted classes:", predictions)
print(f"Training Accuracy: {accuracy * 100:.2f}%")
