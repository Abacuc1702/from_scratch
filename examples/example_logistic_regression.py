import numpy as np
from ml_algorithms.logistic_regression import LogisticRegression


# ======================
# 1. Create Dataset
# ======================

X = np.array([
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
        [5, 6],
        [6, 7]
    ])
y = np.array([0, 0, 0, 1, 1, 1])

# =======================
# 2. Initialize and train the model
# =======================
model = LogisticRegression(learning_rate=0.01, n_iterations=1000)
model.fit(X, y)

# =======================
# 3. Make predictions
# =======================
predictions = model.predict(X)
probas = model.predict_proba(X)

# ======================
# Evaluate Model
# ======================
accuracy = np.mean(predictions == y)

# =====================
# Display results
# =====================
print("Weights:", model.weights)
print("Bias:", model.bias)
print("Predicted probabilities:", probas)
print("Predicted classes:", predictions)
print(f"Training Accuracy: {accuracy * 100:.2f}%")
