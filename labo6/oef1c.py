import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.set_printoptions(precision=4, suppress=True)
np.random.seed(42)
print("Libraries geladen!")

# Implementeer Binary Cross-Entropy loss:
# Gebruik np.clip() om te voorkomen dat je log(0) berekent.
def binary_cross_entropy(y_pred, y_true, epsilon=1e-15):
    """Bereken Binary Cross-Entropy Loss."""
    return -np.mean(y_true * np.log(np.clip(y_pred, epsilon, 1 - epsilon)) + 
                    (1 - y_true) * np.log(np.clip(1 - y_pred, epsilon, 1 - epsilon)))

# Test
y_true_class = np.array([1, 0, 1, 1, 0])
y_pred_probs = np.array([0.9, 0.1, 0.8, 0.7, 0.3])

print(f"BCE: {binary_cross_entropy(y_pred_probs, y_true_class)}")