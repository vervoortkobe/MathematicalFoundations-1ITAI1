import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.set_printoptions(precision=4, suppress=True)
np.random.seed(42)
print("Libraries geladen!")

# Implementeer de Mean Squared Error (MSE) loss functie:
def mse_loss(y_pred, y_true):
    """Bereken Mean Squared Error."""
    return np.mean((y_pred - y_true) ** 2)

# Test
y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_pred = np.array([1.1, 2.2, 2.9, 4.0, 5.2])

# Verwacht: ((0.1)² + (0.2)² + (0.1)² + 0 + (0.2)²) / 5 = 0.02
print(f"MSE: {mse_loss(y_pred, y_true)}")