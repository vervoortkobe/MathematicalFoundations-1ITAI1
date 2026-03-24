import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.set_printoptions(precision=4, suppress=True)
np.random.seed(42)
print("Libraries geladen!")

# Implementeer de gradiënt van de MSE loss ten opzichte van de voorspellingen:
def mse_gradient(y_pred, y_true):
    """Bereken de gradiënt van MSE naar y_pred."""
    return 2 * (y_pred - y_true) / len(y_true)

# Test
y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
y_pred = np.array([1.1, 2.2, 2.9, 4.0, 5.2])

# Test
grad = mse_gradient(y_pred, y_true)
print(f"Gradiënt: {grad}")