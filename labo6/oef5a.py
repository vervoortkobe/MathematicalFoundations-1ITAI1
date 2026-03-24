import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.set_printoptions(precision=4, suppress=True)
np.random.seed(42)
print("Libraries geladen!")

# Genereer data voor binaire classificatie.
np.random.seed(42)
n_samples = 100

# Klasse 0: gecentreerd rond (-1, -1)
X0 = np.random.randn(n_samples // 2, 2) * 0.8 + np.array([-1, -1])
y0 = np.zeros(n_samples // 2)

# Klasse 1: gecentreerd rond (1, 1)
X1 = np.random.randn(n_samples // 2, 2) * 0.8 + np.array([1, 1])
y1 = np.ones(n_samples // 2)

X_class = np.vstack([X0, X1])
y_class = np.hstack([y0, y1])

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(X_class[y_class == 0, 0], X_class[y_class == 0, 1], c='blue', label='Klasse 0')
plt.scatter(X_class[y_class == 1, 0], X_class[y_class == 1, 1], c='red', label='Klasse 1')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.title('Binaire classificatie data')
plt.grid(True, alpha=0.3)
plt.show()