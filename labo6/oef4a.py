import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.set_printoptions(precision=4, suppress=True)
np.random.seed(42)
print("Libraries geladen!")

# Genereer synthetische data voor lineaire regressie: y = 2.5x + 3 + ruis.
np.random.seed(42)
n_samples = 50

# Genereer data
X = np.random.uniform(0, 10, n_samples)
y = 2.5 * X + 3 + np.random.randn(n_samples) * 1.5

# Plot
plt.figure(figsize=(8, 5))
plt.scatter(X, y, alpha=0.7)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Synthetische data: y = 2.5x + 3 + ruis')
plt.grid(True, alpha=0.3)
plt.show()