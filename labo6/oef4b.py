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

# Implementeer lineaire regressie met gradient descent. Het model is ^y = w * x + b.
class LinearRegressionGD:
    def __init__(self, learning_rate=0.01):
        self.lr = learning_rate
        self.w = 0.0
        self.b = 0.0
        self.loss_history = []
    
    def predict(self, X):
        """Voorspel y = w*X + b."""
        return self.w * X + self.b
    
    def compute_loss(self, X, y):
        """Bereken MSE loss."""
        return np.mean((self.predict(X) - y) ** 2)
    
    def compute_gradients(self, X, y):
        """Bereken gradiënten ∂L/∂w en ∂L/∂b."""
        y_pred = self.predict(X)
        n = len(y)
        dw = (2/n) * np.sum((y_pred - y) * X)    # ∂MSE/∂w = (2/n) Σ(ŷᵢ-yᵢ)xᵢ
        db = (2/n) * np.sum(y_pred - y)          # ∂MSE/∂b = (2/n) Σ(ŷᵢ-yᵢ)
        return dw, db
    
    def fit(self, X, y, n_epochs=100):
        """Train het model met gradient descent."""
        for _ in range(n_epochs):
            gradients = self.compute_gradients(X, y)
            self.w -= self.lr * gradients[0] # dw
            self.b -= self.lr * gradients[1] # db
            self.loss_history.append(self.compute_loss(X, y))
        return self

# Train
model = LinearRegressionGD(learning_rate=0.01)
model.fit(X, y, n_epochs=200)

print(f"Geleerd: w = {model.w:.4f}, b = {model.b:.4f}")
print(f"Werkelijk: w = 2.5, b = 3")