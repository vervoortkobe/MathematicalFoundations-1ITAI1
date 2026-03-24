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

# Visualiseer:
# 1. De data met de geleerde lijn
# 2. De loss curve tijdens training
# 3. De evolutie van w en b

# Visualisatie 1: Data + geleerde lijn
plt.figure(figsize=(12, 4))

# Subplot 1: Data scatter + regressielijn
plt.subplot(1, 3, 1)
plt.scatter(X, y, alpha=0.7, color='blue', label='Data')
X_plot = np.linspace(0, 10, 100)
y_pred = model.w * X_plot + model.b
plt.plot(X_plot, y_pred, 'r-', linewidth=2, label=f'Geleerd: y={model.w:.2f}x+{model.b:.2f}')
plt.plot(X_plot, 2.5*X_plot + 3, 'g--', linewidth=1, label='Echt: y=2.5x+3')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Data + Regressielijn')
plt.legend()
plt.grid(True, alpha=0.3)

# Subplot 2: Loss curve tijdens training
plt.subplot(1, 3, 2)
plt.plot(model.loss_history, 'b-', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Loss tijdens Training')
plt.yscale('log')  # Log-schaal voor betere zichtbaarheid
plt.grid(True, alpha=0.3)

# Subplot 3: Evolutie w en b
plt.subplot(1, 3, 3)
epochs = np.arange(len(model.loss_history))
plt.plot(epochs, model.loss_history, 'b-', linewidth=2, label='Loss')
plt.axhline(y=2.5, color='r', linestyle='--', alpha=0.7, label='w=2.5')
plt.axhline(y=3.0, color='g', linestyle='--', alpha=0.7, label='b=3.0')
plt.xlabel('Epoch')
plt.ylabel('Waarde')
plt.title('Evolutie w, b en Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print finale resultaten
print(f"\nFinale resultaten na 200 epochs:")
print(f"w = {model.w:.4f}  (verwacht: 2.5000)")
print(f"b = {model.b:.4f}  (verwacht: 3.0000)")
print(f"Finale MSE = {model.loss_history[-1]:.4f}")
