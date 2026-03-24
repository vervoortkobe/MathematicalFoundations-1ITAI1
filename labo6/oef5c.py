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

# Implementeer logistische regressie met gradient descent.
# Model: p = sigmoid(w1*x1 + w2*x2 + b)
# Loss: Binary Cross-Entropy
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

class LogisticRegressionGD:
    def __init__(self, learning_rate=0.1):
        self.lr = learning_rate
        self.w = None
        self.b = 0.0
        self.loss_history = []
    
    def predict_proba(self, X):
        """Voorspel kansen."""
        return sigmoid(X @ self.w + self.b)
    
    def predict(self, X):
        """Voorspel klassen (0 of 1)."""
        return (self.predict_proba(X) >= 0.5).astype(int)
    
    def fit(self, X, y, n_epochs=100):
        """Train met gradient descent."""
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        for _ in range(n_epochs):
            y_pred = self.predict_proba(X)
            dw = (1 / n_samples) * X.T @ (y_pred - y)
            db = (1 / n_samples) * np.sum(y_pred - y)
            self.w -= self.lr * dw
            self.b -= self.lr * db
        return self

# Train
log_model = LogisticRegressionGD(learning_rate=0.5)
log_model.fit(X_class, y_class, n_epochs=200)

# Evalueer
predictions = log_model.predict(X_class)
accuracy = np.mean(predictions == y_class)
print(f"Nauwkeurigheid: {accuracy * 100:.1f}%")

# Visualiseer de decision boundary.
x_min, x_max = X_class[:, 0].min() - 1, X_class[:, 0].max() + 1
y_min, y_max = X_class[:, 1].min() - 1, X_class[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
Z = log_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, cmap='RdYlBu', alpha=0.7)
plt.scatter(X_class[y_class == 0, 0], X_class[y_class == 0, 1], c='blue', label='Klasse 0')
plt.scatter(X_class[y_class == 1, 0], X_class[y_class == 1, 1], c='red', label='Klasse 1')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.title('Decision Boundary')
plt.grid(True, alpha=0.3)
plt.show()