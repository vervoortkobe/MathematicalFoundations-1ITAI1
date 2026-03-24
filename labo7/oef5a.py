import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)
np.random.seed(42)

print("Libraries geladen!")

# Train een netwerk met 1 hidden layer op een XOR-achtig probleem:
# - Input: 2D punten
# - Output: 1 als punt in kwadrant 1 of 3, 0 anders

# Genereer XOR-achtige data
np.random.seed(42)
n_samples = 200

X = np.random.randn(n_samples, 2)
y = ((X[:, 0] > 0) == (X[:, 1] > 0)).astype(float)  # XOR-achtig

# Visualiseer
plt.figure(figsize=(8, 6))
plt.scatter(X[y==0, 0], X[y==0, 1], c='blue', label='Klasse 0')
plt.scatter(X[y==1, 0], X[y==1, 1], c='red', label='Klasse 1')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('XOR-achtig probleem')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Bouw en train je netwerk
# --- Layer Implementaties ---

class LinearLayer:
    def __init__(self, input_dim, output_dim):
        self.W = np.random.randn(input_dim, output_dim) * 0.1
        self.b = np.zeros((1, output_dim))
        self.dW = None
        self.db = None
    
    def forward(self, X):
        self.X = X
        self.output = X @ self.W + self.b
        return self.output
    
    def backward(self, dout):
        self.dW = self.X.T @ dout
        self.db = np.sum(dout, axis=0, keepdims=True)
        dX = dout @ self.W.T
        return dX

class ReLULayer:
    def __init__(self):
        self.mask = None
    def forward(self, X):
        self.mask = (X > 0)
        return X * self.mask
    def backward(self, dout):
        return dout * self.mask

class Sigmoid:
    def forward(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output
    def backward(self, dout):
        s = self.output
        return dout * s * (1 - s)

# --- Training Setup ---

# Hyperparameters
hidden_size = 4
learning_rate = 0.5
epochs = 5000

# Initialiseer lagen
layer1 = LinearLayer(2, hidden_size)
relu = ReLULayer()
layer2 = LinearLayer(hidden_size, 1)
sigmoid = Sigmoid()

losses = []

# Voor training hebben we y nodig in shape (n_samples, 1)
y_true = y.reshape(-1, 1)

print("Training gestart...")

for epoch in range(epochs):
    # 1. Forward Pass
    z1 = layer1.forward(X)
    a1 = relu.forward(z1)
    z2 = layer2.forward(a1)
    y_pred = sigmoid.forward(z2)
    
    # 2. Bereken Loss (Binary Cross Entropy)
    # Epsilon om log(0) te voorkomen
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    losses.append(loss)
    
    # 3. Backward Pass
    # dL/dy_pred voor BCE loss
    dout = (y_pred - y_true) / (y_pred * (1 - y_pred) * len(X))
    
    dz2 = sigmoid.backward(dout)
    da1 = layer2.backward(dz2)
    dz1 = relu.backward(da1)
    dx = layer1.backward(dz1)
    
    # 4. Gewichten updaten (Gradient Descent)
    layer1.W -= learning_rate * layer1.dW
    layer1.b -= learning_rate * layer1.db
    layer2.W -= learning_rate * layer2.dW
    layer2.b -= learning_rate * layer2.db
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

print("Training voltooid!")

# --- Visualisatie ---

# Plot loss curve
plt.figure(figsize=(8, 4))
plt.plot(losses)
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("BCE Loss")
plt.grid(True)
plt.show()

# Plot Decision Boundary
def predict(X_in):
    a1 = relu.forward(layer1.forward(X_in))
    return (sigmoid.forward(layer2.forward(a1)) > 0.5).astype(float)

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

Z_boundary = predict(np.c_[xx.ravel(), yy.ravel()])
Z_boundary = Z_boundary.reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z_boundary, alpha=0.3, cmap='RdBu')
plt.scatter(X[y==0, 0], X[y==0, 1], c='blue', label='Klasse 0')
plt.scatter(X[y==1, 0], X[y==1, 1], c='red', label='Klasse 1')
plt.title("Decision Boundary")
plt.legend()
plt.show()