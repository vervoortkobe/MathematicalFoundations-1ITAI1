import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

np.set_printoptions(precision=4, suppress=True)
np.random.seed(42)

print("Libraries geladen!")

# Maak een systematische vergelijking van:
# Effect van batch size (16, 32, 64, 128)
# Effect van learning rate (0.01, 0.1, 0.5, 1.0)
# Plot de learning curves.

# Jouw experimenten hier

# Laad MNIST
print("MNIST laden...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X, y = mnist.data / 255.0, mnist.target.astype(int)

X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

# --- Layer Implementaties ---

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy_loss(probs, y_true):
    N = probs.shape[0]
    return -np.sum(np.log(probs[np.arange(N), y_true] + 1e-10)) / N

def cross_entropy_gradient(probs, y_true):
    N = probs.shape[0]
    grad = probs.copy()
    grad[np.arange(N), y_true] -= 1
    return grad / N

class LinearLayer:
    def __init__(self, input_dim, output_dim):
        self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
        self.b = np.zeros((1, output_dim))
    
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

# --- Experimentatie Functie ---

def train_experiment(batch_size, learning_rate, epochs=3):
    print(f"Training met batch_size={batch_size}, learning_rate={learning_rate}...")
    
    # Layer initialisatie
    layer1 = LinearLayer(784, 128)
    relu = ReLULayer()
    layer2 = LinearLayer(128, 10)
    
    loss_history = []
    
    for epoch in range(epochs):
        indices = np.random.permutation(len(X_train))
        X_train_sh = X_train[indices]
        y_train_sh = y_train[indices]
        
        epoch_loss = 0
        num_batches = 0
        
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train_sh[i:i+batch_size]
            y_batch = y_train_sh[i:i+batch_size]
            
            # Forward
            z1 = layer1.forward(X_batch)
            a1 = relu.forward(z1)
            z2 = layer2.forward(a1)
            probs = softmax(z2)
            
            # Loss tracking
            loss = cross_entropy_loss(probs, y_batch)
            epoch_loss += loss
            num_batches += 1
            
            # Backward
            dout = cross_entropy_gradient(probs, y_batch)
            dz2 = layer2.backward(dout)
            da1 = relu.backward(dz2)
            layer1.backward(da1)
            
            # Update
            layer1.W -= learning_rate * layer1.dW
            layer1.b -= learning_rate * layer1.db
            layer2.W -= learning_rate * layer2.dW
            layer2.b -= learning_rate * layer2.db
            
        loss_history.append(epoch_loss / num_batches)
        
    return loss_history

# Experimenten
epochs = 5
batch_sizes = [16, 32, 64, 128]
learning_rates = [0.01, 0.1, 0.5, 1.0]

plt.figure(figsize=(12, 5))

# 1. Effect van Batch Size (LR = 0.1)
plt.subplot(1, 2, 1)
fixed_lr = 0.1
for bs in batch_sizes:
    history = train_experiment(bs, fixed_lr, epochs=epochs)
    plt.plot(range(1, epochs + 1), history, label=f'BS={bs}')

plt.title(f'Effect van Batch Size (LR={fixed_lr})')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# 2. Effect van Learning Rate (BS = 64)
plt.subplot(1, 2, 2)
fixed_bs = 64
for lr in learning_rates:
    history = train_experiment(fixed_bs, lr, epochs=epochs)
    plt.plot(range(1, epochs + 1), history, label=f'LR={lr}')

plt.title(f'Effect van Learning Rate (BS={fixed_bs})')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
