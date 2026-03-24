import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)
np.random.seed(42)

print("Libraries geladen!")

# Train een neuraal netwerk op MNIST. Gebruik de layer classes die je hebt geïmplementeerd.

# Laad MNIST
from sklearn.datasets import fetch_openml

print("MNIST laden...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X, y = mnist.data / 255.0, mnist.target.astype(int)

X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

print(f"Training: {X_train.shape}, Test: {X_test.shape}")

# Softmax en Cross-Entropy (gegeven)
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

# --- Layer Implementaties ---

class LinearLayer:
    def __init__(self, input_dim, output_dim):
        # He-initialisatie of Xavier-initialisatie is beter voor diepe netwerken, 
        # maar een kleine random waarde werkt ook voor dit lab.
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

# --- Netwerk Initialisatie ---

input_dim = 784
hidden_dim = 128
output_dim = 10

layer1 = LinearLayer(input_dim, hidden_dim)
relu = ReLULayer()
layer2 = LinearLayer(hidden_dim, output_dim)

# --- Training Loop ---

learning_rate = 0.5
batch_size = 128
epochs = 5

print("\nTraining gestart...")

for epoch in range(epochs):
    # Shuffle de trainingsdata elk epoch
    indices = np.random.permutation(len(X_train))
    X_train_sh = X_train[indices]
    y_train_sh = y_train[indices]
    
    total_loss = 0
    total_correct = 0
    
    for i in range(0, len(X_train), batch_size):
        X_batch = X_train_sh[i:i+batch_size]
        y_batch = y_train_sh[i:i+batch_size]
        
        # 1. Forward Pass
        z1 = layer1.forward(X_batch)
        a1 = relu.forward(z1)
        z2 = layer2.forward(a1)
        probs = softmax(z2)
        
        # 2. Loss & Accuracy
        total_loss += cross_entropy_loss(probs, y_batch) * len(X_batch)
        total_correct += np.sum(np.argmax(probs, axis=1) == y_batch)
        
        # 3. Backward Pass
        dout = cross_entropy_gradient(probs, y_batch)
        dz2 = layer2.backward(dout)
        da1 = relu.backward(dz2)
        dx = layer1.backward(da1)
        
        # 4. Update Gewichten
        layer1.W -= learning_rate * layer1.dW
        layer1.b -= learning_rate * layer1.db
        layer2.W -= learning_rate * layer2.dW
        layer2.b -= learning_rate * layer2.db
        
    train_acc = total_correct / len(X_train)
    train_loss = total_loss / len(X_train)
    
    # Test set evaluatie
    z1_test = layer1.forward(X_test)
    a1_test = relu.forward(z1_test)
    z2_test = layer2.forward(a1_test)
    probs_test = softmax(z2_test)
    test_acc = np.mean(np.argmax(probs_test, axis=1) == y_test)
    
    print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

print("\nTraining voltooid!")

# Toon enkele voorbeelden met voorspellingen
plt.figure(figsize=(10, 4))
for i in range(5):
    idx = np.random.randint(len(X_test))
    img = X_test[idx].reshape(28, 28)
    
    # Forward pass voor één sample
    z1 = layer1.forward(X_test[idx:idx+1])
    a1 = relu.forward(z1)
    z2 = layer2.forward(a1)
    pred = np.argmax(softmax(z2))
    
    plt.subplot(1, 5, i+1)
    plt.imshow(img, cmap='gray')
    plt.title(f"Label: {y_test[idx]}\nPred: {pred}")
    plt.axis('off')
plt.show()

# Experimenteer met hyperparameters:
# Probeer verschillende hidden layer sizes (64, 128, 256)
# Probeer verschillende learning rates (0.1, 0.5, 1.0)
# Wat geeft de beste test accuracy?

# Jouw experimenten hier

# --- Experimentatie Functie ---

def train_and_evaluate(hidden_dim, learning_rate, epochs=3, batch_size=128):
    # Initialiseer lagen voor dit experiment
    layer1 = LinearLayer(784, hidden_dim)
    relu = ReLULayer()
    layer2 = LinearLayer(hidden_dim, 10)
    
    for epoch in range(epochs):
        indices = np.random.permutation(len(X_train))
        X_train_sh = X_train[indices]
        y_train_sh = y_train[indices]
        
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train_sh[i:i+batch_size]
            y_batch = y_train_sh[i:i+batch_size]
            
            # Forward
            z1 = layer1.forward(X_batch)
            a1 = relu.forward(z1)
            z2 = layer2.forward(a1)
            probs = softmax(z2)
            
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
            
    # Test set evaluatie
    z1_test = layer1.forward(X_test)
    a1_test = relu.forward(z1_test)
    z2_test = layer2.forward(a1_test)
    probs_test = softmax(z2_test)
    test_acc = np.mean(np.argmax(probs_test, axis=1) == y_test)
    
    return test_acc

# Hyperparameters om te testen
hidden_sizes = [64, 128, 256]
learning_rates = [0.1, 0.5, 1.0]

results = []

print("\n--- Start Hyperparameter Experimenten ---")
print(f"{'Hidden Size':<12} | {'LR':<5} | {'Test Accuracy':<15}")
print("-" * 40)

best_acc = 0
best_config = None

for h in hidden_sizes:
    for lr in learning_rates:
        acc = train_and_evaluate(h, lr, epochs=3) # 3 epochs is genoeg voor trend
        results.append((h, lr, acc))
        print(f"{h:<12} | {lr:<5} | {acc:.4f}")
        
        if acc > best_acc:
            best_acc = acc
            best_config = (h, lr)

print("-" * 40)
print(f"Beste configuratie: Hidden Size = {best_config[0]}, Learning Rate = {best_config[1]} met Acc = {best_acc:.4f}")

# Visualiseer resultaten (optioneel: heatmap)
plt.figure(figsize=(10, 6))
for h in hidden_sizes:
    accs = [r[2] for r in results if r[0] == h]
    plt.plot(learning_rates, accs, label=f'Hidden Size: {h}', marker='o')

plt.xlabel('Learning Rate')
plt.ylabel('Test Accuracy')
plt.title('Hyperparameter Experimenten op MNIST')
plt.legend()
plt.grid(True)
plt.show()
