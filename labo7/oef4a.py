import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)
np.random.seed(42)

print("Libraries geladen!")

# Implementeer een LinearLayer class met forward en backward methodes. De laag berekent: z = X @ W + b

class LinearLayer:
    def __init__(self, input_dim, output_dim):
        # Initialiseer W en b
        # We gebruiken een kleine random waarde voor W en nullen voor b
        self.W = np.random.randn(input_dim, output_dim) * 0.01
        self.b = np.zeros(output_dim)
        self.dW = None
        self.db = None
    
    def forward(self, X):
        # X shape: (batch_size, input_dim)
        # Return shape: (batch_size, output_dim)
        # Sla X op voor backward!
        self.X = X
        self.output = X @ self.W + self.b
        return self.output
    
    def backward(self, dout):
        # dout shape: (batch_size, output_dim)
        # dL/dW = X.T @ dout
        # dL/db = som van dout over de batch (axis=0)
        # dL/dX = dout @ W.T
        
        self.dW = self.X.T @ dout
        self.db = np.sum(dout, axis=0)
        
        dX = dout @ self.W.T
        return dX

# Test
print("\n=== Test LinearLayer ===")
input_size = 3
output_size = 2
batch_size = 4

layer = LinearLayer(input_size, output_size)
X = np.random.randn(batch_size, input_size)

# Forward pass
Z = layer.forward(X)
print(f"Input shape:  {X.shape}")
print(f"Output shape: {Z.shape}")
print(f"Output Z:\n{Z}")

# Backward pass
dout = np.ones_like(Z) # Stel dL/dZ = 1 voor alle elementen
dX = layer.backward(dout)

print(f"\ndX shape: {dX.shape}")
print(f"dW shape: {layer.dW.shape}")
print(f"db shape: {layer.db.shape}")

# Verifieer shapes
assert Z.shape == (batch_size, output_size)
assert dX.shape == (batch_size, input_size)
assert layer.dW.shape == (input_size, output_size)
assert layer.db.shape == (output_size,)

print("\nShapes zijn correct!")