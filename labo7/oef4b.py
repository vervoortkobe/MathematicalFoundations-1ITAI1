import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)
np.random.seed(42)

print("Libraries geladen!")

# Implementeer een ReLULayer class.

class ReLULayer:
    def __init__(self):
        self.mask = None
        
    def forward(self, X):
        # ReLU: max(0, X)
        # Sla een mask op waar X > 0 voor de backward pass
        self.mask = (X > 0)
        self.output = X * self.mask
        return self.output
    
    def backward(self, dout):
        # Alleen de elementen waar X > 0 was, laten de gradiënt door
        dX = dout * self.mask
        return dX

# Test
print("\n=== Test ReLULayer ===")
relu = ReLULayer()
X = np.array([[-1.0, 2.0, 0.0], [4.0, -5.0, 6.0]])

# Forward pass
out = relu.forward(X)
print(f"Input X:\n{X}")
print(f"Output ReLU(X):\n{out}")

# Backward pass
dout = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
dX = relu.backward(dout)
print(f"\nGradiënt dout:\n{dout}")
print(f"Gradiënt dX:\n{dX}")

# Verifieer correctness
expected_out = np.maximum(0, X)
expected_dX = dout * (X > 0)

assert np.all(out == expected_out)
assert np.all(dX == expected_dX)

print("\nReLU implementatie is CORRECT!")