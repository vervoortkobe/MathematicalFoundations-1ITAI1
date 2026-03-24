import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)
np.random.seed(42)

print("Libraries geladen!")

# Bereken met de hand (en verifieer met code):
# 1. Alle tussenresultaten in de forward pass
# 2. ∂L/∂y, ∂L/∂h, ∂L/∂w2, ∂L/∂b2, ∂L/∂w1, ∂L/∂b1

# Gegeven waarden
x = 2
w1, b1 = 0.5, 0.1
w2, b2 = -0.3, 0.2
y_target = 0.5

def relu(z): return np.maximum(0, z)
def relu_prime(z): return 1.0 * (z > 0)
def sigmoid(z): return 1 / (1 + np.exp(z))
def sigmoid_prime(z): return sigmoid(z) * (1 - sigmoid(z))

# Forward pass
h = relu(w1*x + b1)
y = sigmoid(w2*h + b2)
L = (y - y_target)**2

# Backward pass
dL_dy = 2*(y - y_target)
dL_dh = dL_dy * w2 * sigmoid_prime(w2*h + b2)
dL_dw2 = dL_dy * h * sigmoid_prime(w2*h + b2)
dL_db2 = dL_dy * sigmoid_prime(w2*h + b2)
dL_dw1 = dL_dh * x * relu_prime(w1*x + b1)
dL_db1 = dL_dh * relu_prime(w1*x + b1)

print("Forward pass:")
print(f"h = {h:.4f}")
print(f"y = {y:.4f}")
print(f"L = {L:.6f}")
print()

print("Backward pass:")
print(f"∂L/∂y = {dL_dy:.4f}")
print(f"∂L/∂h = {dL_dh:.4f}")
print(f"∂L/∂w2 = {dL_dw2:.4f}")
print(f"∂L/∂b2 = {dL_db2:.4f}")
print(f"∂L/∂w1 = {dL_dw1:.4f}")
print(f"∂L/∂b1 = {dL_db1:.4f}")
print()