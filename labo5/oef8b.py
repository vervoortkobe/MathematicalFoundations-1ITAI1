import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)
print("Libraries geladen!")

# Beschouw een mini-netwerk met twee neuronen in serie:
# h = σ(w₁x + b₁) (hidden neuron)
# y = σ(w₂h + b₂) (output neuron)

# Bereken ∂y/∂w₁ met de kettingregel. Je zult merken dat dit afhangt van ∂y/∂h en ∂h/∂w₁.
def sigmoid(z):
	return 1 / (1 + np.exp(-z))

# Parameters
w1 = 0.7
b1 = -0.2
w2 = 1.3
b2 = 0.5
x = 0.8

# Voorwaartse propagatie
z1 = w1 * x + b1
h = sigmoid(z1)
z2 = w2 * h + b2
y = sigmoid(z2)

# Afgeleiden
# σ'(z) = σ(z) * (1 - σ(z))
def sigmoid_deriv(z):
	s = sigmoid(z)
	return s * (1 - s)

# Kettingregel:
# ∂y/∂w₁ = ∂y/∂h * ∂h/∂w₁
# ∂y/∂h = w₂ * σ'(z₂)
# ∂h/∂w₁ = x * σ'(z₁)

dy_dh = w2 * sigmoid_deriv(z2)
dh_dw1 = x * sigmoid_deriv(z1)
dy_dw1 = dy_dh * dh_dw1

print(f"y = {y:.4f}")
print(f"∂y/∂w₁ = {dy_dw1:.4f}")

# Implementeer het mini-netwerk en bereken alle gradiënten:
# ∂y/∂w₂, ∂y/∂b₂
# ∂y/∂w₁, ∂y/∂b₁
# Verifieer numeriek.

# Analytische gradiënten
dy_dw2 = h * sigmoid_deriv(z2)
dy_db2 = sigmoid_deriv(z2)
dy_db1 = dy_dh * sigmoid_deriv(z1)

print(f"∂y/∂w₂ = {dy_dw2:.4f}")
print(f"∂y/∂b₂ = {dy_db2:.4f}")
print(f"∂y/∂b₁ = {dy_db1:.4f}")

# Numerieke verificatie (finite differences)
def forward(x, w1, b1, w2, b2):
	z1 = w1 * x + b1
	h = sigmoid(z1)
	z2 = w2 * h + b2
	y = sigmoid(z2)
	return y

eps = 1e-5

# ∂y/∂w₂
num_dy_dw2 = (forward(x, w1, b1, w2 + eps, b2) - forward(x, w1, b1, w2, b2)) / eps
# ∂y/∂b₂
num_dy_db2 = (forward(x, w1, b1, w2, b2 + eps) - forward(x, w1, b1, w2, b2)) / eps
# ∂y/∂w₁
num_dy_dw1 = (forward(x, w1 + eps, b1, w2, b2) - forward(x, w1, b1, w2, b2)) / eps
# ∂y/∂b₁
num_dy_db1 = (forward(x, w1, b1 + eps, w2, b2) - forward(x, w1, b1, w2, b2)) / eps

print("\nNumerieke gradiënten (finite differences):")
print(f"∂y/∂w₂ ≈ {num_dy_dw2:.4f}")
print(f"∂y/∂b₂ ≈ {num_dy_db2:.4f}")
print(f"∂y/∂w₁ ≈ {num_dy_dw1:.4f}")
print(f"∂y/∂b₁ ≈ {num_dy_db1:.4f}")