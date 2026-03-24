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