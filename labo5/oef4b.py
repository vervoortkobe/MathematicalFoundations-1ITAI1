import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)
print("Libraries geladen!")

# Gegeven f(x, y, z) = x²y + yz² + xz, bereken alle drie de partiële afgeleiden en de gradiënt ∇f. Evalueer op het punt (1, 2, 3).
def f(x, y, z):
    return x**2 * y + y * z**2 + x * z

# Partiële afgeleiden
# ∂f/∂x = 2xy + z
# ∂f/∂y = x² + z²
# ∂f/∂z = 2yz + x

def partial_x(x, y, z):
    return 2 * x * y + z

def partial_y(x, y, z):
    return x**2 + z**2

def partial_z(x, y, z):
    return 2 * y * z + x

# Evalueer op het punt (1, 2, 3)
x, y, z = 1.0, 2.0, 3.0
print("Partiële afgeleiden:")
print(f"∂f/∂x = {partial_x(x, y, z):.4f}")
print(f"∂f/∂y = {partial_y(x, y, z):.4f}")
print(f"∂f/∂z = {partial_z(x, y, z):.4f}")

print("Gradiënt ∇f:")
print(f"∇f = ({partial_x(x, y, z):.4f}, {partial_y(x, y, z):.4f}, {partial_z(x, y, z):.4f})")