import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)
print("Libraries geladen!")

# Bereken de partiële afgeleiden ∂f/∂x en ∂f/∂y voor:
# 1. f(x, y) = x² + y²
# df/dx = 2x
# df/dy = 2y

# 2. f(x, y) = 3xy + x - 2y
# df/dx = 3y + 1
# df/dy = 3x - 2

# 3. f(x, y) = e^(xy)
# df/dx = y * e^(xy)
# df/dy = x * e^(xy)

# 4. f(x, y) = x²y³
# df/dx = 2xy³
# df/dy = 3x²y²


# Implementeer en verifieer numeriek.
# Numerieke partiële afgeleiden helper
def numerical_partial_x(f, x, y, h=1e-5):
    return (f(x + h, y) - f(x, y)) / h

def numerical_partial_y(f, x, y, h=1e-5):
    return (f(x, y + h) - f(x, y)) / h

# Test de functies
def f1(x, y):
    return x**2 + y**2

def f2(x, y):
    return 3*x*y + x - 2*y

def f3(x, y):
    return np.exp(x*y)

def f4(x, y):
    return x**2 * y**3

x_test, y_test = 1.0, 2.0
print("Numerieke partiële afgeleiden:")
print(f"f1: df/dx = {numerical_partial_x(f1, x_test, y_test):.4f}, df/dy = {numerical_partial_y(f1, x_test, y_test):.4f}")
print(f"f2: df/dx = {numerical_partial_x(f2, x_test, y_test):.4f}, df/dy = {numerical_partial_y(f2, x_test, y_test):.4f}")
print(f"f3: df/dx = {numerical_partial_x(f3, x_test, y_test):.4f}, df/dy = {numerical_partial_y(f3, x_test, y_test):.4f}")
print(f"f4: df/dx = {numerical_partial_x(f4, x_test, y_test):.4f}, df/dy = {numerical_partial_y(f4, x_test, y_test):.4f}")