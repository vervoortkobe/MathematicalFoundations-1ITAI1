import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)
print("Libraries geladen!")

# Bereken de afgeleide met de kettingregel voor:
# h'(x) = f'(g(x)) * g'(x)

# 1. f(x) = (2x + 1)³
# g(x) = 2x + 1
# f(x) = u³, waarbij u = g(x)
# f'(u) = 3u²
# g'(x) = 2
# h'(x) = f'(g(x)) * g'(x) = 3(2x + 1)² * 2 = 6(2x + 1)²

# 2. g(x) = e^(x²)
# f(x) = e^u, waarbij u = x²
# f'(u) = e^u * du/dx = e^u * 2x
# g'(x) = 2x
# h'(x) = f'(g(x)) * g'(x) = e^(x²) * 2x

# 3. h(x) = sin(3x) (afgeleide van sin is cos)
# f(x) = sin(u), waarbij u = 3x
# f'(u) = cos(u) * du/dx = cos(u) * 3
# g(x) = 3x
# h'(x) = f'(g(x)) * g'(x) = cos(3x) * 3

# 4. k(x) = √(x² + 1)
# f(x) = √u, waarbij u = x² + 1
# f'(u) = (1/2)u^(-1/2) * du/dx = (1/2)(x² + 1)^(-1/2) * 2x
# g(x) = x² + 1
# h'(x) = f'(g(x)) * g'(x) = (1/2)(x² + 1)^(-1/2) * 2x

# Implementeer de functies en hun afgeleiden, en verifieer numeriek.
def numerical_derivative(f, x, h=1e-5):
  return (f(x + h) - f(x)) / h

f = lambda x: (2*x + 1)**3
g = lambda x: np.exp(x**2)
h = lambda x: np.sin(3*x)
k = lambda x: np.sqrt(x**2 + 1)
f_prime = lambda x: 6 * (2*x + 1)**2
g_prime = lambda x: np.exp(x**2) * 2*x
h_prime = lambda x: np.cos(3*x) * 3
k_prime = lambda x: (1/2) * (x**2 + 1)**(-1/2) * 2*x

x_values = [1, 2, 3]
for x in x_values:
  print(f"h'({x}) = {h_prime(x):.4f}, numeriek: {numerical_derivative(h, x):.4f}")
  print(f"k'({x}) = {k_prime(x):.4f}, numeriek: {numerical_derivative(k, x):.4f}")
  print()