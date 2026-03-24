import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)
print("Libraries geladen!")

def numerical_derivative(f, x, h=1e-5):
  return (f(x + h) - f(x)) / h

# Bereken de afgeleide analytisch (met de hand) voor de volgende functies:
# f(x) = 5x⁴
# f'(x) = 20x³

# g(x) = x³ - 2x² + 4x - 1
# g'(x) = 3x² - 4x + 4

# h(x) = 3/x (hint: schrijf als 3x⁻¹)
# h'(x) = -3x⁻²

# k(x) = √x (hint: schrijf als x^(1/2))
# k'(x) = (1/2)x^(-1/2)

# m(x) = 2eˣ + 3x²
# m'(x) = 2eˣ + 6x

# Implementeer dan zowel de originele functies als hun afgeleiden in Python en verifieer met numerieke afgeleiden.

# Definieer de functies en hun afgeleiden
f = lambda x: 5 * x**4
g = lambda x: x**3 - 2 * x**2 + 4 * x - 1
h = lambda x: 3 / x
k = lambda x: x**0.5
m = lambda x: 2 * np.exp(x) + 3 * x**2

# Bereken de analytische afgeleiden
f_prime = lambda x: 20 * x**3
g_prime = lambda x: 3 * x**2 - 4 * x + 4
h_prime = lambda x: -3 / x**2
k_prime = lambda x: 0.5 / x**0.5
m_prime = lambda x: 2 * np.exp(x) + 6 * x

# Verifieer met numerieke afgeleiden
x_values = [1, 2, 3]
for x in x_values:
  print(f"f'({x}) = {f_prime(x):.4f}, numeriek: {numerical_derivative(f, x):.4f}")
  print(f"g'({x}) = {g_prime(x):.4f}, numeriek: {numerical_derivative(g, x):.4f}")
  print(f"h'({x}) = {h_prime(x):.4f}, numeriek: {numerical_derivative(h, x):.4f}")
  print(f"k'({x}) = {k_prime(x):.4f}, numeriek: {numerical_derivative(k, x):.4f}")
  print(f"m'({x}) = {m_prime(x):.4f}, numeriek: {numerical_derivative(m, x):.4f}")
  print()
