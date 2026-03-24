import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)
print("Libraries geladen!")

# Implementeer een functie numerical_derivative(f, x, h=1e-5) die de afgeleide van f op punt x numeriek benadert met de forward difference methode: (f(x+h) - f(x)) / h.
def numerical_derivative(f, x, h=1e-5):
    return (f(x + h) - f(x)) / h

# Implementeer ook de central difference methode: (f(x+h) - f(x-h)) / (2h). Vergelijk de nauwkeurigheid met de forward difference methode voor dezelfde h-waarden.
def central_difference(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)

# Test je functie op f(x) = x³ op het punt x = 2. De analytische afgeleide is f’(x) = 3x², dus f’(2) = 12.
f = lambda x: x**3
x = 2
numerical_deriv_forward = numerical_derivative(f, x)
numerical_deriv_central = central_difference(f, x)
analytic_deriv = 3 * x**2
print(f"Numeriek (forward): {numerical_deriv_forward:.4f}, Analytisch: {analytic_deriv:.4f}")
print(f"Numeriek (central): {numerical_deriv_central:.4f}, Analytisch: {analytic_deriv:.4f}")