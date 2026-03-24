import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)
print("Libraries geladen!")

# Implementeer een functie numerical_derivative(f, x, h=1e-5) die de afgeleide van f op punt x numeriek benadert met de forward difference methode: (f(x+h) - f(x)) / h.
def numerical_derivative(f, x, h=1e-5):
    return (f(x + h) - f(x)) / h

# Test je functie op f(x) = x³ op het punt x = 2. De analytische afgeleide is f’(x) = 3x², dus f’(2) = 12.
f = lambda x: x**3
x = 2
numerical_deriv = numerical_derivative(f, x)
analytic_deriv = 3 * x**2
print(f"Numeriek: {numerical_deriv:.4f}, Analytisch: {analytic_deriv:.4f}")