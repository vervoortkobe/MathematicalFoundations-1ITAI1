import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)
print("Libraries geladen!")

# Implementeer een functie numerical_derivative(f, x, h=1e-5) die de afgeleide van f op punt x numeriek benadert met de forward difference methode: (f(x+h) - f(x)) / h.
def numerical_derivative(f, x, h=1e-5):
    return (f(x + h) - f(x)) / h

# Test de functie met f(x) = sin(x) op x = pi/4 en vergelijk met de analytische afgeleide cos(pi/4).
f = np.sin
x = np.pi / 4
numerical_deriv = numerical_derivative(f, x)
analytic_deriv = np.cos(x)
print(f"Numeriek: {numerical_deriv:.4f}, Analytisch: {analytic_deriv:.4f}")