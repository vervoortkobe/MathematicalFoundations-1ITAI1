import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.set_printoptions(precision=4, suppress=True)
np.random.seed(42)
print("Libraries geladen!")

# Implementeer gradient descent voor een 1D functie. Het algoritme is:
def gradient_descent_1d(f, df_dx, x_init, learning_rate, n_iterations):
    """Voer gradient descent uit voor een 1D functie.
    
    Returns:
        x_final: de gevonden x waarde
        history: lijst van alle x waarden tijdens optimalisatie
    """
    # x t+1 = x t - eta * df/dx (x t)
    # x = x_init - learning_rate * df_dx(x_init)
    x_final = x_init
    history = [x_final]
    for i in range(n_iterations):
        grad = df_dx(x_final)
        x_final = x_final - learning_rate * grad
        history.append(x_final)
    return x_final, history

# Test op f(x) = x^2 - 4x + 5 (minimum bij x=2).
# Functie en afgeleide
def f(x):
    return x**2 - 4*x + 5

def df_dx(x):
    return 2*x - 4

# Test
x_final, history = gradient_descent_1d(f, df_dx, x_init=6.0, learning_rate=0.1, n_iterations=50)
print(f"Gevonden minimum: x = {x_final:.4f}")
print(f"Verwacht minimum: x = 2")