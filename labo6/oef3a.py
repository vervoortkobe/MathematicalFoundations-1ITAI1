import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.set_printoptions(precision=4, suppress=True)
np.random.seed(42)
print("Libraries geladen!")

# Breid gradient descent uit naar 2D. Minimaliseer f(x, y) = (x - 2)^2 + (y - 3)^2 (minimum bij (2, 3)).
def gradient_descent_2d(f, grad_f, xy_init, learning_rate, n_iterations):
    """Voer gradient descent uit voor een 2D functie.
    
    Args:
        f: functie f(x, y)
        grad_f: gradiënt functie die [∂f/∂x, ∂f/∂y] teruggeeft
        xy_init: startpunt [x, y]
        learning_rate: stapgrootte
        n_iterations: aantal iteraties
    
    Returns:
        xy_final: gevonden minimum [x, y]
        history: lijst van alle [x, y] waarden
    """
    xy_final = np.array(xy_init)
    history = [xy_final]
    for _ in range(n_iterations):
        grad = grad_f(xy_final[0], xy_final[1])
        xy_final = xy_final - learning_rate * grad
        history.append(xy_final)
    return xy_final, history

# Functie en gradiënt
def f_2d(x, y):
    return (x - 2)**2 + (y - 3)**2

def grad_f_2d(x, y):
    return np.array([2*(x - 2), 2*(y - 3)])

# Test
xy_final, history = gradient_descent_2d(f_2d, grad_f_2d, xy_init=[-1, 6], 
                                         learning_rate=0.1, n_iterations=50)
print(f"Gevonden minimum: ({xy_final[0]:.4f}, {xy_final[1]:.4f})")
print(f"Verwacht minimum: (2, 3)")