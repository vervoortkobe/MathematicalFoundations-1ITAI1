import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)
np.random.seed(42)

print("Libraries geladen!")

# Implementeer een gradient_check functie die de analytische gradiënt vergelijkt met de numerieke gradiënt. De functie moet het relatieve verschil teruggeven.

def gradient_check(f, x, analytic_grad, h=1e-5):
    """
    Vergelijk analytische gradiënt met numerieke gradiënt.
    
    Parameters:
    - f: functie die een scalar teruggeeft
    - x: punt waar we de gradiënt controleren
    - analytic_grad: analytisch berekende gradiënt
    - h: stapgrootte voor numerieke gradiënt
    
    Returns:
    - relative_error: relatieve fout tussen beide gradiënten
    """
    numerical_grad = (f(x + h) - f(x - h)) / (2 * h)
    relative_error = np.abs((analytic_grad - numerical_grad) / analytic_grad)
    return relative_error

# Test met f(x) = x²
def f(x):
    return x**2

def test_gradient_check():
    x = 2
    analytic_grad = 4*x
    relative_error = gradient_check(f, x, analytic_grad)
    print(f"Relative error: {relative_error:.4f}")

test_gradient_check()