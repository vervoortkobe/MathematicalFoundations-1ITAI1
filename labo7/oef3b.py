import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)
np.random.seed(42)

print("Libraries geladen!")

class Sigmoid:
    def forward(self, x):
        self.x = x
        self.output = 1 / (1 + np.exp(-x))
        return self.output
    def backward(self, dout):
        # dL/dx = dout * dsigmoid/dx
        # dsigmoid/dx = s * (1 - s) waar s = sigmoid(x)
        s = self.output
        dx = dout * s * (1 - s)
        return dx

class Tanh:
    def forward(self, x):
        self.x = x
        self.output = np.tanh(x)
        return self.output
    def backward(self, dout):
        # dL/dx = dout * dtanh/dx
        # dtanh/dx = 1 - tanh(x)^2
        t = self.output
        dx = dout * (1 - t**2)
        return dx

# Implementeer een gradient_check functie die de analytische gradiënt vergelijkt met de numerieke gradiënt. De functie moet het relatieve verschil teruggeven.

def gradient_check(f, x, analytic_grad, h=1e-5):
    """
    Vergelijk analytische gradiënt met numerieke gradiënt.
    Deze versie werkt voor zowel scalars als arrays (element-wise).
    """
    numerical_grad = (f(x + h) - f(x - h)) / (2 * h)
    # Gebruik een kleine epsilon in de deler om deling door nul te voorkomen
    relative_error = np.abs(analytic_grad - numerical_grad) / (np.maximum(1e-10, np.abs(analytic_grad) + np.abs(numerical_grad)))
    return relative_error

# Gebruik gradient checking om te verifiëren dat je sigmoid backward implementatie correct is.
# We moeten een functie definiëren voor de forward pass om de numerieke gradiënt te kunnen berekenen.
# Omdat Sigmoid element-gewijs werkt, kunnen we het voor een array x tegelijk doen
# zolang de functie die we doorgeven element-gewijs werkt.

print("\n=== Verificatie Sigmoid Backward ===")
x = np.array([-2.0, 0.0, 2.0])
sigmoid = Sigmoid()

# 1. Bereken de analytische gradiënt
# We simuleren een loss functie L = sum(sigmoid(x))
# Dan is dL/dsigmoid = 1 voor alle elementen
out = sigmoid.forward(x)
dout = np.ones_like(out)
analytic_grad = sigmoid.backward(dout)

# 2. Gedefinieerde forward functie voor numerieke gradiënt (puur wiskundig)
def sigmoid_func(z):
    return 1 / (1 + np.exp(-z))

# We willen de gradiënt van de som van de sigmoid waardes: L = sum(sigmoid(x))
def loss_func(z):
    return np.sum(sigmoid_func(z))

# Voor elementwise functies kunnen we de gradiënt van elk element apart controleren.
# De numerieke afgeleide van sigmoid(x) t.o.v. x is (sigmoid(x+h) - sigmoid(x-h)) / 2h.
numerical_grad = (sigmoid_func(x + 1e-5) - sigmoid_func(x - 1e-5)) / (2 * 1e-5)

# Bereken de relatieve fout
rel_error = gradient_check(sigmoid_func, x, analytic_grad)

print(f"Input x: {x}")
print(f"Analytic gradient:  {analytic_grad}")
print(f"Numerical gradient: {numerical_grad}")
print(f"Relative errors:    {rel_error}")

# Check of alle fouten klein genoeg zijn (meestal < 1e-7)
if np.all(rel_error < 1e-7):
    print("Sigmoid backward implementatie is CORRECT!")
else:
    print("Sigmoid backward implementatie is fout.")

# Verifieer ook Tanh voor de volledigheid
print("\n=== Verificatie Tanh Backward ===")
tanh_layer = Tanh()
out_tanh = tanh_layer.forward(x)
dout_tanh = np.ones_like(out_tanh)
analytic_grad_tanh = tanh_layer.backward(dout_tanh)

def tanh_func(z):
    return np.tanh(z)

rel_error_tanh = gradient_check(tanh_func, x, analytic_grad_tanh)
print(f"Tanh Relative errors: {rel_error_tanh}")
if np.all(rel_error_tanh < 1e-7):
    print("Tanh backward implementatie is CORRECT!")
else:
    print("Tanh backward implementatie is fout.")