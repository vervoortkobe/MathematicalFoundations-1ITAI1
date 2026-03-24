import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)
np.random.seed(42)

print("Libraries geladen!")

# Implementeer de sigmoid en tanh activatiefuncties met forward en backward.

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

# Test
print("=== Sigmoid test ===")
x_s = np.array([-2.0, 0.0, 2.0])
sigmoid = Sigmoid()

z_s = sigmoid.forward(x_s)
print("forward(x) =", z_s)

# Simuleer dat dL/dz = 1.0 overal
dout_s = np.array([1.0, 1.0, 1.0])
dx_s = sigmoid.backward(dout_s)
print("backward (dL/dx) =", dx_s)

print("\n=== Tanh test ===")
x_t = np.array([-2.0, 0.0, 2.0])
tanh = Tanh()

z_t = tanh.forward(x_t)
print("forward(x) =", z_t)

# weer dL/dz = 1.0
dout_t = np.array([1.0, 1.0, 1.0])
dx_t = tanh.backward(dout_t)
print("backward (dL/dx) =", dx_t)
