import matplotlib.pyplot as plt
import seaborn as sns
import math
import pandas as pd
import numpy as np

configs = [
    { 'a': 0,  'b': 1,    'cellx': 0, 'celly': 0, 'note': 'normaal' },
    { 'a': -2, 'b': 1,    'cellx': 1, 'celly': 0, 'note': 'verschuiving x-as' },
    { 'a': 0,  'b': 0.55, 'cellx': 0, 'celly': 1, 'note': 'uitgestrekte sigmoide' },
    { 'a': 0,  'b': 3,    'cellx': 1, 'celly': 1, 'note': 'zeer steile sigmoide' }
]

def sigmoide_b0_b1(x, b0, b1):
    return 1 / (1 + (math.e ** -(b0 + b1*x)))

def adapted_sigmoid(a, b, r1, r2):
    x_values = np.linspace(r1, r2, 100)
    y_values = [sigmoide_b0_b1(n, a, b) for n in x_values]
    return pd.DataFrame({ 'x': x_values, 'y': y_values })

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

def plot(i):
    df = adapted_sigmoid(configs[i]['a'], configs[i]['b'], -10, 10)
    axis = axes[configs[i]['cellx']][configs[i]['celly']]
    axis.set_title(f'a = {configs[i]['a']}  en  b = {configs[i]['b']}    ({configs[i]['note']})')
    sns.lineplot(data=df, x="x", y="y", ax=axis)

for i in range(0, 4):
    plot(i)

plt.tight_layout()
plt.show()