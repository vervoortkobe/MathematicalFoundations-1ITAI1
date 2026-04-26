import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# x-as: LIMIT_BAL (gebruik geschaalde waarden voor een mooiere grafiek)
limit_values = np.linspace(10000, 300000, 400)
x = (limit_values - limit_values.mean()) / limit_values.std()

# PAY_0 als klantkenmerk: 0 = geen achterstand, 1 = lichte achterstand, 2 = grotere achterstand
pay0 = 1

# Gewicht van PAY_0 in de lineaire score
pay_weight = 1.0

# Vier scenario's: rechte z = a*x + b + pay_weight*PAY_0
scenarios = {
    "Normale sigmoïde": {"a": 1.0, "b": 0.0, "color": "navy"},
    "Uitgestrekte sigmoïde": {"a": 0.35, "b": 0.0, "color": "darkorange"},
    "Verschoven sigmoïde": {"a": 1.0, "b": -1.5, "color": "forestgreen"},
    "Zeer steile sigmoïde": {"a": 3.0, "b": 0.0, "color": "crimson"},
}

fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
axes = axes.flatten()

for ax, (name, s) in zip(axes, scenarios.items()):
    a = s["a"]
    b = s["b"]
    color = s["color"]

    z = a * x + pay_weight * pay0 + b
    p = sigmoid(z)

    ax.plot(limit_values, z, linestyle="--", color="gray", linewidth=2, label="rechte z")
    ax.axhline(0, color="gray", linestyle=":", alpha=0.7)
    ax.set_title(name)
    ax.set_xlabel("LIMIT_BAL")
    ax.set_ylabel("Lineaire score z")
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(limit_values, p, color=color, linewidth=3, label="kans p(default)")
    ax2.axhline(0.5, color=color, linestyle=":", alpha=0.5)
    ax2.set_ylabel("Kans op default")
    ax2.set_ylim(0, 1)

    ax.legend(loc="upper left")
    ax2.legend(loc="lower right")

plt.suptitle("Rechtes en kansen voor vier logistische scenario's", fontsize=16, y=1.02)
for ax in axes[:2]:
    ax.tick_params(axis='x', labelbottom=True)
plt.tight_layout()
plt.show()