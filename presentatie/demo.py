import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuraties voor a en b en waar in de subplot-grid ze moeten komen
CONFIGS = [
    {"a": 0,   "b": 1,    "row": 0, "col": 0, "note": "normaal"},
    {"a": -2,  "b": 1,    "row": 1, "col": 0, "note": "verschuiving x-as"},
    {"a": 0,   "b": 0.55, "row": 0, "col": 1, "note": "uitgestrekte sigmoïde"},
    {"a": 0,   "b": 3,    "row": 1, "col": 1, "note": "zeer steile sigmoïde"},
]


def sigmoid(x: float, b0: float, b1: float) -> float:
    """Standaard logistische functie met parameters b0 (intercept) en b1 (slope)."""
    return 1.0 / (1.0 + math.e ** -(b0 + b1 * x))


def make_sigmoid_df(a: float, b: float, x_min: float, x_max: float, n_points: int = 200) -> pd.DataFrame:
    """Genereer een DataFrame met x-waarden en bijhorende sigmoïde y-waarden."""
    x_values = np.linspace(x_min, x_max, n_points)
    y_values = [sigmoid(x, a, b) for x in x_values]
    return pd.DataFrame({"x": x_values, "y": y_values})


def main():
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)

    for cfg in CONFIGS:
        df = make_sigmoid_df(cfg["a"], cfg["b"], -10, 10)
        ax = axes[cfg["row"]][cfg["col"]]

        sns.lineplot(data=df, x="x", y="y", ax=ax)
        title = f"a = {cfg['a']}  |  b = {cfg['b']}  ({cfg['note']})"
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("P(y=1 | x)")

    fig.suptitle("Effect van a en b op de sigmoïde", fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()