import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# =========================
# 1. DATA LOADEN
# =========================
data = pd.read_csv("UCI_Credit_Card.csv", sep=";")

if data.iloc[0].astype(str).tolist()[:5] == ['ID', 'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE']:
    data.columns = data.iloc[0]
    data = data.iloc[1:].reset_index(drop=True)

expected_columns = [
    'ID', 'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
    'default.payment.next.month'
]

if len(data.columns) == len(expected_columns):
    data.columns = expected_columns

for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors="coerce")

# =========================
# 2. PREPROCESSING
# =========================
def onehotencode(df, columndict):
    df = df.copy()
    for column, prefix in columndict.items():
        dummies = pd.get_dummies(df[column], prefix=prefix, dtype=int)
        df = pd.concat([df, dummies], axis=1)
        df = df.drop(column, axis=1)
    return df

def cleaner(df):
    df = df.copy()
    if 'ID' in df.columns:
        df = df.drop('ID', axis=1)

    target_col = 'default.payment.next.month'
    df = onehotencode(df, {
        'EDUCATION': 'EDU',
        'MARRIAGE': 'MAR'
    })

    y = df[target_col].copy()
    X = df.drop(target_col, axis=1).copy()
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(X.median(numeric_only=True))
    return X, y

X, y = cleaner(data)

# =========================
# 3. TRAINING
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, random_state=1234, stratify=y
)

scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

model = LogisticRegression(max_iter=2000, random_state=1234)
model.fit(X_train, y_train)

print(f"Train accuracy: {model.score(X_train, y_train)*100:.2f}%")
print(f"Test accuracy:  {model.score(X_test, y_test)*100:.2f}%")

# =========================
# 4. INTERACTIEVE DEMO
#    2 parameters aanpassen
# =========================
"""
Scenario	LIMIT_BAL	PAY_0	Verwachte uitleg
Veilig	200000	0	Hoge kredietlimiet, geen achterstand, dus lagere default-kans.
Midden	50000	1	Gematigd risico, model reageert gematigd.
Risicovol	20000	2	Lage limiet + achterstand, dus hogere default-kans.

- Als PAY_0 stijgt, neemt het risico meestal toe.
- Als LIMIT_BAL lager is, zie je vaak een andere kansinschatting.
- Ik toon nu drie situaties: veilig, gemiddeld en risicovol.
- Zo zie je meteen hoe het model reageert op veranderende input.
"""
limit_balance = 200000   # <-- pas aan: 20000 / 50000 / 200000
pay_0 = 0                # <-- pas aan: 0 (op tijd betaald) / 1 (1 maand achterstallig) / 2 (2 maanden achterstallig)

# vaste voorbeeldwaarden voor de andere features
demo = X.median(numeric_only=True).to_frame().T
demo["LIMIT_BAL"] = limit_balance
demo["PAY_0"] = pay_0

# zorg dat alle kolommen aanwezig zijn
for col in X.columns:
    if col not in demo.columns:
        demo[col] = 0

demo = demo[X.columns]
demo_scaled = pd.DataFrame(scaler.transform(demo), columns=X.columns)

prob_default = model.predict_proba(demo_scaled)[0, 1]
prediction = int(prob_default >= 0.5)

print("\n=== DEMO INPUT ===")
print(f"LIMIT_BAL = {limit_balance}")
print(f"PAY_0     = {pay_0}")
print("\n=== RESULTAAT ===")
print(f"Voorspelde kans op default: {prob_default*100:.2f}%")
print(f"Klassevoorspelling: {'DEFAULT' if prediction == 1 else 'GEEN DEFAULT'}")

# =========================
# 5. VISUELE GRAFIEK
# =========================
limit_values = np.linspace(max(10000, limit_balance - 100000), limit_balance + 100000, 200)
probs = []

for lb in limit_values:
    row = X.median(numeric_only=True).to_frame().T
    row["LIMIT_BAL"] = lb
    row["PAY_0"] = pay_0
    for col in X.columns:
        if col not in row.columns:
            row[col] = 0
    row = row[X.columns]
    row_scaled = pd.DataFrame(scaler.transform(row), columns=X.columns)
    probs.append(model.predict_proba(row_scaled)[0, 1])

plt.figure(figsize=(10, 6))
sns.lineplot(x=limit_values, y=probs, color="navy", linewidth=2)
plt.axvline(limit_balance, color="red", linestyle="--", label=f"Jouw LIMIT_BAL = {limit_balance}")
plt.axhline(prob_default, color="green", linestyle="--", label=f"Kans = {prob_default*100:.2f}%")
plt.scatter([limit_balance], [prob_default], color="black", zorder=5)
plt.title("Effect van LIMIT_BAL op de voorspelde default-kans")
plt.xlabel("LIMIT_BAL")
plt.ylabel("Voorspelde kans op default")
plt.legend()
plt.tight_layout()
plt.show()