import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

# Load data
data = pd.read_csv('UCI_Credit_Card.csv', sep=';')

# If the first row contains the real column names, fix them
if data.iloc[0].astype(str).tolist()[:5] == ['ID', 'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE']:
    data.columns = data.iloc[0]
    data = data.iloc[1:].reset_index(drop=True)

# Standardize column names if the file uses X1, X2, ... or similar
expected_columns = [
    'ID', 'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
    'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
    'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
    'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
    'default.payment.next.month'
]

if len(data.columns) == len(expected_columns):
    data.columns = expected_columns

# Convert everything possible to numeric
for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

print(data.head())
print(data.info())

# Correlation heatmap
numeric_data = data.select_dtypes(include=[np.number])
corr = numeric_data.corr()

plt.figure(figsize=(18, 15))
sns.heatmap(corr, annot=True, fmt='.2f', vmin=-1.0, vmax=1.0, cmap='mako', linewidths=0.5)
plt.title('Correlation')
plt.tight_layout()
plt.show()

# Preprocessing
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

    # Ensure target is the last column name we expect
    target_col = 'default.payment.next.month'
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Check the CSV format and column names.")

    df = onehotencode(
        df,
        {
            'EDUCATION': 'EDU',
            'MARRIAGE': 'MAR'
        }
    )

    y = df[target_col].copy()
    X = df.drop(target_col, axis=1).copy()

    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(X.median(numeric_only=True))

    return X, y

X, y = cleaner(data)
print(X.head())
print(y.head())

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    train_size=0.7,
    random_state=1234,
    stratify=y
)

# Scale only on training data to avoid leakage
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

# Train models
models = {
    LogisticRegression(max_iter=2000, random_state=1234): "Logistic Regression",
    SVC(random_state=1234): "Support Vector Machine",
    MLPClassifier(max_iter=1000, random_state=1234): "Neural Network"
}

for model in models.keys():
    model.fit(X_train, y_train)

for model, name in models.items():
    print(f"{name}: {model.score(X_test, y_test)*100:.2f}%")

# ROC and AUC
model_svc = SVC(kernel='rbf', random_state=1234)
model_svc.fit(X_train, y_train)
y_pred_svm = model_svc.decision_function(X_test)

model_logistic = LogisticRegression(max_iter=2000, random_state=1234)
model_logistic.fit(X_train, y_train)
y_pred_logistic = model_logistic.decision_function(X_test)

logistic_fpr, logistic_tpr, _ = roc_curve(y_test, y_pred_logistic)
auc_logistic = auc(logistic_fpr, logistic_tpr)

svm_fpr, svm_tpr, _ = roc_curve(y_test, y_pred_svm)
auc_svm = auc(svm_fpr, svm_tpr)

plt.figure(figsize=(5, 5), dpi=100)
plt.plot(svm_fpr, svm_tpr, linestyle='-', label=f'SVM (auc = {auc_svm:.3f})')
plt.plot(logistic_fpr, logistic_tpr, marker='.', label=f'Logistic (auc = {auc_logistic:.3f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.tight_layout()
plt.show()