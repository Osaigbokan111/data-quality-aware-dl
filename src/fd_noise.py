import pandas as pd
import numpy as np

# This function injects FD violations by randomly shuffling the RHS
# values for a subset of rows determined by violation_rate.
# It is intentionally lightweight and fast because it operates only
# on the target column (Country Name) rather than the full dataset.
def inject_fd_violations_fast(df, lhs, rhs, violation_rate, rng):
    # If no noise requested, return unchanged
    if violation_rate == 0.0:
        return df

    idx = df.index
    n = int(len(idx) * violation_rate)

    # If noise rate too small to affect any row
    if n == 0:
        return df

    # Randomly select rows where FD will be violated
    violated_idx = rng.choice(idx, n, replace=False)

    # Shuffle RHS values to break the FD
    shuffled_rhs = (
        df.loc[idx, rhs]
        .sample(frac=1.0, random_state=rng.randint(1_000_000))
        .values
    )

    # Apply the shuffled RHS to the selected rows
    df.loc[violated_idx, rhs] = shuffled_rhs[:n]
    return df

# Predicting Country Name using airport + passenger metadata.
TARGET = "Country Name"

FEATURES = [
    "Airport Country Code",
    "Airport Name",
    "Airport Continent",
    "Continents",
    "Arrival Airport",
    "Nationality",
    "Age"
]

# Split into X (features) and y (target)
X = df[FEATURES]
y = df[TARGET]

# TRAIN / TEST SPLIT
from sklearn.model_selection import train_test_split

# Stratified split ensures target distribution is preserved
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Reset indices for clean alignment
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# Convert Country Name to integer labels for ML models
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

# HashingEncoder avoids one-hot explosion for high-cardinality columns.
from category_encoders import HashingEncoder
from sklearn.preprocessing import StandardScaler

# Identify categorical and numeric columns
categorical = X_train.columns[X_train.dtypes == "object"].tolist()
numerical = ["Age"]

# Hash categorical features
encoder = HashingEncoder(n_components=128)
X_train_hashed = encoder.fit_transform(X_train[categorical])
X_test_hashed = encoder.transform(X_test[categorical])

# Scale numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train[numerical])
X_test_scaled = scaler.transform(X_test[numerical])

# Combine hashed categorical + scaled numeric into final matrices
X_train_final = np.hstack([X_train_hashed.values, X_train_scaled])
X_test_final = np.hstack([X_test_hashed.values, X_test_scaled])

from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# MLP neural network classifier
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    learning_rate="adaptive",
    learning_rate_init=0.001,
    max_iter=150,
    early_stopping=True,
    n_iter_no_change=5,
    validation_fraction=0.1,
    random_state=42,
)

# XGBoost classifier
xgb = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
)

from sklearn.metrics import accuracy_score, f1_score

# Evaluate model performance on encoded labels
def evaluate(model, X_test, y_test_enc):
    preds = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test_enc, preds),
        "f1": f1_score(y_test_enc, preds, average="weighted"),
    }

# Test multiple FD violation levels and measure robustness
results = []
rng = np.random.RandomState(42)

for noise in [0.0, 0.05, 0.1, 0.2, 0.5]:

    # Copy clean labels
    y_train_noisy = y_train_enc.copy()

    # Inject FD noise into target labels
    if noise > 0:
        tmp = pd.DataFrame({
            "Airport Country Code": X_train["Airport Country Code"],
            "Country Name": le.inverse_transform(y_train_noisy)
        })

        tmp = inject_fd_violations_fast(
            tmp,
            lhs=["Airport Country Code"],
            rhs="Country Name",
            violation_rate=noise,
            rng=rng
        )

        # Re-encode noisy labels
        y_train_noisy = le.transform(tmp["Country Name"])

    # Train both models on the same noisy labels
    mlp.fit(X_train_final, y_train_noisy)
    xgb.fit(X_train_final, y_train_noisy)

    # Evaluate and store results
    res = {
        "noise": noise,
        "mlp": evaluate(mlp, X_test_final, y_test_enc),
        "xgb": evaluate(xgb, X_test_final, y_test_enc),
    }

    results.append(res)
    print(res)

# Plot results
import matplotlib.pyplot as plt

noise_levels = [r["noise"] for r in results]
mlp_acc = [r["mlp"]["accuracy"] for r in results]
xgb_acc = [r["xgb"]["accuracy"] for r in results]

plt.plot(noise_levels, mlp_acc, marker="o", label="MLP (Adam)")
plt.plot(noise_levels, xgb_acc, marker="s", label="XGBoost")
plt.xlabel("FD Violation Rate")
plt.ylabel("Accuracy")
plt.title("FD Violations vs Accuracy (FAST VERSION)")
plt.legend()
plt.grid(True)
plt.show()
