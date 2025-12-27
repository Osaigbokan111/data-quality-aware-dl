import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# This function injects functional dependency violations
# directly into the *target labels* instead of the full dataset.
# It breaks the FD: Airport Country Code → Country Name
# by randomly shuffling the labels for a subset of rows.
def inject_fd_noise_on_target(y_enc, X_lhs, le, violation_rate, rng):
    """
    Break FD: Airport Country Code -> Country Name
    by shuffling labels for a subset of rows
    """
    # No noise → return clean labels
    if violation_rate == 0.0:
        return y_enc.copy()

    # Number of rows to corrupt
    n = int(len(y_enc) * violation_rate)
    if n == 0:
        return y_enc.copy()

    # Randomly choose rows to violate FD
    idx = np.arange(len(y_enc))
    violated_idx = rng.choice(idx, n, replace=False)

    # Shuffle labels and assign to selected rows
    y_noisy = y_enc.copy()
    shuffled = rng.permutation(y_noisy)
    y_noisy[violated_idx] = shuffled[violated_idx]

    return y_noisy

# Predicting Country Name
TARGET = "Country Name"

# Full feature set contains FD RHS columns
FULL_FEATURES = [
    "Airport Country Code",
    "Airport Name",
    "Arrival Airport",
    "Airport Continent",
    "Continents",
    "Nationality",
    "Age"
]

# Canonical feature set
# FD LHS
# weak auxiliary signal
CANONICAL_FEATURES = [
    "Airport Country Code",  
    "Age"                    
]

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from category_encoders import HashingEncoder
from sklearn.preprocessing import StandardScaler

# Extract X and y
X = df[FULL_FEATURES]
y = df[TARGET]

# Stratified split to preserve target distribution
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

# Encode target labels
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)


# Hash categorical features + scale numeric features
# Returns NumPy matrices ready for ML models
def preprocess(X_train, X_test):
    # Identify categorical and numeric columns
    categorical = X_train.columns[X_train.dtypes == "object"].tolist()
    numerical = ["Age"]

    # HashingEncoder avoids one-hot explosion
    encoder = HashingEncoder(n_components=128)
    scaler = StandardScaler()

    # Transform categorical features
    Xtr_cat = encoder.fit_transform(X_train[categorical])
    Xte_cat = encoder.transform(X_test[categorical])

    # Scale numeric features
    Xtr_num = scaler.fit_transform(X_train[numerical])
    Xte_num = scaler.transform(X_test[numerical])

    # Combine hashed categorical + scaled numeric
    return (
        np.hstack([Xtr_cat.values, Xtr_num]),
        np.hstack([Xte_cat.values, Xte_num]),
    )


from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# Build fresh model instances for each experiment run
def build_models():
    # Neural network classifier
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        solver="adam",
        learning_rate="adaptive",
        max_iter=150,
        early_stopping=True,
        n_iter_no_change=5,
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
    return mlp, xgb


from sklearn.metrics import accuracy_score

# FD noise levels to test
noise_levels = [0.0, 0.05, 0.1, 0.2, 0.5]

# Random generator for reproducibility
rng = np.random.RandomState(42)

results = []

# Run experiment for both feature sets
for feature_name, FEATURES in [
    ("FULL", FULL_FEATURES),
    ("CANONICAL", CANONICAL_FEATURES),
]:

    # Preprocess features ONCE per feature set
    Xtr, Xte = preprocess(X_train[FEATURES], X_test[FEATURES])

    # Loop over FD noise levels
    for noise in noise_levels:

        # Inject FD noise into target labels
        y_train_noisy = inject_fd_noise_on_target(
            y_train_enc,
            X_train["Airport Country Code"],
            le,
            noise,
            rng
        )

        # Build fresh models
        mlp, xgb = build_models()

        # Train models on noisy labels
        mlp.fit(Xtr, y_train_noisy)
        xgb.fit(Xtr, y_train_noisy)

        # Evaluate on clean test set
        res = {
            "features": feature_name,
            "noise": noise,
            "mlp_acc": accuracy_score(y_test_enc, mlp.predict(Xte)),
            "xgb_acc": accuracy_score(y_test_enc, xgb.predict(Xte)),
        }

        results.append(res)
        print(res)



# Plot results
res_df = pd.DataFrame(results)

# Plot accuracy curves for both models and both feature sets
for model in ["mlp_acc", "xgb_acc"]:
    for feat in ["FULL", "CANONICAL"]:
        subset = res_df[res_df["features"] == feat]
        plt.plot(
            subset["noise"],
            subset[model],
            marker="o",
            label=f"{model.upper()} - {feat}"
        )

plt.xlabel("FD Violation Rate")
plt.ylabel("Accuracy")
plt.title("Experiment 2: Canonical vs Full Features under FD Noise")
plt.legend()
plt.grid(True)
plt.show()
