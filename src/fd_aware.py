import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from category_encoders import HashingEncoder
from sklearn.preprocessing import StandardScaler

# Target and features
# Predicting the country name
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

# Extract features and target
X = df[FEATURES]
y = df[TARGET]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Reset indices for clean alignment
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

# Encode target labels as integers
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

# Identify categorical and numeric columns
categorical = X_train.columns[X_train.dtypes == "object"].tolist()
numerical = ["Age"]

# Hash categorical features
encoder = HashingEncoder(n_components=128)
X_train_cat = encoder.fit_transform(X_train[categorical])
X_test_cat = encoder.transform(X_test[categorical])

# Scale numeric features
scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train[numerical])
X_test_num = scaler.transform(X_test[numerical])

# Combine hashed categorical and scaled numeric features
X_train_final = np.hstack([X_train_cat.values, X_train_num])
X_test_final = np.hstack([X_test_cat.values, X_test_num])

# Convert to PyTorch tensors
Xtr = torch.tensor(X_train_final, dtype=torch.float32)
ytr = torch.tensor(y_train_enc, dtype=torch.long)

Xte = torch.tensor(X_test_final, dtype=torch.float32)
yte = torch.tensor(y_test_enc, dtype=torch.long)

# Store LHS attribute for FD regularization
country_codes = X_train["Airport Country Code"].values

# FD-aware MLP model
class FDMLP(nn.Module):
    """Simple 3-layer MLP classifier."""
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# Encourages predictions to be consistent within each Airport Country Code group.
def fd_loss(logits, country_codes):
    probs = torch.softmax(logits, dim=1)
    loss = 0.0

    # Unique LHS values Airport Country Codes
    unique_codes = np.unique(country_codes)

    for code in unique_codes:
        # Indices of rows with this code
        # Continue, no FD constraint for single-row groups
        idx = np.where(country_codes == code)[0]
        if len(idx) <= 1:
            continue  
        group_probs = probs[idx]
        mean_prob = group_probs.mean(dim=0)

        # Penalize deviation from group mean distribution
        loss += ((group_probs - mean_prob) ** 2).mean()

    return loss / len(unique_codes)

def train(model, fd_weight=0.0, epochs=20):
    opt = optim.Adam(model.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()

    for _ in range(epochs):
        opt.zero_grad()

        logits = model(Xtr)

        # Standard cross-entropy loss
        loss = ce(logits, ytr)

        # Add FD regularization if enabled
        if fd_weight > 0:
            loss += fd_weight * fd_loss(logits, country_codes)

        loss.backward()
        opt.step()

# Measures how often predictions violate the FD:
# Airport Country Code → Country Name
def fd_violation_rate(preds, codes):
    dfp = pd.DataFrame({
        "code": codes,
        "pred": preds
    })

    violations = 0
    for _, g in dfp.groupby("code"):
        if g["pred"].nunique() > 1:
            violations += 1

    return violations / dfp["code"].nunique()

# Train baselne and FD-aware models
input_dim = Xtr.shape[1]
num_classes = len(le.classes_)

# Baseline model no FD regularization
baseline = FDMLP(input_dim, num_classes)
train(baseline, fd_weight=0.0)

# FD-aware model with FD regularization
fd_model = FDMLP(input_dim, num_classes)
train(fd_model, fd_weight=1.0)

# Models evaluation
with torch.no_grad():
    base_preds = baseline(Xte).argmax(dim=1).numpy()
    fd_preds = fd_model(Xte).argmax(dim=1).numpy()

print("Baseline accuracy:", accuracy_score(yte, base_preds))
print("FD-aware accuracy:", accuracy_score(yte, fd_preds))

print("Baseline FD violation rate:",
      fd_violation_rate(base_preds, X_test["Airport Country Code"].values))

print("FD-aware FD violation rate:",
      fd_violation_rate(fd_preds, X_test["Airport Country Code"].values))
