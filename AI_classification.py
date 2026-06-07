# =============================================================================
# DecodeLabs | Batch 2026
# Project 2: Handwritten Digit Classification
# Algorithm : Random Forest
# Dataset   : Digits Dataset
# Pipeline  : Load → Scale → Split → Train → Predict → Evaluate
# =============================================================================

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score
)

# =============================================================================
# PHASE 1 — INPUT
# =============================================================================

digits = load_digits()

X = digits.data
y = digits.target

class_names = [str(i) for i in digits.target_names]

print("=" * 60)
print(" DecodeLabs | Project 2 — Digit Classification")
print("=" * 60)

print("\n[DATASET]")

print(f" Total Samples : {X.shape[0]}")
print(f" Features      : {X.shape[1]}")
print(f" Classes       : {class_names}")

# Scaling object
scaler = StandardScaler()

# =============================================================================
# PHASE 2 — PROCESS
# =============================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    shuffle=True,
    random_state=42
)

print("\n[SPLIT]")
print(f" Training Samples : {len(X_train)}")
print(f" Testing Samples  : {len(X_test)}")

# Feature Scaling
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =============================================================================
# RANDOM FOREST MODEL
# =============================================================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train_scaled, y_train)

print("\n[MODEL]")
print(" Algorithm : Random Forest")
print(" Trees     : 100")
print(" Training  : Complete")

# Predictions
predictions = model.predict(X_test_scaled)

# =============================================================================
# PHASE 3 — OUTPUT
# =============================================================================

accuracy = accuracy_score(y_test, predictions)

f1 = f1_score(
    y_test,
    predictions,
    average="weighted"
)

cm = confusion_matrix(
    y_test,
    predictions
)

report = classification_report(
    y_test,
    predictions
)

print("\n" + "=" * 60)
print(" RESULTS ")
print("=" * 60)

print(f"\n[ACCURACY]  {accuracy * 100:.2f}%")

print(f"\n[F1 SCORE]  {f1:.4f}")

print("\n[CONFUSION MATRIX]")
print(cm)

print("\n[CLASSIFICATION REPORT]")
print(report)

# =============================================================================
# SAMPLE PREDICTIONS
# =============================================================================

print("=" * 60)
print(" SAMPLE TEST PREDICTIONS ")
print("=" * 60)

for i in range(10):

    actual = y_test[i]

    predicted = predictions[i]

    status = "✓" if actual == predicted else "✗"

    print(
        f" Sample {i+1:2} | "
        f"Actual: {actual} | "
        f"Predicted: {predicted} | "
        f"{status}"
    )

print("\nDecodeLabs | Project 2 Complete")
print("=" * 60)