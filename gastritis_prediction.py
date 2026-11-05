
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    KFold, StratifiedKFold, LeaveOneOut, cross_validate
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# 1. Load dataset
path = r"C:\Users\likit\Downloads\smart_gastritis_risk_prediction_synthetic_dataset.csv"
df = pd.read_csv(path)

target = "Gastritis_Risk_Label"

if target not in df.columns:
    raise ValueError("Gastritis_Risk_Label column not found in dataset.")

df = df.dropna(subset=[target]).drop_duplicates()

id_cols = [
    c for c in df.columns
    if c.lower() in ["patient_id", "id", "unnamed: 0"]
]

X = df.drop(columns=[target] + id_cols)
y = df[target].astype(str)

# 2. Prepare numerical and categorical features
num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()

preprocessor = ColumnTransformer([
    ("numeric", SimpleImputer(strategy="median"), num_cols),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_cols)
])

# 3. Random Forest model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ))
])

# 4. Set up cross-validation methods
minimum_class_count = y.value_counts().min()
k = min(5, int(minimum_class_count))

if k < 2:
    raise ValueError(
        "At least 2 records are required in every risk class "
        "for stratified cross-validation."
    )

methods = {
    f"K-Fold ({k} folds)": KFold(
        n_splits=k, shuffle=True, random_state=42
    ),
    f"Stratified K-Fold ({k} folds)": StratifiedKFold(
        n_splits=k, shuffle=True, random_state=42
    ),
    "LOOCV": LeaveOneOut()
}

# 5. Compare cross-validation methods
results = []

print("\nGASTRITIS RISK - CROSS-VALIDATION COMPARISON")
print("Algorithm: Random Forest Classifier")
print("-" * 55)

for name, cv in methods.items():
    print(f"\nEvaluating {name}...")

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=["accuracy", "f1_weighted"],
        n_jobs=1
    )

    accuracy = scores["test_accuracy"].mean()
    f1 = scores["test_f1_weighted"].mean()

    results.append({
        "Method": name,
        "Accuracy (%)": accuracy * 100,
        "Weighted F1 (%)": f1 * 100
    })

    print(f"Mean Accuracy: {accuracy * 100:.2f}%")
    print(f"Mean Weighted F1: {f1 * 100:.2f}%")

results_df = pd.DataFrame(results)

print("\nFINAL COMPARISON")
print(results_df.round(2).to_string(index=False))

# 6. Plot comparison
plot_df = results_df.set_index("Method")

plot_df.plot(kind="bar", figsize=(10, 6))
plt.title("Random Forest: Cross-Validation Comparison")
plt.ylabel("Score (%)")
plt.xlabel("Cross-Validation Method")
plt.ylim(0, 105)
plt.xticks(rotation=15, ha="right")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# 7. Train final model on the complete dataset
model.fit(X, y)

# 8. Predict risk for a new patient
print("\nNEW PATIENT RISK PREDICTION")
print("Press Enter to use a default value.")

patient = {}

for col in X.columns:
    if col in num_cols:
        default = X[col].median()
        value = input(f"{col} (default {default}): ").strip()

        try:
            patient[col] = float(value) if value else default
        except ValueError:
            print("Invalid number. Using default.")
            patient[col] = default

    else:
        choices = X[col].dropna().astype(str).unique().tolist()
        mode_values = X[col].mode()
        default = str(mode_values.iloc[0]) if not mode_values.empty else ""

        print("Available categories:", choices)
        value = input(f"{col} (default {default}): ").strip()
        patient[col] = value if value else default

new_patient = pd.DataFrame([patient], columns=X.columns)

predicted_risk = model.predict(new_patient)[0]

print("\nPREDICTION RESULT")
print("Predicted Gastritis Risk:", predicted_risk)

if hasattr(model, "predict_proba"):
    probabilities = model.predict_proba(new_patient)[0]
    classes = model.named_steps["classifier"].classes_

    print("\nPredicted class probabilities:")
    for label, probability in zip(classes, probabilities):
        print(f"{label}: {probability * 100:.2f}%")

print("\nNote: This is a dataset-based prediction, not a medical diagnosis.")
