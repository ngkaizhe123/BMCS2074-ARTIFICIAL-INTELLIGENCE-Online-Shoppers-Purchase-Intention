"""
run_pipeline.py
----------------
Runs the EDA (src/eda.py) and preprocessing (src/data_preprocessing.py)
pipelines end to end and writes the cleaned dataset to:
data/processed/cleaned_online_shoppers_intention.csv
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier

from eda import run_eda
from data_preprocessing import (
    run_preprocessing_pipeline,
    preprocess_data,
    build_preprocessor,
    get_smotenc,
)
from utils import save_cleaned_dataset, evaluate_model, print_metrics

# ── Paths setup ─────────────────────────────────────────────────────────────
data_path = project_root / "data" / "raw" / "online_shoppers_intention.csv"
plots_dir = project_root / "report_assets" / "plots" / "eda"
processed_path = (
    project_root / "data" / "processed" / "cleaned_online_shoppers_intention.csv"
)

# ── STEP 1: EDA ─────────────────────────────────────────────────────────────
print("\n" + "#" * 70)
print("# STEP 1: EDA")
print("#" * 70)
df_raw = run_eda(filepath=data_path, save_dir=plots_dir, show=False)

# ── STEP 2: PREPROCESSING ──────────────────────────────────────────────────
print("\n" + "#" * 70)
print("# STEP 2: PREPROCESSING & SAVE TO CSV")
print("#" * 70)

# Run the preprocessing pipeline to clean the full dataset
df_clean = run_preprocessing_pipeline(df_raw, outlier_method="none")

# Save to the single cleaned CSV file in data/processed/
save_cleaned_dataset(df_clean, processed_path)

# ── STEP 3: DEMO PIPELINE (SMOTENC -> Preprocessor -> Classifier) ────────────
print("\n" + "#" * 70)
print("# STEP 3: DEMO PIPELINE (SMOTENC -> Preprocessor -> Classifier)")
print("#" * 70)

# Retrieve train/test splits dynamically using preprocess_data
X_train, X_test, y_train, y_test, preprocessor = preprocess_data(
    filepath=processed_path,  # Load from the freshly saved cleaned CSV
    outlier_method="none",
    test_size=0.2,
    random_state=42,
    transform=False,
    scale_numerical=False,
)

print(f"X_train: {X_train.shape}   X_test: {X_test.shape}")
print(
    f"y_train purchase rate: {y_train.mean():.4f}   y_test purchase rate: {y_test.mean():.4f}"
)

# SMOTENC MUST be placed BEFORE preprocessor so it sees original Month, VisitorType,
# and Weekend string/bool values instead of one-hot encoded binary columns.
demo_pipeline = ImbPipeline(
    steps=[
        ("smotenc", get_smotenc(X_train)),
        ("preprocess", build_preprocessor(scale_numerical=False)),
        ("clf", RandomForestClassifier(random_state=42, n_estimators=200, n_jobs=-1)),
    ]
)

demo_pipeline.fit(X_train, y_train)

# Evaluate using src.utils helper
metrics = evaluate_model(demo_pipeline, X_test, y_test)
print_metrics("RandomForest + SMOTENC Baseline", metrics)

print("\n" + "=" * 70)
print(f"[OK] [run_pipeline] Done!")
print(f"   Cleaned dataset saved to : {processed_path.resolve()}")
print(f"   Plots saved to           : {plots_dir.resolve()}")
print("=" * 70)
