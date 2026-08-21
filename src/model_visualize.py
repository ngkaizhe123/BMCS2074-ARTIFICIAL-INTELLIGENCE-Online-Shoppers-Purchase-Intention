"""
model_visualize.py
------------------
Evaluates all trained models in saved_models/, saves their metrics to a JSON file,
and generates static visualization charts (Metric Comparisons, Confusion Matrices, ROC Curves)
to report_assets/plots/ for the Streamlit dashboard.
"""

import json
import sys
import textwrap
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.ticker as ticker

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data_preprocessing import preprocess_data
from src.utils import evaluate_model, plot_confusion_matrix, plot_roc_curve

SAVED_DIR = project_root / "saved_models"
PLOT_DIR = project_root / "report_assets" / "plots"
METRICS_PATH = project_root / "report_assets" / "metrics.json"

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[*] Scanning {SAVED_DIR} for models...")
    models = {}
    if SAVED_DIR.exists():
        for pkl in sorted(SAVED_DIR.glob("*.pkl")):
            skip_names = {"preprocessor", "scaler"}
            stem = pkl.stem.lower()
            if any(s in stem for s in skip_names):
                continue
            nice_name = pkl.stem.replace("_", " ").title()
            models[nice_name] = {
                "path": pkl,
                "stem": pkl.stem.split("_")[0],
            }

    if not models:
        print("[!] No trained models found. Exiting.")
        return

    print(f"[*] Found {len(models)} models. Loading test data...")
    X_train, X_test, y_train, y_test, _ = preprocess_data(
        filepath=str(project_root / "data" / "raw" / "online_shoppers_intention.csv"),
        transform=False,
    )

    all_metrics = {}
    
    sns.set_theme(style="whitegrid")
    
    # ── 1. EVALUATE MODELS & GENERATE INDIVIDUAL PLOTS ──────────────────────
    print("Generating individual model diagnostics (Confusion Matrix & ROC)...")
    for name, info in models.items():
        print(f"    -> Evaluating {name}...")
        try:
            model = joblib.load(info["path"])
            metrics = evaluate_model(model, X_test, y_test)

            cm = metrics["Confusion Matrix"]
            all_metrics[name] = {
                "stem": info["stem"],
                "Accuracy": float(metrics["Accuracy"]),
                "Precision": float(metrics["Precision"]),
                "Recall": float(metrics["Recall"]),
                "F1 Score": float(metrics["F1"]),
                "AUC": float(metrics["AUC"]) if metrics["AUC"] is not None else None,
                "Confusion Matrix": cm.tolist() if hasattr(cm, "tolist") else cm,
                "Classification Report": metrics["Classification Report"],
            }
            
            safe_name = info["stem"]
            
            # Confusion Matrix
            fig_cm = plot_confusion_matrix(model, X_test, y_test)
            fig_cm.axes[0].set_title(f"Confusion Matrix ({name})", fontweight="bold")
            fig_cm.savefig(PLOT_DIR / f"confusion_matrix_{safe_name}.png", dpi=300, bbox_inches="tight")
            plt.close(fig_cm)
            
            # ROC Curve
            if metrics["AUC"] is not None:
                fig_roc = plot_roc_curve(model, X_test, y_test)
                fig_roc.axes[0].set_title(f"ROC Curve ({name})", fontweight="bold")
                fig_roc.savefig(PLOT_DIR / f"roc_curve_{safe_name}.png", dpi=300, bbox_inches="tight")
                plt.close(fig_roc)
                
        except Exception as e:
            print(f"    [!] Error evaluating {name}: {e}")

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=4)
    print(f"[*] Metrics successfully saved to {METRICS_PATH}")

    # ── 2. MODEL PERFORMANCE COMPARISONS (CHARTS) ──────────────────────────
    print("Generating Model Comparison Charts...")
    df_metrics = pd.DataFrame(all_metrics).T
    if df_metrics.empty:
        print("[!] No metrics computed. Skipping charts.")
        return
        
    wrapped_labels = [textwrap.fill(str(label), width=12) for label in df_metrics.index]

    # Chart: Multi-metric Comparison (Accuracy, Precision, Recall, F1)
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1 Score"]
    
    df_plot = df_metrics[metrics_to_plot].astype(float).reset_index().melt(id_vars="index", var_name="Metric", value_name="Score")
    df_plot.rename(columns={"index": "Model"}, inplace=True)
    
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=df_plot, x="Model", y="Score", hue="Metric", palette="Set2", ax=ax)
    
    ax.set_xticks(range(len(wrapped_labels)))
    ax.set_xticklabels(wrapped_labels, rotation=0, ha="center")
    
    ax.set_xlabel("Models", fontweight="bold")
    ax.set_ylabel("Score (0.0 - 1.0)", fontweight="bold")
    ax.set_title("Model Comparison - Primary Classification Metrics", fontweight="bold", fontsize=14)
    ax.set_ylim(0, 1.15)
    ax.legend(title="Metric", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f"{height:.3f}",
                (p.get_x() + p.get_width() / 2., height),
                ha='center', va='bottom',
                xytext=(0, 4),
                textcoords='offset points',
                fontsize=8,
                rotation=90
            )

    plt.tight_layout()
    plt.savefig(PLOT_DIR / "model_comparison_metrics.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✅ Saved model_comparison_metrics.png")

    # Chart: AUC Comparison
    if "AUC" in df_metrics.columns and not df_metrics["AUC"].isnull().all():
        fig, ax = plt.subplots(figsize=(10, 7))
        auc_data = df_metrics["AUC"].dropna().astype(float)
        valid_labels = [textwrap.fill(str(l), width=12) for l in auc_data.index]
        
        sns.barplot(
            x=valid_labels,
            y=auc_data.values,
            hue=valid_labels,
            palette="magma",
            legend=False,
            ax=ax,
        )
        ax.set_xlabel("Models", fontweight="bold")
        ax.set_ylabel("ROC AUC Score", fontweight="bold")
        ax.set_title(
            "Model Comparison - ROC AUC (Higher is Better)",
            fontweight="bold", fontsize=14
        )
        
        ax.set_ylim(0, max(auc_data.values) * 1.15)
        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                ax.annotate(
                    f"{height:.4f}",
                    (p.get_x() + p.get_width() / 2.0, height),
                    ha="center", va="bottom",
                    xytext=(0, 5), textcoords="offset points",
                    fontsize=10
                )
        plt.tight_layout()
        plt.savefig(PLOT_DIR / "model_comparison_auc.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("✅ Saved model_comparison_auc.png")

    print("\n✅ All visualization charts generated successfully!")

if __name__ == "__main__":
    main()
