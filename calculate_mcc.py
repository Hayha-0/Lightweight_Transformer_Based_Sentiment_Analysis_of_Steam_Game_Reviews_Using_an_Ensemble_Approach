import pandas as pd
from sklearn.metrics import matthews_corrcoef
from pathlib import Path

BASE = Path(r"C:\Users\User\Documents\SteamRecommendationProject\scripts\results")

FILES = {
    "EuroBERT-210M": BASE / "eurobert210m" / "predictions eurobert.csv",
    "MiniCPM4-0.5B": BASE / "minicpm4_05b" / "predictions minicpm4.csv",
    "Qwen3-0.6B": BASE / "qwen06b" / "predictions qwen.csv",
    "Gemma 3 270M": BASE / "gemma270m" / "predictions gemma.csv",
    "Ensemble": BASE / "xai" / "ensemble_predictions.csv",
}

results = []

for model_name, file_path in FILES.items():

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    if not file_path.exists():
        print(f"ERROR: File not found:\n{file_path}")
        continue

    df = pd.read_csv(file_path)

    print("Columns found:")
    print(list(df.columns))

    # Individual model prediction files
    if "true_label" in df.columns and "pred_label" in df.columns:
        y_true = df["true_label"]
        y_pred = df["pred_label"]

    # Try common alternatives for ensemble file
    else:
        true_candidates = [
            "true_label", "label", "y_true",
            "actual", "ground_truth"
        ]

        pred_candidates = [
            "pred_label", "predicted_label",
            "prediction", "y_pred",
            "ensemble_pred", "ensemble_prediction"
        ]

        true_col = next(
            (col for col in true_candidates if col in df.columns),
            None
        )

        pred_col = next(
            (col for col in pred_candidates if col in df.columns),
            None
        )

        if true_col is None or pred_col is None:
            print("\nCould not identify true/prediction columns.")
            print("Paste the columns above here.")
            continue

        print(f"Using true labels: {true_col}")
        print(f"Using predictions: {pred_col}")

        y_true = df[true_col]
        y_pred = df[pred_col]

    # Remove missing values if any
    valid = pd.DataFrame({
        "true": y_true,
        "pred": y_pred
    }).dropna()

    mcc = matthews_corrcoef(
        valid["true"],
        valid["pred"]
    )

    print(f"\nMCC = {mcc:.4f}")

    results.append({
        "Model": model_name,
        "MCC": round(mcc, 4)
    })

print("\n" + "=" * 60)
print("FINAL MCC RESULTS")
print("=" * 60)

if results:
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    output_path = BASE / "mcc_results.csv"
    results_df.to_csv(output_path, index=False)

    print(f"\nSaved to:\n{output_path}")

print("\nDONE")