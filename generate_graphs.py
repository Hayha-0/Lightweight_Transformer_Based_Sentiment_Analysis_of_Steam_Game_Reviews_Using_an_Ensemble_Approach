"""
generate_graphs.py
Run this AFTER all 4 train_*.py scripts have finished. It reads each
results/<model>/metrics.json and produces the cross-model comparison figures
required by the project spec: a metric comparison bar chart, inference time
comparison, memory usage comparison, and training time comparison.

Run:  python generate_graphs.py
"""

import os
import json
import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = "results"
GRAPHS_DIR = "graphs"

MODEL_DISPLAY_NAMES = {
    "eurobert210m": "EuroBERT-210M",
    "gemma270m": "Gemma-3-270M",
    "qwen06b": "Qwen3-0.6B",
}


def load_all_metrics():
    rows = []
    for folder, display_name in MODEL_DISPLAY_NAMES.items():
        path = os.path.join(RESULTS_DIR, folder, "metrics.json")
        if not os.path.exists(path):
            print(f"WARNING: {path} not found — skipping {display_name}. "
                  f"Did train_{folder}.py finish successfully?")
            continue
        with open(path) as f:
            m = json.load(f)
        m["display_name"] = display_name
        rows.append(m)
    if not rows:
        raise RuntimeError("No metrics.json files found in results/*/. Run the training scripts first.")
    return pd.DataFrame(rows)


def bar_chart(df, column, title, ylabel, filename, pct=False):
    plt.figure(figsize=(7, 5))
    vals = df[column] * 100 if pct else df[column]
    bars = plt.bar(df["display_name"], vals, color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=15)
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.2f}",
                  ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, filename), dpi=150)
    plt.close()


def grouped_metric_chart(df, filename="model_comparison_metrics.png"):
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    x = range(len(df))
    width = 0.15
    plt.figure(figsize=(9, 5))
    for i, metric in enumerate(metrics):
        plt.bar([p + i * width for p in x], df[metric], width=width, label=metric)
    plt.xticks([p + 2 * width for p in x], df["display_name"], rotation=15)
    plt.ylabel("Score")
    plt.title("Model Comparison — Test Set Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPHS_DIR, filename), dpi=150)
    plt.close()


def main():
    os.makedirs(GRAPHS_DIR, exist_ok=True)
    df = load_all_metrics()
    df.to_csv(os.path.join(GRAPHS_DIR, "all_models_summary.csv"), index=False)

    grouped_metric_chart(df)
    bar_chart(df, "avg_inference_ms_per_sample", "Inference Time Comparison",
              "ms / sample", "inference_time_comparison.png")
    if "peak_gpu_memory_MB" in df.columns and df["peak_gpu_memory_MB"].notna().any():
        bar_chart(df, "peak_gpu_memory_MB", "Peak GPU Memory Usage Comparison",
                  "MB", "memory_usage_comparison.png")
    if "training_wall_time_sec" in df.columns:
        df["training_wall_time_min"] = df["training_wall_time_sec"] / 60
        bar_chart(df, "training_wall_time_min", "Training Time Comparison",
                  "minutes", "training_time_comparison.png")

    print("Summary:")
    print(df[["display_name", "accuracy", "precision", "recall", "f1", "roc_auc"]].to_string(index=False))
    print(f"\nBest model by F1: {df.loc[df['f1'].idxmax(), 'display_name']}")
    print("Figures saved to:", GRAPHS_DIR)


if __name__ == "__main__":
    main()
