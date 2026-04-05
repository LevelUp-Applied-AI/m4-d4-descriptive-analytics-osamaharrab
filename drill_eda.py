"""Core Skills Drill — Descriptive Analytics

Compute summary statistics, plot distributions, and create a correlation
heatmap for the sample sales dataset.

Usage:
    python drill_eda.py
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def compute_summary(df):
    """Compute summary statistics for all numeric columns."""
    numeric_df = df.select_dtypes(include=[np.number])

    summary = pd.DataFrame({
        "count": numeric_df.count(),
        "mean": numeric_df.mean(),
        "median": numeric_df.median(),
        "std": numeric_df.std(),
        "min": numeric_df.min(),
        "max": numeric_df.max(),
    }).T

    summary.to_csv("output/summary.csv")
    return summary


def plot_distributions(df, columns, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, column in zip(axes, columns):
        sns.histplot(df[column], kde=True, ax=ax)
        ax.set_title(f"Distribution of {column}")
        ax.set_xlabel(column)
        ax.set_ylabel("Frequency")

    for ax in axes[len(columns):]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)

def plot_correlation(df, output_path):
    """Compute Pearson correlation matrix and visualize as a heatmap."""
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr(method="pearson")

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    """Load data, compute summary, and generate all plots."""
    os.makedirs("output", exist_ok=True)

    df = pd.read_csv("data/sample_sales.csv")

    compute_summary(df)

    numeric_columns = df.select_dtypes(include=[np.number]).columns[:4].tolist()
    plot_distributions(df, numeric_columns, "output/distributions.png")

    plot_correlation(df, "output/correlation.png")


if __name__ == "__main__":
    main()