"""
Q4 - Task 1: Experimental Protocol  ->  ALL THREE DATASETS
    housing_features.csv  (Case A)
    gene_features.csv     (Case B)
    sensor_features.csv   (Case C)

Running this single script performs Task 1 end-to-end for all three
datasets in one go. For each dataset we:
    1. Generate a correlation heatmap to inspect collinearity / structure.
    2. Train OLS, Ridge, and Lasso models across a logarithmic grid of
       lambda (regularization strength) values.
    3. Record, for every (model, lambda) configuration:
           - Test MSE
           - Test R^2
           - The full coefficient vector (beta)

Unlike Q1/Q2, the lab explicitly allows scikit-learn for the
regularization sections (Q3 onward), so we use sklearn's Ridge/Lasso
instead of deriving closed-form solutions by hand.

NOTE ON LAMBDA vs ALPHA:
    The lab calls the regularization strength "lambda". scikit-learn's
    Ridge/Lasso classes call the same quantity "alpha" - same number, same
    penalty term, just a different name. We keep the variable name `lam`
    throughout to match the lab's notation, and pass it in as `alpha=lam`
    when constructing sklearn models.

WHY ONE SHARED FUNCTION INSTEAD OF COPY-PASTING THE STEPS 3 TIMES:
    Task 1's procedure is IDENTICAL across all three datasets - only the
    file path and train/test split size differ. Writing the sweep logic
    once (in `run_task1_experiment` below) and calling it three times
    means there's only one place to fix bugs or tweak the lambda grid,
    and all three datasets stay perfectly consistent with each other.

OUTPUT (written to the current working directory when you run this script):
    q4_housing_corr_heatmap.png   q4_housing_results.csv
    q4_gene_corr_heatmap.png      q4_gene_results.csv
    q4_sensor_corr_heatmap.png    q4_sensor_results.csv
Each *_results.csv has one row per (model, lambda) with Test MSE, Test R2,
intercept, and one coefficient column per feature - this is what Task 2's
Case A/B/C questions will be answered from.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# LAMBDA GRID
# ---------------------------------------------------------------------------
# The lab's Task 1 spec gives this exact example sequence:
#       lambda in {0, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100}
# We hardcode it explicitly (rather than generating it with np.geomspace)
# for one important reason: several Case Study questions in Task 2 ask
# about SPECIFIC lambda values - e.g. Case A asks about "Lasso at lambda=10",
# Case B asks to compare "lambda=0.3 vs lambda=1.0", and Case C asks about
# "Ridge at lambda=30" and "Lasso at lambda=1". If we used an *approximate*
# geometric grid instead, none of those exact values would necessarily land
# in our results table, forcing us to re-fit one-off models later. Hardcoding
# the lab's own example sequence guarantees every value Task 2 will ask
# about is already sitting in this grid.
LAMBDA_GRID = np.array([0.0, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0])


def plot_correlation_heatmap(df, title, save_path):
    """
    Computes and saves a correlation heatmap for `df` (expected to contain
    both feature columns and the target column, so we can visually inspect
    both feature-feature collinearity and feature-target relationships in
    a single figure).
    """
    corr_matrix = df.corr()

    # Figure size scales gently with the number of variables so that the
    # gene dataset (24 features + target = 25x25) doesn't render as an
    # unreadable, tiny annotated grid the way a fixed 7x6 figure would.
    n = corr_matrix.shape[0]
    figsize = (max(7, 0.35 * n), max(6, 0.3 * n))

    plt.figure(figsize=figsize)
    sns.heatmap(
        corr_matrix,
        annot=(n <= 12),     # only print numeric values if the grid is small
                              # enough to stay readable; for the 25x25 gene
                              # matrix, numbers would be illegible clutter -
                              # color alone is enough to spot the block there.
        fmt=".2f",
        cmap="coolwarm",      # diverging colormap: blue = negative, red = positive
        vmin=-1, vmax=1,      # fix the color scale to the full correlation range
        square=True,
        cbar_kws={"shrink": 0.8},
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved correlation heatmap -> {save_path}")


def evaluate_model(model, model_name, lam, X_train, y_train, X_test, y_test, feature_names):
    """
    Fits `model` on the training data, evaluates it on the test data, and
    returns a dict (one "row") capturing everything Task 1 asks us to record:
    Test MSE, Test R^2, and the full coefficient vector.

    We deliberately do NOT record Train MSE/R^2 here, since the lab's Task 1
    spec only requires Test MSE and Test R^2 - but train metrics can be added
    later (e.g. for Case A's baseline-comparison question) by reusing this
    same function and predicting on X_train instead.
    """
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)

    row = {
        "model": model_name,
        "lambda": lam,
        "test_mse": mean_squared_error(y_test, y_pred_test),
        "test_r2": r2_score(y_test, y_pred_test),
        # model.intercept_ is w0 (not part of the "coefficient vector" beta,
        # but kept around for sanity checks / later questions).
        "intercept": model.intercept_,
    }
    # Spread the coefficients into named columns (coef_feature_1, coef_feature_2, ...)
    # rather than storing them as a single list, so the resulting CSV is easy
    # to filter/sort by any individual feature's coefficient in Task 2.
    for name, coef in zip(feature_names, model.coef_):
        row[f"coef_{name}"] = coef

    return row


def run_task1_experiment(csv_path, dataset_label, test_size, random_state=42, lasso_max_iter=20000):
    """
    Runs the full Q4 Task 1 protocol for ONE dataset:
        1. Load CSV (assumes the last column is "target", all others are features).
        2. Plot + save a correlation heatmap.
        3. Train/test split.
        4. Sweep OLS (lambda=0) / Ridge / Lasso across LAMBDA_GRID.
        5. Save + return the results as a tidy DataFrame.

    Parameters
    ----------
    csv_path : str
        Path to the dataset's CSV file.
    dataset_label : str
        Short name used for print statements, plot titles, and output
        filenames (e.g. "housing", "gene", "sensor").
    test_size : float
        Fraction of rows held out for testing. Passed straight to
        train_test_split. We let the CALLER decide this per-dataset because
        the three datasets have very different sample sizes (5000 vs 80 vs
        100 rows) and a single fixed value wouldn't suit all of them well.
    random_state : int
        Seed for the train/test split, fixed for reproducibility - this
        matters because Case A/B/C's follow-up questions (stress tests,
        two-stage refits, hybrid pipelines) all build on this exact split.
    lasso_max_iter : int
        Iteration budget for Lasso's coordinate-descent solver. Raised above
        sklearn's default (1000) to avoid convergence warnings, especially
        at small lambda values where the solver needs more steps to settle.

    Returns
    -------
    results_df : pd.DataFrame
        One row per (model, lambda) configuration with Test MSE, Test R^2,
        intercept, and per-feature coefficients.
    (X_train, X_test, y_train, y_test) : tuple of np.ndarray
        The split data, returned so Task 2's case-study code can reuse the
        EXACT same split (e.g. the two-stage Lasso->OLS refit in Case B, or
        the hybrid Lasso->Ridge pipeline in Case C) without re-splitting and
        accidentally introducing a different train/test partition.
    """
    # --- Step 1: Load data ---
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c != "target"]
    X = df[feature_cols].values
    y = df["target"].values
    print(f"[{dataset_label}] Loaded {csv_path} -> X shape: {X.shape}, y shape: {y.shape}")

    # --- Step 2: Correlation heatmap ---
    # Computed on the FULL dataset (features + target together), before any
    # splitting. This is purely descriptive/exploratory, not part of model
    # fitting, so there's no train/test leakage concern in just looking at it.
    plot_correlation_heatmap(
        df,
        title=f"{dataset_label.capitalize()} Dataset: Feature/Target Correlation Heatmap",
        save_path=f"q4_{dataset_label}_corr_heatmap.png",
    )

    # --- Step 3: Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"[{dataset_label}] Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    # --- Step 4: Sweep OLS / Ridge / Lasso across the lambda grid ---
    results = []
    for lam in LAMBDA_GRID:
        if lam == 0.0:
            # lambda = 0 means "no penalty at all" -> this IS ordinary least
            # squares. We fit it once and label it "OLS" rather than fitting
            # Ridge(alpha=0)/Lasso(alpha=0), because:
            #   - Ridge(alpha=0) is mathematically identical to OLS anyway.
            #   - Lasso(alpha=0) is numerically unstable in sklearn (the
            #     coordinate-descent solver assumes alpha > 0) and raises a
            #     warning. Using LinearRegression avoids that entirely.
            ols = LinearRegression()
            results.append(evaluate_model(ols, "OLS", lam, X_train, y_train, X_test, y_test, feature_cols))
            continue

        # --- Ridge (L2 penalty) ---
        ridge = Ridge(alpha=lam, random_state=random_state)
        results.append(evaluate_model(ridge, "Ridge", lam, X_train, y_train, X_test, y_test, feature_cols))

        # --- Lasso (L1 penalty) ---
        lasso = Lasso(alpha=lam, random_state=random_state, max_iter=lasso_max_iter)
        results.append(evaluate_model(lasso, "Lasso", lam, X_train, y_train, X_test, y_test, feature_cols))

    # --- Step 5: Package + save results ---
    results_df = pd.DataFrame(results).sort_values(["model", "lambda"]).reset_index(drop=True)

    out_path = f"q4_{dataset_label}_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"[{dataset_label}] Saved results table -> {out_path}")

    return results_df, (X_train, X_test, y_train, y_test)


# =============================================================================
# DATASET 1: housing_features.csv  (Case A - "Back to the ordinary")
# =============================================================================
# Shape: 5000 rows, 5 features. LOTS of samples relative to very few
# features -> the "easy" regime where Case A's questions expect
# regularization to barely move the needle versus plain OLS.
# test_size=0.2 -> 4000 train / 1000 test.
housing_results, housing_splits = run_task1_experiment(
    csv_path="housing_features.csv",
    dataset_label="housing",
    test_size=0.2,
    random_state=42,
)
print("\n=== Housing: Results table (Test MSE / Test R2 / coefficients) ===")
print(housing_results.to_string(index=False))


# =============================================================================
# DATASET 2: gene_features.csv  (Case B - "The hidden gene")
# =============================================================================
# Shape: 80 rows, 24 features. The OPPOSITE regime from housing: very few
# samples relative to a much larger number of features, with a known block
# of highly-correlated, mostly-irrelevant genes (indices 4-10) mixed in
# among a few true "driver" genes. This is exactly where Lasso's feature
# selection and Ridge's shrinkage are expected to behave very differently.
# test_size=0.2 -> 64 train / 16 test (a small test set is unavoidable given
# only 80 total rows, but it's enough to compare models on common ground).
gene_results, gene_splits = run_task1_experiment(
    csv_path="gene_features.csv",
    dataset_label="gene",
    test_size=0.2,
    random_state=42,
)
print("\n=== Gene: Results table (Test MSE / Test R2 / coefficients) ===")
print(gene_results.to_string(index=False))


# =============================================================================
# DATASET 3: sensor_features.csv  (Case C - "Imperfect Clones")
# =============================================================================
# Shape: 100 rows, 10 features, organized into two clusters of redundant
# sensors (1-4 and 5-8) plus two external/noise sensors (9-10). This is the
# setting for comparing Ridge's "grouping" / averaging behavior on redundant
# features against Lasso's tendency to arbitrarily pick one sensor per
# cluster and zero out the rest.
# test_size=0.2 -> 80 train / 20 test.
sensor_results, sensor_splits = run_task1_experiment(
    csv_path="sensor_features.csv",
    dataset_label="sensor",
    test_size=0.2,
    random_state=42,
)
print("\n=== Sensor: Results table (Test MSE / Test R2 / coefficients) ===")
print(sensor_results.to_string(index=False))