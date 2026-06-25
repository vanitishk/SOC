"""
 Q2: Classification Under Class Imbalance (Thyroid Data)
=========================================================================
Implements Tasks A-F:
  A. Preprocessing (encoding, missingness, KNN imputation, standardisation)
  B. Baseline logistic regression + stratified CV confusion matrix
  C. ROC curve + AUC
  D. Threshold adjustment (cost-minimising, fixed-sensitivity, Youden's J)
  E. Class-weighted logistic loss (inverse-frequency weighting)
  F. Data-level strategies: random undersampling, random oversampling, SMOTE

Logistic regression itself is implemented from scratch with batch gradient
descent (consistent with Q1 of the lab), since the point of the exercise is
to understand the mechanics of the optimisation and class-imbalance fixes,
not to call sklearn.LogisticRegression as a black box. sklearn is used only
for the *evaluation* machinery (StratifiedKFold, confusion_matrix, KNNImputer)
which the lab explicitly allows / requires.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt

RNG = np.random.RandomState(42)   # single shared RNG -> reproducible folds/sampling

# -------------------------------------------------------------------------
# TASK A: PREPROCESSING
# -------------------------------------------------------------------------

# Column names in the exact order they appear in thyroid.data, taken from
# the accompanying thyroid.names file. The UCI file packs the patient ID
# into the same field as the class label, separated by '|', so we will
# split that off separately.
COLUMNS = [
    "age", "sex", "on_thyroxine", "query_on_thyroxine", "on_antithyroid_meds",
    "sick_flag", "pregnant", "thyroid_surgery", "I131_treatment",
    "query_hypothyroid", "query_hyperthyroid", "lithium", "goitre", "tumor",
    "hypopituitary", "psych", "TSH_measured", "TSH", "T3_measured", "T3",
    "TT4_measured", "TT4", "T4U_measured", "T4U", "FTI_measured", "FTI",
    "TBG_measured", "TBG", "referral_source", "class_and_id",
]

BINARY_COLS = [
    "sex", "on_thyroxine", "query_on_thyroxine", "on_antithyroid_meds",
    "sick_flag", "pregnant", "thyroid_surgery", "I131_treatment",
    "query_hypothyroid", "query_hyperthyroid", "lithium", "goitre", "tumor",
    "hypopituitary", "psych", "TSH_measured", "T3_measured", "TT4_measured",
    "T4U_measured", "FTI_measured", "TBG_measured",
]

NUMERIC_COLS = ["age", "TSH", "T3", "TT4", "T4U", "FTI", "TBG"]
CATEGORICAL_COLS = ["referral_source"]  # multi-valued categorical -> one-hot


def load_raw_data(path: str) -> pd.DataFrame:
    """Read the raw .data file and split the packed 'class|id' field."""
    df = pd.read_csv(path, header=None, names=COLUMNS)

    # The dataset uses '?' as the missing-value sentinel everywhere instead
    # of NaN, so we replace it up front -> lets pandas/numpy treat it as a
    # genuine missing value (enables .isna(), imputers, etc).
    df = df.replace("?", np.nan)

    # 'class_and_id' looks like "negative.|3733" -> split on '|' to recover
    # the actual class label and discard the trailing record ID (not a
    # predictive feature, it's just a row identifier from the original DB).
    split_col = df["class_and_id"].str.split("|", expand=True)
    df["class_label"] = split_col[0]
    df = df.drop(columns=["class_and_id"])

    # Build the binary target: 1 = thyroid sick, 0 = negative (euthyroid).
    # Note: the dataset actually contains a few rare sub-classes (e.g.
    # "compensated hypothyroid.") in the full multi-class version; in this
    # binarised file class_label is only "sick." or "negative.", so we map
    # any label that is NOT "negative." to the positive class to be safe.
    df["y"] = (df["class_label"] != "negative.").astype(int)
    df = df.drop(columns=["class_label"])
    return df


def encode_binary_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Map the dataset's f/t flags (and sex M/F) to 0/1 indicator columns."""
    df = df.copy()
    for col in BINARY_COLS:
        if col == "sex":
            # Encode sex as a single indicator (1 = Male, 0 = Female).
            # Missing sex (rare) is left as NaN for the imputer to handle.
            df[col] = df[col].map({"M": 1, "F": 0})
        else:
            df[col] = df[col].map({"t": 1, "f": 0})
    return df


def drop_high_missingness(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """
    Drop columns missing in more than `threshold` fraction of rows.
    TBG is famously >90% missing in this dataset (the lab measured it in
    only a handful of patients), so it gets dropped here. We also drop the
    *_measured flag for TBG since, once TBG itself is gone, the flag is a
    near-constant column carrying almost no information.
    """
    missing_frac = df.isna().mean()
    cols_to_drop = missing_frac[missing_frac > threshold].index.tolist()
    print(f"Dropping high-missingness columns (>{threshold:.0%} missing): {cols_to_drop}")
    df = df.drop(columns=cols_to_drop)
    return df


def one_hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode the multi-valued categorical 'referral_source' column."""
    present_cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]
    # drop_first=False: we keep a column for every category. With logistic
    # regression + an explicit bias term inside our own gradient-descent
    # implementation (rather than relying on a dummy-encoded intercept
    # trick), keeping all dummies is fine and easier to standardise.
    df = pd.get_dummies(df, columns=present_cat_cols, dtype=float)
    return df


def knn_impute(X_train: np.ndarray, X_test: np.ndarray, n_neighbors: int = 5):
    """
    Impute missing numeric values with KNNImputer.
    CRITICAL: the imputer is `fit` only on the training fold and then
    `transform`-ed on both train and test. Fitting on the full dataset
    (including the test fold) would let test-set information leak into
    the neighbour-search used to fill training values -> optimistic bias
    in cross-validated performance. This is why imputation happens *inside*
    the cross-validation loop in `run_pipeline`, not once on the whole df.
    """
    imputer = KNNImputer(n_neighbors=n_neighbors)
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    return X_train_imputed, X_test_imputed


def standardise(X_train: np.ndarray, X_test: np.ndarray):
    """
    Standardise to zero mean / unit variance, fit on train fold only
    (same train/test leakage argument as knn_impute above).

    Why standardisation matters for gradient descent specifically: features
    here live on wildly different scales (age ~ 0-90, TSH ~ 0-100s, binary
    flags are 0/1). Gradient descent takes a *single* learning rate shared
    across all weight coordinates. If TSH has a much larger scale than the
    binary flags, the loss surface becomes a long thin ellipse rather than
    a circular bowl in weight-space: the gradient along the large-scale
    feature's axis is huge while the gradient along small-scale features is
    tiny, forcing you to either use a tiny learning rate (slow convergence
    on every other feature) or risk overshooting/diverging along the
    large-scale axis. Standardising puts every feature on a comparable
    scale so a single learning rate converges efficiently in all directions.
    """
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    sigma[sigma == 0] = 1.0  # guard against degenerate constant columns -> avoid /0
    X_train_std = (X_train - mu) / sigma
    X_test_std = (X_test - mu) / sigma
    return X_train_std, X_test_std


# -------------------------------------------------------------------------
# LOGISTIC REGRESSION VIA GRADIENT DESCENT (with optional class weights)
# -------------------------------------------------------------------------

def sigmoid(z: np.ndarray) -> np.ndarray:
    # Clip z to avoid overflow in exp() for very large |z| (keeps the
    # function numerically stable without changing its mathematical value
    # in any region that matters: sigmoid saturates to 0/1 well before
    # |z| = 500 anyway).
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def add_bias_column(X: np.ndarray) -> np.ndarray:
    """Prepend a column of ones so w[0] acts as the intercept/bias term."""
    return np.hstack([np.ones((X.shape[0], 1)), X])


def train_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    lr: float = 0.1,
    n_iters: int = 2000,
    sample_weight: np.ndarray = None,
    tol: float = 1e-6,
):
    """
    Fit w via batch gradient descent on the (optionally weighted) binary
    cross-entropy loss, using y in {0, 1} and probabilistic predictions
    p = sigmoid(w^T x).

    Weighted loss (Task E):
        L_w = -(1/N) * sum_i [ w1 * y_i * log(p_i) + w0 * (1-y_i) * log(1-p_i) ]

    A very convenient fact (used below) is that the *gradient* of this
    weighted cross-entropy w.r.t. w has exactly the same clean form as the
    unweighted case, just with each sample's residual (p_i - y_i) scaled by
    that sample's weight:
        grad = (1/N) * X^T @ ( s_i * (p_i - y_i) )
    where s_i = w1 if y_i == 1 else w0. This is why sample_weight is applied
    by elementwise-multiplying the residual vector rather than needing a
    separate code path.
    """
    Xb = add_bias_column(X)
    n_samples, n_features = Xb.shape
    w = np.zeros(n_features)  # zero init is fine for convex logistic loss

    if sample_weight is None:
        sample_weight = np.ones(n_samples)

    loss_history = []
    for it in range(n_iters):
        z = Xb @ w
        p = sigmoid(z)

        # Per-sample weighted residual (p - y). For y=1 samples this gets
        # scaled by w1, for y=0 samples by w0 -> this is exactly how the
        # class weights pull the decision boundary towards better recall
        # on the minority class: misclassified positives now contribute a
        # proportionally larger gradient, pushing w harder to fix them.
        residual = sample_weight * (p - y)

        grad = (Xb.T @ residual) / n_samples
        w_new = w - lr * grad

        # Weighted cross-entropy loss (for monitoring/plotting convergence).
        eps = 1e-12  # avoid log(0)
        loss = -np.mean(
            sample_weight * (y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        )
        loss_history.append(loss)

        # Early stopping once the step size becomes negligible -> same
        # "stopping criterion" idea as Task 1/2 of Q1.
        if np.linalg.norm(w_new - w) < tol:
            w = w_new
            break
        w = w_new

    return w, loss_history


def predict_proba(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Return P(y=1 | x) for each row of X."""
    Xb = add_bias_column(X)
    return sigmoid(Xb @ w)


def predict_label(w: np.ndarray, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (predict_proba(w, X) >= threshold).astype(int)


# -------------------------------------------------------------------------
# CROSS-VALIDATED EVALUATION PIPELINE (shared by Tasks B, D, E, F)
# -------------------------------------------------------------------------

def run_cv_pipeline(
    X_raw: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    sample_weight_fn=None,
    resampling_fn=None,
    lr: float = 0.1,
    n_iters: int = 2000,
    threshold: float = 0.5,
):
    """
    Generic stratified-CV driver.

    Stratified K-fold (rather than plain K-fold) is essential here: with
    only ~6% positive cases, a random split could easily put very few (or
    zero) positives in some folds, making recall/precision estimates on
    that fold meaningless/undefined. StratifiedKFold guarantees each fold
    keeps (approximately) the same 6/94 class ratio as the full dataset, so
    the cross-validated confusion matrix is a fair, low-variance estimate
    of out-of-sample performance.

    Imputation and standardisation are deliberately performed INSIDE the
    loop, fit on X_train only, to prevent test-fold leakage (see comments
    on knn_impute/standardise above). Resampling (Task F) and sample
    weighting (Task E) are also applied to the TRAINING fold only -- the
    test fold must always reflect the real, imbalanced population so that
    the reported confusion matrix matches what you'd see in deployment.

    `sample_weight_fn(y_train) -> weights array`  : used for Task E.
    `resampling_fn(X_train, y_train) -> (X_res, y_res)` : used for Task F.

    Returns: aggregated confusion matrix (summed over folds), plus
    out-of-fold (concatenated) y_true/y_proba for ROC/AUC (Task C).
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    total_cm = np.zeros((2, 2), dtype=int)
    all_y_true, all_y_proba = [], []

    for train_idx, test_idx in skf.split(X_raw, y):
        X_train_raw, X_test_raw = X_raw[train_idx], X_raw[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # --- per-fold imputation & standardisation (fit on train only) ---
        X_train_imp, X_test_imp = knn_impute(X_train_raw, X_test_raw)
        X_train_std, X_test_std = standardise(X_train_imp, X_test_imp)

        # --- optional data-level resampling (Task F), TRAIN FOLD ONLY ---
        if resampling_fn is not None:
            X_train_std, y_train = resampling_fn(X_train_std, y_train)

        # --- optional class weighting (Task E) ---
        weights = sample_weight_fn(y_train) if sample_weight_fn is not None else None

        # --- fit and evaluate ---
        w, _ = train_logistic_regression(
            X_train_std, y_train, lr=lr, n_iters=n_iters, sample_weight=weights
        )
        y_proba = predict_proba(w, X_test_std)
        y_pred = (y_proba >= threshold).astype(int)

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        total_cm += cm

        all_y_true.append(y_test)
        all_y_proba.append(y_proba)

    return total_cm, np.concatenate(all_y_true), np.concatenate(all_y_proba)


def print_confusion(cm: np.ndarray, title: str):
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    acc = (tp + tn) / cm.sum()
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    print(f"\n--- {title} ---")
    print(f"Confusion matrix [[TN FP] [FN TP]]:\n{cm}")
    print(f"Accuracy={acc:.4f}  Recall={recall:.4f}  Precision={precision:.4f}  "
          f"F1={f1:.4f}  FPR={fpr:.4f}")
    return dict(acc=acc, recall=recall, precision=precision, f1=f1, fpr=fpr)


# -------------------------------------------------------------------------
# TASK B: BASELINE LOGISTIC REGRESSION
# -------------------------------------------------------------------------

def task_b_baseline(X, y):
    cm, y_true, y_proba = run_cv_pipeline(X, y, n_splits=5)
    stats = print_confusion(cm, "Task B: Baseline Logistic Regression (tau=0.5)")
    # Interpretation printed for the report:
    # FN (sick patients missed) is typically the larger error count relative
    # to FP, because the loss/gradient is dominated by the ~94% majority
    # (negative) class -> the decision boundary drifts towards predicting
    # "negative" by default. Accuracy alone hides this: a trivial classifier
    # that always predicts "negative" scores ~94% accuracy while catching
    # zero sick patients, so accuracy cannot distinguish a clinically useful
    # model from a clinically useless one in this imbalanced setting.
    return cm, y_true, y_proba, stats


# -------------------------------------------------------------------------
# TASK C: ROC CURVE + AUC
# -------------------------------------------------------------------------

def task_c_roc(y_true, y_proba, save_path="roc_curve.png"):
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"Logistic Regression (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="grey", label="Random guessing")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Thyroid Sick Detection")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\nTask C: AUC = {roc_auc:.4f} (plot saved to {save_path})")

    # Discussion (for report):
    # - ROC is built from TPR = TP/(TP+FN) and FPR = FP/(FP+TN): each rate is
    #   normalised WITHIN its own true class (positives only for TPR,
    #   negatives only for FPR). Since neither rate is divided by the total
    #   sample size, the curve doesn't shift just because one class has far
    #   more examples than the other -- unlike accuracy, which is dominated
    #   by whichever class is numerically larger.
    # - A high AUC does NOT guarantee good minority-class performance at a
    #   *specific* operating threshold. AUC summarises ranking quality
    #   across *all* thresholds, but in practice you must pick one tau to
    #   deploy; at that tau, recall/precision on the minority class can
    #   still be poor even if AUC is high (e.g. if most of the ranking
    #   quality comes from separating "easy" majority points rather than
    #   correctly ordering the minority points near the boundary).
    return fpr, tpr, thresholds, roc_auc


# -------------------------------------------------------------------------
# TASK D: THRESHOLD ADJUSTMENT
# -------------------------------------------------------------------------

def youdens_index_threshold(y_true, y_proba):
    """tau* = argmax_tau (Sensitivity(tau) + Specificity(tau) - 1) = argmax(TPR - FPR)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    j_scores = tpr - fpr  # Specificity = 1 - FPR, so Sens+Spec-1 = TPR - FPR
    best_idx = np.argmax(j_scores)
    return thresholds[best_idx], j_scores[best_idx]


def fixed_sensitivity_threshold(y_true, y_proba, target_recall: float = 0.95):
    """
    Smallest threshold achieving at least `target_recall`. As tau decreases
    we predict the positive class more liberally, so recall is
    monotonically non-increasing in tau; we scan thresholds returned by
    roc_curve (sorted from high to low) and pick the lowest tau that still
    clears the target recall, i.e. the most conservative tau that satisfies
    the medical-screening requirement.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    valid = np.where(tpr >= target_recall)[0]
    if len(valid) == 0:
        return thresholds[-1]  # fallback: most liberal threshold available
    # roc_curve returns thresholds in decreasing order; among those meeting
    # the recall target we want the *largest* tau (most conservative / fewest
    # false positives) that still satisfies it.
    return thresholds[valid].max()


def cost_minimizing_threshold(y_true, y_proba, cost_fn: float = 5.0, cost_fp: float = 1.0):
    """
    tau* = argmin_tau [ C_FN * FN(tau) + C_FP * FP(tau) ].
    cost_fn is set higher than cost_fp by default because, in a medical
    screening context, missing a sick patient (FN) is typically far more
    costly than a false alarm (FP) that simply triggers a follow-up test.
    We brute-force scan a grid of thresholds since FN(tau)/FP(tau) are
    piecewise-constant step functions of tau (not differentiable), so
    gradient-based optimisation doesn't apply -- a direct grid search over
    observed probability values is the standard approach.
    """
    candidate_taus = np.unique(y_proba)
    best_tau, best_cost = 0.5, np.inf
    for tau in candidate_taus:
        y_pred = (y_proba >= tau).astype(int)
        fn = np.sum((y_true == 1) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        cost = cost_fn * fn + cost_fp * fp
        if cost < best_cost:
            best_cost, best_tau = cost, tau
    return best_tau, best_cost


def task_d_threshold_adjustment(X, y):
    # Re-run the baseline CV pipeline to get out-of-fold probabilities,
    # then explore thresholds against those held-out predictions (we never
    # tune tau using the same fold's labels that we then "test" on -- the
    # CV-aggregated y_true/y_proba below are already out-of-sample for every
    # point, so this stays a fair evaluation of thresholding alone, which
    # does not require retraining the model).
    cm_default, y_true, y_proba, _ = task_b_baseline(X, y)

    tau_youden, j = youdens_index_threshold(y_true, y_proba)
    tau_sens95 = fixed_sensitivity_threshold(y_true, y_proba, target_recall=0.95)
    tau_cost, cost = cost_minimizing_threshold(y_true, y_proba, cost_fn=5.0, cost_fp=1.0)

    print(f"\nTask D thresholds found: Youden tau={tau_youden:.4f} (J={j:.4f}), "
          f"Fixed-sensitivity(0.95) tau={tau_sens95:.4f}, "
          f"Cost-minimising tau={tau_cost:.4f} (cost={cost:.1f})")

    for name, tau in [("Youden's J", tau_youden),
                       ("Fixed sensitivity 0.95", tau_sens95),
                       ("Cost-minimising", tau_cost)]:
        y_pred = (y_proba >= tau).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        print_confusion(cm, f"Task D: {name} (tau={tau:.3f})")

    return tau_youden, tau_sens95, tau_cost


# -------------------------------------------------------------------------
# TASK E: CLASS-WEIGHTED LOSS
# -------------------------------------------------------------------------

def inverse_frequency_weight_fn(y_train: np.ndarray) -> np.ndarray:
    """
    w1 = N / (2*N1), w0 = N / (2*N0)  -- inverse-frequency class weights,
    applied per-sample (each sample gets its own class's weight, broadcast
    across the fold). The factor of 2 normalises things so that, if the
    classes WERE balanced (N1 = N0 = N/2), both weights collapse to 1 and
    weighting has no effect -- i.e. it only kicks in to compensate for
    actual imbalance.
    """
    n = len(y_train)
    n1 = np.sum(y_train == 1)
    n0 = np.sum(y_train == 0)
    w1 = n / (2 * n1)
    w0 = n / (2 * n0)
    weights = np.where(y_train == 1, w1, w0)
    return weights


def task_e_class_weighted(X, y):
    cm, y_true, y_proba = run_cv_pipeline(
        X, y, n_splits=5, sample_weight_fn=inverse_frequency_weight_fn
    )
    stats = print_confusion(cm, "Task E: Class-Weighted Logistic Regression (inverse frequency)")
    # Discussion: compare stats['recall'] / stats['fpr'] against Task B's
    # baseline numbers. Weighting almost always raises recall (fewer sick
    # patients missed) at the cost of a higher FPR (more healthy patients
    # flagged for follow-up) -- the classic precision/recall trade-off, just
    # achieved by re-shaping the loss instead of moving tau after the fact.
    return cm, y_true, y_proba, stats


# -------------------------------------------------------------------------
# TASK F: DATA-LEVEL STRATEGIES
# -------------------------------------------------------------------------

def random_undersample(X_train, y_train, rng=RNG):
    """Keep all minority samples; randomly subsample majority down to the same count."""
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]
    # Sample WITHOUT replacement, exactly |pos_idx| majority examples.
    sampled_neg_idx = rng.choice(neg_idx, size=len(pos_idx), replace=False)
    keep_idx = np.concatenate([pos_idx, sampled_neg_idx])
    rng.shuffle(keep_idx)  # shuffle so the optimiser doesn't see a block of all-1s then all-0s
    return X_train[keep_idx], y_train[keep_idx]


def random_oversample(X_train, y_train, rng=RNG):
    """Keep all majority samples; randomly duplicate (with replacement) minority up to the same count."""
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]
    # Sample WITH replacement since we need more positive examples than
    # actually exist -- duplication is the entire point of oversampling.
    sampled_pos_idx = rng.choice(pos_idx, size=len(neg_idx), replace=True)
    keep_idx = np.concatenate([neg_idx, sampled_pos_idx])
    rng.shuffle(keep_idx)
    return X_train[keep_idx], y_train[keep_idx]


def smote_oversample(X_train, y_train, k: int = 5, rng=RNG):
    """
    Minimal from-scratch SMOTE: for each minority sample, find its k nearest
    minority neighbours (Euclidean distance on the standardised features),
    pick one at random, and interpolate a synthetic point along the segment
    joining them. We generate enough synthetic points to bring the minority
    class count up to the majority class count, then concatenate with all
    real (majority + minority) samples.
    """
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]
    X_pos = X_train[pos_idx]
    n_pos, n_neg = len(pos_idx), len(neg_idx)
    n_to_generate = n_neg - n_pos
    if n_to_generate <= 0:
        return X_train, y_train  # already balanced/majority-minority reversed

    # Pairwise distance matrix among minority points only (small relative to
    # the full dataset since the minority class is rare, so this is cheap).
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b, computed via broadcasting to
    # avoid an explicit Python double loop over all pairs.
    sq_norms = np.sum(X_pos ** 2, axis=1)
    dist_sq = sq_norms[:, None] + sq_norms[None, :] - 2 * (X_pos @ X_pos.T)
    np.fill_diagonal(dist_sq, np.inf)  # a point is never its own neighbour
    dist_sq = np.maximum(dist_sq, 0)   # guard against tiny negative values from floating-point error

    k_eff = min(k, n_pos - 1)
    # argsort each row and take the k_eff closest -> indices of nearest neighbours per minority point.
    neighbor_idx = np.argsort(dist_sq, axis=1)[:, :k_eff]

    synthetic = np.zeros((n_to_generate, X_train.shape[1]))
    for s in range(n_to_generate):
        i = rng.randint(n_pos)                          # pick a random minority "anchor" point x_i
        j = neighbor_idx[i, rng.randint(k_eff)]          # pick one of its k nearest minority neighbours x_j
        lam = rng.uniform(0, 1)                          # interpolation factor lambda ~ U(0,1)
        synthetic[s] = X_pos[i] + lam * (X_pos[j] - X_pos[i])  # x_new = x_i + lambda*(x_j - x_i)

    X_resampled = np.vstack([X_train, synthetic])
    y_resampled = np.concatenate([y_train, np.ones(n_to_generate, dtype=int)])
    # Shuffle so synthetic points (all appended at the end) aren't seen as a
    # contiguous block by anything downstream that might be order-sensitive.
    perm = rng.permutation(len(y_resampled))
    return X_resampled[perm], y_resampled[perm]


def task_f_data_level(X, y):
    results = {}
    for name, fn in [
        ("Random Undersampling", random_undersample),
        ("Random Oversampling", random_oversample),
        ("SMOTE", smote_oversample),
    ]:
        cm, y_true, y_proba = run_cv_pipeline(X, y, n_splits=5, resampling_fn=fn)
        stats = print_confusion(cm, f"Task F: {name}")
        results[name] = stats
    return results


# -------------------------------------------------------------------------
# MAIN: PUT IT ALL TOGETHER
# -------------------------------------------------------------------------

def build_feature_matrix(data_path: str):
    df = load_raw_data(data_path)
    df = encode_binary_flags(df)
    df = drop_high_missingness(df, threshold=0.5)
    df = one_hot_encode(df)

    y = df["y"].to_numpy()
    X_df = df.drop(columns=["y"])
    X = X_df.to_numpy(dtype=float)
    print(f"Final feature matrix shape: {X.shape}, positive rate = {y.mean():.3%}")
    return X, y, X_df.columns.tolist()


if __name__ == "__main__":
    X, y, feature_names = build_feature_matrix("thyroid.data")

    # Task B
    cm_b, y_true_b, y_proba_b, stats_b = task_b_baseline(X, y)

    # Task C
    task_c_roc(y_true_b, y_proba_b)

    # Task D
    task_d_threshold_adjustment(X, y)

    # Task E
    task_e_class_weighted(X, y)

    # Task F
    task_f_data_level(X, y)