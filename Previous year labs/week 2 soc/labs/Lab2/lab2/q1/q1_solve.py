#!/usr/bin/env python3
"""
q1_solve.py

Implement ALL functions below.
Do NOT import sklearn.
"""

import numpy as np
from typing import List, Tuple


# -------------------------
# Utilities
# -------------------------
def add_bias(X: np.ndarray) -> np.ndarray:
    """
    Add bias (column of ones) as first column.
    See main.py for usage.
    """
    # TODO
    N = X.shape[0]
    return np.hstack((np.ones((N, 1)), X))


def mse(y: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error."""
    # TODO
    return np.mean((y - y_pred) ** 2)


def standardize_train(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize features using training statistics.
    Returns standardized X, mean, and stddev.
    See main.py for usage.
    """
    # TODO
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    
    # Avoid division by zero for constant features
    std_safe = np.copy(std)
    std_safe[std_safe == 0.0] = 1.0 
    
    X_std = (X - mean) / std_safe
    return X_std, mean, std


def standardize_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Apply training standardization.
    See main.py for usage.
    """
    # TODO
    std_safe = np.copy(std)
    std_safe[std_safe == 0.0] = 1.0
    return (X - mean) / std_safe


# -------------------------
# Ridge Regression
# -------------------------
def ridge_regression_closed_form(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """
    Closed-form ridge regression:
        (X^T X + λD) w = X^T y
    where D[0,0] = 0 (bias not regularized).
    """
    # TODO
    d = X.shape[1]
    D = np.eye(d)
    D[0, 0] = 0.0  # Bias not regularized
    
    XTX = X.T @ X
    penalty = lam * D
    
    # Solve (X^T X + lambda D) w = X^T y
    w = np.linalg.solve(XTX + penalty, X.T @ y)
    return w


# -------------------------
# Cross-validation
# -------------------------
def k_fold_split(N: int, k: int) -> List[np.ndarray]:
    """
        k-fold split after shuffling
        Returns list of k arrays of indices.
    """
    # TODO
    indices = np.random.permutation(N)
    return np.array_split(indices, k)


def ridge_cv(X: np.ndarray, y: np.ndarray, lam: float, k: int) -> float:
    """
    k-fold CV MSE for ridge.
    Use the k_fold_split function above to get the folds
    Use ridge_regression_closed_form to fit the model.
    Parameters:
        X: (N, D) training data
        y: (N,) training targets
        lam: regularization parameter
        k: number of folds
    Returns average MSE across folds.
    """
    # TODO
    N = X.shape[0]
    folds = k_fold_split(N, k)
    mses = []
    
    for i in range(k):
        # The validation fold is the i-th split
        val_idx = folds[i]
        
        # The training folds are all other splits combined
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        # Fit model on training folds
        w = ridge_regression_closed_form(X_train, y_train, lam)
        
        # Evaluate on validation fold
        y_pred = X_val @ w
        mses.append(mse(y_val, y_pred))
        
    return float(np.mean(mses))

# -------------------------
# Hyperparameter search
# -------------------------
def grid_search_lambdas(
    X: np.ndarray, y: np.ndarray,
    lambdas: np.ndarray, k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate each λ using CV.
    Parameters:
        X: (N, D) training data
        y: (N,) training targets
        lambdas: (M,) array of λ values to evaluate
        k: number of folds
    Returns:
        lambdas: (M,) same as input
        mses: (M,) average CV MSE for each λ
    """
    # TODO
    mses = np.zeros(len(lambdas))
    for i, lam in enumerate(lambdas):
        mses[i] = ridge_cv(X, y, lam, k)
    return lambdas, mses


def random_search_lambdas(
    X: np.ndarray, y: np.ndarray,
    n_iter: int,
    low_exp: float,
    high_exp: float,
    k: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Sample λ = 10^u, where u ~ Uniform(low_exp, high_exp).
    Parameters:
        X: (N, D) training data
        y: (N,) training targets
        n_iter: number of λ values to sample
        low_exp: lower bound of exponent
        high_exp: upper bound of exponent
        k: number of folds
    Returns:
        lambdas: (n_iter,) sampled λ values
        mses: (n_iter,) average CV MSE for each λ
    """
    # TODO
   # Sample exponents from uniform distribution
    u = np.random.uniform(low_exp, high_exp, size=n_iter)
    lambdas = 10 ** u
    
    mses = np.zeros(n_iter)
    for i, lam in enumerate(lambdas):
        mses[i] = ridge_cv(X, y, lam, k)
        
    return lambdas, mses
