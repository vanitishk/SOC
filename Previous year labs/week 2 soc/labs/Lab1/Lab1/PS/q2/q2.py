# do not modify the imports below
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Polynomial regression implementations
# ============================================================

def poly_features(x, degree):
    """
    x : (N,)
    degree: int

    Returns Phi : (N, degree+1)
        Phi[i] = [1, x_i, x_i^2, ..., x_i^degree]
    """
    # TODO
    column = []
    for d in range(degree + 1):
        column.append(x ** d)
    Phi = np.column_stack(column)
    return Phi
   

def fit_ols(Phi, y):
    """
    Phi : (N, p+1)
    y   : (N,)

    Returns w : (p+1,)
    """
    # TODO
    a= Phi.T @ Phi
    b= Phi.T @ y
    inv_a = np.linalg.inv(a)
    w = inv_a @ b
    return w


def predict(x, degree, w):
    """
    x : (N, )
    degree: int
    w : (degree+1,)

    Returns y_hat : (N,)
    """
    # TODO
    Phi = poly_features(x, degree)
    y_hat = Phi @ w
    return y_hat

# ============================================================
# k-fold cross-validation
# ============================================================

def mse(y, y_hat):
    # TODO
    return np.mean((y-y_hat) ** 2)



def k_fold_cv(x, y, degree, k=5):
    """
    x: (N,)
    y: (N,)
    degree: int
    k: int

    Returns avg_val_mse : float
    """
    # TODO:
    # 1. shuffle data
    # 2. split into k folds
    # 3. train on k-1, validate on 1
    # 4. return average validation MSE
    idx = np.random.permutation(len(x))
    x = x[idx]
    y = y[idx]
    fold_size = len(x) // k
    val_mse = []
    for i in range(k):
        start = i * fold_size
        end = (i + 1) * fold_size if i != k - 1 else len(x)
        x_val, y_val = x[start:end], y[start:end]
        x_train = np.concatenate((x[:start], x[end:]))
        y_train = np.concatenate((y[:start], y[end:]))
        Phi_train = poly_features(x_train, degree)
        w = fit_ols(Phi_train, y_train)
        y_val_hat = predict(x_val, degree, w)
        val_mse.append(mse(y_val, y_val_hat))
    return  np.mean(val_mse)



def evaluate_degrees(x, y, D_max, k=5):
    """
    x : (N,) array
    y : (N,) array
    D_max : int
    k : int

    Returns
        train_mse : list of length D_max
        cv_mse    : list of length D_max
    """
    # TODO :
    # For each d in 1..D_max:
    #   - construct Phi
    #   - fit OLS
    #   - compute training MSE
    #   - compute k-fold CV MSE
    # Store results in lists and return them
    train_mse =[]
    cv_mse = []
    for d in range(1, D_max +1):
        Phi = poly_features(x, d )
        w = fit_ols(Phi, y)
        y_hat = predict(x, d, w)
        train_mse.append(mse(y , y_hat))
        cv_mse.append(k_fold_cv(x,y,d,k))
    return train_mse, cv_mse


def select_degree(cv_mse):
    """
    cv_mse : list
    Returns best_degree : int
    """
    # TODO
    return np.argmin(cv_mse)+1


def fit_final_model(x, y, degree):
    """
    x : (N,) array
    y : (N,) array
    degree : int

    Returns w : (degree+1,) array
    """
    # TODO
    pp= poly_features(x, degree)
    m = fit_ols(pp, y)
    return m


def plot_errors(train_mse, cv_mse):
    D = len(train_mse)
    plt.plot(range(1, D + 1), train_mse, label="Training MSE")
    plt.plot(range(1, D + 1), cv_mse, label="CV MSE")
    plt.xlabel("Polynomial Degree")
    plt.ylabel("MSE")
    plt.legend()
    plt.show()

# ============================================================
# Data loading
# ============================================================

def load_data():
    """
    Load the train dataset

    Returns
    -------
    X_train, y_train (q2_train.csv)
    """
    # TODO
    data = np.loadtxt("q2_train.csv", delimiter=',', skiprows =1) 
    x_train = data[: , 0]
    y_train = data[: , 1]
    return x_train, y_train


# ============================================================
# Main experiment (DO NOT MODIFY, AUTOGRADER TESTS WILL RUN SOME OTHER MAIN)
# ============================================================
if __name__ == "__main__":
    np.random.seed(1234)

    x_train, y_train = load_data()
    D_max = 10
    k = 5

    train_mse, cv_mse = evaluate_degrees(x_train, y_train, D_max, k)

    best_d = select_degree(cv_mse)
    w = fit_final_model(x_train, y_train, best_d)

    print("Optimal degree:", best_d)
    print("Weights:", w)

    plot_errors(train_mse, cv_mse)