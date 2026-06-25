import numpy as np

def get_knn_accuracy(X_train, y_train, X_test, y_test, k, metric='L2', method='broadcasting'):
    N, D = X_test.shape
    M = X_train.shape[0]
    
    if method == 'expansion':
        if metric == 'L2':
            # Space: O(NM) - Only possible for L2
            dot_product = np.dot(X_test, X_train.T)
            test_sq = np.sum(X_test**2, axis=1, keepdims=True)
            train_sq = np.sum(X_train**2, axis=1)
            dists = np.sqrt(np.maximum(0, test_sq + train_sq - 2 * dot_product))
        else:
            raise ValueError("Expansion trick is only mathematically valid for L2 metric.")
        
    
    #the next few methods are tradeoffs, the standard for method uses much lesser space, but is slower, try to observe this tradeoff by playing
    #around with the sizes of the array to check this
    elif method == 'broadcasting':
        # Space: O(NMD), faster by stride allocation, please read up on how broadcasting works, is extremely useful
        ord_val = 2 if metric == 'L2' else 1
        dists = np.linalg.norm(X_test[:, np.newaxis, :] - X_train, ord=ord_val, axis=2)

    elif method == 'tiled':
        # Space: O(NMD) with explicit memory allocation
        X_test_3d = np.tile(X_test[:, np.newaxis, :], (1, M, 1))
        X_train_3d = np.tile(X_train[np.newaxis, :, :], (N, 1, 1))
        ord_val = 2 if metric == 'L2' else 1
        dists = np.linalg.norm(X_test_3d - X_train_3d, ord=ord_val, axis=2)

    nn_indices = np.argsort(dists, axis=1)[:, :k]
    nn_labels = y_train[nn_indices]
    
    predictions = np.sign(np.sum(nn_labels, axis=1))
    predictions[predictions == 0] = 1 
    
    return np.mean(predictions == y_test)


#the rest of the code for the experiments can be implemented on your own, just write the main function to run the experiments with hyperparameterization

def get_knn_accuracy_linf(X_train, y_train, X_test, y_test, k):
    """
    Helper function specifically handling the L_infinity (Chebyshev) distance metric 
    for Task C using pure NumPy broadcasting.
    """
    # L_infinity distance: max_j |x_test,j - x_train,j|
    dists = np.max(np.abs(X_test[:, np.newaxis, :] - X_train), axis=2)
    
    nn_indices = np.argsort(dists, axis=1)[:, :k]
    nn_labels = y_train[nn_indices]
    
    predictions = np.sign(np.sum(nn_labels, axis=1))
    predictions[predictions == 0] = 1 
    
    return np.mean(predictions == y_test)


if __name__ == "__main__":
    # ---------------------------------------------------------
    # 0. Generation of Mock Dataset for Demonstration
    # ---------------------------------------------------------
    np.random.seed(42)
    N_train, N_test, D_features = 200, 50, 15
    
    # Simulating vastly different scales across features (as noticed by Giyu)
    scales = np.random.uniform(0.1, 100.0, size=(1, D_features))
    X_train_raw = np.random.normal(0, 1, size=(N_train, D_features)) * scales
    X_test_raw = np.random.normal(0, 1, size=(N_test, D_features)) * scales
    
    # Generate labels (+1 or -1)
    y_train = np.random.choice([-1, 1], size=N_train)
    y_test = np.random.choice([-1, 1], size=N_test)

    print("=================== TASK A ===================")
    # Vary parameter k = 1, 2, 5, 10, 100 on raw data using L2 distance
    k_list_A = [1, 2, 5, 10, 100]
    for k in k_list_A:
        train_acc = get_knn_accuracy(X_train_raw, y_train, X_train_raw, y_train, k, metric='L2')
        test_acc = get_knn_accuracy(X_train_raw, y_train, X_test_raw, y_test, k, metric='L2')
        print(f"k = {k:3d} | Train Accuracy: {train_acc*100:6.2f}% | Test Accuracy: {test_acc*100:6.2f}%")

    print("\n=================== TASK B ===================")
    # 1. Standardizing Data across dimensions to mean 0 and variance 1
    mean_train = np.mean(X_train_raw, axis=0)
    var_train = np.var(X_train_raw, axis=0)
    std_train = np.sqrt(var_train)
    
    # Handle possible division by zero if a feature is completely constant
    std_train[std_train == 0] = 1.0 

    X_train_std = (X_train_raw - mean_train) / std_train
    X_test_std = (X_test_raw - mean_train) / std_train

    # 2. Supervised Feature Selection via Pearson Correlation Coefficient
    # r_j = [Sum((x_ij - mean_x_j) * (y_i - mean_y))] / sqrt(Sum(x_ij - mean_x_j)^2 * Sum(y_i - mean_y)^2)
    mean_y_train = np.mean(y_train)
    
    numerator = np.sum((X_train_raw - mean_train) * (y_train - mean_y_train)[:, np.newaxis], axis=0)
    denominator = np.sqrt(np.sum((X_train_raw - mean_train)**2, axis=0) * np.sum((y_train - mean_y_train)**2))
    
    # Vectorized calculation of absolute Pearson correlation coefficients
    r_j = np.abs(numerator / denominator)
    
    # Rank features by sorting indices descendingly based on absolute correlation
    ranked_features = np.argsort(r_j)[::-1]
    
    print("Feature rankings (highest correlation first):", ranked_features)
    print(f"Running experiments for Task B (k = 20):")
    
    # Vary the number of top dimensions 'm' chosen
    for m in range(1, D_features + 1):
        top_m_indices = ranked_features[:m]
        
        X_train_selected = X_train_std[:, top_m_indices]
        X_test_selected = X_test_std[:, top_m_indices]
        
        test_acc_m = get_knn_accuracy(X_train_selected, y_train, X_test_selected, y_test, k=20, metric='L2')
        print(f"Top m = {m:2d} features | Test Accuracy: {test_acc_m*100:6.2f}%")

    print("\n=================== TASK C ===================")
    # Run experiments on fully standardized data using L1 and predicted L_infinity
    k_task_c = 20
    
    acc_l2 = get_knn_accuracy(X_train_std, y_train, X_test_std, y_test, k=k_task_c, metric='L2')
    acc_l1 = get_knn_accuracy(X_train_std, y_train, X_test_std, y_test, k=k_task_c, metric='L1')
    acc_linf = get_knn_accuracy_linf(X_train_std, y_train, X_test_std, y_test, k=k_task_c)
    
    print(f"Using Standardized Dataset (k = {k_task_c}):")
    print(f"L2 Metric (Euclidean) Distance Accuracy:  {acc_l2*100:6.2f}%")
    print(f"L1 Metric (Manhattan) Distance Accuracy:  {acc_l1*100:6.2f}%")
    print(f"L_inf Metric (Chebyshev) Distance Accuracy: {acc_linf*100:6.2f}%")