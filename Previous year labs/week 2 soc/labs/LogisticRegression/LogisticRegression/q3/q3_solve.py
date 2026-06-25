###
# q3_solve.py... Fill in the code in the described sections to implement the MultiClass classifiers.
###

import numpy as np


def make_labels(y: np.ndarray) -> np.ndarray:
    """
    Conver string labels to labels of the form {0, 1, ..., K-1}.
    Parameters
    ----------
    y : np.ndarray
        Original string labels (N,)
    Returns
    -------
    y_numeric : np.ndarray
        Numeric labels {0, 1, ..., K-1} (N,)
    dict_num_to_str : dict
        Dictionary mapping numeric labels to original string labels
    """

    unique_labels = np.unique(y)
    dict_label_to_num = {label: idx for idx, label in enumerate(unique_labels)}
    dict_num_to_str = {idx: label for label, idx in dict_label_to_num.items()}
    y_numeric = np.array([dict_label_to_num[label] for label in y])
    return y_numeric, dict_num_to_str

def make_labels_inverse(y_numeric: np.ndarray, dict_num_to_str: dict) -> np.ndarray:
    """
    Convert numeric labels back to string labels.
    Parameters
    ----------
    y_numeric : np.ndarray
        Numeric labels {0, 1, ..., K-1} (N,)
    dict_num_to_str : dict
        Dictionary mapping numeric labels to original string labels
    Returns
    -------
    y_str : np.ndarray
        Original string labels (N,)
    """
    return np.array([dict_num_to_str[num] for num in y_numeric])





# Logistic regression implementation.. Copied from the inlab
# Will be used in the One-vs-Rest and One-vs-One classifiers
# If you were not able to solve the inlab then you should type this out once again.
class LogisticRegressionGD:
    """
    Logistic Regression classifier using batch gradient descent.
    """

    def __init__(self, lr=0.01, max_iter=1000, tol=1e-6):
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.w = None
        self.logistic_loss_history = []


    def logistic_gradient(self, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute gradient of logistic loss.

        Parameters
        ----------
        w : np.ndarray
            Weights (d,)
        X : np.ndarray
            Feature matrix (N, d)
        y : np.ndarray
            Labels {-1, +1} (N,)

        Returns
        -------
        grad : np.ndarray
            Gradient of shape (d,)
        """

        # TODO: Compute logistic gradient
        # Useful concepts: Keeping track of dimensions, normalization by number of samples, vectorized implementation
        # Gotachas: The exp function can lead to overflow. So clip the input to exp to a reasonable range. Also, remember that the labels are in {-1, +1} and not {0, 1}.
        # np.clip for clipping, np.exp, np.dot / @ for matrix multiplication, division by the number of samples for normalization (Can be gotten by np.shape)
        # Check the shape of the output gradient to make sure it is (d,) and not (d, 1) or (1, d)
        N = X.shape[0]
        z = X @ w  # (N,)
        yz = np.clip(y * z, -500, 500)
        sig = 1.0 / (1.0 + np.exp(yz))  # this equals sigmoid(-y*z)
        grad = (X.T @ (-y * sig)) / N
        return grad


    def compute_logistic_loss(self, w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute logistic loss.

        Parameters
        ----------
        w : np.ndarray
            Weights (d,)
        X : np.ndarray
            Feature matrix (N, d)
        y : np.ndarray
            Labels {-1, +1} (N,)

        Returns
        -------
        loss : float
            Logistic loss
        """

        # TODO: Compute logistic loss
        # Again, keep track of dimension, take mean over samples, and be careful about overflow when computing the exp function.
        # np.exp, np.dot / @ for matrix multiplication, np.mean for taking mean over samples, np.clip for clipping the input to exp
        # Check that the output is a float
        z = X @ w
        yz = np.clip(y * z, -500, 500)
        loss = np.mean(np.log(1.0 + np.exp(-yz)))
        return float(loss)


    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit logistic regression using batch gradient descent, while adding logistic loss to history.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (N, d)
        y : np.ndarray
            Labels {-1, +1} (N,)
        """

        # TODO: Implement batch gradient descent for logistic regression
        # Begin by Initializing weights (To zeros is often a safe choice).
        # Then do a for loop over max_iter iterations, and in each iteration compute the gradient and update the weights. Also compute the logistic loss and add it to the history.
        # If the size of the update (norm of the gradient) is less than tol, then break out of the loop.
        # np.zeros for initializing weights, np.linalg.norm for computing the norm of the gradient, self.logistic_loss_history.append for adding loss to history
        # If all other functions are correct than this should be straightforward to implement. Checking the exact correctness of this function by outputs is somewhat difficult, but you can check that the logistic loss is decreasing over iterations.
        N, d = X.shape
        self.w = np.zeros(d)
        for _ in range(self.max_iter):
            grad = self.logistic_gradient(self.w, X, y)
            loss = self.compute_logistic_loss(self.w, X, y)
            self.logistic_loss_history.append(loss)
            self.w = self.w - self.lr * grad
            if np.linalg.norm(grad) < self.tol:
                break

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels {-1, +1}
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix (N, d)
        
        Returns
        -------
        y_pred : np.ndarray
            Predicted labels (N,)
        """

        # TODO: Implement predict method
        # Again check the dimensions. The output should be a 1D array of shape (N,) and not (N, 1) or (1, N)
        # The predicted labels should be in {-1, +1}. You can use np.where to convert the output of the linear function to predicted labels.
        z = X @ self.w
        y_pred = np.where(z >= 0, 1, -1)
        return y_pred



class OneVsRestClassifier:
    """
    One-vs-Rest multi-class classifier using LogisticRegressionGD as the base binary classifier.
    """

    def __init__(self, lr=0.01, max_iter=1000, tol=1e-6):
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.classifiers = dict()  # Dictionary to hold the binary classifiers for each class

    def create_binary_labels(self, y: np.ndarray, class_k: int) -> np.ndarray:
        """
        Create binary labels for class k in One-vs-Rest.

        Parameters
        ----------
        y : np.ndarray
            Original labels {0, 1, ..., K-1} (N,)
        class_k : int
            The class for which we are creating binary labels

        Returns
        -------
        y_binary : np.ndarray
            Binary labels {-1, +1} (N,)
        """

        #TODO Implement this function to create binary labels for class_k. The samples belonging to class_k should be labeled +1 and all other samples should be labeled -1. You can use np.where for this.
        # Remember that the input labels are in {0, 1, ..., K-1} and not {-1, +1}. The output should be in {-1, +1}. Also check the shape of the output to make sure it is (N,) and not (N, 1) or (1, N)
        y_binary = np.where(y == class_k, 1, -1)
        return y_binary

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit One-vs-Rest classifiers.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (N, d)
        y : np.ndarray
            Labels {0, 1, ..., K-1} (N,)
        """

        # TODO: Implement fit method for One-vs-Rest classifier
        # For each class k in {0, 1, ..., K-1}, create a binary label vector where samples of class k are labeled +1 and all other samples are labeled, then train a LogisticRegressionGD classifier on the data with these binary labels, and store the trained classifier in the self.classifiers dictionary with key k.
        # You can get the unique classes using which function? (Search the docs!)
        classes = np.unique(y)
        for class_k in classes:
            y_binary = self.create_binary_labels(y, class_k)
            clf = LogisticRegressionGD(lr=self.lr, max_iter=self.max_iter, tol=self.tol)
            clf.fit(X, y_binary)
            self.classifiers[class_k] = clf

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels {0, 1, ..., K-1}.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (N, d)

        Returns
        -------
        y_pred : np.ndarray
            Predicted class labels (N,)
        """

        # TODO: Implement predict method for One-vs-Rest classifier
        # For each classifier corresponding to class k, compute the decision function (w^T x) for the input samples, and then predict the class with the highest decision function value. 
        # To do this in a vectorized way you can create a matrix of shape ? and then take argmax over ? which axis?
        # Useful functions: np.dot or @ for matrix multiplication, np.argmax for taking argmax, np.array for creating a matrix to hold the decision function values for each class
        # Again check the shape of the output to make sure it is (N,) and not (N, 1) or (1, N)
        # You can assume you have a model for each class in self.classifiers, and you can access the weights of the logistic regression model for class k using self.classifiers[k].w
        classes = sorted(self.classifiers.keys())
        scores = np.array([X @ self.classifiers[k].w for k in classes]).T  # (N, K)
        best_idx = np.argmax(scores, axis=1)
        y_pred = np.array(classes)[best_idx]
        return y_pred


class OneVsOneClassifier:
    """
    One-vs-One multi-class classifier using LogisticRegressionGD as the base binary classifier.
    """

    def __init__(self, lr=0.01, max_iter=1000, tol=1e-6):
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.classifiers = dict()  # Dictionary to hold the binary classifiers for each pair of classes

    def create_binary_labels(self, y: np.ndarray, class_i: int, class_j: int) -> np.ndarray:
        """
        Create binary labels for classes i and j in One-vs-One.

        Parameters
        ----------
        y : np.ndarray
            Original labels {0, 1, ..., K-1} (N,)
        class_i : int
            The first class for which we are creating binary labels
        class_j : int
            The second class for which we are creating binary labels

        Returns
        -------
        y_binary : np.ndarray
            Binary labels {-1, +1} (N,)
        """

        #TODO Implement this function to create binary labels for classes i and j. The samples belonging to class_i should be labeled +1, the samples belonging to class_j should be labeled -1.
        # The samples belonging to other should be labeled 0, and should not be used for training the classifier for classes i and j. You can remove these samples later
        # Remember that the input labels are in {0, 1, ..., K-1} and not {-1, +1}. The output should be in {-1, 0, +1}. Also check the shape of the output to make sure it is (N,) and not (N, 1) or (1, N)
        # Use np.where to solve this part
        y_binary = np.where(y == class_i, 1, np.where(y == class_j, -1, 0))
        return y_binary

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit One-vs-One classifiers.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (N, d)
        y : np.ndarray
            Labels {0, 1, ..., K-1} (N,)
        """

        # TODO: Implement fit method for One-vs-One classifier
        # For each pair of classes (i, j) in {0, 1, ..., K-1}, create a binary label vector where samples of class i are labeled +1, 
        # samples of class j are labeled -1, and all other samples are labeled 0. Then train a LogisticRegressionGD classifier on the data with these binary labels, 
        # but only use the samples belonging to class i and class j for training (i.e. ignore samples labeled 0). 
        # Store the trained classifier in the self.classifiers dictionary with key (i, j).
        # Avoid training duplicate classifiers for pairs (i, j)!
        # You can use indexes to select only the samples belonging to class i and class j from the training data.
        for class_i in np.unique(y):
            for class_j in np.unique(y):
                if class_i < class_j:
                    y_binary = self.create_binary_labels(y, class_i, class_j)
                    mask = y_binary != 0
                    X_sub = X[mask]
                    y_sub = y_binary[mask]
                    clf = LogisticRegressionGD(lr=self.lr, max_iter=self.max_iter, tol=self.tol)
                    clf.fit(X_sub, y_sub)
                    self.classifiers[(class_i, class_j)] = clf
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels {0, 1, ..., K-1}.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (N, d)

        Returns
        -------
        y_pred : np.ndarray
            Predicted class labels (N,)
        """

        # For each classifier corresponding to pair of classes (i, j), compute the decision function (w^T x) for the input samples,
        #  and then predict either class i or class j based on the sign of the decision function.
        # After that you need to convert the (N,) output of the classifiers to votes for each class, and then predict the class with the most votes.
        # At the end take argmax over appropriate axis.


        # vote_matrix = fill and uncomment
        classes = sorted(set(c for pair in self.classifiers.keys() for c in pair))
        class_to_idx = {c: idx for idx, c in enumerate(classes)}
        vote_matrix = np.zeros((X.shape[0], len(classes)))
        for (class_i, class_j), clf in self.classifiers.items():
            pred_ij = clf.predict(X)  # This will be an array of shape (N,) with values in {-1, +1}
            # Now you need to convert pred_ij to votes for class_i and class_j.
            vote_matrix[pred_ij == 1, class_to_idx[class_i]] += 1
            vote_matrix[pred_ij == -1, class_to_idx[class_j]] += 1

        # Finally, predict the class with the most votes for each sample. You can use np.argmax for this.
        # y_pred = np.argmax() fill in properly
        best_idx = np.argmax(vote_matrix, axis=1)
        y_pred = np.array(classes)[best_idx]
        return y_pred



class SoftmaxClassifier:
    """
    Softmax multi-class classifier.
    """

    def __init__(self, lr=0.01, max_iter=1000, tol=1e-6):
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.W = None  # Weight matrix of shape (d, K)
        self.softmax_loss_history = []

    def convert_to_one_hot(self, y: np.ndarray) -> np.ndarray:
        """
        Convert labels {0, 1, ..., K-1} to one-hot encoding.

        Parameters
        ----------
        y : np.ndarray
            Original labels {0, 1, ..., K-1} (N,)

        Returns
        -------
        y_one_hot : np.ndarray
            One-hot encoded labels (N, K)
        """

        # For this you can use ? to create an identity matrix of size K and then index it with labels y to get the one-hot encoding. 
        # Check the shape of the output to make sure it is (N, K) and not (K, N) or (N,) or (K,)
        K = self.K if getattr(self, "K", None) is not None else int(np.max(y)) + 1
        y_one_hot = np.eye(K)[y]
        return y_one_hot


    def softmax_gradient(self, W: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute gradient of softmax loss.

        Parameters
        ----------
        W : np.ndarray
            Weights (d, K)
        X : np.ndarray
            Feature matrix (N, d)
        y : np.ndarray
            Labels {0, 1, ..., K-1} (N,)

        Returns
        -------
        grad : np.ndarray
            Gradient of shape (d, K)
        """

        # Much easier if y is converted to one-hot encoding.

        y_one_hot = self.convert_to_one_hot(y)
        # Now compute the predicted probabilities using the softmax function, and then compute the gradient of
        # the softmax loss with respect to the weights W.
        # Useful functions: np.dot or @ for matrix multiplication, np.exp for computing the exponential
        # You can use keepdims with np.sum when computing the softmax probabilities to make sure the dimensions work out for broadcasting.
        
        # Z = 
        # softmax_probs = 
        # grad =
        N = X.shape[0]
        Z = X @ W  # (N, K)
        Z = Z - np.max(Z, axis=1, keepdims=True)  # numerical stability
        exp_Z = np.exp(Z)
        softmax_probs = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        grad = (X.T @ (softmax_probs - y_one_hot)) / N
        return grad

    def compute_softmax_loss(self, W: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute softmax loss.

        Parameters
        ----------
        W : np.ndarray
            Weights (d, K)
        X : np.ndarray
            Feature matrix (N, d)
        y : np.ndarray
            Labels {0, 1, ..., K-1} (N,)

        Returns
        -------
        loss : float
            Softmax loss
        """

        # Again, much easier if y is converted to one-hot encoding.

        y_one_hot = self.convert_to_one_hot(y)
        # Now compute the predicted probabilities using the softmax function, and then compute the softmax loss.
        # Useful functions: np.dot or @ for matrix multiplication, 
        # np.exp for computing the exponential, np.sum for summing over classes and samples, np.mean for taking mean over samples, 
        # np.clip for clipping the input to exp to avoid overflow
        
        # Z = 
        # softmax_probs = 
        # loss = 
        N = X.shape[0]
        Z = X @ W  # (N, K)
        Z = Z - np.max(Z, axis=1, keepdims=True)  # numerical stability
        Z = np.clip(Z, -500, 500)
        exp_Z = np.exp(Z)
        softmax_probs = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        log_probs = np.log(np.clip(softmax_probs, 1e-12, 1.0))
        loss = -np.mean(np.sum(y_one_hot * log_probs, axis=1))
        return float(loss)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit softmax classifier using batch gradient descent, while adding softmax loss to history.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (N, d)
        y : np.ndarray
            Labels {0, 1, ..., K-1} (N,)
        
            For this you can follow a similar procedure as the fit method for logistic regression, but use the softmax_gradient and softmax_loss instead.
        """
        N, d = X.shape
        self.K = int(np.max(y)) + 1
        self.W = np.zeros((d, self.K))
        for _ in range(self.max_iter):
            grad = self.softmax_gradient(self.W, X, y)
            loss = self.compute_softmax_loss(self.W, X, y)
            self.softmax_loss_history.append(loss)
            self.W = self.W - self.lr * grad
            if np.linalg.norm(grad) < self.tol:
                break
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels {0, 1, ..., K-1}.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (N, d)

        Returns
        -------
        y_pred : np.ndarray
            Predicted class labels (N,)
        """

        # Compute the predicted probabilities using the softmax function, and then predict the class with the highest probability for each sample.
        # Useful functions: np.dot or @ for matrix multiplication, np.exp for computing the exponential, np.argmax for taking argmax over classes to get predicted labels.
        # Again check the shape and clip before computing exp.
        Z = X @ self.W
        Z = Z - np.max(Z, axis=1, keepdims=True)
        Z = np.clip(Z, -500, 500)
        exp_Z = np.exp(Z)
        softmax_probs = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        y_pred = np.argmax(softmax_probs, axis=1)
        return y_pred