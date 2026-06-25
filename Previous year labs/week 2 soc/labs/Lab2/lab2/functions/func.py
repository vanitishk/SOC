import numpy as np
from typing import Callable, Optional, Tuple, List, Any, Union

class func:
    def __init__(self):
        pass
    def __call__(self, x: np.ndarray) -> np.ndarray: # type: ignore
        return self.eval(x)
    def eval(self, x: np.ndarray) -> np.ndarray:# type: ignore
        pass
    def grad(self, x: np.ndarray) -> np.ndarray: # type: ignore
        pass
    def hessian(self, x: np.ndarray) -> np.ndarray: # type: ignore
        pass 


class LSLR (func):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = X
        self.y = y
        self.n_samples, self.n_features = X.shape
        super().__init__()

    def eval(self, x: np.ndarray) -> np.ndarray: #type: ignore
        w = x
        residuals = self.X @ w - self.y
        return 0.5 * np.mean(residuals ** 2)
    
    def grad(self, x: np.ndarray) -> np.ndarray: # type: ignore
        w = x
        residuals = self.X @ w - self.y
        return (self.X.T @ residuals) / self.n_samples
    
    def hessian(self, x: np.ndarray) -> np.ndarray:  # type: ignore
        return (self.X.T @ self.X) / self.n_samples


class rosenbrock(func):
    def __init__(self, a: float = 1.0, b: float = 100.0) -> None:
        self.a = a
        self.b = b
        super().__init__()

    def eval(self, x: np.ndarray) -> np.ndarray: # type: ignore
        # f(x,y) = (a - x_0)^2 + b(x_1 - x_0^2)^2
        return (self.a - x[0])**2 + self.b * (x[1] - x[0]**2)**2

    def grad(self, x: np.ndarray) -> np.ndarray: # type: ignore
        # Partial derivative with respect to x_0
        df_dx0 = -2 * (self.a - x[0]) - 4 * self.b * x[0] * (x[1] - x[0]**2)
        # Partial derivative with respect to x_1
        df_dx1 = 2 * self.b * (x[1] - x[0]**2)
        
        return np.array([df_dx0, df_dx1])

    def hessian(self, x: np.ndarray) -> np.ndarray: # type: ignore
        # Second-order partial derivatives
        d2f_dx0_dx0 = 2 - 4 * self.b * (x[1] - 3 * x[0]**2)
        d2f_dx0_dx1 = -4 * self.b * x[0]
        d2f_dx1_dx0 = -4 * self.b * x[0]
        d2f_dx1_dx1 = 2 * self.b
        
        return np.array([
            [d2f_dx0_dx0, d2f_dx0_dx1],
            [d2f_dx1_dx0, d2f_dx1_dx1]
        ])

class rot_anisotropic(func):
    def __init__(self, U: np.ndarray, V: np.ndarray, S: np.ndarray, b: np.ndarray) -> None:
        self.U = U
        self.V = V
        self.S = S
        self.b = b
        
        # Precompute the Q matrix (Q = U * S * V^T) to save computation 
        # during eval, grad, and hessian calls
        self.Q = self.U @ self.S @ self.V.T
        super().__init__()

    def eval(self, x: np.ndarray) -> np.ndarray: # type: ignore
        # f(x) = x^T Q x - b^T x
        return x.T @ self.Q @ x - self.b.T @ x

    def grad(self, x: np.ndarray) -> np.ndarray: # type: ignore
        # Gradient of (x^T Q x - b^T x) is (Q + Q^T)x - b
        return (self.Q + self.Q.T) @ x - self.b

    def hessian(self, x: np.ndarray) -> np.ndarray: # type: ignore
        # Hessian of (x^T Q x - b^T x) is Q + Q^T
        return self.Q + self.Q.T

if __name__ == "__main__":
    pass
