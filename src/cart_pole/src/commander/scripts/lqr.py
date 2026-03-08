import numpy as np
from scipy.linalg import solve_continuous_are

def lqr(A, B, Q, R):
    """
    Continuous-time LQR controller.
    Returns state-feedback gain K.
    """
    # Solve Riccati equation
    P = solve_continuous_are(A, B, Q, R)

    # Compute LQR gain
    K = 1/R * B.T @ P
    return K
