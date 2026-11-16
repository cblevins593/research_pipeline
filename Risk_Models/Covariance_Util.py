import numpy as np
import pandas as pd

def create_covariance_matrix(X: pd.DataFrame, ledoit_wolf: bool = False, ewma_halflife: int|float=float('inf'), identity_only: bool = False) -> pd.DataFrame:

    """
    Calculate the covariance matrix of a given DataFrame.

    Parameters:
    -----------
    X : pd.DataFrame
        Input DataFrame where rows are observations and columns are variables.
    ledoit_wolf : bool, optional
        If True, apply Ledoit-Wolf shrinkage to the covariance matrix. Default is False.
    ewma_halflife : int or float, optional
        Half-life for Exponentially Weighted Moving Average (EWMA) if ledoit_wolf is True. Default is float('inf').
    identity_only : bool, optional
        If True, return only the diagonal elements of the covariance matrix. Default is False.

    Returns:
    --------
    pd.DataFrame
        Covariance matrix of the input DataFrame with the columns of the input DataFrame..
    """
    
    if X.isna().any().any():
        null_cols = X.columns[X.isna().any()].tolist()
        raise ValueError(f"create_covariance_matrix input dataFrame contains missing values in columns: {null_cols}")
    
    lam = 0.5 ** (1.0 / ewma_halflife)
    columns=X.columns

    #Convert to numpy array for calculations
    X_array=np.array(X)
    T,N=X_array.shape

    # Calculate the weights for EWMA
    raw_weights = lam ** np.arange(T - 1, -1, -1)
    weights = raw_weights / np.sum(raw_weights)

    # Calculate the effective sample size for EWMA
    sum_sq_weights = np.sum(weights ** 2)
    T_eff = 1.0 / sum_sq_weights

    # Center the data around the weighted mean for each column
    weighted_mean = np.sum(X_array * weights[:, np.newaxis], axis=0)
    X_centered = X_array - weighted_mean

    # Calculate the EWMA sample covariance matrix
    if ledoit_wolf: #Ledoit_wolf makes the ddof=0
        scale_factor=1
    else:
        scale_factor = 1.0 / (1.0 - sum_sq_weights)
        
    S = scale_factor * (X_centered.T @ (X_centered * weights[:, np.newaxis]))

    if ledoit_wolf:
        # Compute the Constant Variance / Zero Correlation Target Matrix (F)
        mean_var = np.mean(np.diag(S))
        F = mean_var * np.identity(N)

        # Compute Misalignment (gamma) squared frobenius norm between the sample matrix S and target matrix F
        gamma = np.sum((S - F) ** 2)

        if gamma == 0:
            output = S

        else:
            # Z is an array of shape (T, N, N) where each of T slices is the outer product of the centered returns at time t
            # dev is a 3D array where each slice is the difference between the outer product at time t and the sample covariance matrix S
            # Deviation of individual outerproducts from the aggregate sample covariance matrix which is the aggregate of all outer products
            Z = np.einsum('ti,tj->tij', X_centered, X_centered) # Shape(T,N,N)
            dev = Z - S[np.newaxis, :, :] # Shape(T,N,N)

    
            # Sum the squared deviations, scaled by the EWMA weights
            # Result (pi) is an aggregate measure of sampling uncertainty
            pi_mat = np.sum((dev ** 2) * weights[:, np.newaxis, np.newaxis], axis=0) # Shape(N,N)
            pi = np.sum(pi_mat) * (scale_factor ** 2) / T_eff


            # Calculate the optimal shrinkage intensity (delta) bound the result between 0 and 1
            delta_star=pi/gamma
            delta = max(0.0, min(1.0, delta_star))

            # Create the final shrunk covariance matrix
            shrunk_cov = (1.0 - delta) * S + delta * F
            output=shrunk_cov

    # If not ledoit wolf then output the sample covariance matrix
    else:
        output=S

    if identity_only:
        output=np.diag(np.diag(output))


    return pd.DataFrame(output,columns = columns, index = columns)
    