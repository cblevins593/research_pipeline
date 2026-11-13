import numpy as np

def ewma_ledoit_wolf(X, lam=0.7071):
    """
    Implements an Exponentially Weighted Moving Average (EWMA) Ledoit-Wolf 
    linear shrinkage estimator toward a scaled-identity target.
    
    Optimized for large asset universes (N >> T) and short memory half-lives.
    
    Parameters:
    -----------
    X   : np.ndarray (T x N) -> Matrix of T time periods and N assets.
    lam : float              -> EWMA decay factor (e.g., 0.7071 for a 2-month half-life).
    
    Returns:
    --------
    shrunk_cov : np.ndarray (N x N) -> The final invertible EWMA-shrunk covariance matrix.
    delta      : float              -> The optimal shrinkage intensity applied (between 0 and 1).
    """
    T, N = X.shape
    if T <= 1:
        raise ValueError("Time periods (T) must be greater than 1.")

    # 1. Generate Exponential Weights
    # Weights decay backwards in time: today is t=T-1 (weight = 1), oldest is t=0 (weight = lam**(T-1))
    raw_weights = lam ** np.arange(T - 1, -1, -1)
    weights = raw_weights / np.sum(raw_weights)  # Normalize so weights sum to 1
    
    # Calculate Effective Sample Size (T_eff) for EWMA: 1 / sum(w_t^2)
    # This correctly scales down our sample size when an aggressive decay is used.
    sum_sq_weights = np.sum(weights ** 2)
    T_eff = 1.0 / sum_sq_weights

    # 2. Compute Weighted Mean and Center the Data
    weighted_mean = np.sum(X * weights[:, np.newaxis], axis=0)
    X_centered = X - weighted_mean

    # 3. Compute EWMA Sample Covariance Matrix (S)
    scale_factor = 1.0 / (1.0 - sum_sq_weights)
    S = scale_factor * (X_centered.T @ (X_centered * weights[:, np.newaxis]))

    # --- N > T Safeguard ---
    # Add a microscopic ridge penalty to the diagonal to ensure strict numerical positivity
    # and prevent floating-point precision errors during inversion with N=500 assets.
    np.fill_diagonal(S, np.diag(S) + 1e-8)

    # 4. Compute the Constant Variance / Zero Correlation Target Matrix (F)
    # Ledoit-Wolf (2004) Target A: Average variance on the diagonal, zero off-diagonal
    mean_var = np.mean(np.diag(S))
    F = mean_var * np.identity(N)

    # 5. Compute Misalignment (gamma)
    # Squared Frobenius norm between the sample matrix S and target matrix F
    gamma = np.sum((S - F) ** 2)
    if gamma == 0:
        return S, 0.0

    # 6. Compute Exponentially Weighted Estimation Noise (pi)
    # Using np.einsum to broadcast the daily outer products instantly into a 3D tensor (T x N x N).
    # This completely eliminates slow Python loops over time steps.
    Z = np.einsum('ti,tj->tij', X_centered, X_centered)
    
    # Deviation of individual daily cross-products from the aggregate sample covariance matrix
    dev = Z - S[np.newaxis, :, :]
    
    # Sum the squared deviations, scaled by the EWMA temporal weights
    pi_mat = np.sum((dev ** 2) * weights[:, np.newaxis, np.newaxis], axis=0)
    pi = np.sum(pi_mat) * (scale_factor ** 2) / T_eff

    # 7. Compute Asymptotic Covariance (rho)
    # For a scaled identity target, rho collapses to the sum of the variances of the diagonal entries
    phi_diag = np.sum(np.diag(pi_mat)) * (scale_factor ** 2) / T_eff
    rho = phi_diag 

    # 8. Calculate and Cap the Optimal Shrinkage Intensity (delta)
    delta_star = (pi - rho) / gamma
    delta = max(0.0, min(1.0, delta_star))

    # 9. Blend the Matrices to create the final Shrunk Covariance
    shrunk_cov = (1.0 - delta) * S + delta * F

    return shrunk_cov, delta


# ==========================================
# --- Demonstration & Stress Test Setup ---
# ==========================================
if __name__ == "__main__":
    # Simulating your configuration:
    # 60 months of historical lookback data (T = 60) for 500 assets (N = 500)
    T_lookback = 60
    N_assets = 500
    
    # Setting up a 2-month half-life: lambda = 0.5 ** (1.0 / 2.0)
    half_life_months = 2.0
    target_lambda = 0.5 ** (1.0 / half_life_months) # Roughly 0.7071
    
    print(f"--- Configuration ---")
    print(f"Asset Count (N)         : {N_assets}")
    print(f"Lookback Window (T)     : {T_lookback} months")
    print(f"Target Half-Life        : {half_life_months} months")
    print(f"Calculated Lambda (λ)   : {target_lambda:.4f}\n")

    # Generate synthetic random returns data
    np.random.seed(42)
    mock_returns = np.random.randn(T_lookback, N_assets) * 0.05 # 5% volatility scale
    
    # Run the EWMA Ledoit-Wolf Estimator
    cov_matrix, optimal_delta = ewma_ledoit_wolf(mock_returns, lam=target_lambda)
    
    print(f"--- Matrix Outputs ---")
    print(f"Output Matrix Shape    : {cov_matrix.shape}")
    print(f"Optimal Shrinkage (δ)  : {optimal_delta:.4f}")
    
    # Check if the matrix is invertible (will fail if singular/underdetermined)
    try:
        inv_cov = np.linalg.inv(cov_matrix)
        print(f"Matrix Inversion Test  : SUCCESS (The matrix is stable and fully invertible!)")
    except np.linalg.LinAlgError:
        print(f"Matrix Inversion Test  : FAILED")