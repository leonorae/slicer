def compute_intrinsic_dimensionality(explained_variance_ratio, threshold=0.9):
    """
    Compute number of components needed to reach threshold variance.
    Also compute participation ratio (continuous measure).
    """
    # Cumulative variance
    cumsum = np.cumsum(explained_variance_ratio)
    
    # Number of components to reach threshold
    n_to_threshold = np.searchsorted(cumsum, threshold) + 1
    
    # Participation ratio: (sum λ_i)² / sum λ_i²
    # Where λ_i are eigenvalues (normalized)
    # Higher = more dimensions active
    eigenvals = explained_variance_ratio * len(explained_variance_ratio)  # denormalize approx
    participation_ratio = (np.sum(eigenvals) ** 2) / (np.sum(eigenvals ** 2) + 1e-8)
    
    return {
        'n_to_90_percent': n_to_threshold,
        'participation_ratio': participation_ratio,
        'explained_variance_ratio': explained_variance_ratio
    }