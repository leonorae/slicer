def compute_position_dependence(per_token_activations, position_labels):
    """
    Measure how much variance is explained by position.
    For learned embeddings, we expect high position predictivity.
    For RoPE, we expect low position predictivity in the residual stream.
    """
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score
    
    # per_token_activations: (n_sequences, seq_len, hidden_dim)
    # Flatten to (n_tokens, hidden_dim)
    n_seq, seq_len, hidden = per_token_activations.shape
    flat_acts = per_token_activations.reshape(-1, hidden)
    
    # Create position labels (0 to seq_len-1 repeated for each sequence)
    pos_labels = np.tile(np.arange(seq_len), n_seq)
    
    # Train linear model to predict position from activations
    model = Ridge(alpha=1.0)
    
    # Cross-validated R² score
    scores = cross_val_score(model, flat_acts, pos_labels, 
                             cv=5, scoring='r2')
    
    # Also compute mutual information for non-linear dependence
    from sklearn.feature_selection import mutual_info_regression
    
    # Sample for MI (computationally expensive)
    sample_idx = np.random.choice(len(flat_acts), min(5000, len(flat_acts)), replace=False)
    mi = mutual_info_regression(flat_acts[sample_idx], pos_labels[sample_idx])
    
    return {
        'position_r2_mean': np.mean(scores),
        'position_r2_std': np.std(scores),
        'mutual_info_mean': np.mean(mi),
        'mutual_info_max': np.max(mi)
    }