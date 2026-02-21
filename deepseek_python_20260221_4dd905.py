def compute_cluster_quality(activations_dict, layer_idx=None):
    """
    For each component type (attention, MLP, residual), measure:
    - Intra-component similarity
    - Cross-component separation
    """
    results = {}
    
    # Group by component type across layers
    attn_acts = []
    mlp_acts = []
    residual_acts = []
    
    for name, acts in activations_dict.items():
        if 'attn_out' in name:
            attn_acts.append(acts)
        elif 'mlp_out' in name:
            mlp_acts.append(acts)
        elif 'post_mlp' in name or 'post_attn' in name:
            residual_acts.append(acts)
    
    # Combine across layers if desired
    attn_combined = np.vstack([a.reshape(-1, a.shape[-1]) for a in attn_acts])
    mlp_combined = np.vstack([a.reshape(-1, a.shape[-1]) for a in mlp_acts])
    residual_combined = np.vstack([a.reshape(-1, a.shape[-1]) for a in residual_acts])
    
    # Compute pairwise distances
    results['attn_vs_mlp'] = compute_separability(attn_combined, mlp_combined)
    results['attn_vs_residual'] = compute_separability(attn_combined, residual_combined)
    results['mlp_vs_residual'] = compute_separability(mlp_combined, residual_combined)
    
    return results