def compute_separability(attn_activations, mlp_activations):
    """
    Compute separability between attention and MLP point clouds.
    Uses silhouette score or centroid distance ratio.
    """
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    
    # Combine and label
    n_attn = len(attn_activations)
    n_mlp = len(mlp_activations)
    
    combined = np.vstack([attn_activations, mlp_activations])
    labels = np.array([0] * n_attn + [1] * n_mlp)
    
    # Standardize
    scaler = StandardScaler()
    combined_scaled = scaler.fit_transform(combined)
    
    # Silhouette score (higher = more separable, range -1 to 1)
    sil_score = silhouette_score(combined_scaled, labels)
    
    # Alternative: compute distance between centroids / average intra-cluster distance
    centroid_attn = np.mean(attn_activations, axis=0)
    centroid_mlp = np.mean(mlp_activations, axis=0)
    centroid_dist = np.linalg.norm(centroid_attn - centroid_mlp)
    
    # Average distance to own centroid
    intra_attn = np.mean([np.linalg.norm(x - centroid_attn) for x in attn_activations])
    intra_mlp = np.mean([np.linalg.norm(x - centroid_mlp) for x in mlp_activations])
    avg_intra = (intra_attn + intra_mlp) / 2
    
    separability_ratio = centroid_dist / (avg_intra + 1e-8)
    
    return {
        'silhouette_score': sil_score,
        'separability_ratio': separability_ratio,
        'centroid_distance': centroid_dist,
        'avg_intra_distance': avg_intra
    }