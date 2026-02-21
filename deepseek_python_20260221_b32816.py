class PCAAnalyzer:
    def __init__(self, n_components=50):
        self.n_components = n_components
        self.pca = None
        self.explained_variance_ratio = None
        
    def fit_transform(self, activations_dict):
        """Fit PCA on all activations and transform each set"""
        results = {}
        
        for name, tensor in activations_dict.items():
            # Flatten tokens: (n_tokens, hidden_dim)
            # tensor shape: (batch, seq_len, hidden_dim)
            n_tokens = tensor.shape[0] * tensor.shape[1]
            flat = tensor.view(n_tokens, -1).numpy()
            
            # Fit PCA
            self.pca = PCA(n_components=min(self.n_components, flat.shape[1]))
            transformed = self.pca.fit_transform(flat)
            
            # Store results
            results[name] = {
                'transformed': transformed,
                'explained_variance_ratio': self.pca.explained_variance_ratio_,
                'components': self.pca.components_,
                'mean': self.pca.mean_
            }
            
            # Reshape back to sequence structure for visualization
            results[name]['per_token'] = transformed.reshape(tensor.shape[0], tensor.shape[1], -1)
        
        return results