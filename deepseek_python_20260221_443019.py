class UMAPAnalyzer:
    def __init__(self, n_components=2, n_neighbors=15, min_dist=0.1):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.reducer = None
        
    def fit_transform(self, activations_dict, sample_size=5000):
        """Fit UMAP on a sample and transform all data"""
        results = {}
        
        # First, sample tokens from all layers to fit the reducer
        all_samples = []
        for name, tensor in activations_dict.items():
            flat = tensor.view(-1, tensor.shape[-1]).numpy()
            # Random sample
            if len(flat) > sample_size:
                idx = np.random.choice(len(flat), sample_size, replace=False)
                all_samples.append(flat[idx])
            else:
                all_samples.append(flat)
        
        # Combine samples
        combined = np.vstack(all_samples)
        
        # Fit UMAP on combined sample
        self.reducer = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            random_state=42
        )
        self.reducer.fit(combined)
        
        # Transform each activation set
        for name, tensor in activations_dict.items():
            flat = tensor.view(-1, tensor.shape[-1]).numpy()
            transformed = self.reducer.transform(flat)
            
            results[name] = {
                'transformed': transformed,
                'per_token': transformed.reshape(tensor.shape[0], tensor.shape[1], -1)
            }
        
        return results