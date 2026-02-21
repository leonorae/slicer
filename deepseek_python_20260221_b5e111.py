class GeometryReport:
    def __init__(self, model_name):
        self.model_name = model_name
        self.results = {
            'model_name': model_name,
            'pca': {},
            'umap': {},
            'metrics': {}
        }
    
    def add_pca_results(self, pca_results):
        self.results['pca'] = pca_results
        
        # Extract key metrics
        for name, result in pca_results.items():
            if 'post_mlp' in name:  # Focus on final representations
                self.results['metrics']['intrinsic_dimensionality'] = \
                    compute_intrinsic_dimensionality(result['explained_variance_ratio'])
    
    def add_umap_results(self, umap_results):
        self.results['umap'] = umap_results
    
    def add_separability_metrics(self, activations_dict):
        self.results['metrics']['separability'] = \
            compute_cluster_quality(activations_dict)
    
    def add_position_metrics(self, per_token_acts, seq_len):
        # Need to pass position-labeled data
        if 'post_mlp_layer_last' in per_token_acts:
            self.results['metrics']['position_dependence'] = \
                compute_position_dependence(per_token_acts['post_mlp_layer_last'], 
                                           np.arange(seq_len))
    
    def save(self):
        with open(f'reports/{self.model_name}_report.json', 'w') as f:
            # Convert numpy arrays to lists for JSON
            report_dict = json_numpy_converter(self.results)
            json.dump(report_dict, f, indent=2)
        
        # Also save a human-readable summary
        with open(f'reports/{self.model_name}_summary.txt', 'w') as f:
            f.write(f"Geometric Fertility Report: {self.model_name}\n")
            f.write("="*50 + "\n\n")
            
            if 'intrinsic_dimensionality' in self.results['metrics']:
                idim = self.results['metrics']['intrinsic_dimensionality']
                f.write(f"Intrinsic Dimensionality:\n")
                f.write(f"  Components to 90% variance: {idim['n_to_90_percent']}\n")
                f.write(f"  Participation ratio: {idim['participation_ratio']:.2f}\n\n")
            
            if 'separability' in self.results['metrics']:
                sep = self.results['metrics']['separability']
                f.write(f"Attention-MLP Separability:\n")
                f.write(f"  Silhouette score: {sep['attn_vs_mlp']['silhouette_score']:.3f}\n")
                f.write(f"  Separability ratio: {sep['attn_vs_mlp']['separability_ratio']:.3f}\n\n")
            
            if 'position_dependence' in self.results['metrics']:
                pos = self.results['metrics']['position_dependence']
                f.write(f"Position Dependence:\n")
                f.write(f"  R² predicting position: {pos['position_r2_mean']:.3f}\n")