class GeometryVisualizer:
    def __init__(self, save_dir='./viz_output'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
    def plot_pca_scree(self, pca_results, model_name):
        """Plot explained variance ratio vs components"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        for name, result in pca_results.items():
            evr = result['explained_variance_ratio']
            cumsum = np.cumsum(evr)
            
            axes[0].plot(range(1, len(evr)+1), evr, label=name, alpha=0.7)
            axes[1].plot(range(1, len(evr)+1), cumsum, label=name, alpha=0.7)
        
        axes[0].set_xlabel('Principal Component')
        axes[0].set_ylabel('Explained Variance Ratio')
        axes[0].set_title('Scree Plot')
        axes[0].legend()
        
        axes[1].set_xlabel('Number of Components')
        axes[1].set_ylabel('Cumulative Explained Variance')
        axes[1].set_title('Cumulative Variance')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(self.save_dir / f'{model_name}_pca_scree.png', dpi=150)
        plt.show()
    
    def plot_umap_components(self, umap_results, model_name, 
                              color_by='layer', highlight_attention=True):
        """
        Plot UMAP projections, coloring by layer or component type.
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Flatten axes
        axes = axes.flatten()
        
        for idx, (name, result) in enumerate(umap_results.items()):
            if idx >= len(axes):
                break
                
            transformed = result['transformed']  # (n_tokens, 2)
            
            # Determine colors based on name
            if 'attn_out' in name:
                color = 'red'
                label = 'Attention'
            elif 'mlp_out' in name:
                color = 'blue'
                label = 'MLP'
            elif 'residual' in name or 'post' in name:
                color = 'green'
                label = 'Residual'
            else:
                color = 'gray'
                label = 'Other'
            
            axes[idx].scatter(transformed[:, 0], transformed[:, 1], 
                             c=color, s=1, alpha=0.5, label=label)
            axes[idx].set_title(name)
            axes[idx].legend()
        
        plt.tight_layout()
        plt.savefig(self.save_dir / f'{model_name}_umap_components.png', dpi=150)
        plt.show()
    
    def plot_layer_evolution(self, umap_results, model_name, layer_indices=None):
        """
        Show how representations evolve through layers.
        """
        # Filter for specific layer types (e.g., post-MLP residual at each layer)
        layer_acts = {}
        for name, result in umap_results.items():
            if 'post_mlp' in name and 'layer' in name:
                # Extract layer number
                match = re.search(r'layer_(\d+)_post_mlp', name)
                if match:
                    layer = int(match.group(1))
                    layer_acts[layer] = result['transformed']
        
        # Plot in order
        layers = sorted(layer_acts.keys())
        n_layers = len(layers)
        cols = 4
        rows = (n_layers + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 3*rows))
        axes = axes.flatten()
        
        for idx, layer in enumerate(layers):
            if idx >= len(axes):
                break
            acts = layer_acts[layer]
            axes[idx].scatter(acts[:, 0], acts[:, 1], s=1, alpha=0.5)
            axes[idx].set_title(f'Layer {layer}')
        
        # Hide unused subplots
        for idx in range(len(layers), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'{model_name}: Layerwise Evolution (Post-MLP)')
        plt.tight_layout()
        plt.savefig(self.save_dir / f'{model_name}_layer_evolution.png', dpi=150)
        plt.show()