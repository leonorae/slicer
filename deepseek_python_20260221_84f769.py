class GeometricFertilityPipeline:
    def __init__(self, model_name, device='cuda'):
        self.model_name = model_name
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.extractor = ActivationExtractor(self.model, model_name)
        self.visualizer = GeometryVisualizer()
        self.report = GeometryReport(model_name)
        
    def run(self, prompts, max_length=128, batch_size=4, 
            n_pca_components=50, umap_sample_size=5000):
        """
        Run full pipeline on a set of prompts.
        """
        print(f"Running geometric analysis on {self.model_name}")
        
        # Step 1: Extract activations
        print("Extracting activations...")
        activations = self.extractor.extract(prompts, max_length, batch_size)
        
        # Step 2: PCA analysis
        print("Running PCA...")
        pca = PCAAnalyzer(n_components=n_pca_components)
        pca_results = pca.fit_transform(activations)
        self.report.add_pca_results(pca_results)
        
        # Step 3: UMAP analysis
        print("Running UMAP...")
        umap = UMAPAnalyzer()
        umap_results = umap.fit_transform(activations, sample_size=umap_sample_size)
        self.report.add_umap_results(umap_results)
        
        # Step 4: Compute metrics
        print("Computing metrics...")
        self.report.add_separability_metrics(activations)
        
        # Position metrics need per-token structure
        # We'll compute on a representative layer
        for name, acts in activations.items():
            if 'post_mlp' in name and 'layer_5' in name:  # Middle layer
                self.report.add_position_metrics(
                    {name: acts.numpy()}, 
                    acts.shape[1]  # seq_len
                )
                break
        
        # Step 5: Generate visualizations
        print("Generating visualizations...")
        self.visualizer.plot_pca_scree(pca_results, self.model_name)
        self.visualizer.plot_umap_components(umap_results, self.model_name)
        self.visualizer.plot_layer_evolution(umap_results, self.model_name)
        
        # Step 6: Save report
        print("Saving report...")
        self.report.save()
        
        return {
            'activations': activations,
            'pca_results': pca_results,
            'umap_results': umap_results,
            'report': self.report
        }