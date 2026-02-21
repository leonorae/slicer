class ActivationExtractor:
    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name
        self.activations = defaultdict(list)
        self.hooks = []
        
    def register_hooks(self):
        """Register forward hooks at each extraction point"""
        # For each layer, attach hooks to specific submodules
        for layer_idx, layer in enumerate(self.model.transformer.h):
            # Pre-attention norm (if using pre-norm architecture)
            if hasattr(layer, 'ln_1'):
                hook = layer.ln_1.register_forward_hook(
                    self._make_hook(f'layer_{layer_idx}_pre_attn_norm')
                )
                self.hooks.append(hook)
            
            # Attention output (before residual)
            if hasattr(layer.attn, 'c_proj'):
                # For GPT-2 style
                hook = layer.attn.c_proj.register_forward_hook(
                    self._make_hook(f'layer_{layer_idx}_attn_out')
                )
                self.hooks.append(hook)
            
            # Post-attention residual (we can compute this, but maybe capture after add)
            # MLP similarly...
            
    def _make_hook(self, name):
        def hook(module, input, output):
            # Detach and move to CPU to save GPU memory
            self.activations[name].append(output.detach().cpu())
        return hook
    
    def extract(self, text_prompts, max_length=128, batch_size=4):
        """Run forward passes and collect activations"""
        self.activations.clear()
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        for i in range(0, len(text_prompts), batch_size):
            batch = text_prompts[i:i+batch_size]
            inputs = tokenizer(batch, return_tensors="pt", 
                               padding=True, truncation=True, 
                               max_length=max_length)
            
            with torch.no_grad():
                _ = self.model(**inputs)
        
        # Concatenate all collected activations
        for key in self.activations:
            self.activations[key] = torch.cat(self.activations[key], dim=0)
        
        return self.activations