"""
Interface to the Hugging Face Hub for genomic models.
"""

import os
from typing import Dict, List, Optional, Union

import torch
from transformers import AutoConfig, AutoModel, AutoModelForSequenceClassification, AutoTokenizer


class ModelHub:
    """
    Interface to access pre-trained genomic models from Hugging Face.
    
    This class provides functionality to:
    1. List available pre-trained genomic models
    2. Load models and tokenizers from Hugging Face Hub
    3. Get detailed information about models
    
    The list of supported models focuses on those that have been tested and 
    are known to work well with genomic sequences. However, users can also
    use other models from Hugging Face, with a warning that they may not be
    optimized for genomic data.
    
    Note: When loading models, special token handling (such as padding tokens)
    is managed by the SequenceEmbedder class, which ensures that all required
    tokens are available regardless of the model used.
    """
    
    # List of popular genomic models that have been tested with this package
    SUPPORTED_GENOMIC_MODELS = [
        "AIRI-Institute/gena-lm-bert-base",
        "AIRI-Institute/gena-lm-bigbird-base-t2t",
        "InstaDeepAI/nucleotide-transformer-500m-human-ref",
        "InstaDeepAI/nucleotide-transformer-2.5b-1000g",
        "zhihan1996/DNA_bert_6",
    ]
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the model hub.
        
        Args:
            cache_dir: Directory to cache downloaded models.
                      If None, uses the default Hugging Face cache.
        """
        self.cache_dir = cache_dir
        if cache_dir is not None:
            os.makedirs(cache_dir, exist_ok=True)
    
    def list_genomic_models(self) -> List[str]:
        """
        List available pre-trained genomic models.
        
        Returns:
            List of model names available for use.
        """
        return self.SUPPORTED_GENOMIC_MODELS
    
    def load_model(
        self, 
        model_name: str, 
        for_classification: bool = False,
        num_labels: int = 2,
        trust_remote_code: bool = True,
        **kwargs
    ) -> Union[torch.nn.Module, Dict[str, torch.nn.Module]]:
        """
        Load a pre-trained model from Hugging Face.
        
        Args:
            model_name: Name of the model to load.
            for_classification: Whether to load the model for sequence classification.
            num_labels: Number of labels for classification (default: 2 for binary).
            trust_remote_code: Whether to trust and execute code from the model repository.
            **kwargs: Additional arguments to pass to the model constructor.
        
        Returns:
            If for_classification is False:
                A dictionary containing the tokenizer and model.
            If for_classification is True:
                A dictionary containing the tokenizer and classification model.
                
        Note:
            Some models may not have all special tokens (like pad_token) defined.
            These issues are handled by the SequenceEmbedder class which ensures
            proper token configuration before using the tokenizer.
        """
        # Check if the model name is in the list of supported models
        if model_name not in self.SUPPORTED_GENOMIC_MODELS:
            print(f"Warning: {model_name} is not in the list of supported genomic models.")
            print("It may not be optimized for genomic sequences.")
        
        # Attempt to load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=self.cache_dir,
            trust_remote_code=trust_remote_code,
            **kwargs
        )
        
        # Load the model
        if for_classification:
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=num_labels,
                cache_dir=self.cache_dir,
                trust_remote_code=trust_remote_code,
                **kwargs
            )
        else:
            model = AutoModel.from_pretrained(
                model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=trust_remote_code,
                **kwargs
            )
        
        return {
            "tokenizer": tokenizer,
            "model": model
        }

    def get_model_info(self, model_name: str) -> Dict:
        """
        Get detailed information about a model.
        
        Args:
            model_name: Name of the model to get information for.
        
        Returns:
            Dictionary with model information.
        """
        # Load the configuration to get model details
        config = AutoConfig.from_pretrained(
            model_name,
            cache_dir=self.cache_dir,
            trust_remote_code=True
        )
        
        # Safe function to get attributes with fallbacks
        def safe_get(obj, attr, default="Unknown"):
            if hasattr(obj, attr):
                val = getattr(obj, attr)
                if val is not None:
                    if isinstance(val, list) and len(val) > 0:
                        return val[0]
                    elif not isinstance(val, list):
                        return val
            return default
        
        # Get basic model information
        model_info = {
            "name": model_name,
            "architecture": safe_get(config, "architectures"),
            "hidden_size": safe_get(config, "hidden_size"),
            "num_hidden_layers": safe_get(config, "num_hidden_layers"),
            "vocab_size": safe_get(config, "vocab_size"),
        }
        
        return model_info 