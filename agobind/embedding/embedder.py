"""
Functionality for embedding genomic sequences using pre-trained models.
"""

import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizer

from agobind.models.hub import ModelHub


class SequenceEmbedder:
    """
    Embed genomic sequences using pre-trained models from Hugging Face.
    """
    
    def __init__(
        self,
        model_name: str,
        output_dir: Optional[str] = None,
        device: Optional[str] = None,
        pooling_strategy: str = "mean",
        layer: int = -1,
        **kwargs
    ):
        """
        Initialize the sequence embedder.
        
        Args:
            model_name: Name of the pre-trained model to use for embedding.
            output_dir: Directory to save embeddings to.
            device: Device to run the model on ("cpu", "cuda", "mps").
            pooling_strategy: Strategy to pool token embeddings into sequence embeddings.
                             Options: "mean", "max", "cls" (use [CLS] token embedding).
            layer: Which layer of the model to use for embeddings.
                  -1 means the last layer, -2 means the second to last, etc.
            **kwargs: Additional arguments to pass to the model loader.
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.pooling_strategy = pooling_strategy
        self.layer = layer
        
        if self.output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Load model and tokenizer
        model_hub = ModelHub(**kwargs)
        model_dict = model_hub.load_model(
            model_name, 
            for_classification=False,
            **kwargs
        )
        
        self.tokenizer = model_dict["tokenizer"]
        self.model = model_dict["model"].to(self.device)
        
        # Ensure the tokenizer has necessary tokens for padding
        self._ensure_special_tokens()
        
        # Set model to evaluation mode
        self.model.eval()
    
    def _ensure_special_tokens(self):
        """
        Ensure the tokenizer has all necessary special tokens, like padding token.
        This is required for batched processing of sequences.
        """
        # Check if pad token is missing and add it if needed
        if self.tokenizer.pad_token is None:
            # First try using eos_token as pad_token if it exists
            if self.tokenizer.eos_token is not None:
                print(f"No pad token found in tokenizer. Using eos_token '{self.tokenizer.eos_token}' as pad_token.")
                self.tokenizer.pad_token = self.tokenizer.eos_token
            # Otherwise try using sep_token
            elif hasattr(self.tokenizer, 'sep_token') and self.tokenizer.sep_token is not None:
                print(f"No pad token found in tokenizer. Using sep_token '{self.tokenizer.sep_token}' as pad_token.")
                self.tokenizer.pad_token = self.tokenizer.sep_token
            # Otherwise try using cls_token
            elif hasattr(self.tokenizer, 'cls_token') and self.tokenizer.cls_token is not None:
                print(f"No pad token found in tokenizer. Using cls_token '{self.tokenizer.cls_token}' as pad_token.")
                self.tokenizer.pad_token = self.tokenizer.cls_token
            # As a last resort, add a new [PAD] token
            else:
                print("No suitable token found for padding. Adding '[PAD]' as a special token.")
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                # If we added new tokens, we need to resize the model embeddings
                self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Check for and set other essential tokens if missing
        need_to_resize = False
        
        # Check for CLS token (for sequence classification)
        if hasattr(self.tokenizer, 'cls_token') and self.tokenizer.cls_token is None:
            if self.tokenizer.bos_token is not None:
                print(f"No cls token found in tokenizer. Using bos_token '{self.tokenizer.bos_token}' as cls_token.")
                self.tokenizer.cls_token = self.tokenizer.bos_token
            else:
                print("No suitable token found for CLS. Adding '[CLS]' as a special token.")
                special_tokens = {'cls_token': '[CLS]'}
                self.tokenizer.add_special_tokens(special_tokens)
                need_to_resize = True
        
        # Check for SEP token (for sequence pair tasks)
        if hasattr(self.tokenizer, 'sep_token') and self.tokenizer.sep_token is None:
            if self.tokenizer.eos_token is not None:
                print(f"No sep token found in tokenizer. Using eos_token '{self.tokenizer.eos_token}' as sep_token.")
                self.tokenizer.sep_token = self.tokenizer.eos_token
            else:
                print("No suitable token found for SEP. Adding '[SEP]' as a special token.")
                special_tokens = {'sep_token': '[SEP]'}
                self.tokenizer.add_special_tokens(special_tokens)
                need_to_resize = True
        
        # If we added any new tokens, resize model embeddings
        if need_to_resize:
            self.model.resize_token_embeddings(len(self.tokenizer))
    
    def embed_sequences(
        self,
        sequences: List[str],
        batch_size: int = 32,
        return_numpy: bool = True,
        show_progress: bool = True,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Embed a list of genomic sequences.
        
        Args:
            sequences: List of genomic sequences to embed.
            batch_size: Batch size for processing.
            return_numpy: Whether to return a numpy array (True) or PyTorch tensor (False).
            show_progress: Whether to show a progress bar.
        
        Returns:
            Array or tensor of embeddings with shape (num_sequences, embedding_dim).
        """
        all_embeddings = []
        
        # Process in batches to avoid OOM errors
        iterator = range(0, len(sequences), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Embedding sequences")
        
        for i in iterator:
            # Get batch of sequences
            batch_sequences = sequences[i:i+batch_size]
            
            # Tokenize the sequences
            inputs = self.tokenizer(
                batch_sequences,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                try:
                    outputs = self.model(**inputs, output_hidden_states=True)
                except RuntimeError as e:
                    if "CUDA" in str(e) or "GPU" in str(e):
                        print(f"CUDA error occurred: {e}")
                        print("Falling back to CPU...")
                        self.device = "cpu"
                        self.model = self.model.to(self.device)
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        outputs = self.model(**inputs, output_hidden_states=True)
                    else:
                        raise
            
            # Get hidden states from the specified layer
            if hasattr(outputs, "hidden_states"):
                hidden_states = outputs.hidden_states[self.layer]
            else:
                # Fallback to last_hidden_state if hidden_states not available
                hidden_states = outputs.last_hidden_state
            
            # Apply pooling strategy
            if self.pooling_strategy == "mean":
                # Mean pooling: average over all tokens excluding padding
                attention_mask = inputs["attention_mask"]
                batch_embeddings = self._mean_pooling(hidden_states, attention_mask)
            elif self.pooling_strategy == "max":
                # Max pooling: take max value over all tokens
                batch_embeddings = torch.max(hidden_states, dim=1)[0]
            elif self.pooling_strategy == "cls":
                # CLS pooling: use [CLS] token embedding (first token)
                batch_embeddings = hidden_states[:, 0, :]
            else:
                raise ValueError(f"Unsupported pooling strategy: {self.pooling_strategy}")
            
            # Move to CPU to save GPU memory
            all_embeddings.append(batch_embeddings.cpu())
        
        # Concatenate all batches
        all_embeddings = torch.cat(all_embeddings, dim=0)
        
        # Convert to numpy if requested
        if return_numpy:
            all_embeddings = all_embeddings.numpy()
        
        return all_embeddings
    
    def _mean_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Mean pooling operation to get sequence-level embeddings.
        
        Args:
            token_embeddings: Embeddings for each token in the sequences.
                            Shape: (batch_size, sequence_length, hidden_size)
            attention_mask: Attention mask to avoid pooling padding tokens.
                          Shape: (batch_size, sequence_length)
        
        Returns:
            Sequence embeddings with shape (batch_size, hidden_size).
        """
        # Expand attention_mask to match token_embeddings dimensions
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # Apply mask and compute mean
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        return sum_embeddings / sum_mask
    
    def save_embeddings(self, embeddings: Union[np.ndarray, torch.Tensor], filename: str) -> str:
        """
        Save embeddings to disk.
        
        Args:
            embeddings: Embeddings to save.
            filename: Name of the file to save embeddings to.
        
        Returns:
            Path to the saved embeddings file.
        """
        if self.output_dir is None:
            raise ValueError("output_dir must be specified to save embeddings.")
        
        # Convert to numpy if needed
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        
        # Create full path
        filepath = os.path.join(self.output_dir, filename)
        
        # Add .npy extension if not present
        if not filepath.endswith(".npy"):
            filepath += ".npy"
        
        # Save the embeddings
        np.save(filepath, embeddings)
        
        return filepath
    
    def load_embeddings(self, filename: str) -> np.ndarray:
        """
        Load embeddings from disk.
        
        Args:
            filename: Name of the file to load embeddings from.
        
        Returns:
            Numpy array of embeddings.
        """
        if self.output_dir is None:
            raise ValueError("output_dir must be specified to load embeddings.")
        
        # Create full path
        filepath = os.path.join(self.output_dir, filename)
        
        # Add .npy extension if not present
        if not filepath.endswith(".npy"):
            filepath += ".npy"
        
        # Check if file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Embeddings file not found: {filepath}")
        
        # Load the embeddings
        return np.load(filepath) 