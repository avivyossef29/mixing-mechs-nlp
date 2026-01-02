"""
Model wrappers for binding task evaluation.

This module provides an abstract base class and concrete implementations
for different language models, following the same pattern as Pipeline(ABC)
in CausalAbstraction/neural/pipeline.py.
"""

from abc import ABC, abstractmethod
from typing import Optional, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTNeoXTokenizerFast


class BindingModelWrapper(ABC):
    """Abstract base class for model wrappers in binding task evaluation.
    
    Follows the same pattern as Pipeline(ABC) in CausalAbstraction/neural/pipeline.py.
    Each concrete implementation handles model-specific loading and prompt formatting.
    """
    
    def __init__(self, model_id: str, **kwargs: Any):
        """
        Initialize the model wrapper.
        
        Args:
            model_id: HuggingFace model identifier
            **kwargs: Additional arguments passed to _load_model()
        """
        self.model_id = model_id
        self.model = None
        self.tokenizer = None
        self._load_model(**kwargs)
    
    @abstractmethod
    def _load_model(self, **kwargs: Any):
        """Load model and tokenizer with model-specific parameters."""
        pass
    
    @abstractmethod
    def format_prompt(self, prompt: str, queried_object: Optional[str] = None) -> str:
        """
        Format prompt according to model's expected format.
        
        Args:
            prompt: Raw input prompt from the binding task
            queried_object: Optional queried object name (for context)
            
        Returns:
            Formatted prompt string ready for the model
        """
        pass
    
    def generate_response(self, prompt: str, max_new_tokens: int = 10) -> str:
        """
        Generate response from model.
        
        Args:
            prompt: Formatted prompt string
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated response text
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )
    
    def get_display_name(self) -> str:
        """Return display name for the model."""
        return self.model_id
    
    def get_num_layers(self) -> int:
        """Return number of layers in the model."""
        if hasattr(self.model.config, 'n_layers'):
            return self.model.config.n_layers
        elif hasattr(self.model.config, 'num_hidden_layers'):
            return self.model.config.num_hidden_layers
        elif hasattr(self.model.config, 'n_layer'):
            return self.model.config.n_layer
        else:
            return -1


class MPTModelWrapper(BindingModelWrapper):
    """Wrapper for MPT instruct models with custom prompt formatting."""
    
    def _load_model(self, **kwargs: Any):
        """Load MPT model with fallback tokenizer logic."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",  # Avoid flash attention issues
        )
    
    def format_prompt(self, prompt: str, queried_object: Optional[str] = None) -> str:
        """Format using MPT instruct format."""
        return (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{prompt}\n\n"
            "### Response:\n "
        )

class MambaModelWrapper(BindingModelWrapper):
    """Unified wrapper for all Mamba variants (Falcon3, OpenHermes, Codestral, Mamba2)."""
    
    def _load_model(self, **kwargs: Any):
        torch_dtype = kwargs.get('torch_dtype', torch.bfloat16)
        device_map = kwargs.get('device_map', 'auto')
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            token=False
        )
    
    def format_prompt(self, prompt: str, queried_object: Optional[str] = None) -> str:
        """Auto-detect prompt format using chat template."""
        if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
            messages = [{"role": "user", "content": prompt}]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        # Fallback for base models without chat template
        return prompt
    



def create_model_wrapper(model_id: str, **kwargs: Any) -> BindingModelWrapper:
    """Factory function to create the appropriate BindingModelWrapper."""
    mid = model_id.lower()
    if "mpt" in mid:
        return MPTModelWrapper(model_id, **kwargs)
    elif "mamba" in mid or "zamba" in mid:
        return MambaModelWrapper(model_id, **kwargs)
    else:
        raise ValueError(f"Unsupported model ID: {model_id}")
