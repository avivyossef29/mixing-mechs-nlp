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
        torch_dtype = kwargs.get('torch_dtype', torch.bfloat16)
        device_map = kwargs.get('device_map', 'auto')
        
        # MPT tokenizer handling with fallbacks
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                token=False
            )
        except Exception:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_id,
                    trust_remote_code=False,
                    token=False
                )
            except Exception:
                self.tokenizer = GPTNeoXTokenizerFast.from_pretrained(
                    self.model_id, 
                    token=False
                )
        
        # Load model with eager attention fallback
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=torch_dtype,
                device_map=device_map,
                attn_implementation="eager",
                trust_remote_code=True,
                token=False
            )
        except Exception:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(self.model_id, trust_remote_code=True, token=False)
            if hasattr(config, 'attn_config'):
                config.attn_config['attn_impl'] = 'torch'
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                config=config,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=True,
                token=False
            )
    
    def format_prompt(self, prompt: str, queried_object: Optional[str] = None) -> str:
        """Format using MPT instruct format."""
        return (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{prompt}\n\n"
            "### Response:\n "
        )
    
    def get_display_name(self) -> str:
        return f"MPT-7b ({self.model_id})"


class FalconMambaModelWrapper(BindingModelWrapper):
    """Wrapper for Falcon3-Mamba models using chat templates."""
    
    def _load_model(self, **kwargs):
        """Load Falcon3-Mamba model with standard parameters."""
        torch_dtype = kwargs.get('torch_dtype', torch.bfloat16)
        device_map = kwargs.get('device_map', 'auto')
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            low_cpu_mem_usage=False
        )
        # after from_pretrained
        num_meta = sum(p.device.type == "meta" for p in self.model.parameters())
        print("meta params:", num_meta)

    
    def format_prompt(self, prompt: str, queried_object: Optional[str] = None) -> str:
        """
        Format prompt using chat template (Falcon3-Mamba supports this).
        
        Based on the HuggingFace model card, Falcon3-Mamba uses chat templates.
        """
        messages = [
            {
                "role": "system",
                "content": "You are a helpful friendly assistant Falcon3 from TII, try to follow instructions as much as possible."
            },
            {"role": "user", "content": prompt}
        ]
        
        if hasattr(self.tokenizer, 'apply_chat_template'):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # Fallback if chat template not available
            return f"{messages[0]['content']}\n\nUser: {prompt}\n\nAssistant: "
    def generate_response(self, prompt: str, max_new_tokens: int = 10) -> str:
        # Pick a real device from the first non-meta parameter
        try:
            param_device = next(p.device for p in self.model.parameters() if p.device.type != "meta")
        except StopIteration:
            raise RuntimeError("All model parameters are on meta; model weights did not load correctly.")

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(param_device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        return self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    
    def get_display_name(self) -> str:
        """Return display name for Falcon3-Mamba model."""
        return "Falcon3-Mamba-7B-Instruct"


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
    
    def get_display_name(self) -> str:
        """Smart display name based on model ID."""
        mid = self.model_id.lower()
        if "falcon3-mamba" in mid or "falcon3_mamba" in mid:
            return f"Falcon3-Mamba ({self.model_id})"
        elif "openhermes" in mid:
            return f"Mamba-OpenHermes ({self.model_id})"
        elif "codestral" in mid:
            return f"Mamba-Codestral ({self.model_id})"
        elif "mamba2" in mid:
            return f"Mamba2 ({self.model_id})"
        else:
            return f"Mamba ({self.model_id})"



# class MambaOpenhermesModelWrapper(BindingModelWrapper):
#     """Wrapper for Mamba OpenHermes models."""
    
#     def _load_model(self, **kwargs: Any):
#         torch_dtype = kwargs.get('torch_dtype', torch.bfloat16)
#         device_map = kwargs.get('device_map', 'auto')
        
#         self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=False)
#         self.model = AutoModelForCausalLM.from_pretrained(
#             self.model_id,
#             torch_dtype=torch_dtype,
#             device_map=device_map,
#             token=False
#         )
    
#     def format_prompt(self, prompt: str, queried_object: Optional[str] = None) -> str:
#         """Format using chat template."""
#         if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
#             messages = [{"role": "user", "content": prompt}]
#             return self.tokenizer.apply_chat_template(
#                 messages,
#                 tokenize=False,
#                 add_generation_prompt=True
#             )
#         # Fallback for OpenHermes/ChatML style if template is missing but we know it's Hermes
#         return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
#     def get_display_name(self) -> str:
#         return f"Mamba-OpenHermes ({self.model_id})"



def create_model_wrapper(model_id: str, **kwargs: Any) -> BindingModelWrapper:
    """Factory function to create the appropriate BindingModelWrapper."""
    # mid = model_id.lower()
    # if "mpt" in mid:
    #     return MPTModelWrapper(model_id, **kwargs)
    # # elif "Falcon3" in mid:
    # else:
    return FalconMambaModelWrapper(model_id, **kwargs)
    # elif "mamba" in mid:  # Catches all Mamba variants
    #     return MambaModelWrapper(model_id, **kwargs)
    # else:
    #     # Generic fallback
    #     return MambaModelWrapper(model_id, **kwargs)
