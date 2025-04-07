import torch
import torch.nn as nn
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from functools import partial
from typing import Optional, List, Type, Callable, Sequence
from transformers import (
    AutoTokenizer,
    AutoConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    LlamaForCausalLM,
    LlamaTokenizerFast,
    LlamaConfig,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.modeling_outputs import CausalLMOutputWithPast


from model.base_prompter import PromptBuilder
from model.prompt_llama2 import LLaMa2ChatPromptBuilder

class LLaMa2LLMBackbone(nn.Module):
    def __init__(
        self,
        llm_max_length: int = 2048,
        hf_token: Optional[str] = None,
        inference_mode: bool = False,
        use_flash_attention_2: bool = True,
    ) -> None:
        super().__init__()
        self.llm_max_length = llm_max_length
        self.inference_mode = inference_mode

        # Get model configuration from preset
        # hf_hub_path = model_config["hf_hub_path"]
        hf_hub_path = "meta-llama/llama2-7b"

        # Model initialization
        if not inference_mode:
            self.llm = LlamaForCausalLM.from_pretrained(
                hf_hub_path,
                token=hf_token,
                use_flash_attention_2=use_flash_attention_2,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
            )
        else:
            llm_config = AutoConfig.from_pretrained(hf_hub_path, token=hf_token)
            self.llm = LlamaForCausalLM._from_config(llm_config)

        # Configuration settings
        self.llm.config.use_cache = False if not inference_mode else True
        if not inference_mode:
            self.llm.enable_input_require_grads()

        # Tokenizer initialization
        self.tokenizer = AutoTokenizer.from_pretrained(
            hf_hub_path,
            model_max_length=self.llm_max_length,
            token=hf_token,
            padding_side="right"
        )

        # Handle special tokens for LLaMA2
        self.tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        self.llm.config.pad_token_id = self.tokenizer.pad_token_id
        self.llm.resize_token_embeddings(len(self.tokenizer), pad_to_multiple_of=64)

    def get_tokenizer(self) -> PreTrainedTokenizerBase:
        return self.tokenizer

    def get_fsdp_wrapping_policy(self) -> Callable:
        return partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={LlamaDecoderLayer}
        )

    def enable_gradient_checkpointing(self) -> None:
        self.llm.gradient_checkpointing_enable()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> CausalLMOutputWithPast:
        return self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def embed_input_ids(self, input_ids: torch.LongTensor) -> torch.Tensor:
        return self.llm.get_input_embeddings()(input_ids)

    @property
    def prompt_builder_fn(self):
        return LLaMa2ChatPromptBuilder
        
    @property
    def transformer_layer_cls(self) -> Type[nn.Module]:
        return LlamaDecoderLayer

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return torch.bfloat16

    @property
    def last_layer_finetune_modules(self) -> Sequence[nn.Module]:
        return (self.llm.model.embed_tokens, self.llm.model.layers[-1], self.llm.lm_head)

    @property
    def embed_dim(self) -> int:
        return self.llm.config.hidden_size

    @property
    def pad_token_id(self) -> int:
        return self.tokenizer.pad_token_id
