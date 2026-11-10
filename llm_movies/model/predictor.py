import torch
from peft import LoraConfig, get_peft_model
from torch import Tensor, nn
from transformers import AutoModelForCausalLM, PreTrainedModel

from llm_movies.model.config import TrainConfig
from llm_movies.model.head import QuantileHead


class MoviePredictor(nn.Module):
    """LoRA-adapted decoder, last-token pooling, quantile head: (input_ids, mask) -> (B, 2, 3)."""

    def __init__(self, lm: nn.Module, head: QuantileHead):
        super().__init__()
        self.lm = lm  # PEFT-wrapped CausalLM; only its decoder runs, so no vocab-size logits are computed
        self.head = head

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        if input_ids.shape != attention_mask.shape:
            raise ValueError(f"input_ids {tuple(input_ids.shape)} and attention_mask {tuple(attention_mask.shape)} differ")
        decoder = self.lm.get_base_model().get_decoder()  # LoRA layers are injected in place, so they are active here
        hidden = decoder(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
        last = attention_mask.sum(1) - 1  # right padding: the last real token
        pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), last]
        with torch.autocast(device_type=hidden.device.type, enabled=False):  # autocast would cast the head's inputs back to fp16
            return self.head(pooled.float())


def build_predictor(cfg: TrainConfig, lm: PreTrainedModel | None = None, device: str | torch.device | None = None) -> MoviePredictor:
    """`device` defaults to cuda when available. fp16 base, gradient checkpointing and the move happen only for CUDA."""
    cuda = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu")).type == "cuda"
    if lm is None:
        lm = AutoModelForCausalLM.from_pretrained(cfg.base_model, dtype=torch.float16 if cuda else torch.float32)
    for p in lm.parameters():
        p.requires_grad_(False)
    if cuda:
        lm.config.use_cache = False
        lm.gradient_checkpointing_enable()
        lm.enable_input_require_grads()  # frozen embeddings would otherwise cut the checkpointed graph
    lora = LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout, target_modules="all-linear", bias="none"
    )
    peft_lm = get_peft_model(lm, lora)
    for n, p in peft_lm.named_parameters():
        if "lora_" in n:
            p.data = p.data.float()  # adapters in fp32 even on an fp16 base
    head = QuantileHead(lm.config.hidden_size)
    model = MoviePredictor(peft_lm, head)
    if cuda:
        model.to(device or "cuda")
    return model


def trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
