from pathlib import Path

import pytest

GOLDEN_PROMPTS = Path(__file__).parent / "golden" / "prompts"


@pytest.fixture(scope="session")
def golden_prompts() -> list[str]:
    return [p.read_text(encoding="utf-8") for p in sorted(GOLDEN_PROMPTS.glob("*.txt"))]


@pytest.fixture(scope="session")
def tiny_tokenizer(golden_prompts):
    # transformers/tokenizers live in the train extra: import lazily so a plain sync still collects other tests
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.WordLevel(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.WordLevelTrainer(special_tokens=["<pad>", "<eos>", "<unk>"])
    tok.train_from_iterator(golden_prompts, trainer)
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>")
    fast.padding_side = "right"
    return fast


@pytest.fixture
def tiny_lm(tiny_tokenizer):
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    config = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=len(tiny_tokenizer),
        pad_token_id=tiny_tokenizer.pad_token_id,
        eos_token_id=tiny_tokenizer.eos_token_id,
    )
    return LlamaForCausalLM(config).eval()
