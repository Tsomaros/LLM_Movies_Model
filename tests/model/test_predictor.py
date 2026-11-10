import pytest
import torch

from llm_movies.model.config import TrainConfig
from llm_movies.model.predictor import build_predictor, trainable_parameters

CFG = TrainConfig(run_name="t", base_model="unused")


@pytest.fixture
def predictor(tiny_lm):
    return build_predictor(CFG, lm=tiny_lm).eval()


def _enc(tok, texts, **kw):
    return tok(texts, return_tensors="pt", padding=True, **kw)


def test_output_shape(predictor, tiny_tokenizer, golden_prompts):
    enc = _enc(tiny_tokenizer, golden_prompts[:3])
    out = predictor(enc["input_ids"], enc["attention_mask"])
    assert out.shape == (3, 2, 3)
    assert out.dtype == torch.float32


def test_only_lora_and_head_trainable(predictor):
    trainable = {n for n, p in predictor.named_parameters() if p.requires_grad}
    assert trainable
    for n, p in predictor.named_parameters():
        if "lora_" in n or n.startswith("head."):
            assert p.requires_grad, n
            assert p.dtype == torch.float32, n
        else:
            assert not p.requires_grad, n  # every base weight is frozen
    assert trainable == {n for n in trainable if "lora_" in n or n.startswith("head.")}
    assert any("lora_A" in n for n in trainable) and any(n.startswith("head.") for n in trainable)
    assert trainable_parameters(predictor) == sum(p.numel() for n, p in predictor.named_parameters() if p.requires_grad)


def test_lora_targets_all_linear_layers(predictor):
    names = {n for n, _ in predictor.named_parameters() if "lora_A" in n}
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
        assert any(proj in n for n in names), proj
    assert not any("lm_head" in n for n in names)


def test_lora_is_active_in_forward_path(predictor, tiny_tokenizer, golden_prompts):
    predictor.train()
    enc = _enc(tiny_tokenizer, golden_prompts[:2])
    predictor(enc["input_ids"], enc["attention_mask"]).sum().backward()
    lora_b = [(n, p) for n, p in predictor.named_parameters() if "lora_B" in n]
    assert lora_b
    for n, p in lora_b:
        assert p.grad is not None and p.grad.abs().sum() > 0, n
    for n, p in predictor.named_parameters():
        if "lora_" not in n and not n.startswith("head."):
            assert p.grad is None, n


def test_padding_does_not_change_prediction(predictor, tiny_tokenizer, golden_prompts):
    short, long_ = min(golden_prompts, key=len), max(golden_prompts, key=len)
    alone = _enc(tiny_tokenizer, [short])
    batch = _enc(tiny_tokenizer, [short, long_])
    assert batch["input_ids"].shape[1] > alone["input_ids"].shape[1]  # the short one really is padded
    assert batch["attention_mask"][0].sum() == alone["attention_mask"].sum()
    with torch.no_grad():
        a = predictor(alone["input_ids"], alone["attention_mask"])
        b = predictor(batch["input_ids"], batch["attention_mask"])
    torch.testing.assert_close(a[0], b[0], atol=1e-5, rtol=0)


def test_truncated_prompt_pools_last_kept_token(predictor, tiny_tokenizer, golden_prompts):
    n = 12
    prompt = max(golden_prompts, key=len)
    full_ids = tiny_tokenizer(prompt)["input_ids"]
    assert len(full_ids) > n + 3
    trunc = tiny_tokenizer([prompt], return_tensors="pt", truncation=True, max_length=n)
    assert trunc["input_ids"].shape[1] == n and trunc["attention_mask"].sum() == n
    cut = torch.tensor([full_ids[:n]])
    with torch.no_grad():
        got = predictor(trunc["input_ids"], trunc["attention_mask"])
        expected = predictor(cut, torch.ones_like(cut))
        # wrong pooling: the last position of a right-padded copy holds a pad token
        pad = torch.full((1, 3), tiny_tokenizer.pad_token_id)
        padded = torch.cat([cut, pad], dim=1)
        wrong = predictor(padded, torch.ones_like(padded))  # mask all ones -> pools a pad position
    torch.testing.assert_close(got, expected, atol=1e-6, rtol=0)
    assert not torch.allclose(got, wrong, atol=1e-4)

    # mixed batch: one row truncated at n, one shorter row padded; each row must pool its own last real token
    mixed = tiny_tokenizer([prompt, "Release: 4 November"], return_tensors="pt", padding=True, truncation=True, max_length=n)
    lens = mixed["attention_mask"].sum(1)
    assert lens[0] == n and 0 < lens[1] < n  # row 0 truncated, row 1 shorter and right-padded
    with torch.no_grad():
        rows = predictor(mixed["input_ids"], mixed["attention_mask"])
        for i in range(2):
            ids = mixed["input_ids"][i : i + 1, : lens[i]]
            torch.testing.assert_close(rows[i], predictor(ids, torch.ones_like(ids))[0], atol=1e-5, rtol=0)


def test_rejects_left_padding_free_inputs_shape(predictor):
    with pytest.raises(ValueError):
        predictor(torch.zeros(2, 4, dtype=torch.long), torch.ones(2, 5, dtype=torch.long))


def test_head_stays_fp32_under_autocast(predictor, tiny_tokenizer, golden_prompts):
    enc = _enc(tiny_tokenizer, golden_prompts[:2])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = predictor(enc["input_ids"], enc["attention_mask"])
    assert out.dtype == torch.float32


def test_build_predictor_on_cpu_keeps_fp32_and_no_checkpointing(tiny_lm):
    model = build_predictor(CFG, lm=tiny_lm, device="cpu")
    assert all(p.dtype == torch.float32 for p in model.parameters())
    assert not tiny_lm.is_gradient_checkpointing
