import torch


def test_tiny_fixtures_forward(tiny_tokenizer, tiny_lm, golden_prompts):
    batch = tiny_tokenizer(golden_prompts[:2], return_tensors="pt", padding=True)
    assert tiny_tokenizer.padding_side == "right"
    assert tiny_tokenizer.pad_token_id is not None and tiny_tokenizer.eos_token_id is not None
    # the prompts differ in length, so padding really happened (right side: trailing zeros in the mask)
    mask = batch["attention_mask"]
    assert (mask == 0).any() and (mask.sum(1) != mask.shape[1]).any()
    assert (batch["input_ids"][mask == 0] == tiny_tokenizer.pad_token_id).all()
    # the vocabulary was trained on these prompts, so nothing maps to <unk>
    assert not (batch["input_ids"] == tiny_tokenizer.unk_token_id).any()
    with torch.no_grad():
        out = tiny_lm(**batch, output_hidden_states=True)
    assert out.hidden_states[-1].shape == (2, batch["input_ids"].shape[1], 32)
    assert out.logits.shape[-1] == len(tiny_tokenizer)
    assert torch.isfinite(out.hidden_states[-1]).all() and torch.isfinite(out.logits).all()
