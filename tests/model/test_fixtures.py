import torch


def test_tiny_fixtures_forward(tiny_tokenizer, tiny_lm, golden_prompts):
    batch = tiny_tokenizer(golden_prompts[:2], return_tensors="pt", padding=True)
    assert tiny_tokenizer.padding_side == "right"
    assert tiny_tokenizer.pad_token_id is not None and tiny_tokenizer.eos_token_id is not None
    with torch.no_grad():
        out = tiny_lm(**batch, output_hidden_states=True)
    assert out.hidden_states[-1].shape == (2, batch["input_ids"].shape[1], 32)
    assert out.logits.shape[-1] == len(tiny_tokenizer)
