from llm_movies.eval.speed import cpu_forward_seconds


def test_speed_runs_on_tiny_model(tiny_lm, tiny_tokenizer):
    seconds = cpu_forward_seconds((tiny_lm, tiny_tokenizer), max_len=16, batch=3, int8=True)
    assert isinstance(seconds, float) and seconds > 0


def test_speed_without_int8_leaves_model_untouched(tiny_lm, tiny_tokenizer):
    before = type(tiny_lm.model.layers[0].self_attn.q_proj)
    assert cpu_forward_seconds((tiny_lm, tiny_tokenizer), max_len=16, batch=2, int8=False) > 0
    assert type(tiny_lm.model.layers[0].self_attn.q_proj) is before
