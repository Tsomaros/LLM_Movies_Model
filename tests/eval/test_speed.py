from types import SimpleNamespace

import torch

from llm_movies.eval import speed
from llm_movies.eval.speed import cpu_forward_seconds


def _record_shapes(model, calls):
    model.register_forward_pre_hook(lambda m, args, kwargs: calls.append(tuple(kwargs["input_ids"].shape)), with_kwargs=True)


def test_speed_runs_on_tiny_model(tiny_lm, tiny_tokenizer, monkeypatch):
    calls, quantized = [], []
    real = torch.ao.quantization.quantize_dynamic

    def spy(model, specs, dtype):
        out = real(model, specs, dtype=dtype)
        has_int8 = any(isinstance(m, torch.ao.nn.quantized.dynamic.Linear) for m in out.modules())
        quantized.append((specs, dtype, has_int8))
        _record_shapes(out, calls)
        return out

    monkeypatch.setattr(torch.ao.quantization, "quantize_dynamic", spy)
    seconds = cpu_forward_seconds((tiny_lm, tiny_tokenizer), max_len=16, batch=3, int8=True)
    assert isinstance(seconds, float) and seconds > 0
    assert quantized == [({torch.nn.Linear}, torch.qint8, True)]  # Linear layers really became int8 layers
    assert calls == [(3, 16)] * 6  # one warm-up plus five timed passes of batch x max_len


def test_speed_is_the_median_of_five_timings(tiny_lm, tiny_tokenizer, monkeypatch):
    ticks = iter([0, 5, 10, 11, 20, 23, 30, 130, 140, 142])  # durations 5, 1, 3, 100, 2: median 3, mean 22.2
    # exactly 10 clock reads: a timed warm-up would exhaust the iterator and fail
    monkeypatch.setattr(speed, "time", SimpleNamespace(perf_counter=lambda: next(ticks)))
    assert cpu_forward_seconds((tiny_lm, tiny_tokenizer), max_len=8, batch=2, int8=False) == 3


def test_speed_without_int8_leaves_model_untouched(tiny_lm, tiny_tokenizer):
    calls = []
    _record_shapes(tiny_lm, calls)
    before = type(tiny_lm.model.layers[0].self_attn.q_proj)
    assert cpu_forward_seconds((tiny_lm, tiny_tokenizer), max_len=16, batch=2, int8=False) > 0
    assert type(tiny_lm.model.layers[0].self_attn.q_proj) is before
    assert calls == [(2, 16)] * 6  # the original model ran, unquantized
