from types import SimpleNamespace

import torch

from llm_movies.eval import speed
from llm_movies.eval.speed import cpu_forward_seconds


def _record_shapes(model, calls):
    model.register_forward_pre_hook(lambda m, args, kwargs: calls.append(tuple(kwargs["input_ids"].shape)), with_kwargs=True)


def test_speed_runs_on_tiny_model(tiny_lm, tiny_tokenizer, monkeypatch):
    calls, quantized = [], []
    real = torch.ao.quantization.quantize_dynamic

    def spy(model, specs, dtype, inplace=False):
        out = real(model, specs, dtype=dtype, inplace=inplace)
        has_int8 = any(isinstance(m, torch.ao.nn.quantized.dynamic.Linear) for m in out.modules())
        quantized.append((specs, dtype, has_int8, model is tiny_lm.model))
        _record_shapes(out, calls)
        return out

    monkeypatch.setattr(torch.ao.quantization, "quantize_dynamic", spy)
    seconds = cpu_forward_seconds((tiny_lm, tiny_tokenizer), max_len=16, batch=3, int8=True)
    assert isinstance(seconds, float) and seconds > 0
    # Linear layers really became int8 layers, in a copy of the decoder (the caller's ready model is not quantized)
    assert quantized == [({torch.nn.Linear}, torch.qint8, True, True)]
    assert not any(isinstance(m, torch.ao.nn.quantized.dynamic.Linear) for m in tiny_lm.modules())
    assert calls == [(3, 16)] * 6  # one warm-up plus five timed passes of batch x max_len


def test_speed_is_the_median_of_five_timings(tiny_lm, tiny_tokenizer, monkeypatch):
    ticks = iter([0, 5, 10, 11, 20, 23, 30, 130, 140, 142])  # durations 5, 1, 3, 100, 2: median 3, mean 22.2
    # exactly 10 clock reads: a timed warm-up would exhaust the iterator and fail
    monkeypatch.setattr(speed, "time", SimpleNamespace(perf_counter=lambda: next(ticks)))
    assert cpu_forward_seconds((tiny_lm, tiny_tokenizer), max_len=8, batch=2, int8=False) == 3


def test_speed_without_int8_leaves_model_untouched(tiny_lm, tiny_tokenizer):
    calls = []
    _record_shapes(tiny_lm.model, calls)  # the decoder is what runs
    before = type(tiny_lm.model.layers[0].self_attn.q_proj)
    assert cpu_forward_seconds((tiny_lm, tiny_tokenizer), max_len=16, batch=2, int8=False) > 0
    assert type(tiny_lm.model.layers[0].self_attn.q_proj) is before
    assert calls == [(2, 16)] * 6  # the original model ran, unquantized


def test_speed_never_computes_vocabulary_logits(tiny_lm, tiny_tokenizer):
    fired = []
    tiny_lm.lm_head.register_forward_hook(lambda *a: fired.append(1))
    decoder_outputs = []
    tiny_lm.model.register_forward_hook(lambda m, args, kwargs, out: decoder_outputs.append(out), with_kwargs=True)
    for int8 in (False, True):
        cpu_forward_seconds((tiny_lm, tiny_tokenizer), max_len=8, batch=2, int8=int8)
    assert fired == []  # the lm_head never ran
    assert decoder_outputs and not any(hasattr(o, "logits") for o in decoder_outputs)


def test_speed_loads_by_id_with_dtype_not_torch_dtype(tiny_lm, monkeypatch):
    from transformers import AutoModelForCausalLM

    seen = {}

    def fake_from_pretrained(model_id, **kwargs):
        seen["id"], seen["kwargs"] = model_id, kwargs
        return tiny_lm

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", fake_from_pretrained)
    assert cpu_forward_seconds("some/model", max_len=8, batch=2, int8=False) > 0
    assert seen["id"] == "some/model"
    assert seen["kwargs"]["dtype"] is torch.float32
    assert seen["kwargs"]["low_cpu_mem_usage"] is True
    assert "torch_dtype" not in seen["kwargs"]


def test_speed_by_id_quantizes_the_decoder_in_place(tiny_lm, monkeypatch):
    from transformers import AutoModelForCausalLM

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", lambda model_id, **kw: tiny_lm)
    decoder = tiny_lm.model
    calls, inplace_flags, targets = [], [], []
    real = torch.ao.quantization.quantize_dynamic

    def spy(model, specs, dtype, inplace=False):
        inplace_flags.append(inplace)
        targets.append(model)
        return real(model, specs, dtype=dtype, inplace=inplace)

    monkeypatch.setattr(torch.ao.quantization, "quantize_dynamic", spy)
    _record_shapes(decoder, calls)
    assert cpu_forward_seconds("some/model", max_len=8, batch=2, int8=True) > 0
    assert inplace_flags == [True]  # a model loaded here is quantized in place: no second fp32 copy
    assert targets == [decoder]  # the timed module is the decoder, not the full CausalLM
    assert not [m for m in decoder.modules() if type(m) is torch.nn.Linear]  # no fp32 Linear left in the timed module
    # 2 layers x (q, k, v, o, gate, up, down) = 14 int8 Linear layers
    assert sum(isinstance(m, torch.ao.nn.quantized.dynamic.Linear) for m in decoder.modules()) == 14
    assert calls == [(2, 8)] * 6
