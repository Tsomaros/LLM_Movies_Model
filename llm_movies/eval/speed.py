"""CPU forward-pass speed, to check the free-CPU Space can serve the base model.

python -m llm_movies.eval.speed --model ID --max-len L [--batch 13] [--no-int8]

    cpu_forward_seconds(model, max_len, batch=13, int8=True) -> float, the median of 5 timed forward passes after one untimed warm-up

`model` is a Hub id (str) or a ready (model, tokenizer) pair. Batch 13 = one prediction plus 12 effect groups. Linear
layers are dynamically quantized to int8 when `int8` is true.

The timing covers the decoder forward pass the predictor uses (`get_decoder()`, hidden states, `use_cache=False`), with
no lm_head and no vocabulary logits. A Hub id is loaded in fp32 (`low_cpu_mem_usage`) and quantized in place. A ready
model is not changed: only `.to("cpu").eval()` is applied, and with int8 a quantized copy of its decoder is timed.
"""

from __future__ import annotations

import argparse
import statistics
import time
import warnings

RUNS = 5


def cpu_forward_seconds(model, max_len: int, batch: int = 13, int8: bool = True) -> float:
    import torch

    loaded_here = isinstance(model, str)
    if loaded_here:
        from transformers import AutoModelForCausalLM

        lm = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float32, low_cpu_mem_usage=True)
    else:
        lm = model[0]
    decoder = lm.get_decoder().to("cpu").eval()  # the base *Model: what the predictor runs, no lm_head
    del lm  # for a model loaded here this was the last reference: the lm_head is released (tied weights stay in the decoder)
    if int8:
        # in place (no second fp32 copy) for a model loaded here; a ready model's decoder is copied so the caller's stays fp32
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="torch.ao.quantization is deprecated", category=DeprecationWarning)
            decoder = torch.ao.quantization.quantize_dynamic(decoder, {torch.nn.Linear}, dtype=torch.qint8, inplace=loaded_here)
    ids = torch.randint(0, decoder.config.vocab_size, (batch, max_len))
    mask = torch.ones_like(ids)
    times = []
    with torch.inference_mode():
        decoder(input_ids=ids, attention_mask=mask, use_cache=False)  # untimed warm-up: first-call allocation and kernel setup
        for _ in range(RUNS):
            start = time.perf_counter()
            decoder(input_ids=ids, attention_mask=mask, use_cache=False)
            times.append(time.perf_counter() - start)
    return statistics.median(times)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True)
    p.add_argument("--max-len", type=int, required=True)
    p.add_argument("--batch", type=int, default=13)
    p.add_argument("--no-int8", action="store_true")
    args = p.parse_args(argv)
    import torch

    seconds = cpu_forward_seconds(args.model, args.max_len, args.batch, not args.no_int8)
    print(f"{args.model}: {seconds:.2f} s per forward pass (batch {args.batch}, max_len {args.max_len}, {torch.get_num_threads()} torch threads)")


if __name__ == "__main__":
    main()
