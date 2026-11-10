"""CPU forward-pass speed, to check the free-CPU Space can serve the base model.

python -m llm_movies.eval.speed --model ID --max-len L [--batch 13] [--no-int8]

    cpu_forward_seconds(model, max_len, batch=13, int8=True) -> float, the median of 5 timed forward passes

`model` is a Hub id (str) or a ready (model, tokenizer) pair. Batch 13 = one prediction plus 12 effect groups. Linear
layers are dynamically quantized to int8 when `int8` is true.
"""

from __future__ import annotations

import argparse
import statistics
import time
import warnings

RUNS = 5


def cpu_forward_seconds(model, max_len: int, batch: int = 13, int8: bool = True) -> float:
    import torch

    if isinstance(model, str):
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float32)
    else:
        model = model[0]
    model = model.to("cpu").eval()
    if int8:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="torch.ao.quantization is deprecated", category=DeprecationWarning)
            model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    ids = torch.randint(0, model.config.vocab_size, (batch, max_len))
    mask = torch.ones_like(ids)
    times = []
    with torch.inference_mode():
        for _ in range(RUNS):
            start = time.perf_counter()
            model(input_ids=ids, attention_mask=mask)
            times.append(time.perf_counter() - start)
    return statistics.median(times)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True)
    p.add_argument("--max-len", type=int, required=True)
    p.add_argument("--batch", type=int, default=13)
    p.add_argument("--no-int8", action="store_true")
    args = p.parse_args(argv)
    print(f"{cpu_forward_seconds(args.model, args.max_len, args.batch, not args.no_int8):.2f} s per forward pass")


if __name__ == "__main__":
    main()
