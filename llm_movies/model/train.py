"""The training loop: LoRA + quantile head, AdamW (two learning rates), warmup then cosine, grad accumulation.

    python -m llm_movies.model.train --config configs/train/<name>.yaml [--data DIR]

Position is counted in optimizer steps: steps_per_epoch = ceil(ceil(N / batch_size) / grad_accum), a trailing partial
accumulation still makes a step. A resumed run re-derives the epoch and the offset inside it from `state["step"]`, rebuilds
that epoch's sampler from (seed, epoch) and skips the batches already trained on, so it equals an uninterrupted run. The
validation metric is the pinball loss over the whole validation frame (scaled units) / 2; a lower one saves "best".
A non-finite loss skips the step (counted); 10 in a row stop the run with stopped_reason "nan". metrics.jsonl (in
`out_dir`, default outputs/<run_name>/) gets one line per validation; W&B is used only when WANDB_API_KEY is set.
"""

import argparse
import json
import math
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from llm_movies.data import hub_store
from llm_movies.data.export import FILMS_FILE
from llm_movies.eval.metrics import revenue_metrics, score_metrics
from llm_movies.model.checkpoint import load_checkpoint, restore_training_state, resume_state, save_checkpoint
from llm_movies.model.config import TrainConfig, load_train_config
from llm_movies.model.data import Collator, FilmDataset, TargetScaler, load_frames, revenue_sampler
from llm_movies.model.loss import pinball_loss
from llm_movies.model.predictor import build_predictor

MAX_NAN_STEPS = 10  # in a row


def _autocast(device):
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda")


@torch.no_grad()
def evaluate(model, dataset: FilmDataset, scaler: TargetScaler, tokenizer, cfg: TrainConfig, device) -> tuple[float, dict]:
    """(pinball over the whole frame / 2 in scaled units, {"revenue": ..., "score": ...} in original units). Leaves the
    model in train mode."""
    model.eval()
    loader = DataLoader(dataset, batch_size=cfg.batch_size, collate_fn=Collator(tokenizer, cfg.max_len), num_workers=0)
    preds = []
    for b in loader:
        with _autocast(device):
            preds.append(model(b["input_ids"].to(device), b["attention_mask"].to(device)).float().cpu())
    model.train()
    pred = torch.cat(preds)
    loss = pinball_loss(pred, dataset.target, dataset.mask, dataset.weight).item() / 2
    mean, std = scaler.mean, scaler.std
    original = scaler.inverse(pred).numpy()
    metrics = {}
    for k, name in enumerate(("revenue", "score")):
        known = dataset.mask[:, k].numpy()
        if not known.any():
            continue  # no validation rows for this target
        true = dataset.target[:, k].numpy()[known] * std[k] + mean[k]
        if k == 0:
            metrics[name] = revenue_metrics(original[known, 0], true)
        else:
            metrics[name] = score_metrics(original[known, 1], true, dataset.weight[:, 1].numpy()[known])
    return loss, metrics


def _schedule(total: int):
    warm = max(1, round(0.03 * total))

    def factor(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))

    return factor


def _chunks(items: list, n: int) -> list[list]:
    return [items[i : i + n] for i in range(0, len(items), n)]


def train(
    cfg: TrainConfig, data_dir: Path, *, api=None, lm=None, tokenizer=None, device=None, out_dir: Path | None = None
) -> dict:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    api = api or hub_store.make_api()
    out_dir = Path(out_dir or Path("outputs") / cfg.run_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    train_df, val_df = load_frames(data_dir, cfg)
    if len(val_df) == 0 or len(train_df) == 0:
        raise ValueError("the train and validation frames must both have rows")

    resumed = resume_state(cfg, api) is not None
    if resumed:
        model, scaler, state = load_checkpoint(cfg, "last", api, lm)
    else:
        for seed_fn in (torch.manual_seed, np.random.seed):
            seed_fn(cfg.seed)
        model, scaler = build_predictor(cfg, lm), TargetScaler.fit(train_df)
        state = {"step": 0, "epoch": 0, "best_metric": None, "bad_checks": 0, "metrics": {}, "best_step": None, "best_metrics": {}}
    model.to(device).train()

    lora = [p for p in model.lm.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{"params": lora, "lr": cfg.lr_lora}, {"params": list(model.head.parameters()), "lr": cfg.lr_head}],
        weight_decay=cfg.weight_decay,
    )
    micro = math.ceil(len(train_df) / cfg.batch_size)
    steps_per_epoch = math.ceil(micro / cfg.grad_accum)
    total = steps_per_epoch * cfg.epochs
    eval_interval = max(1, round(steps_per_epoch * cfg.eval_every))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _schedule(total))
    grad_scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None
    if resumed:
        restore_training_state(cfg, api, optimizer=optimizer, grad_scaler=grad_scaler, lr_scheduler=scheduler)

    val_set = FilmDataset(val_df, scaler, cfg, train=False)
    collate = Collator(tokenizer, cfg.max_len)
    wandb_run = None
    if os.environ.get("WANDB_API_KEY"):
        import wandb  # lazy: only when asked for

        wandb_run = wandb.init(project=cfg.wandb_project, name=cfg.run_name, config=asdict(cfg), resume="allow")

    def checkpoint(kind):
        save_checkpoint(model, scaler, cfg, state, kind, api, optimizer=optimizer, grad_scaler=grad_scaler, lr_scheduler=scheduler)

    def validate():
        loss, metrics = evaluate(model, val_set, scaler, tokenizer, cfg, device)
        state["metrics"] = {"val_loss": loss, **metrics}
        line = {"step": state["step"], "epoch": state["epoch"], "lr": scheduler.get_last_lr(), **state["metrics"]}
        with open(out_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(line) + "\n")
        if wandb_run is not None:
            wandb_run.log({"val_loss": loss, **{f"{t}/{k}": v for t, m in metrics.items() for k, v in m.items()}}, step=state["step"])
        best = state["best_metric"]
        if math.isfinite(loss) and (best is None or loss < best):
            state.update(best_metric=loss, best_step=state["step"], best_metrics=metrics, bad_checks=0)
            save_checkpoint(model, scaler, cfg, state, "best", api)
        else:
            state["bad_checks"] += 1

    reason, error, nan_run, nan_steps = None, None, 0, 0
    if state["bad_checks"] >= cfg.patience:
        reason = "patience"
    while reason is None and state["step"] < total:
        epoch, offset = divmod(state["step"], steps_per_epoch)
        dataset = FilmDataset(train_df, scaler, cfg, train=True)
        dataset.set_epoch(epoch)
        order = list(revenue_sampler(train_df, cfg.revenue_oversample, cfg.seed, epoch))
        groups = _chunks(_chunks(order, cfg.batch_size), cfg.grad_accum)[offset:]  # skip the steps already trained on
        for group in groups:
            optimizer.zero_grad(set_to_none=True)
            finite = True
            for indices in group:
                b = {k: v.to(device) for k, v in collate([dataset[i] for i in indices]).items()}
                with _autocast(device):
                    pred = model(b["input_ids"], b["attention_mask"])
                loss = pinball_loss(pred, b["target"], b["mask"], b["weight"])
                if not torch.isfinite(loss):
                    finite = False
                    break
                scaled = loss / len(group)
                (grad_scaler.scale(scaled) if grad_scaler else scaled).backward()
            if finite:
                nan_run = 0
                if grad_scaler:
                    grad_scaler.step(optimizer)  # skips the update itself on inf grads; that is not a NaN step
                    grad_scaler.update()
                else:
                    optimizer.step()
            else:
                optimizer.zero_grad(set_to_none=True)
                nan_run, nan_steps = nan_run + 1, nan_steps + 1
            scheduler.step()
            state["step"] += 1
            state["epoch"] = state["step"] // steps_per_epoch
            if nan_run >= MAX_NAN_STEPS:
                reason = "nan"
                error = f"stopped after {MAX_NAN_STEPS} consecutive steps with a non-finite loss (NaN or inf); check the data, learning rates and precision"
                break
            validated = state["step"] % eval_interval == 0 or state["step"] == total
            if validated:
                validate()
            if validated or state["step"] % cfg.save_every_steps == 0:
                checkpoint("last")
            if validated and state["bad_checks"] >= cfg.patience:
                reason = "patience"
                break
    reason = reason or "epochs"
    if wandb_run is not None:
        wandb_run.finish()
    return {
        "best_metric": state["best_metric"], "best_step": state["best_step"], "stopped_reason": reason,
        "metrics": state["best_metrics"], "step": state["step"], "nan_steps": nan_steps, "error": error,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the movie predictor; resumes from the Hub when a run exists.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, help="folder with films.parquet (default: downloaded from the config's data_repo)")
    args = parser.parse_args(argv)
    cfg = load_train_config(args.config)
    with tempfile.TemporaryDirectory() as tmp:
        data = args.data
        if data is None:  # only films.parquet; the locked file is never fetched
            data = Path(tmp)
            hub_store.download_file(cfg.data_repo, FILMS_FILE, data)
        summary = train(cfg, data)
    print(json.dumps(summary, indent=2))
    if summary["error"]:
        raise SystemExit(summary["error"])


if __name__ == "__main__":
    main()
