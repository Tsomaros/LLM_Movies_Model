# Colab launchers for the data pipeline

Three notebooks (01 to 03) run the data pipeline on Google Colab. They hold no logic: each one clones this repo, runs
`uv sync --frozen`, reads its secrets, and then calls `python -m llm_movies....` commands. All the code is in the
package and its tests.

Nothing is kept on the Colab disk and Google Drive is not used. The raw data lives in a **private Hugging Face
dataset repo, `Tsomaros/llm-movies-raw`**, created on the first upload. A session that dies loses only the work since the
last upload.

## Before you start

- Use a **CPU runtime** (Runtime > Change runtime type > CPU) for all three. Nothing here needs a GPU.
- Colab Secrets (key icon in the left bar, notebook access switched on). Names only, never paste a value into a notebook:
  - `TMDB_READ_TOKEN`: the TMDB read access token (notebook 01).
  - `HF_TOKEN`: a Hugging Face token with **write** access (all three notebooks read it, because every one talks to the repo).
- Both Hub repos are private. `hub_store` refuses to upload to `llm-movies-raw` if it is public, and `export --push` refuses the same for
  `Tsomaros/llm-movies-data` (its `locked.parquet` holds the test labels). If either was ever made public, make it private first.

The raw repo is laid out as:

```
ids/movie_ids.json.gz         the TMDB id export, downloaded once (01)
tmdb/                         tmdb_NNNN.jsonl, missing_NNNN.txt, manifest.json (01)
wd_films/, wd_people/         Wikidata shards (02)
reports/data_summary.md       the data report (03)
```

## Run order

1. **`01_pull_tmdb.ipynb`**: one run pulls every chunk, in order, and uploads the shard and the manifest after each chunk.
   Leave `LIMIT` and `MAX_CHUNKS` empty for the full run. The first run downloads the id export and keeps it in the repo; keep
   that one export for the whole pull, because a fresh export shifts ids between chunks.
2. **`02_pull_wikidata.ipynb`**, after 01 has finished all chunks. It builds the film and person id lists from the TMDB shards, then
   pulls Wikidata facts for films and for people, uploading the shards every 30 minutes and at the end (each upload is a commit holding a copy of one growing file, so the interval is long to keep the repo's history small).
3. **`03_build_and_export.ipynb`**, after 02. Downloads everything, builds `movies.db` on the Colab disk, exports the parquet dataset,
   writes the report and uploads it. `PUSH` stays off until you want the dataset on the Hub. No copy of the database is kept; it
   rebuilds from the repo.

## What stops a bad run

In Colab a failed `!command` does not stop "Run all". So each notebook's last cell is **one chain joined with `&&`**: if any step
fails (say a download from the Hub), the rest do not run. The code also refuses to write a wrong state when run by hand:

- the manifest records `chunk_size` and the chunk count, and `tmdb_pull --all` refuses to start if they differ from the current config and
  id list;
- before pulling, `--all` reads the Hub's manifest and refuses if it lists done chunks the local manifest lacks (the download failed);
- `ids/movie_ids.json.gz` in the repo is never replaced by a different file: `hub_store upload` refuses unless `--replace` is given (the
  notebooks never pass it; an identical file is a no-op, so rerunning works);
- `shard_ids` refuses to build the id lists while the pull is unfinished, unless `--partial` (the `PARTIAL` form in notebook 02, for
  a sample run only);
- `hub_store` retries Hub errors for about two minutes before it raises, and never uploads `.env*`, `.git`, `.venv` or cache folders, so a
  wrong `--dir` cannot leak a secret.

## How to resume

**Rerun the same notebook.** That is the whole story. It downloads what the repo holds and carries on.

- **01:** finished chunks are skipped, and ids already in the shard of the chunk in progress are not asked again. The chunk in progress
  at the moment of a crash is the only thing lost (about 2 minutes of work at the measured speed, see below).
- **02:** both pulls skip ids already in their shard; at most the last 30 minutes are redone.
- **03:** rebuilds from scratch each time.
- A sample run (`LIMIT` set in 01) leaves its chunks unmarked, so a later full run completes them. `MAX_CHUNKS` stops 01 after that many
  chunks; running it again carries on.

## How many chunks, and how to find out

A chunk holds 5,000 ids (`chunk_size` in `config/data.yaml`). The id export of 2026-11-05 had 1,204,179 usable ids (1,269,754 lines
minus adult and video entries), so **241 chunks** (0 to 240, the last with 4,179 ids). The count moves a little each day, and the
export is fixed once downloaded. To check it, in a notebook after the setup cells:

```
!uv run python -c "from llm_movies.data.tmdb_ids import read_ids; print(-(-len(read_ids('/content/raw/ids/movie_ids.json.gz')) // 5000))"
```

The pull is complete when `tmdb/manifest.json` in the raw repo has one entry per chunk; rerunning 01 with nothing left prints
`{"chunks": []}`.

## How long it takes and how big it is

Measured in a local dry run on 2026-11-05 (500 ids from each of chunks 0 and 5 of the old 50,000 sized chunks, on a Windows PC; Colab's
network will differ). Anything else is an estimate, marked so.

| stage | measured | full run (estimate from the measurement) |
| --- | --- | --- |
| 01 export download | 28 MB in about 4 s | trivial |
| 01 TMDB pull | 500 ids in about 103 s: about 4.9 ids/s. The pull asks one id at a time, so TMDB's response time limits it, not the 40 per second cap | about 17 minutes per 5,000-id chunk; about 70 hours for all 241 |
| 02 id lists | 1,000 films gave 24,408 people ids in under 2 s | seconds |
| 02 Wikidata, films | 1,000 ids in 60 s (500 per query, about 30 s each) | about 20 hours for 1.2 million ids |
| 02 Wikidata, people | 24,408 ids in 368 s (about 66 ids/s) | about 5 hours per 1.2 million ids; the people count of the full set is unknown (24,408 people came from 1,000 films, but those films are skewed to big casts) |
| 03 build, export, report | 995 films: build 2.5 s, export 2.3 s, report 0.5 s | not measured at scale. Earlier projections: export up to 38 minutes and 2.2 GB. Build memory could not be measured at this size |

Size of the raw TMDB shards, measured: about **48 KB per film in chunk 0** (the oldest ids, which are mostly famous films with long casts)
and about **6 KB per film in chunk 5** (2010s ids). Over 1.2 million films that is at least about 7 GB (at 6 KB; an estimate, since the
real mix is unknown, and it could be more if the early ids are as large as chunk 0's). Notebook 03 downloads all of it to the Colab disk.

Colab sessions end after a few hours at most, so expect many sessions for 01 and 02. Every full-run figure is a straight multiplication
of a 500 or 1,000 id sample, and the first real chunks will correct it.

## After the run

- `reports/data_summary.md` is in the raw repo (and printed by the notebook). **Copy it into this project repo by hand** as
  `reports/data_summary.md` and commit it; the Colab clone is never pushed back.
- `min_votes` is decided from the report's "Score films by min_votes" table. Then change `min_votes` in `config/data.yaml`, commit and
  push it, and re-run **notebook 03** from the top (its setup cells pull the new config; the database is not kept between sessions, so the
  build runs again). `min_votes` only affects the export, but any other change in `config/data.yaml` (split years, `base_model_cutoff`)
  needs the rebuild too: `export` stops if the database was built with the old values.
- The first push needs the Colab Secret `HF_TOKEN` and the dataset repo `Tsomaros/llm-movies-data` to be **private or absent**.

# Phase 2: model choice

Two more notebooks pick the base model. They also hold no logic (`llm_movies.data.db_snapshot`, `llm_movies.eval.model_choice`,
`llm_movies.eval.speed`) and need only the Colab Secret `HF_TOKEN`, not `TMDB_READ_TOKEN`. The candidate list lives in one place,
`CANDIDATES` in `llm_movies/eval/model_choice.py`; `12_speed.ipynb` repeats it, and a test keeps the two equal.

| model | notes |
| --- | --- |
| `Qwen/Qwen3-0.6B-Base`, `Qwen/Qwen3-1.7B-Base` | the **-Base** (pretrained) checkpoints; Qwen ids without `-Base` are post-trained and not used |
| `HuggingFaceTB/SmolLM2-1.7B` | open |
| `meta-llama/Llama-3.2-1B` | gated: licence step below |
| `google/gemma-3-1b-pt` | gated: licence step below |
| `google/gemma-4-E2B` | **conditional** (Apache 2.0): kept only if its CPU time allows. About 5.1B parameters with embeddings, so expect the slowest CPU time. It may be multimodal and fail to load as a plain causal LM; both notebooks put it last, so a failure costs nothing earlier |

## What each runtime installs

The first three notebooks run `uv sync --frozen` (no PyTorch). Notebooks 10 and 12 need `transformers` (the `train` extra) and a PyTorch
build, and the two come as mutually exclusive extras in `pyproject.toml`:

| where | sync command | PyTorch |
| --- | --- | --- |
| 10 part A (CPU runtime) | `uv sync --frozen` | none |
| 10 part B (T4 GPU runtime) | `uv sync --frozen --extra train --extra cu132` | CUDA 13.2 build, `torch 2.15.0+cu132` (Turing sm_75 is in its supported list) |
| 12 (CPU runtime) | `uv sync --frozen --extra train --extra cpu` | CPU build |
| tests, CI, this PC | `uv sync --extra train --extra cpu` | CPU build |

The GPU sync downloads the CUDA wheels (roughly 3 to 5 GB with the cuDNN, cuBLAS and NCCL libraries; an estimate, not measured) and takes
a few minutes on Colab. Part B has its own sync cell, so a T4 session runs Setup, Secrets, then part B only; a CPU session running part A
never installs CUDA. `uv run` keeps the extras installed by the last `uv sync`, so the commands do not need `--extra` themselves.

## Run order

1. **`10_model_choice.ipynb`, part A, on a CPU runtime, once.** Rebuilds `movies.db` from the raw shards and uploads it to the raw repo
   as `db/movies.db`. If the repo already holds it, the command says so and does nothing (no download, no build).
2. **`10_model_choice.ipynb`, part B, on a T4 GPU runtime.** Change the runtime and run the Setup and Secrets cells again (the new
   machine is empty). **Before the part B sync, run `!nvidia-smi` in a scratch cell. The top-right "CUDA Version" must be 13.2 or higher.
   If it is lower, stop and report it; do not run the probes on the CPU.** Then run the part B sync cell and the part B probe cell. Do not
   re-run the part A sync cell after part B's sync: `uv sync` is exact and would remove torch. The probe downloads `db/movies.db` only,
   then probes each model: director recall (150 films per release quarter from 2019Q1 to 2026Q2, at least 100 votes) and revenue
   memorization (300 validation films), the same films for every model. After **each** model it uploads that model's two CSVs and
   `summary.json` / `summary.md` to `model_choice/` in the raw repo. **To resume, rerun part B**: a model already `ok` is skipped; a
   failed one is tried again. Without a GPU it stops with a message.
3. **`12_speed.ipynb` on a CPU runtime.** Times one forward pass (batch 13, 768 tokens, int8) per candidate, in the order of the list
   above, and prints one line per model. Nothing is uploaded; copy the numbers into `docs/model_choice.md`.

## The gated models (Llama 3.2 and Gemma 3)

Before the first run, on huggingface.co, signed in with the **same account that owns `HF_TOKEN`**, open
`meta-llama/Llama-3.2-1B` and `google/gemma-3-1b-pt` and accept each licence. The token also needs **read access to gated repos**
(a fine-grained token: tick "Read access to contents of all public gated repos you can access"). For these notebooks `HF_TOKEN` also
needs **write access to the raw repo**. If a licence is not accepted, notebook 10 records that model as failed with the Hub's error
message (never the token) and goes on to the next; accept the licence and rerun part B. In notebook 12 the gated models come after the
open ones, so a refused licence stops the chain only after the open models are timed.

The `db/movies.db` snapshot is frozen and exists for the Task 10 probes only. Run part A only after the full pull is complete (it builds
from whatever shards exist). `ensure` never replaces an existing snapshot and `fetch` does not check the build config, so after
`base_model_cutoff` or the split years change its split labels are stale: later phases must rebuild (notebook 03). To refresh the snapshot,
delete `db/movies.db` from the raw repo first. A part B rerun with different settings (for example a smoke run with `--per-quarter 2`)
replaces that model's entry and CSVs in `model_choice/`: after a smoke run, rerun with the defaults (the Hub history keeps old versions).

The raw repo gains two folders:

```
db/movies.db                  the built database, uploaded once by 10 part A
model_choice/                 <model>_director.csv, <model>_revenue.csv, summary.json, summary.md (10 part B)
```

## Expected run times (ESTIMATES, nothing here was measured on Colab)

| step | estimate | how it was derived |
| --- | --- | --- |
| 10 part A, shard download | the raw shards are at least about 7 GB (see the size note above) | the same measured per-film sizes as above, times 1.2 million films |
| 10 part A, build | on the order of an hour | a measured 2.5 s for 995 films, multiplied straight up to 1.2 million films (about 50 minutes); not measured at scale, and memory at that size is unknown |
| 10 part B, per model | roughly 5 to 15 minutes, plus the first-time model download (about 1 to 4 GB) | 30 quarters x 150 = up to 4,500 director prompts plus 300 revenue prompts, 12 new tokens each, greedy, batches of 16 in fp16: about 300 batches of about 12 decoding steps. The seconds per step on a T4 is a guess, not a measurement |
| 10 part B, all six | about 1 to 1.5 hours if all load | six times the per-model estimate; a rerun resumes after the last uploaded model |
| 12, per model | tens of minutes | 6 forward passes (1 warm-up and 5 timed) of 13 x 768 = 9,984 tokens; a forward pass costs about 2 x parameters x tokens FLOPs, about 20 TFLOP for a 1B model; a free Colab CPU with 2 cores may sustain on the order of 50 to 100 GFLOP/s, so minutes per pass |
| 12, all six | several hours; the 5.1B Gemma 4 E2B alone may take much longer than the rest | the sum of the per-model estimates |

By the arithmetic above, a batch-13 x 768 forward pass is likely minutes per pass on a 2-vCPU Colab runtime even for the 0.6B model
(an estimate), and the 5.1B Gemma 4 E2B may not fit in 12 GB RAM (also unmeasured).

Memory on the CPU runtime (about 12 GB) is a risk for notebook 12: a model is loaded in fp32 (4 bytes per parameter) and quantized in place, so
no second copy stays alive. Only the decoder is timed, as the predictor runs it: no vocabulary logits are computed (they alone would have
been 13 x 768 x 151,936 x 4 bytes, about 6 GB for Qwen) and the lm_head is released after loading. The 1.7B models need about 7 GB for the
fp32 weights, and Gemma 4 E2B (about 5.1B parameters with embeddings) may still not fit, which is a reason it is last.
