# Base model choice

Purpose: Task 10 of Phase 2. Pick one model family and two sizes as the base for the LoRA + quantile-head predictor
(pretrained base checkpoints, not instruct). Rules: the training cutoff we use is the **later** of the stated and the
empirical cutoff (so the leak-safe holdout starts after it); the smaller E1 model must run batch 13 in **<= 20 s on CPU**
so prediction plus explanation stays near the 30 s target.

## Step 1: model cards

| Model id | Licence | Fine-tune + public Space OK? | Gated? | Stated training cutoff | Parameters | Context | Notes |
|---|---|---|---|---|---|---|---|
| [Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base) | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) | Yes / Yes | No | not stated | 0.6B total, 0.44B non-embedding | 32,768 | Pretrained base. Vocab size not stated on card. |
| [Qwen/Qwen3-1.7B-Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base) | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) | Yes / Yes | No (no gate shown) | not stated | 1.7B total, 1.4B non-embedding | 32,768 | Pretrained base. Vocab size not stated on card. |
| [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) | [Llama 3.2 Community License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE) | Yes / Yes, with obligations | **Yes** (accept licence on HF) | "December 2023" | 1.23B | 128k | Pretrained base. Vocab size not stated on card. |
| [google/gemma-3-1b-pt](https://huggingface.co/google/gemma-3-1b-pt) | [Gemma Terms of Use](https://ai.google.dev/gemma/terms) | Yes / Yes, with obligations | **Yes** (accept conditions on HF) | "August 2024" (Gemma 3 model card on ai.google.dev; not on the HF card) | 1.0B | 32K input | Pretrained (`-pt`). 2T training tokens. Vocab size not stated on card. |
| [HuggingFaceTB/SmolLM2-1.7B](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B) | [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) | Yes / Yes | No (no gate shown) | not stated | 1.7B | not stated | Base (not `-Instruct`). 11T tokens. |
| [google/gemma-4-E2B](https://huggingface.co/google/gemma-4-E2B) (**conditional**) | [Apache 2.0, as linked from the card](https://ai.google.dev/gemma/docs/gemma_4_license) | Licence text allows both (unconfirmed for weights, see notes) | unconfirmed | "with a cutoff date of January 2025" | 2.3B effective, 5.1B with embeddings | 128K | Vocab 262K. Base vs instruct unconfirmed. Include only if licence and CPU time fit. |

### Per-model notes (sources read on 2026-11-11)

- **Qwen3-0.6B-Base / Qwen3-1.7B-Base.** Sources: [0.6B-Base card](https://huggingface.co/Qwen/Qwen3-0.6B-Base),
  [1.7B-Base card](https://huggingface.co/Qwen/Qwen3-1.7B-Base), [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)
  (published 2025-05-14; the cards state 36T tokens, 119 languages, no cutoff date). The ids in the plan
  (`Qwen/Qwen3-0.6B`, `Qwen/Qwen3-1.7B`) are the **post-trained** checkpoints (card: "Pretraining & Post-training";
  model tree lists the `-Base` repo as parent). The pretrained base ids are `Qwen/Qwen3-0.6B-Base` and
  `Qwen/Qwen3-1.7B-Base`; the plan and `config` must use the `-Base` ids. Gating of the 1.7B-Base page was not stated
  explicitly; no gate notice was shown. Tech report was not read in full; the cutoff is therefore "not stated" on the card.
- **Llama-3.2-1B.** Sources: [card](https://huggingface.co/meta-llama/Llama-3.2-1B),
  [licence](https://github.com/meta-llama/llama-models/blob/main/models/llama3_2/LICENSE),
  [use policy](https://www.llama.com/llama3_2/use-policy). Released 2024-09-25. Licence permits derivative works and
  hosting. Obligations: ship a copy of the licence, show "Built with Llama" on the related site/UI/docs, include the
  Notice text, follow the Acceptable Use Policy, and any model trained or fine-tuned from it that is distributed must have a
  name starting with "Llama" (so a published fine-tune on the Hub would be named `Llama-...`). A licence request to Meta
  is only needed above 700M monthly users. The card's vocab size is not stated (do not assume).
- **gemma-3-1b-pt.** Sources: [card](https://huggingface.co/google/gemma-3-1b-pt),
  [terms](https://ai.google.dev/gemma/terms), [Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy).
  Release date: card says 2025 only. Terms allow modification and distribution; a public hosted Space counts as
  "Distribution". Obligations: pass the use restrictions (Prohibited Use Policy) on to users, give recipients a copy of the
  terms, mark modified files, Notice file for non-hosted distribution. Model derivatives (including models trained on its
  outputs) fall under the same terms, so the published LoRA weights carry them. Cutoff: the HF card states none, but the official
  [Gemma 3 model card](https://ai.google.dev/gemma/docs/core/model_card_3), Training Dataset section, says "The knowledge
  cutoff date for the training data was August 2024." Checked 2026-11-11. The HF card links the Gemma 3 report at
  https://goo.gle/Gemma3Report (not read). This family-level statement is assumed to cover the 1B-pt size.
- **SmolLM2-1.7B.** Sources: [card](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B). Paper published 2025-02-04; model
  release date not stated on the card. Card states 11T tokens (FineWeb-Edu, DCLM, The Stack, math and code) and no cutoff
  or context length (the paper may state them; not read).
- **Gemma 4 E2B (conditional).** Sources: [card](https://huggingface.co/google/gemma-4-E2B),
  [licence page](https://ai.google.dev/gemma/docs/gemma_4_license); technical report arXiv 2607.02770 (card lists
  "Published Jul 2", year 2026; not read). Card states the release has "pre-trained and instruction-tuned variants" but does
  not say which one `google/gemma-4-E2B` is (no `-it` suffix suggests pretrained; **unconfirmed**), and usage examples load
  `-it`. The linked licence page is Apache 2.0, which allows modification and public hosting with licence copy, change
  notices and kept attributions; it does not mention AI weights explicitly. Gating was not stated on the page
  (**unconfirmed**; check on Hugging Face while logged in). "Effective" 2.3B parameters differ from 5.1B with embeddings, so
  CPU cost must be measured, not inferred.

## Step 2-3: measurements (to be filled after the Colab runs)

Runs of 2026-11-11.
- Probes: notebook 10 part B, Colab T4. Each model was asked about 4171 films (director) and 300 films (revenue). The
  outputs are `model_choice/` in `Tsomaros/llm-movies-raw`.
- Speed: notebook 12, Colab CPU runtime, int8 dynamic quantization, decoder only, **1 torch thread**, padded to the full
  768 tokens.

| Model | Stated cutoff | Empirical cutoff | Director recall 2019-20 | Revenue known % | CPU seconds (batch 13, max_len 768) | Cutoff used |
|---|---|---|---|---|---|---|
| Qwen3-0.6B-Base | not stated (Qwen3 released 2025-04-29) | not informative (2025-12-31 computed) | 0.006 | 9.7% | 100.5 | pending decision |
| Qwen3-1.7B-Base | not stated (Qwen3 released 2025-04-29) | not informative (2026-06-30 computed) | 0.022 | 12.3% | 249.3 | pending decision |
| Llama-3.2-1B | December 2023 | 2022-12-31 | 0.190 | 23.0% | 163.0 | 2023-12-31 |
| gemma-3-1b-pt | August 2024 | invalid run | invalid run (0.000) | invalid run (0.0%) | 137.5 | 2024-08-31 |
| SmolLM2-1.7B | not stated | 2022-06-30 | 0.089 | 14.3% | 271.1 | 2022-06-30 |
| gemma-4-E2B (conditional) | January 2025 | 2024-12-31 | 0.235 | 25.0% | no result (run ended; probably out of memory) | 2025-01-31 |

Director recall per year (2019 → 2026):

| Model | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|
| Qwen3-0.6B-Base | 0.008 | 0.004 | 0.007 | 0.002 | 0.007 | 0.004 | 0.002 | 0.000 |
| Qwen3-1.7B-Base | 0.035 | 0.007 | 0.027 | 0.022 | 0.020 | 0.012 | 0.006 | 0.005 |
| SmolLM2-1.7B | 0.130 | 0.043 | 0.057 | 0.033 | 0.020 | 0.016 | 0.012 | 0.016 |
| Llama-3.2-1B | 0.223 | 0.152 | 0.175 | 0.138 | 0.062 | 0.030 | 0.027 | 0.032 |
| gemma-4-E2B | 0.260 | 0.206 | 0.227 | 0.207 | 0.236 | 0.205 | 0.080 | 0.065 |

What the numbers mean:

- **gemma-3-1b-pt: invalid probe run, not zero memorization.**
  - It scored exactly 0 on both probes, every year, and has no empirical cutoff.
  - Gemma 4 went through the same code and gave normal results, so the probe code itself works.
  - The probable cause: `make_generator` loads models in fp16 on CUDA, and Gemma 3 is known to overflow in fp16.
  - Not confirmed: the answer CSVs have not been inspected.
  - Rerun it in fp32 only if Gemma 3 stays a contender.
- **Qwen empirical cutoffs: not informative.**
  - The threshold is half the 2019-20 recall, which comes to about 0.003 for 0.6B and 0.011 for 1.7B. A single correct
    answer in a quarter of roughly 150 films clears it, so the latest quarter with one lucky hit wins.
  - For 1.7B the computed date, 2026-06-30, is after Qwen3's own release on 2025-04-29
    ([announcement](https://qwenlm.github.io/blog/qwen3/), which states no cutoff).
  - Applied literally, the "later of stated and empirical" rule would leave the holdout almost empty.
  - The method does work when recall is high: Gemma 4's empirical 2024-12-31 matches its stated January 2025.
- **CPU speed: no candidate meets the 20 s rule.**
  - The fastest, Qwen3-0.6B, takes 100 s per batch-13 pass, about 7.7 s per film.
  - These were measured on 1 thread with full 768-token padding. Real prompts and the Space's 2 vCPUs will be somewhat
    faster, but not 5 times faster.
- **Memorization and leakage.** Low recall means the model knows little about 2019+ films, so less of what it knows
  could leak into the test set. That favours Qwen.
- **One family, two sizes.** Among the measured models, only Qwen3 (0.6B and 1.7B) gives one family with two sizes.

## Step 4: decision

Pending: decided with Dimitris after the measurements.
