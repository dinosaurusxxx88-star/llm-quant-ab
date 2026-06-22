# llm-quant-ab

**Did your 4-bit quantization cost quality? Measure it instead of guessing.**

## What it is

A **blind A/B test bench** that compares a *candidate* model (e.g. your local
NVFP4 checkpoint served by [vLLM](https://github.com/vllm-project/vllm)) against
a *full-precision reference* (the Mistral API by default) on text-analysis
tasks, to estimate the quality degradation — if any — caused by quantization.

It covers two task families:

- **Ground-truth tasks** — extraction, classification, and reading-comprehension
  QA, scored automatically against gold labels (no judge needed).
- **Open-ended analysis** — summarization, entity analysis, faithful rewriting,
  scored by a **two-sided pairwise LLM judge** (no gold needed).

## Article

This bench backs the article *« La quantification 4-bit coûte-t-elle de la qualité ? »*
/ *"Does 4-bit quantization cost quality?"*

- 🇫🇷 [lien]
- 🇬🇧 [link]

*(placeholders to fill once the article is online)*

## Install

[uv](https://docs.astral.sh/uv/) is recommended.

```bash
# run directly without installing anything (recommended)
uv run --with openai ab_eval.py --help
uv run --with datasets --with requests build_evalset.py --help

# or a managed environment
uv sync --extra data
```

Requires Python ≥ 3.10. The only runtime dependency for the harness is
`openai`; `datasets` + `requests` are needed only to build an eval set.

## Quickstart

```bash
# 1. Build an open-source eval set (Wikipedia FR + PIAF), on your machine
uv run --with datasets --with requests build_evalset.py \
    --judge-n 40 --qa-n 10 --out evalset.jsonl

# 2. Run the blind A/B comparison
export MISTRAL_API_KEY=...      # reference endpoint (api.mistral.ai)
export JUDGE_API_KEY=...        # independent judge endpoint
uv run --with openai ab_eval.py \
    --eval evalset.jsonl \
    --candidate-base-url http://localhost:8000/v1 \
    --judge-base-url https://api.openai.com/v1 --judge-model gpt-4.1 \
    --out results
```

Environment variables (keys are **never** hard-coded):

| Variable            | Used for                                   | Default   |
| ------------------- | ------------------------------------------ | --------- |
| `MISTRAL_API_KEY`   | reference model (api.mistral.ai)           | required  |
| `JUDGE_API_KEY`     | judge endpoint (for `judge` tasks)         | required* |
| `CANDIDATE_API_KEY` | local vLLM candidate                       | `EMPTY`   |

\* required only if the eval set contains `judge` tasks.

The candidate model defaults to `auto` (the first model served by your vLLM
endpoint). Outputs are written to `results.json` and `results.csv`. Start small
(e.g. 10 examples) to validate the pipeline before scaling to 50–150, since each
run calls the reference and judge APIs.

## Methodology

- **Gold scoring (no judge).** Deterministic metrics computed directly against
  gold labels:
  - `classification` → exact match on the normalized label,
  - `extraction` → **set-based F1** over the expected items (model emits a JSON list),
  - `qa` → **SQuAD-style token F1** against the reference answer.
- **Open-ended analysis → two-sided pairwise judge.** For `judge` tasks there is
  no gold. Each candidate/reference pair is judged **in both orderings** (A-vs-B
  and B-vs-A); a verdict counts as a win **only if it is consistent across both
  orders**, otherwise it's a tie. This cancels the judge's position bias. The
  judge replies with compact JSON (`{"verdict": "A"|"B"|"tie", ...}`).
- Calls use `temperature=0` and a configurable `reasoning_effort`
  (`--reasoning none|high|off`, default `none`). The harness warns if the judge
  model matches the candidate or reference — always use an **independent** judge.

Summary metrics include, per gold task type, the candidate/reference means and
their delta; and for judge tasks the candidate win rate plus a
`no_degradation_rate` (candidate wins + ties).

## Data sources & licenses

- **PIAF** (`etalab-ia/piaf`, **MIT**) — French reading-comprehension QA with
  gold answers, used for the `qa` tasks.
- **French Wikipedia** (**CC BY-SA 4.0**) via the MediaWiki API — ~1000-word
  articles used as open-ended `judge` tasks. Attribution (title + URL) is kept
  in each example's `_source` field.

`eval_examples.jsonl` is a **template** showing one example per task type —
replace it with your own data (or generate a real set via `build_evalset.py`).

## Limitations

- The default reference is the **Mistral API**, not a controlled local BF16
  checkpoint — so the comparison mixes quantization effects with any
  serving/version differences on the reference side.
- **"No significant difference" ≠ "identical."** Small eval sets have wide
  confidence intervals.
- Absolute **F1 levels on PIAF are a metric artifact** (token overlap), not a
  measure of usefulness — read the *delta* candidate−reference, not the raw score.
- **Single judge** → model-specific bias is not cancelled (only position bias is).
- `reasoning_effort=none` by default; raise it if your use case relies on
  chain-of-thought.
- Wikipedia texts are general-purpose — swap in domain texts for a representative
  eval.

## License

MIT — see [LICENSE](LICENSE). © 2026 Sébastien Burel.
