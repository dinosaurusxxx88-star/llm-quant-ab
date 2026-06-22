#!/usr/bin/env python3
"""
ab_eval.py — A/B quality evaluation harness.

Compares a CANDIDATE model (your local NVFP4 served by vLLM) against a
REFERENCE model (full-precision Mistral Small 4 via the Mistral API) on
text-analysis tasks, to estimate quality degradation from quantization.

Task types per eval example (field "type"):
  - "extraction"     : gold is a list of expected items -> set-based F1
  - "classification" : gold is a single label string    -> exact match
  - "qa"             : gold is a reference answer string -> SQuAD token F1
  - "judge"          : no gold; blind TWO-SIDED pairwise LLM-as-judge
                       (each pair is judged in both orderings; only a
                        consistent verdict counts as a win, else tie)

Outputs per-example results + an aggregate summary, and writes
<out>.json and <out>.csv.

API keys are read from environment variables (never hard-code them):
  MISTRAL_API_KEY   -> reference (api.mistral.ai)
  JUDGE_API_KEY     -> judge endpoint
  CANDIDATE_API_KEY -> local vLLM (optional; defaults to "EMPTY")

Quick start (uv):
  export MISTRAL_API_KEY=...
  export JUDGE_API_KEY=...
  uv run --with openai ab_eval.py \
      --eval eval_examples.jsonl \
      --judge-base-url https://api.openai.com/v1 \
      --judge-model gpt-4.1 \
      --out results

Cost note: this calls the Mistral API and the judge API. Cost scales with the
number of examples (and with reasoning_effort). Start small (e.g. 10) to
validate the pipeline before scaling to 50-150.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import string
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import Any, Optional

try:
    from openai import OpenAI
except ImportError:
    sys.exit("Missing dependency: pip install openai  (or run with: uv run --with openai ab_eval.py ...)")


# --------------------------- text normalization ---------------------------

_PUNCT = str.maketrans("", "", string.punctuation)


def normalize(s: str) -> str:
    s = (s or "").lower().strip()
    s = s.translate(_PUNCT)
    s = re.sub(r"\s+", " ", s)
    return s


def tokenize(s: str) -> list[str]:
    return normalize(s).split()


# --------------------------- scoring ---------------------------

def exact_match(pred: str, gold: str) -> float:
    return 1.0 if normalize(pred) == normalize(gold) else 0.0


def token_f1(pred: str, gold: str) -> float:
    p, g = tokenize(pred), tokenize(gold)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    overlap = sum((Counter(p) & Counter(g)).values())
    if overlap == 0:
        return 0.0
    prec, rec = overlap / len(p), overlap / len(g)
    return 2 * prec * rec / (prec + rec)


def _coerce_items(value: Any) -> list[str]:
    """Turn a prediction or gold into a list of normalized item strings."""
    if isinstance(value, list):
        items = value
    elif isinstance(value, str):
        try:  # model was asked to emit a JSON list
            parsed = json.loads(value)
            items = parsed if isinstance(parsed, list) else [parsed]
        except json.JSONDecodeError:
            items = re.split(r"[\n,;]+", value)  # fallback split
    else:
        items = [value]
    out = []
    for it in items:
        if isinstance(it, (dict, list)):
            it = json.dumps(it, ensure_ascii=False, sort_keys=True)
        n = normalize(str(it))
        if n:
            out.append(n)
    return out


def set_f1(pred: Any, gold: Any) -> float:
    p, g = set(_coerce_items(pred)), set(_coerce_items(gold))
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    tp = len(p & g)
    if tp == 0:
        return 0.0
    prec, rec = tp / len(p), tp / len(g)
    return 2 * prec * rec / (prec + rec)


# --------------------------- model calls ---------------------------

@dataclass
class Provider:
    name: str
    client: Any
    model: str


def make_provider(name: str, base_url: str, api_key: str, model: str) -> Provider:
    client = OpenAI(base_url=base_url, api_key=api_key or "EMPTY")
    if model == "auto":
        model = client.models.list().data[0].id
    return Provider(name=name, client=client, model=model)


def complete(prov: Provider, system: Optional[str], user: str, temperature: float,
             max_tokens: int, reasoning_effort: Optional[str], seed: Optional[int],
             retries: int = 4) -> str:
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    kwargs: dict[str, Any] = dict(model=prov.model, messages=messages,
                                  temperature=temperature, max_tokens=max_tokens)
    if seed is not None:
        kwargs["seed"] = seed
    if reasoning_effort:
        kwargs["extra_body"] = {"reasoning_effort": reasoning_effort}

    last: Optional[Exception] = None
    for attempt in range(retries):
        try:
            resp = prov.client.chat.completions.create(**kwargs)
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(min(2 ** attempt, 20))
    raise RuntimeError(f"{prov.name} call failed after {retries} tries: {last}")


# --------------------------- judge ---------------------------

JUDGE_SYS = (
    "You are a strict, impartial evaluator. You compare two AI answers (A and B) "
    "to the SAME analysis task and decide which one better fulfills the instruction: "
    "more accurate, complete, faithful to the source text, and well-reasoned. "
    "Ignore style and length unless they affect quality. "
    'Reply ONLY with compact JSON: {"verdict":"A"|"B"|"tie","reason":"<one short sentence>"}. '
    "Use 'tie' only when the two answers are genuinely of equal quality."
)


def judge_once(judge: Provider, task: str, ans_a: str, ans_b: str) -> str:
    """Return 'A', 'B', or 'tie'."""
    user = (f"# Task\n{task}\n\n# Answer A\n{ans_a}\n\n# Answer B\n{ans_b}\n\n"
            "Which answer is better? Reply with the JSON verdict only.")
    raw = complete(judge, JUDGE_SYS, user, temperature=0.0, max_tokens=200,
                   reasoning_effort=None, seed=0)
    try:
        verdict = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))["verdict"]
    except Exception:  # noqa: BLE001
        verdict = "tie"
    return {"a": "A", "b": "B"}.get(str(verdict).strip().lower(), "tie")


def judge_two_sided(judge: Provider, task: str, cand: str, ref: str) -> str:
    """Judge both orderings; a verdict counts only if consistent across orders.
    Returns 'candidate', 'reference', or 'tie'."""
    v1 = {"A": "candidate", "B": "reference", "tie": "tie"}[judge_once(judge, task, cand, ref)]
    v2 = {"A": "reference", "B": "candidate", "tie": "tie"}[judge_once(judge, task, ref, cand)]
    return v1 if (v1 == v2 and v1 != "tie") else "tie"


# --------------------------- eval driver ---------------------------

@dataclass
class Result:
    id: str
    type: str
    candidate_output: str = ""
    reference_output: str = ""
    candidate_score: Optional[float] = None
    reference_score: Optional[float] = None
    judge_winner: Optional[str] = None
    error: Optional[str] = None


def run_example(ex: dict, cand: Provider, ref: Provider, judge: Optional[Provider],
                temperature: float, max_tokens: int, reasoning: Optional[str],
                seed: Optional[int]) -> Result:
    r = Result(id=str(ex.get("id", "")), type=ex.get("type", "judge"))
    system, user = ex.get("system"), ex["input"]
    try:
        r.candidate_output = complete(cand, system, user, temperature, max_tokens, reasoning, seed)
        r.reference_output = complete(ref, system, user, temperature, max_tokens, reasoning, seed)
        if r.type == "classification":
            r.candidate_score = exact_match(r.candidate_output, ex["gold"])
            r.reference_score = exact_match(r.reference_output, ex["gold"])
        elif r.type == "extraction":
            r.candidate_score = set_f1(r.candidate_output, ex["gold"])
            r.reference_score = set_f1(r.reference_output, ex["gold"])
        elif r.type == "qa":
            gold = ex["gold"] if isinstance(ex["gold"], str) else json.dumps(ex["gold"], ensure_ascii=False)
            r.candidate_score = token_f1(r.candidate_output, gold)
            r.reference_score = token_f1(r.reference_output, gold)
        elif r.type == "judge":
            if judge is None:
                raise RuntimeError("judge task but no judge configured")
            r.judge_winner = judge_two_sided(judge, user, r.candidate_output, r.reference_output)
        else:
            raise RuntimeError(f"unknown task type: {r.type}")
    except Exception as e:  # noqa: BLE001
        r.error = str(e)
    return r


def load_eval(path: str) -> list[dict]:
    examples = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            ex = json.loads(line)
            ex.setdefault("id", f"ex{i}")
            examples.append(ex)
    return examples


def summarize(results: list[Result]) -> dict:
    gold = [r for r in results if r.candidate_score is not None and not r.error]
    judged = [r for r in results if r.judge_winner is not None and not r.error]
    summary: dict[str, Any] = {"n_total": len(results),
                               "n_errors": sum(1 for r in results if r.error)}
    if gold:
        by_type: dict[str, list[Result]] = {}
        for r in gold:
            by_type.setdefault(r.type, []).append(r)
        summary["gold"] = {}
        for t, rs in by_type.items():
            cand = sum(r.candidate_score for r in rs) / len(rs)
            refr = sum(r.reference_score for r in rs) / len(rs)
            summary["gold"][t] = {"n": len(rs), "candidate_mean": round(cand, 4),
                                  "reference_mean": round(refr, 4), "delta": round(cand - refr, 4)}
    if judged:
        c = Counter(r.judge_winner for r in judged)
        n = len(judged)
        summary["judge"] = {"n": n, "candidate_win": c.get("candidate", 0),
                            "tie": c.get("tie", 0), "reference_win": c.get("reference", 0),
                            "candidate_win_rate": round(c.get("candidate", 0) / n, 4),
                            "no_degradation_rate": round((c.get("candidate", 0) + c.get("tie", 0)) / n, 4)}
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="A/B quality eval: local NVFP4 vs Mistral API reference.")
    ap.add_argument("--eval", required=True, help="path to JSONL eval file")
    ap.add_argument("--candidate-base-url", default="http://localhost:8000/v1")
    ap.add_argument("--candidate-model", default="auto", help="'auto' = first model served by vLLM")
    ap.add_argument("--reference-base-url", default="https://api.mistral.ai/v1")
    ap.add_argument("--reference-model", default="mistral-small-latest")
    ap.add_argument("--judge-base-url", default=os.environ.get("JUDGE_BASE_URL", ""))
    ap.add_argument("--judge-model", default=os.environ.get("JUDGE_MODEL", ""))
    ap.add_argument("--reasoning", default="none", choices=["none", "high", "off"],
                    help="reasoning_effort sent to candidate+reference; 'off' omits the param entirely")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=None, help="pass only if BOTH endpoints support it")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    reasoning = None if args.reasoning == "off" else args.reasoning

    cand = make_provider("candidate", args.candidate_base_url,
                         os.environ.get("CANDIDATE_API_KEY", "EMPTY"), args.candidate_model)
    ref_key = os.environ.get("MISTRAL_API_KEY")
    if not ref_key:
        sys.exit("Set MISTRAL_API_KEY (reference endpoint).")
    ref = make_provider("reference", args.reference_base_url, ref_key, args.reference_model)

    examples = load_eval(args.eval)
    judge: Optional[Provider] = None
    if any(ex.get("type", "judge") == "judge" for ex in examples):
        if not args.judge_base_url or not args.judge_model:
            sys.exit("Eval contains 'judge' tasks: set --judge-base-url and --judge-model (and JUDGE_API_KEY).")
        judge = make_provider("judge", args.judge_base_url,
                             os.environ.get("JUDGE_API_KEY", "EMPTY"), args.judge_model)
        if judge.model in (cand.model, ref.model):
            print("WARNING: judge matches candidate or reference — use an INDEPENDENT judge to avoid bias.", file=sys.stderr)

    print(f"Candidate : {cand.model} @ {args.candidate_base_url}")
    print(f"Reference : {ref.model} @ {args.reference_base_url}")
    if judge:
        print(f"Judge     : {judge.model} @ {args.judge_base_url}")
    print(f"Examples  : {len(examples)} | reasoning={args.reasoning} temp={args.temperature} workers={args.workers}\n")

    results: list[Result] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_example, ex, cand, ref, judge, args.temperature,
                            args.max_tokens, reasoning, args.seed): ex for ex in examples}
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            results.append(r)
            if r.error:
                tag = f"ERROR: {r.error}"
            elif r.candidate_score is not None:
                tag = f"score c={r.candidate_score:.3f} r={r.reference_score:.3f}"
            else:
                tag = f"judge={r.judge_winner}"
            print(f"[{i}/{len(examples)}] {r.id} ({r.type}) {tag}")

    results.sort(key=lambda r: r.id)
    summary = summarize(results)

    with open(f"{args.out}.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": [asdict(r) for r in results]},
                  f, ensure_ascii=False, indent=2)
    with open(f"{args.out}.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "type", "candidate_score", "reference_score", "judge_winner", "error"])
        for r in results:
            w.writerow([r.id, r.type, r.candidate_score, r.reference_score, r.judge_winner, r.error or ""])

    print("\n===== SUMMARY =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nWrote {args.out}.json and {args.out}.csv")


if __name__ == "__main__":
    main()
