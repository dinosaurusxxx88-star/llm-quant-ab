#!/usr/bin/env python3
"""
build_evalset.py — assemble a ~50-example eval set for ab_eval.py from
OPEN-LICENSE sources (no copyright headaches; gold labels where possible).

Sources:
  * French Wikipedia (CC BY-SA 4.0) via the MediaWiki API
      -> ~1000-word articles as "judge" tasks (open-ended analysis, no gold).
        Default pull is from "Catégorie:Article de qualité" (featured articles:
        long, clean, encyclopedic). Use --source random for random articles.
  * PIAF (etalab-ia/piaf, MIT) reading comprehension over French Wikipedia
      -> "qa" tasks WITH gold answers (scored by token F1).

Runs on YOUR machine (needs open internet + `datasets` + `requests`):

  uv run --with datasets --with requests build_evalset.py \
      --judge-n 40 --qa-n 10 --out evalset.jsonl

Then feed it to the A/B harness:

  uv run --with openai ab_eval.py --eval evalset.jsonl \
      --judge-base-url https://api.openai.com/v1 --judge-model gpt-4.1

License note: Wikipedia text is CC BY-SA 4.0 — each example keeps its source
URL in "_source" for attribution. Fine for internal eval use.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import re
import sys
import time
from typing import Any, Iterable, Optional

WIKI_API = "https://fr.wikipedia.org/w/api.php"
# MediaWiki API etiquette: send a descriptive User-Agent. Edit the contact.
UA = "Haruni-evalset-builder/1.0 (https://haruni.net; contact@haruni.net) python-requests"

ANALYSIS_PROMPTS = [
    "Résume en trois points les idées principales du texte, puis indique le point le plus important et justifie.",
    "Identifie les entités clés (personnes, lieux, organisations, dates) du texte et explique brièvement le rôle de chacune.",
    "Rédige un résumé fidèle en 4 à 5 phrases, sans ajouter aucune information absente du texte.",
    "Liste les affirmations factuelles vérifiables du texte, puis signale toute ambiguïté ou information manquante.",
    "Explique de quoi traite le texte à un lecteur non spécialiste, en 5 phrases maximum.",
]

JUDGE_SYS = ("Tu es un analyste rigoureux. Réponds de façon concise, structurée "
             "et strictement fidèle au texte fourni.")
QA_SYS = ("Réponds à la question de façon factuelle et concise, en t'appuyant "
          "uniquement sur le texte.")


# --------------------------- text helpers ---------------------------

def truncate_words(text: str, cap: int) -> str:
    """Truncate to <= cap words, backtracking to the last sentence end."""
    words = text.split()
    if len(words) <= cap:
        return text.strip()
    clipped = " ".join(words[:cap])
    cut = max(clipped.rfind("."), clipped.rfind("!"), clipped.rfind("?"))
    return (clipped[:cut + 1] if cut > 0 else clipped).strip()


# --------------------------- Wikipedia (CC BY-SA) ---------------------------

def _wiki_get(session: Any, params: dict) -> dict:
    r = session.get(WIKI_API, params={**params, "format": "json"}, timeout=30)
    r.raise_for_status()
    return r.json()


def _fetch_extract(session: Any, pageid: int) -> Optional[tuple[str, str]]:
    data = _wiki_get(session, {"action": "query", "prop": "extracts",
                               "explaintext": 1, "pageids": pageid, "redirects": 1})
    pages = data.get("query", {}).get("pages", {})
    page = pages.get(str(pageid)) or (next(iter(pages.values())) if pages else None)
    if not page:
        return None
    extract = (page.get("extract") or "").strip()
    return (page.get("title", ""), extract) if extract else None


def _article(session: Any, pageid: int, min_words: int, cap: int) -> Optional[dict]:
    got = _fetch_extract(session, pageid)
    if not got:
        return None
    title, extract = got
    if len(extract.split()) < min_words:
        return None
    text = truncate_words(extract, cap)
    return {"title": title, "url": f"https://fr.wikipedia.org/?curid={pageid}",
            "text": text, "n_words": len(text.split())}


def fetch_category(session: Any, category: str, n: int, min_words: int, cap: int) -> list[dict]:
    out: list[dict] = []
    cmcontinue, guard = None, 0
    while len(out) < n and guard < 30:
        guard += 1
        params: dict[str, Any] = {"action": "query", "list": "categorymembers",
                                  "cmtitle": category, "cmnamespace": 0,
                                  "cmlimit": 50, "cmtype": "page"}
        if cmcontinue:
            params["cmcontinue"] = cmcontinue
        data = _wiki_get(session, params)
        for m in data.get("query", {}).get("categorymembers", []):
            if len(out) >= n:
                break
            art = _article(session, m["pageid"], min_words, cap)
            if art:
                out.append(art)
                print(f"  + [{len(out)}/{n}] {art['title']} ({art['n_words']} mots)")
            time.sleep(0.1)
        cmcontinue = data.get("continue", {}).get("cmcontinue")
        if not cmcontinue:
            break
    return out


def fetch_random(session: Any, n: int, min_words: int, cap: int, max_batches: int = 300) -> list[dict]:
    out: list[dict] = []
    seen: set[int] = set()
    for _ in range(max_batches):
        if len(out) >= n:
            break
        rnd = _wiki_get(session, {"action": "query", "list": "random",
                                  "rnnamespace": 0, "rnlimit": 10})
        for p in rnd.get("query", {}).get("random", []):
            if len(out) >= n:
                break
            pid = p["id"]
            if pid in seen:
                continue
            seen.add(pid)
            art = _article(session, pid, min_words, cap)
            if art:
                out.append(art)
                print(f"  + [{len(out)}/{n}] {art['title']} ({art['n_words']} mots)")
            time.sleep(0.1)
    return out


def build_judge_examples(articles: Iterable[dict]) -> list[dict]:
    cyc = itertools.cycle(ANALYSIS_PROMPTS)
    out = []
    for k, a in enumerate(articles):
        out.append({
            "id": f"wiki-{k:02d}",
            "type": "judge",
            "system": JUDGE_SYS,
            "input": f"{next(cyc)}\n\nTexte :\n{a['text']}",
            "_source": f"Wikipédia FR (CC BY-SA 4.0) — {a['title']} — {a['url']}",
        })
    return out


# --------------------------- PIAF (MIT, gold answers) ---------------------------

def rows_to_qa(rows: Iterable[dict], n: int, min_ctx_words: int = 40) -> list[dict]:
    out = []
    for i, row in enumerate(rows):
        if len(out) >= n:
            break
        ans = (row.get("answers") or {}).get("text") or []
        ctx = (row.get("context") or "").strip()
        q = (row.get("question") or "").strip()
        if not ans or not ctx or not q or len(ctx.split()) < min_ctx_words:
            continue
        out.append({
            "id": f"piaf-{row.get('id', i)}",
            "type": "qa",
            "system": QA_SYS,
            "input": f"Question : {q}\n\nTexte : {ctx}",
            "gold": ans[0],
            "_source": f"PIAF (etalab-ia/piaf, MIT) — {row.get('title', '')}",
        })
    return out


def fetch_piaf_qa(n: int, dataset_id: str = "etalab-ia/piaf") -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset(dataset_id, split="train")
    idxs = list(range(len(ds)))
    random.Random(0).shuffle(idxs)            # deterministic sampling
    return rows_to_qa((ds[i] for i in idxs), n)


# --------------------------- main ---------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Build a ~50-example eval set from open sources.")
    ap.add_argument("--judge-n", type=int, default=40, help="Wikipedia 'judge' examples")
    ap.add_argument("--qa-n", type=int, default=10, help="PIAF 'qa' examples (gold)")
    ap.add_argument("--source", choices=["quality", "random"], default="quality",
                    help="Wikipedia pull: featured articles (quality) or random")
    ap.add_argument("--category", default="Catégorie:Article de qualité")
    ap.add_argument("--min-words", type=int, default=600, help="keep articles with >= this many words")
    ap.add_argument("--max-words", type=int, default=1200, help="truncate articles to this many words")
    ap.add_argument("--piaf-id", default="etalab-ia/piaf")
    ap.add_argument("--out", default="evalset.jsonl")
    args = ap.parse_args()

    examples: list[dict] = []

    if args.judge_n > 0:
        try:
            import requests
        except ImportError:
            sys.exit("Need `requests` for Wikipedia: uv run --with datasets --with requests build_evalset.py ...")
        session = requests.Session()
        session.headers["User-Agent"] = UA
        print(f"Fetching {args.judge_n} Wikipedia FR articles ({args.source}, "
              f"{args.min_words}-{args.max_words} words)...")
        arts = (fetch_category(session, args.category, args.judge_n, args.min_words, args.max_words)
                if args.source == "quality"
                else fetch_random(session, args.judge_n, args.min_words, args.max_words))
        if len(arts) < args.judge_n:
            print(f"  (only {len(arts)} found — try --source random or lower --min-words)")
        examples += build_judge_examples(arts)

    if args.qa_n > 0:
        print(f"Loading {args.qa_n} PIAF qa examples (gold)...")
        try:
            examples += fetch_piaf_qa(args.qa_n, args.piaf_id)
        except Exception as e:  # noqa: BLE001
            print(f"  PIAF load failed ({e}); skipping qa. Install `datasets` and check network.")

    if not examples:
        sys.exit("No examples produced.")

    with open(args.out, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    by_type: dict[str, int] = {}
    for ex in examples:
        by_type[ex["type"]] = by_type.get(ex["type"], 0) + 1
    print(f"\nWrote {len(examples)} examples to {args.out}: {by_type}")
    print("Reminder: Wikipedia content is CC BY-SA 4.0 (attribution kept in '_source').")


if __name__ == "__main__":
    main()
