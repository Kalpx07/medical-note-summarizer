"""
Evaluation framework: base model vs fine-tuned model on the held-out test set.

Metrics:
  - ROUGE-1 / ROUGE-2 / ROUGE-L (rouge-score)
  - BLEU (nltk, smoothed)
  - Structure compliance: does the output contain all 5 required sections?
  - Number fidelity (hallucination proxy): fraction of numbers in the summary
    that actually appear in the source note. Catches invented dosages/vitals.

Modes:
  --mode live    call Replicate for both models on N test examples (costs $)
  --mode offline score prediction files you've already saved
                 (results/preds_base.jsonl, results/preds_finetuned.jsonl,
                  each row: {"prompt":..., "reference":..., "prediction":...})

Usage:
    python src/evaluate.py --mode live --n 25
    python src/evaluate.py --mode offline
"""

import argparse
import json
import re
import statistics
from pathlib import Path

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

from config import DATA_DIR, RESULTS_DIR, REQUIRED_SECTIONS

_smooth = SmoothingFunction().method1
_rouge = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

NUM_RE = re.compile(r"\d+(?:\.\d+)?")


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def rouge_scores(reference: str, prediction: str) -> dict:
    s = _rouge.score(reference, prediction)
    return {k: v.fmeasure for k, v in s.items()}


def bleu_score(reference: str, prediction: str) -> float:
    ref_tokens = reference.lower().split()
    pred_tokens = prediction.lower().split()
    if not pred_tokens:
        return 0.0
    return sentence_bleu([ref_tokens], pred_tokens, smoothing_function=_smooth)


def structure_compliance(prediction: str) -> float:
    """Fraction of required section headers present in the output."""
    present = sum(1 for s in REQUIRED_SECTIONS if s in prediction.upper())
    return present / len(REQUIRED_SECTIONS)


def number_fidelity(source_note: str, prediction: str) -> float:
    """Fraction of numbers in the prediction that appear in the source note.

    A cheap but effective hallucination proxy for clinical text: invented
    dosages, vitals, or lab values show up as numbers absent from the source.
    Returns 1.0 if the prediction contains no numbers.
    """
    source_nums = set(NUM_RE.findall(source_note))
    pred_nums = NUM_RE.findall(prediction)
    if not pred_nums:
        return 1.0
    return sum(1 for n in pred_nums if n in source_nums) / len(pred_nums)


def score_pair(source_note: str, reference: str, prediction: str) -> dict:
    out = rouge_scores(reference, prediction)
    out["bleu"] = bleu_score(reference, prediction)
    out["structure"] = structure_compliance(prediction)
    out["number_fidelity"] = number_fidelity(source_note, prediction)
    return out


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def extract_note(prompt: str) -> str:
    """Pull the raw clinical note back out of the training prompt."""
    if "CLINICAL NOTE:" in prompt:
        return prompt.split("CLINICAL NOTE:", 1)[1].split("STRUCTURED SUMMARY:", 1)[0]
    return prompt


def load_test_set(n: int | None = None) -> list[dict]:
    rows = []
    with open(DATA_DIR / "test.jsonl") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows[:n] if n else rows


def run_live(n: int) -> None:
    from inference import summarize  # imported lazily; needs API token
    test = load_test_set(n)
    for model in ("base", "finetuned"):
        preds = []
        print(f"\nGenerating with {model} model ({len(test)} examples)...")
        for i, row in enumerate(test, 1):
            note = extract_note(row["prompt"])
            try:
                pred = summarize(note, model=model)
            except Exception as e:
                print(f"  [{i}] error: {e}")
                pred = ""
            preds.append({"prompt": row["prompt"],
                          "reference": row["completion"].strip(),
                          "prediction": pred})
            print(f"  [{i}/{len(test)}] done")
        out_path = RESULTS_DIR / f"preds_{model}.jsonl"
        with open(out_path, "w") as f:
            for p in preds:
                f.write(json.dumps(p) + "\n")
        print(f"Saved -> {out_path}")
    run_offline()


def run_offline() -> None:
    report = {}
    for model in ("base", "finetuned"):
        path = RESULTS_DIR / f"preds_{model}.jsonl"
        if not path.exists():
            print(f"Skipping {model}: {path} not found")
            continue
        rows = [json.loads(l) for l in open(path)]
        metric_lists: dict[str, list[float]] = {}
        for r in rows:
            scores = score_pair(extract_note(r["prompt"]), r["reference"], r["prediction"])
            for k, v in scores.items():
                metric_lists.setdefault(k, []).append(v)
        report[model] = {k: statistics.mean(v) for k, v in metric_lists.items()}

    if not report:
        return

    print("\n" + "=" * 62)
    print(f"{'Metric':<18}" + "".join(f"{m:>14}" for m in report))
    print("-" * 62)
    metrics = list(next(iter(report.values())).keys())
    for m in metrics:
        row = f"{m:<18}"
        for model in report:
            row += f"{report[model].get(m, float('nan')):>14.3f}"
        print(row)
    print("=" * 62)

    with open(RESULTS_DIR / "eval_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved -> {RESULTS_DIR / 'eval_report.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["live", "offline"], default="offline")
    parser.add_argument("--n", type=int, default=25, help="test examples for live mode")
    args = parser.parse_args()
    if args.mode == "live":
        run_live(args.n)
    else:
        run_offline()
