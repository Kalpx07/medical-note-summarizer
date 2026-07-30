"""
Run inference against the base Llama 2 model and the fine-tuned model.

Backends (INFERENCE_BACKEND in .env):
    mlx        local on-device inference via mlx-lm (free, default)
    replicate  cloud inference via the Replicate API

Usage (CLI):
    python src/inference.py --model finetuned --note-file some_note.txt
    python src/inference.py --model base --note "Patient is a 54-year-old..."
"""

import argparse
import os
import re
import sys
from pathlib import Path

from config import (
    BASE_CHAT_MODEL,
    GENERATION_PARAMS,
    INFERENCE_BACKEND,
    MLX_ADAPTER_PATH,
    MLX_BASE_MODEL,
)

FINETUNED_MODEL = os.getenv("FINETUNED_MODEL", "")

PROMPT_TEMPLATE = (
    "You are a clinical documentation assistant. Summarize the following "
    "clinical note into a structured summary with exactly these sections: "
    "DIAGNOSIS, KEY FINDINGS, MEDICATIONS, ACTION ITEMS, FOLLOW-UP. "
    "Be concise and factual; do not add information not present in the note.\n\n"
    "CLINICAL NOTE:\n{note}\n\nSTRUCTURED SUMMARY:"
)


def build_prompt(note: str) -> str:
    return PROMPT_TEMPLATE.format(note=note.strip())


# ---------------------------------------------------------------- mlx backend

_MLX_CACHE: dict = {}


def _mlx_load(model: str):
    """Load (and cache) the local base model, with the LoRA adapter for 'finetuned'."""
    if model in _MLX_CACHE:
        return _MLX_CACHE[model]

    from mlx_lm import load

    if not Path(MLX_BASE_MODEL).exists():
        raise RuntimeError(
            f"Local model not found at {MLX_BASE_MODEL}. "
            "Run: python -m mlx_lm convert --hf-path NousResearch/Llama-2-7b-hf "
            f"--mlx-path {MLX_BASE_MODEL} -q"
        )

    adapter = None
    if model == "finetuned":
        if not Path(MLX_ADAPTER_PATH).exists():
            raise RuntimeError(
                f"LoRA adapter not found at {MLX_ADAPTER_PATH}. "
                "Train it first: see src/train_local.py"
            )
        adapter = MLX_ADAPTER_PATH

    _MLX_CACHE[model] = load(MLX_BASE_MODEL, adapter_path=adapter)
    return _MLX_CACHE[model]


def _truncate(text: str) -> str:
    """Base Llama 2 is a raw LM and keeps generating past the summary;
    cut at the first sign it starts a new example or echoes the prompt,
    and never keep anything past the end of the FOLLOW-UP line."""
    for marker in ("</s>", "CLINICAL NOTE:", "You are a clinical"):
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
    # FOLLOW-UP values are plain phrases ("3 months", "48-72 hours for culture
    # results"); anything past the first other punctuation is generation junk.
    text = re.sub(r"(FOLLOW-UP:\s*[A-Za-z0-9\- ]*).*", r"\1", text, flags=re.S)
    # drop stray non-ASCII artifacts the base LM sometimes emits at the end
    text = re.sub(r"[^\x20-\x7E\n]+", "", text)
    return text.strip()


def _summarize_mlx(note: str, model: str) -> str:
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_logits_processors, make_sampler

    lm, tokenizer = _mlx_load(model)
    sampler = make_sampler(
        temp=GENERATION_PARAMS["temperature"],
        top_p=GENERATION_PARAMS["top_p"],
    )
    # repetition penalty stops the model looping on notes that lack
    # sections it saw in every training example (e.g. vitals)
    logits_processors = make_logits_processors(repetition_penalty=1.15)
    out = generate(
        lm,
        tokenizer,
        prompt=build_prompt(note),
        max_tokens=GENERATION_PARAMS["max_new_tokens"],
        sampler=sampler,
        logits_processors=logits_processors,
    )
    return _truncate(out)


# ---------------------------------------------------------- replicate backend

def _summarize_replicate(note: str, model: str) -> str:
    import replicate

    if model == "finetuned":
        if not FINETUNED_MODEL:
            raise RuntimeError("FINETUNED_MODEL not set in .env "
                               "(format: username/model:version-hash)")
        target = FINETUNED_MODEL
    else:
        target = BASE_CHAT_MODEL

    output = replicate.run(
        target,
        input={"prompt": build_prompt(note), **GENERATION_PARAMS},
    )
    # replicate.run streams token chunks for language models
    return "".join(output).strip()


def summarize(note: str, model: str = "finetuned") -> str:
    """Return a structured summary of a clinical note.

    model: "finetuned" uses the LoRA-adapted model, "base" the off-the-shelf one.
    """
    if INFERENCE_BACKEND == "mlx":
        return _summarize_mlx(note, model)
    return _summarize_replicate(note, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["base", "finetuned"], default="finetuned")
    parser.add_argument("--note", help="clinical note text")
    parser.add_argument("--note-file", help="path to a text file containing the note")
    args = parser.parse_args()

    if args.note_file:
        with open(args.note_file) as f:
            note = f.read()
    elif args.note:
        note = args.note
    else:
        sys.exit("Provide --note or --note-file")

    print(summarize(note, model=args.model))
