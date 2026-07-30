"""Upload the GGUF model repo and the Streamlit Space to Hugging Face.

Usage:
    python deploy/upload_space.py
Reads HF_TOKEN from .env. Idempotent: re-running updates both repos.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

TOKEN = os.environ["HF_TOKEN"]
api = HfApi(token=TOKEN)
USER = api.whoami()["name"]

MODEL_REPO = f"{USER}/llama2-med-summarizer-gguf"
SPACE_REPO = f"{USER}/medical-note-summarizer"

MODEL_CARD = """---
license: llama2
base_model: NousResearch/Llama-2-7b-hf
tags: [medical, summarization, lora, gguf, llama-cpp]
---

# Llama 2 7B Medical Note Summarizer (GGUF Q4_K_M)

LoRA fine-tune of Llama 2 7B that converts free-text clinical notes into structured
summaries with fixed sections: DIAGNOSIS, KEY FINDINGS, MEDICATIONS, ACTION ITEMS,
FOLLOW-UP. Adapter fused into the base weights and quantized to Q4_K_M for llama.cpp.

Trained on ~500 fully synthetic notes (MLX LoRA, completion-masked loss).
Eval vs base on 25 held-out notes: ROUGE-1 0.944 vs 0.710, BLEU 0.920 vs 0.351,
structure compliance 1.000 vs 0.976.

Use a repetition penalty (~1.15). Known limitation: notes missing sections the
training data always contained (e.g. vitals) can trigger hallucinated values.

> Educational project. Synthetic data only — not for real patient data or clinical use.

Demo: https://huggingface.co/spaces/{space}
""".replace("{space}", SPACE_REPO)


def main() -> None:
    # ---- model repo ----
    api.create_repo(MODEL_REPO, repo_type="model", exist_ok=True)
    api.upload_file(path_or_fileobj=MODEL_CARD.encode(), path_in_repo="README.md",
                    repo_id=MODEL_REPO, repo_type="model")
    print(f"Uploading GGUF to {MODEL_REPO} (3.8 GB, this is the slow part)...")
    api.upload_file(
        path_or_fileobj=ROOT / "models" / "med-summarizer-Q4_K_M.gguf",
        path_in_repo="med-summarizer-Q4_K_M.gguf",
        repo_id=MODEL_REPO, repo_type="model",
    )
    print(f"Model: https://huggingface.co/{MODEL_REPO}")

    # ---- space ----
    api.create_repo(SPACE_REPO, repo_type="space", space_sdk="docker", exist_ok=True)
    api.upload_folder(folder_path=ROOT / "deploy" / "space", repo_id=SPACE_REPO,
                      repo_type="space")
    print(f"Space: https://huggingface.co/spaces/{SPACE_REPO}")


if __name__ == "__main__":
    main()
