"""
Free local LoRA fine-tuning with mlx-lm (Apple Silicon) — replaces the paid
Replicate managed training in train.py.

Trains a LoRA adapter on the 4-bit quantized base model (QLoRA-style) using
the prompt/completion JSONL in data/mlx/. Loss is computed on the completion
only (--mask-prompt) so the model learns the output schema, not the prompt.

Usage:
    python src/train_local.py            # train with defaults
    python src/train_local.py --iters 200 --batch-size 1   # quick/smaller run
"""

import argparse
import subprocess
import sys
from pathlib import Path

from config import MLX_ADAPTER_PATH, MLX_BASE_MODEL, PROJECT_ROOT

DATA_DIR = PROJECT_ROOT / "data" / "mlx"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=600,
                        help="training iterations (~batch_size*iters/510 epochs)")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-layers", type=int, default=16,
                        help="number of transformer layers to apply LoRA to")
    args = parser.parse_args()

    if not Path(MLX_BASE_MODEL).exists():
        sys.exit(f"Base model missing at {MLX_BASE_MODEL}.\n"
                 "Run: python -m mlx_lm convert --hf-path NousResearch/Llama-2-7b-hf "
                 f"--mlx-path {MLX_BASE_MODEL} -q")

    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", MLX_BASE_MODEL,
        "--train",
        "--data", str(DATA_DIR),
        "--adapter-path", MLX_ADAPTER_PATH,
        "--iters", str(args.iters),
        "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.learning_rate),
        "--num-layers", str(args.num_layers),
        "--mask-prompt",
        "--steps-per-report", "20",
        "--steps-per-eval", "100",
        "--save-every", "100",
        "--grad-checkpoint",
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"\nDone. LoRA adapter saved to {MLX_ADAPTER_PATH}")
    print("Test it:  python src/inference.py --model finetuned --note '<clinical note>'")


if __name__ == "__main__":
    main()
