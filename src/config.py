"""Central configuration. Reads secrets from environment / .env file."""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# --- Replicate ---
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN", "")

# Base model to fine-tune. Check https://replicate.com/meta/llama-2-7b/train
# for the latest trainable version hash before running.
BASE_MODEL_VERSION = os.getenv(
    "BASE_MODEL_VERSION",
    "meta/llama-2-7b:73001d654114dad81ec65da3b834e2f691af1e1526453189b7bf36fb3f32d0f9",
)

# Base chat model used for the "before" comparison at inference time.
BASE_CHAT_MODEL = os.getenv("BASE_CHAT_MODEL", "meta/llama-2-7b-chat")

# Where the trained model gets pushed: "<your-replicate-username>/<model-name>"
# Create the empty model first at https://replicate.com/create
DESTINATION_MODEL = os.getenv("DESTINATION_MODEL", "your-username/llama2-med-summarizer")

# Publicly reachable URL of your train.jsonl (GitHub raw / gist / GCS / S3).
TRAIN_DATA_URL = os.getenv("TRAIN_DATA_URL", "")

# --- Training hyperparameters ---
TRAIN_PARAMS = {
    "num_train_epochs": 3,
    "train_batch_size": 4,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-5,
}

# --- Local MLX backend (free, on-device fine-tuning + inference) ---
# INFERENCE_BACKEND: "mlx" runs everything locally, "replicate" uses the cloud API.
INFERENCE_BACKEND = os.getenv("INFERENCE_BACKEND", "mlx")
MLX_BASE_MODEL = os.getenv("MLX_BASE_MODEL", str(PROJECT_ROOT / "models" / "llama2-7b-4bit"))
MLX_ADAPTER_PATH = os.getenv("MLX_ADAPTER_PATH", str(PROJECT_ROOT / "models" / "adapters"))

# --- Inference ---
GENERATION_PARAMS = {
    "max_new_tokens": 300,
    "temperature": 0.2,
    "top_p": 0.9,
    "stop_sequences": "</s>",
}

REQUIRED_SECTIONS = ["DIAGNOSIS", "KEY FINDINGS", "MEDICATIONS", "ACTION ITEMS", "FOLLOW-UP"]
