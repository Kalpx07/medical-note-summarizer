#!/bin/bash
# =============================================================
#  run_all.sh — one-command runner for the medical summarizer
#  Automates every local step. Pauses only where YOUR accounts
#  are needed (Replicate token, gist URL, model name).
#
#  Usage:  bash run_all.sh
# =============================================================
set -e
cd "$(dirname "$0")"

bold() { printf "\n\033[1m%s\033[0m\n" "$1"; }

bold "STEP 1/6 — Python environment"
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip install -q -r requirements.txt
echo "✓ Dependencies installed"

bold "STEP 2/6 — Dataset"
if [ ! -f data/train.jsonl ]; then
  python data/generate_dataset.py --n 600 --out data/
fi
python src/data_prep.py --file data/train.jsonl
echo "✓ Dataset ready"

bold "STEP 3/6 — Your credentials (.env)"
if [ ! -f .env ]; then cp .env.example .env; fi

# Prompt for anything still unset
grep -q "r8_xxxxxxxxxxxx" .env && {
  echo ""
  echo "You need a Replicate API token."
  echo "  1. Sign up / log in at  https://replicate.com"
  echo "  2. Add a payment method (training costs ~\$3-7 total)"
  echo "  3. Copy a token from    https://replicate.com/account/api-tokens"
  read -p "Paste your Replicate token (starts with r8_): " TOKEN
  sed -i '' "s|REPLICATE_API_TOKEN=.*|REPLICATE_API_TOKEN=$TOKEN|" .env
}

grep -q "gist.githubusercontent.com/you" .env && {
  echo ""
  echo "Your training data must be at a public URL."
  echo "  1. Go to https://gist.github.com  (log in with GitHub)"
  echo "  2. Upload/paste the file  data/train.jsonl  → Create public gist"
  echo "  3. Click the 'Raw' button and copy that URL"
  read -p "Paste the RAW gist URL: " URL
  sed -i '' "s|TRAIN_DATA_URL=.*|TRAIN_DATA_URL=$URL|" .env
}

grep -q "DESTINATION_MODEL=your-username" .env && {
  echo ""
  echo "Create an empty destination model:"
  echo "  1. Go to https://replicate.com/create"
  echo "  2. Name it: llama2-med-summarizer  (private is fine)"
  read -p "Your Replicate username: " RUSER
  sed -i '' "s|DESTINATION_MODEL=.*|DESTINATION_MODEL=$RUSER/llama2-med-summarizer|" .env
}

echo "✓ .env configured"
echo ""
echo "NOTE: verify the base model version hash at"
echo "  https://replicate.com/meta/llama-2-7b/train"
echo "If it differs from BASE_MODEL_VERSION in .env, update it, then re-run."
read -p "Press Enter to start training (this bills your Replicate account ~\$2-6)... "

bold "STEP 4/6 — Fine-tuning on Replicate (15-40 min)"
python src/train.py
echo ""
read -p "Paste the fine-tuned version string printed above (username/model:hash): " FT
sed -i '' "s|FINETUNED_MODEL=.*|FINETUNED_MODEL=$FT|" .env
echo "✓ Fine-tuned model saved to .env"

bold "STEP 5/6 — Evaluation (your resume numbers, ~\$0.50-1)"
python src/evaluate.py --mode live --n 25
echo ""
echo "✓ Real metrics saved to results/eval_report.json"
echo "  → Copy these numbers into docs/RESUME_BULLETS.md"

bold "STEP 6/6 — Demo app"
echo "Launching Streamlit at http://localhost:8501 (Ctrl+C to stop)"
streamlit run app/streamlit_app.py
