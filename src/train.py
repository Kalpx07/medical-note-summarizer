"""
Kick off Llama 2 7B fine-tuning on Replicate and monitor progress.

Prerequisites:
  1. REPLICATE_API_TOKEN set in .env
  2. Empty destination model created at https://replicate.com/create
  3. train.jsonl uploaded somewhere publicly reachable (GitHub raw URL works),
     set as TRAIN_DATA_URL in .env

Usage:
    python src/train.py                # start training
    python src/train.py --status ID    # check an existing training
"""

import argparse
import sys
import time

import replicate

from config import BASE_MODEL_VERSION, DESTINATION_MODEL, TRAIN_DATA_URL, TRAIN_PARAMS


def start_training() -> None:
    if not TRAIN_DATA_URL:
        sys.exit("TRAIN_DATA_URL is not set. Upload data/train.jsonl to a public "
                 "URL (e.g. a GitHub gist raw link) and set it in .env")

    print(f"Base model:  {BASE_MODEL_VERSION}")
    print(f"Destination: {DESTINATION_MODEL}")
    print(f"Train data:  {TRAIN_DATA_URL}")
    print(f"Params:      {TRAIN_PARAMS}\n")

    training = replicate.trainings.create(
        version=BASE_MODEL_VERSION,
        input={"train_data": TRAIN_DATA_URL, **TRAIN_PARAMS},
        destination=DESTINATION_MODEL,
    )
    print(f"Training started: {training.id}")
    print(f"Monitor at: https://replicate.com/trainings\n")
    monitor(training.id)


def monitor(training_id: str) -> None:
    terminal = {"succeeded", "failed", "canceled"}
    while True:
        training = replicate.trainings.get(training_id)
        print(f"[{time.strftime('%H:%M:%S')}] status: {training.status}")
        if training.status in terminal:
            break
        time.sleep(30)

    if training.status == "succeeded":
        print("\nTraining complete.")
        print(f"Fine-tuned version: {training.output.get('version') if training.output else DESTINATION_MODEL}")
        print("Set FINETUNED_MODEL in .env to this version string, e.g.:")
        print(f'  FINETUNED_MODEL="{DESTINATION_MODEL}:<version-hash>"')
    else:
        print(f"\nTraining ended with status: {training.status}")
        if getattr(training, "error", None):
            print(f"Error: {training.error}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", help="training ID to monitor instead of starting new")
    args = parser.parse_args()
    if args.status:
        monitor(args.status)
    else:
        start_training()
