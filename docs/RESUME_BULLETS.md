# Resume Bullets — Medical Note Summarizer

## Project title line

**Clinical Note Summarizer — Fine-tuned Llama 2 7B** | Python, Replicate, LoRA, Streamlit, Hugging Face
*GitHub: <link> | Demo: <link> | HF: <link>*

## Bullets (pick 3–4; replace bracketed numbers with YOUR actual eval results)

**Core version:**

- Fine-tuned Llama 2 7B using LoRA (Replicate managed training) on 600 synthetic clinical notes to generate structured summaries with a fixed 5-section schema, improving output structure compliance from [X]% (base model) to [Y]%
- Designed an evaluation framework combining ROUGE-1/2/L and BLEU with two custom clinical metrics — structure compliance and number fidelity (a hallucination check for invented dosages/vitals) — achieving [Z] ROUGE-L vs [W] for the base model on a held-out test set
- Built an end-to-end MLOps pipeline: synthetic data generation with validation, prompt/completion formatting, cloud training orchestration with status monitoring, and automated base-vs-fine-tuned benchmarking
- Deployed a Streamlit app for side-by-side base/fine-tuned comparison with live quality metrics, and published the dataset and model card to Hugging Face Hub

**Shorter version (if space is tight):**

- Fine-tuned Llama 2 7B (LoRA via Replicate) for clinical note summarization; lifted structure compliance [X]%→[Y]% and ROUGE-L by [N] points over the base model, measured via a custom eval framework with hallucination detection
- Shipped the full pipeline — synthetic dataset generation, training orchestration, evaluation, Streamlit demo, and Hugging Face publishing — in modular, production-style Python

## Rules for filling in the numbers

1. **Run the eval first** (`python src/evaluate.py --mode live --n 25`), then copy real numbers from `results/eval_report.json`. Never invent metrics — interviewers drill into them.
2. Expect roughly: base model structure compliance 20–50%, fine-tuned 90–100%. That delta is your headline.
3. Express ROUGE as "0.62 ROUGE-L" or "improved ROUGE-L from 0.31 to 0.62" — both are fine.
4. If a number is unimpressive, lead with the one that is (structure compliance almost always is).

## Skills section additions

`LLM Fine-tuning (LoRA)` · `Llama 2` · `Replicate` · `Model Evaluation (ROUGE/BLEU)` · `Hallucination Detection` · `Streamlit` · `Hugging Face Hub`

## LinkedIn post template (for visibility)

> Shipped my latest AI engineering project: fine-tuning Llama 2 7B to turn messy clinical notes into structured summaries. 🩺
>
> The interesting part wasn't the training — it was the evaluation. Generic metrics like ROUGE won't tell you if your model invented a medication dosage, so I built a "number fidelity" check that flags any number in the summary that doesn't exist in the source note.
>
> Results: structure compliance went from [X]% (base Llama 2) to [Y]% after fine-tuning on just 600 examples.
>
> Full pipeline on GitHub: data generation → LoRA fine-tuning on Replicate → eval framework → Streamlit demo → Hugging Face.
> [link]
