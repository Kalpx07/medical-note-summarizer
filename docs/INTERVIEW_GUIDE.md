# Interview Guide — How to Talk About This Project

## The 60-second pitch (memorize the shape, not the words)

> "I fine-tuned Llama 2 7B to convert free-text clinical notes into structured summaries with a fixed schema — diagnosis, key findings, medications, action items, follow-up. I generated a synthetic dataset of 600 SOAP-style notes to avoid any PHI issues, formatted it as prompt/completion pairs, and used Replicate's managed LoRA training so I could focus my effort on data quality and evaluation rather than GPU management. The most interesting part was evaluation: standard ROUGE scores don't catch clinical failure modes, so I added two custom metrics — structure compliance, which checks the output actually follows the required schema, and number fidelity, which flags any number in the summary that doesn't appear in the source note, a cheap but effective hallucination detector for invented dosages and vitals. The base model followed the schema [X]% of the time; after fine-tuning it was [Y]%."

## Deep-dive questions and strong answers

**Q: Why fine-tuning instead of just prompting?**
> Three reasons for this task. First, format reliability — the base model follows the 5-section schema inconsistently even with careful prompting, and downstream systems parsing the output need 100% schema adherence. Second, cost and latency at scale — a fine-tuned 7B model with a short prompt beats stuffing few-shot examples into every request. Third, I wanted to demonstrate the full training loop. That said, I'd always prototype with prompting first and only fine-tune once prompting plateaus — I measured that plateau, which is what the base-model column in my eval report shows.

**Q: Why LoRA and not full fine-tuning?**
> LoRA trains small low-rank adapter matrices instead of all 7B parameters — roughly 0.1–1% of the weights. For a narrow task like format adherence and style, that's more than enough capacity, it's 10–100x cheaper, trains in minutes instead of hours, and it's far less prone to catastrophic forgetting of general language ability.

**Q: Why Replicate instead of training locally?**
> My machine is an M-series MacBook Air — technically it can run QLoRA on a 7B model, but slowly and with thermal throttling. Replicate gives managed A40 GPUs, so training took under an hour for a few dollars. The engineering judgment I'd defend: the differentiated work in this project is data design and evaluation, not GPU orchestration. In a company setting, I'd make the same build-vs-buy call at this scale.

**Q: How did you build the dataset? Why synthetic?**
> Real clinical notes are PHI — using them would need de-identification and IRB-level care that's inappropriate for a portfolio project. So I wrote a generator with 10 condition templates (diabetes, pneumonia, depression, etc.), each with realistic symptom, vital, lab, medication, and plan pools, randomly composed into SOAP-style notes. The summary is generated from the same structured source, so references are guaranteed faithful. The limitation I'd volunteer: synthetic data has less linguistic diversity than real notes, so real-world performance would be lower — the honest next step is evaluating on a de-identified public corpus like MIMIC (with proper credentialed access).

**Q: Walk me through your evaluation.**
> Held-out test set, never seen in training. Four metric families: ROUGE-1/2/L for content overlap, BLEU for precision-oriented overlap, then two custom metrics. Structure compliance — the fraction of the 5 required section headers present — because that's the actual product requirement. And number fidelity — the fraction of numbers in the output that exist in the source note. In clinical text, hallucinated numbers (a wrong dosage, an invented blood pressure) are the highest-severity failure, and ROUGE can score well while still containing one. I run both base and fine-tuned models over the same test set and diff every metric.

**Q: What were the failure modes / what surprised you?**
> Base Llama 2 chat mostly fails by over-talking — it adds preambles like "Sure, here's a summary" and drops sections, which kills structure compliance. The fine-tuned model locks onto the format almost immediately. What surprised me is how few examples format adherence needs — the model learns the schema long before it maximizes content quality. Also, the leading space on completions matters for tokenization — a small detail from Replicate's docs that affects training quality.

**Q: How would you productionize this?**
> Four things. One: input validation and PHI scrubbing before anything hits the model. Two: guardrails on output — reject or retry any summary failing structure compliance or number fidelity below a threshold; that's cheap because my metrics are pure functions. Three: monitoring — log latency, cost per summary, metric distributions over time to catch drift. Four: human-in-the-loop review for low-confidence outputs, with clinician feedback flowing back into the training set. And for anything real-world clinical: regulatory review — this class of tool can be a medical device depending on claims.

**Q: What would you do differently / next?**
> Evaluate on real de-identified data; add an LLM-as-judge evaluation for faithfulness beyond number matching; try a modern base (Llama 3, Mistral) and compare; and run a prompting-vs-fine-tuning cost/quality frontier analysis to make the fine-tuning decision quantitative rather than assumed.

## Trap questions — don't get caught

- **"What was your ROUGE score?"** — Know your exact numbers from `results/eval_report.json`. Saying "I don't remember" undoes the whole project.
- **"Is this safe for real patients?"** — No, and say so immediately: synthetic data, no clinical validation, educational purpose. Owning the limitation reads as maturity.
- **"Did you really train a model, or call an API?"** — Be precise: "Replicate runs managed LoRA training — I own the data pipeline, hyperparameters, and evaluation; they own the GPUs. Same division of labor as using SageMaker or Vertex at a company."
- **"600 examples — isn't that tiny?"** — For LoRA on a narrow, schema-focused task, it's within the effective range (hundreds to low thousands). The eval shows it works; more data helps content quality, not format adherence.

## One-line answers for rapid-fire rounds

- **LoRA?** Low-rank adapters on frozen weights — fine-tune ~1% of parameters for ~100% of the task gain.
- **Why temperature 0.2 at inference?** Summarization wants determinism and faithfulness, not creativity.
- **Prompt/completion vs chat format?** Replicate's Llama 2 trainer expects prompt/completion JSONL; the instruction lives inside the prompt.
- **Biggest lesson?** Evaluation design is the project. Anyone can call a training API; knowing whether the model got better — and in which failure modes — is the engineering.
