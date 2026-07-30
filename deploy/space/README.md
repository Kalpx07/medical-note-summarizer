---
title: Medical Note Summarizer
emoji: 🩺
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8501
pinned: false
license: llama2
---

# 🩺 Medical Note Summarizer

Llama 2 7B fine-tuned with LoRA to convert free-text clinical notes into structured
summaries (DIAGNOSIS, KEY FINDINGS, MEDICATIONS, ACTION ITEMS, FOLLOW-UP).

Runs the fused Q4_K_M GGUF with llama.cpp on free CPU hardware — generation takes
1-2 minutes per summary.

> ⚠️ Educational demo trained on fully synthetic data. Not for real patient data
> or clinical decisions.

Fine-tuned vs base evaluation on 25 held-out synthetic notes: ROUGE-1 0.944 vs 0.710,
BLEU 0.920 vs 0.351, structure compliance 1.000 vs 0.976.
