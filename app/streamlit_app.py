"""
Streamlit UI: paste a clinical note, get structured summaries from the base
Llama 2 model and the fine-tuned model side by side, with live quality checks.

Run:
    streamlit run app/streamlit_app.py
"""

import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import streamlit as st

from evaluate import structure_compliance, number_fidelity
from inference import summarize

st.set_page_config(page_title="Medical Note Summarizer", page_icon="🩺", layout="wide")

# ---------------------------------------------------------------- styling

st.markdown("""
<style>
/* ---- page ---- */
.stApp {
    background: linear-gradient(180deg, #f0f9ff 0%, #ffffff 340px);
}
.block-container { padding-top: 3.5rem; max-width: 1200px; }
#MainMenu, footer { visibility: hidden; }

/* ---- hero ---- */
.med-hero {
    background: linear-gradient(135deg, #0284c7 0%, #0ea5e9 60%, #38bdf8 100%);
    border-radius: 18px;
    padding: 2.2rem 2.4rem 2rem;
    color: #ffffff;
    margin-top: .8rem;
    margin-bottom: 1.6rem;
    box-shadow: 0 8px 24px rgba(2, 132, 199, .25);
}
.med-hero h1 {
    color: #ffffff; font-size: 2rem; font-weight: 700;
    margin: 0 0 .35rem; padding: 0; letter-spacing: -.02em;
}
.med-hero p { color: #e0f2fe; font-size: 1.02rem; margin: 0; }
.med-badges { margin-top: 1.2rem; display: flex; flex-wrap: wrap; gap: .7rem; }
.med-badge {
    display: inline-block; background: rgba(255,255,255,.16);
    border: 1px solid rgba(255,255,255,.35); border-radius: 999px;
    padding: .35rem 1rem; font-size: .78rem; font-weight: 600;
    color: #ffffff;
}

/* ---- cards ---- */
.med-card {
    background: #ffffff; border: 1px solid #bae6fd; border-radius: 14px;
    padding: 1.1rem 1.3rem; margin-bottom: .8rem;
    box-shadow: 0 2px 8px rgba(14, 165, 233, .07);
}
.med-card-title {
    color: #0369a1; font-size: .8rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: .08em; margin-bottom: .3rem;
}
.med-card-body { color: #0f172a; font-size: .95rem; line-height: 1.55; }

/* ---- model column headers ---- */
.med-model-head {
    display: flex; align-items: center; gap: .5rem;
    font-size: 1.12rem; font-weight: 700; color: #0f172a;
    padding-bottom: .5rem; border-bottom: 3px solid #0ea5e9;
    margin-bottom: .9rem;
}
.med-model-head.base { border-bottom-color: #94a3b8; }
.med-chip {
    font-size: .7rem; font-weight: 700; border-radius: 999px;
    padding: .18rem .6rem; text-transform: uppercase; letter-spacing: .05em;
}
.med-chip.ft { background: #e0f2fe; color: #0369a1; }
.med-chip.base { background: #f1f5f9; color: #475569; }

/* ---- metric pills ---- */
.med-metric {
    display: inline-block; border-radius: 12px; padding: .55rem 1rem;
    margin: .6rem .6rem 0 0; border: 1px solid #bae6fd; background: #f0f9ff;
}
.med-metric .v { font-size: 1.25rem; font-weight: 700; color: #0284c7; }
.med-metric .l { font-size: .72rem; color: #475569; text-transform: uppercase; letter-spacing: .05em; }

/* ---- disclaimer ---- */
.med-disclaimer {
    background: #fffbeb; border: 1px solid #fde68a; border-radius: 12px;
    padding: .8rem 1.1rem; font-size: .85rem; color: #92400e; margin-top: 1.4rem;
}

/* ---- inputs & buttons ---- */
.stTextArea textarea {
    border: 1.5px solid #bae6fd !important; border-radius: 12px !important;
    font-size: .92rem !important;
}
.stTextArea textarea:focus { border-color: #0ea5e9 !important; box-shadow: 0 0 0 3px rgba(14,165,233,.18) !important; }
.stButton button {
    background: linear-gradient(135deg, #0284c7, #0ea5e9) !important;
    color: #fff !important; border: none !important; border-radius: 10px !important;
    padding: .55rem 1.6rem !important; font-weight: 600 !important;
    box-shadow: 0 4px 12px rgba(2,132,199,.3) !important;
}
.stButton button:hover { filter: brightness(1.07); }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------- header

st.markdown("""
<div class="med-hero">
  <h1>🩺 Medical Note Summarizer</h1>
  <p>Turn free-text clinical notes into clean, structured summaries — powered by
     Llama&nbsp;2&nbsp;7B fine-tuned with LoRA, running fully on-device.</p>
  <div class="med-badges">
    <span class="med-badge">✓ Structured 5-section output</span>
    <span class="med-badge">✓ Hallucination checks</span>
    <span class="med-badge">✓ 100% local &amp; private</span>
  </div>
</div>
""", unsafe_allow_html=True)

SAMPLE_NOTE = """Patient is a 58-year-old male presenting for follow-up of blood sugar control.
HPI: Reports increased thirst and fatigue. Also notes blurred vision.
PMH: Hyperlipidemia, obesity.
Social history: Former smoker, quit 5 years ago. No alcohol.
Vitals: BP 138/86, HR 82, afebrile.
Objective/Results: HbA1c 8.2%. Fasting glucose 162 mg/dL.
Current medications include metformin 1000 mg BID.
Assessment: Findings consistent with type 2 diabetes mellitus.
Plan: add empagliflozin 10 mg daily; repeat HbA1c in 3 months; reinforce dietary counseling. Follow up in 3 months."""

SECTION_ICONS = {
    "DIAGNOSIS": "🔬",
    "KEY FINDINGS": "📋",
    "MEDICATIONS": "💊",
    "ACTION ITEMS": "✅",
    "FOLLOW-UP": "📅",
}
SECTION_RE = re.compile(
    r"^(DIAGNOSIS|KEY FINDINGS|MEDICATIONS|ACTION ITEMS|FOLLOW-UP):\s*(.*)$", re.M)


def render_summary(text: str) -> None:
    """Render each section of a well-formed summary as a card;
    fall back to a plain code block for free-form output."""
    matches = SECTION_RE.findall(text)
    if len(matches) >= 3:
        for name, body in matches:
            icon = SECTION_ICONS.get(name, "•")
            st.markdown(
                f'<div class="med-card">'
                f'<div class="med-card-title">{icon} {name}</div>'
                f'<div class="med-card-body">{body or "—"}</div></div>',
                unsafe_allow_html=True)
    else:
        st.code(text, language=None)


def render_metrics(note: str, output: str) -> None:
    s = structure_compliance(output)
    nf = number_fidelity(note, output)
    st.markdown(
        f'<div class="med-metric"><div class="v">{s:.0%}</div>'
        f'<div class="l">Structure compliance</div></div>'
        f'<div class="med-metric"><div class="v">{nf:.0%}</div>'
        f'<div class="l">Number fidelity</div></div>',
        unsafe_allow_html=True)


# ---------------------------------------------------------------- input

st.markdown("#### 📄 Clinical note")
note = st.text_area("Paste a clinical note", value=SAMPLE_NOTE, height=260,
                    label_visibility="collapsed")

col_run, col_opts = st.columns([1, 3])
with col_run:
    run = st.button("✨ Summarize", type="primary")
with col_opts:
    compare_base = st.checkbox("Compare against base Llama 2 (slower, runs both models)",
                               value=True)

# ---------------------------------------------------------------- results

if run and note.strip():
    cols = st.columns(2 if compare_base else 1, gap="large")

    with cols[0]:
        st.markdown(
            '<div class="med-model-head">Fine-tuned model '
            '<span class="med-chip ft">Ours</span></div>',
            unsafe_allow_html=True)
        with st.spinner("Generating structured summary..."):
            try:
                ft = summarize(note, model="finetuned")
                render_summary(ft)
                render_metrics(note, ft)
            except Exception as e:
                st.error(f"Fine-tuned model error: {e}")

    if compare_base:
        with cols[1]:
            st.markdown(
                '<div class="med-model-head base">Base Llama 2 7B '
                '<span class="med-chip base">Baseline</span></div>',
                unsafe_allow_html=True)
            with st.spinner("Generating baseline summary..."):
                try:
                    base = summarize(note, model="base")
                    render_summary(base)
                    render_metrics(note, base)
                except Exception as e:
                    st.error(f"Base model error: {e}")

# ---------------------------------------------------------------- footer

st.markdown("""
<div class="med-disclaimer">
  ⚠️ <strong>Educational demo only.</strong> Trained on fully synthetic data — do not use
  with real patient information or for clinical decisions.
  <em>Structure compliance</em> = fraction of the 5 required sections present.
  <em>Number fidelity</em> = fraction of numbers in the summary that appear in the source
  note (a hallucination check for invented dosages and vitals).
</div>
""", unsafe_allow_html=True)
