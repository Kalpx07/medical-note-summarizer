"""
Synthetic clinical note dataset generator for fine-tuning.

Generates realistic (but fully synthetic) SOAP-style clinical notes paired with
structured summaries. Output format matches Replicate's Llama 2 fine-tuning
spec: JSONL rows of {"prompt": ..., "completion": ...}.

Usage:
    python data/generate_dataset.py --n 600 --out data/
"""

import argparse
import json
import random
from pathlib import Path

random.seed(42)

# ---------------------------------------------------------------------------
# Building blocks for synthetic notes
# ---------------------------------------------------------------------------

CONDITIONS = [
    {
        "name": "type 2 diabetes mellitus",
        "cc": "follow-up of blood sugar control",
        "symptoms": ["increased thirst", "fatigue", "blurred vision", "frequent urination"],
        "vitals": {"bp": (128, 145, 78, 92), "hr": (72, 96)},
        "labs": ["HbA1c 8.2%", "fasting glucose 162 mg/dL", "HbA1c 7.1%", "fasting glucose 138 mg/dL"],
        "meds": ["metformin 1000 mg BID", "glipizide 5 mg daily", "empagliflozin 10 mg daily"],
        "plan": [
            "increase metformin to 1000 mg BID",
            "add empagliflozin 10 mg daily",
            "repeat HbA1c in 3 months",
            "referral to diabetes educator",
            "reinforce dietary counseling and exercise",
        ],
        "followup": ["3 months", "6 weeks"],
    },
    {
        "name": "hypertension",
        "cc": "elevated home blood pressure readings",
        "symptoms": ["occasional headaches", "no chest pain", "no visual changes", "mild dizziness"],
        "vitals": {"bp": (148, 168, 88, 102), "hr": (68, 88)},
        "labs": ["basic metabolic panel within normal limits", "creatinine 1.0 mg/dL", "potassium 4.1 mEq/L"],
        "meds": ["lisinopril 10 mg daily", "amlodipine 5 mg daily", "hydrochlorothiazide 25 mg daily"],
        "plan": [
            "increase lisinopril to 20 mg daily",
            "add amlodipine 5 mg daily",
            "home BP log twice daily",
            "low-sodium DASH diet counseling",
            "recheck BP in 4 weeks",
        ],
        "followup": ["4 weeks", "2 months"],
    },
    {
        "name": "community-acquired pneumonia",
        "cc": "productive cough and fever for 4 days",
        "symptoms": ["productive cough with yellow sputum", "fever up to 101.8F", "pleuritic chest pain", "shortness of breath on exertion"],
        "vitals": {"bp": (110, 132, 68, 84), "hr": (92, 118)},
        "labs": ["WBC 14.2", "chest X-ray shows right lower lobe infiltrate", "SpO2 94% on room air"],
        "meds": ["azithromycin 500 mg day 1 then 250 mg daily", "amoxicillin-clavulanate 875 mg BID"],
        "plan": [
            "start azithromycin 500 mg day 1 then 250 mg x4 days",
            "supportive care with fluids and antipyretics",
            "return precautions for worsening dyspnea",
            "repeat chest X-ray in 6 weeks",
        ],
        "followup": ["1 week", "6 weeks for repeat imaging"],
    },
    {
        "name": "major depressive disorder",
        "cc": "low mood and poor sleep for 2 months",
        "symptoms": ["depressed mood most days", "early morning awakening", "decreased appetite", "poor concentration at work", "denies suicidal ideation"],
        "vitals": {"bp": (112, 128, 70, 84), "hr": (64, 84)},
        "labs": ["PHQ-9 score 16 (moderately severe)", "TSH within normal limits"],
        "meds": ["sertraline 50 mg daily", "escitalopram 10 mg daily"],
        "plan": [
            "start sertraline 50 mg daily",
            "referral to cognitive behavioral therapy",
            "sleep hygiene counseling",
            "follow up in 4 weeks to assess response",
            "safety plan reviewed",
        ],
        "followup": ["4 weeks", "2 weeks"],
    },
    {
        "name": "asthma exacerbation",
        "cc": "worsening wheeze and nighttime cough",
        "symptoms": ["wheezing", "nighttime cough 3-4x per week", "using rescue inhaler daily", "chest tightness with exercise"],
        "vitals": {"bp": (108, 126, 66, 82), "hr": (76, 102)},
        "labs": ["peak flow 68% of personal best", "SpO2 96% on room air", "lungs with scattered expiratory wheezes"],
        "meds": ["albuterol inhaler PRN", "fluticasone-salmeterol 250/50 BID", "montelukast 10 mg nightly"],
        "plan": [
            "step up to fluticasone-salmeterol 250/50 BID",
            "prednisone 40 mg daily x5 days",
            "review inhaler technique",
            "written asthma action plan provided",
            "follow up in 2 weeks",
        ],
        "followup": ["2 weeks", "1 month"],
    },
    {
        "name": "gastroesophageal reflux disease",
        "cc": "burning chest discomfort after meals",
        "symptoms": ["postprandial burning sensation", "sour taste in mouth", "symptoms worse when lying flat", "no dysphagia or weight loss"],
        "vitals": {"bp": (118, 136, 72, 86), "hr": (66, 88)},
        "labs": ["no alarm features on review", "ECG normal sinus rhythm"],
        "meds": ["omeprazole 20 mg daily", "famotidine 20 mg BID"],
        "plan": [
            "start omeprazole 20 mg daily x8 weeks",
            "avoid late meals, caffeine, and alcohol",
            "elevate head of bed",
            "return if dysphagia, weight loss, or bleeding",
        ],
        "followup": ["8 weeks", "6 weeks"],
    },
    {
        "name": "urinary tract infection",
        "cc": "dysuria and urinary frequency for 2 days",
        "symptoms": ["burning with urination", "urinary frequency", "suprapubic discomfort", "no fever or flank pain"],
        "vitals": {"bp": (110, 130, 68, 84), "hr": (70, 92)},
        "labs": ["urinalysis positive for leukocyte esterase and nitrites", "urine culture pending"],
        "meds": ["nitrofurantoin 100 mg BID", "trimethoprim-sulfamethoxazole DS BID"],
        "plan": [
            "nitrofurantoin 100 mg BID x5 days",
            "increase oral fluid intake",
            "follow up culture results",
            "return if fever or flank pain develops",
        ],
        "followup": ["48-72 hours for culture results", "1 week"],
    },
    {
        "name": "osteoarthritis of the knee",
        "cc": "chronic right knee pain worse with stairs",
        "symptoms": ["aching right knee pain", "morning stiffness lasting 15 minutes", "pain worse with stairs and prolonged walking", "no locking or giving way"],
        "vitals": {"bp": (122, 142, 74, 90), "hr": (64, 86)},
        "labs": ["X-ray shows medial joint space narrowing and osteophytes", "no effusion on exam"],
        "meds": ["acetaminophen 650 mg TID PRN", "topical diclofenac gel", "naproxen 500 mg BID PRN"],
        "plan": [
            "topical diclofenac gel to right knee",
            "referral to physical therapy for quadriceps strengthening",
            "weight management counseling",
            "consider intra-articular steroid injection if no improvement",
        ],
        "followup": ["6 weeks", "3 months"],
    },
    {
        "name": "hypothyroidism",
        "cc": "fatigue and cold intolerance",
        "symptoms": ["persistent fatigue", "cold intolerance", "mild weight gain", "dry skin", "constipation"],
        "vitals": {"bp": (108, 128, 66, 84), "hr": (54, 72)},
        "labs": ["TSH 9.8 mIU/L", "free T4 0.7 ng/dL", "TSH 6.2 mIU/L"],
        "meds": ["levothyroxine 50 mcg daily", "levothyroxine 75 mcg daily"],
        "plan": [
            "start levothyroxine 50 mcg daily on empty stomach",
            "repeat TSH in 6-8 weeks",
            "counseled on medication timing away from calcium and iron",
        ],
        "followup": ["6-8 weeks", "8 weeks"],
    },
    {
        "name": "migraine without aura",
        "cc": "recurrent throbbing headaches",
        "symptoms": ["unilateral throbbing headache", "photophobia and phonophobia", "nausea during episodes", "episodes 4-6x per month lasting 6-12 hours"],
        "vitals": {"bp": (114, 134, 70, 88), "hr": (66, 90)},
        "labs": ["neurological exam non-focal", "no red flag features"],
        "meds": ["sumatriptan 50 mg PRN", "propranolol 40 mg BID", "topiramate 25 mg nightly"],
        "plan": [
            "sumatriptan 50 mg at headache onset",
            "start propranolol 40 mg BID for prophylaxis",
            "headache diary",
            "identify and avoid triggers",
            "follow up in 6 weeks",
        ],
        "followup": ["6 weeks", "2 months"],
    },
]

FIRST_LINE = [
    "Patient is a {age}-year-old {sex} presenting for {cc}.",
    "{age} y/o {sex} here today for {cc}.",
    "The patient, a {age}-year-old {sex}, presents with {cc}.",
]

PMH_POOL = [
    "hyperlipidemia", "obesity", "seasonal allergies", "GERD",
    "anxiety", "prediabetes", "chronic low back pain", "vitamin D deficiency",
]

SOCIAL = [
    "Denies tobacco use, drinks alcohol socially.",
    "Former smoker, quit 5 years ago. No alcohol.",
    "Never smoker. Occasional alcohol use.",
    "Works a desk job, exercises 1-2x weekly.",
]

PROMPT_INSTRUCTION = (
    "You are a clinical documentation assistant. Summarize the following "
    "clinical note into a structured summary with exactly these sections: "
    "DIAGNOSIS, KEY FINDINGS, MEDICATIONS, ACTION ITEMS, FOLLOW-UP. "
    "Be concise and factual; do not add information not present in the note.\n\n"
    "CLINICAL NOTE:\n{note}\n\nSTRUCTURED SUMMARY:"
)


def build_note_and_summary(cond: dict) -> tuple[str, str]:
    age = random.randint(24, 78)
    sex = random.choice(["male", "female"])
    bp_lo_s, bp_hi_s, bp_lo_d, bp_hi_d = cond["vitals"]["bp"]
    bp = f"{random.randint(bp_lo_s, bp_hi_s)}/{random.randint(bp_lo_d, bp_hi_d)}"
    hr = random.randint(*cond["vitals"]["hr"])
    symptoms = random.sample(cond["symptoms"], k=min(3, len(cond["symptoms"])))
    labs = random.sample(cond["labs"], k=min(2, len(cond["labs"])))
    med = random.choice(cond["meds"])
    plan_items = random.sample(cond["plan"], k=min(3, len(cond["plan"])))
    followup = random.choice(cond["followup"])
    pmh = random.sample(PMH_POOL, k=2)

    note_lines = [
        random.choice(FIRST_LINE).format(age=age, sex=sex, cc=cond["cc"]),
        f"HPI: Reports {symptoms[0]} and {symptoms[1]}. Also notes {symptoms[2]}."
        if len(symptoms) >= 3 else f"HPI: Reports {', '.join(symptoms)}.",
        f"PMH: {pmh[0].capitalize()}, {pmh[1]}.",
        f"Social history: {random.choice(SOCIAL)}",
        f"Vitals: BP {bp}, HR {hr}, afebrile unless noted.",
        f"Objective/Results: {labs[0].capitalize()}. {labs[1].capitalize()}." if len(labs) >= 2
        else f"Objective/Results: {labs[0].capitalize()}.",
        f"Current medications include {med}.",
        f"Assessment: Findings consistent with {cond['name']}.",
        "Plan: " + "; ".join(plan_items) + f". Follow up in {followup}.",
    ]
    note = "\n".join(note_lines)

    summary = (
        f"DIAGNOSIS: {cond['name'].capitalize()}\n"
        f"KEY FINDINGS: {symptoms[0].capitalize()}; {symptoms[1]}; "
        f"{labs[0]}; BP {bp}, HR {hr}\n"
        f"MEDICATIONS: {med.capitalize()}\n"
        f"ACTION ITEMS: " + "; ".join(p.capitalize() for p in plan_items) + "\n"
        f"FOLLOW-UP: {followup.capitalize()}"
    )
    return note, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=600, help="total examples")
    parser.add_argument("--out", type=str, default="data/")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(args.n):
        cond = CONDITIONS[i % len(CONDITIONS)]
        note, summary = build_note_and_summary(cond)
        rows.append({
            "prompt": PROMPT_INSTRUCTION.format(note=note),
            "completion": " " + summary,  # leading space helps tokenization
        })

    random.shuffle(rows)
    n_train = int(len(rows) * 0.85)
    n_val = int(len(rows) * 0.075)
    splits = {
        "train.jsonl": rows[:n_train],
        "val.jsonl": rows[n_train:n_train + n_val],
        "test.jsonl": rows[n_train + n_val:],
    }
    for fname, split_rows in splits.items():
        with open(out_dir / fname, "w") as f:
            for r in split_rows:
                f.write(json.dumps(r) + "\n")
        print(f"Wrote {len(split_rows):>4} examples -> {out_dir / fname}")


if __name__ == "__main__":
    main()
