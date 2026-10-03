"""
generate_deliverables.py
Run this LOCALLY (where GPT-2 can be downloaded) from inside the lab folder
that contains creative_prompts.txt, technical_prompts.txt, support_prompts.txt
and model_setup.py:

    python generate_deliverables.py

It creates, with real GPT-2 output:
    submission/parameter_comparison.csv     (Deliverable 2)
    submission/beam_search_comparison.txt   (Deliverable 3)
"""
import csv
import os
import re

from model_setup import generate_text, generate_text_beam, tokenizer

os.makedirs("submission", exist_ok=True)


def load_prompts(path):
    """Read prompts; strips RTF markup in case the file was saved as RTF."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    if raw.lstrip().startswith("{\\rtf"):
        lines = []
        for ln in raw.splitlines():
            if ln.startswith("\\f0"):
                ln = re.sub(r"^\\f0\\fs24\s*(\\cf2.*?kerning0)?\s*", "", ln)
            if ln.startswith("{") or ln.startswith("\\") or not ln.strip():
                continue
            lines.append(ln.strip())
        return lines
    return [ln.strip() for ln in raw.splitlines() if ln.strip()]


creative = load_prompts("creative_prompts.txt")
technical = load_prompts("technical_prompts.txt")
support = load_prompts("support_prompts.txt")

# ---------------------------------------------------------------- Deliverable 2
# Five clearly different configurations (temperature / top_k / top_p / beam).
configs = [
    ("CFG-1", dict(temperature=0.3, top_k=0,  top_p=0.90, beam=1)),   # focused
    ("CFG-2", dict(temperature=0.7, top_k=50, top_p=1.00, beam=1)),   # balanced top-k
    ("CFG-3", dict(temperature=0.9, top_k=0,  top_p=0.95, beam=1)),   # creative nucleus
    ("CFG-4", dict(temperature=1.2, top_k=0,  top_p=0.85, beam=1)),   # high-temp, tight nucleus
    ("CFG-5", dict(temperature=1.0, top_k=0,  top_p=1.00, beam=5)),   # beam search (sampling params ignored)
]
cases = [("Creative Blog", creative[0]),
         ("Technical Summary", technical[0]),
         ("Customer Support", support[0])]

rows = []
for cid, c in configs:
    for use_case, prompt in cases:
        if c["beam"] > 1:
            text = generate_text_beam(prompt, num_beams=c["beam"], max_length=150)
        else:
            text = generate_text(prompt, temperature=c["temperature"], top_k=c["top_k"],
                                 top_p=c["top_p"], max_length=150, seed=42)
        assert len(tokenizer.encode(text)) > 50, "output too short"
        rows.append(dict(configuration_id=cid, temperature=c["temperature"], top_k=c["top_k"],
                         top_p=c["top_p"], beam=c["beam"], use_case=use_case,
                         prompt=prompt, output_sample=text.replace("\n", " ").strip()))

with open("submission/parameter_comparison.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["configuration_id", "temperature", "top_k", "top_p",
                                      "beam", "use_case", "prompt", "output_sample"])
    w.writeheader()
    w.writerows(rows)
print("Saved submission/parameter_comparison.csv")

# ---------------------------------------------------------------- Deliverable 3
prompt = technical[0]
out3 = generate_text_beam(prompt, num_beams=3, max_length=150)
out5 = generate_text_beam(prompt, num_beams=5, max_length=150)

with open("submission/beam_search_comparison.txt", "w", encoding="utf-8") as f:
    f.write("BEAM SEARCH COMPARISON (GPT-2, deterministic decoding)\n")
    f.write("=" * 70 + "\n")
    f.write(f"Prompt (identical for both runs): {prompt}\n\n")
    f.write("OUTPUT A - beam=3\n" + "-" * 70 + "\n" + out3 + "\n\n")
    f.write("OUTPUT B - beam=5\n" + "-" * 70 + "\n" + out5 + "\n\n")
    f.write("ANALYSIS NOTES\n" + "-" * 70 + "\n")
    f.write("""[EDIT after reading your two outputs above - describe what you actually see.]

Guide / talking points:
1. How the outputs differ: compare fluency, repetition, and level of detail
   between beam=3 and beam=5. Wider beams keep more candidate sequences, so
   they tend to find higher-probability (often smoother, more generic) text.
2. Determinism: re-running gives the same text for the same prompt and beam
   width. Sampling (temperature/top-k/top-p) changes on every run.
3. Beam search vs sampling: beam search favours the most probable wording, so
   it is safe and consistent but can be bland or repetitive. Sampling adds
   variety and a more human voice but can drift or invent facts at high
   temperature.
4. Advantages for specific use cases: technical documentation, product specs,
   and templated support replies benefit from reproducible, low-risk output;
   creative blog intros are usually better served by sampling.
""")
print("Saved submission/beam_search_comparison.txt")
