"""
model_setup.py  -  Deliverable 1: Model Setup Code
LLM Decoding Parameters Lab (Module 1)

Text-generation helpers for GPT-2 with adjustable decoding parameters:
temperature, top-k, top-p (sampling) and num_beams (beam search).

Usage:
    python model_setup.py          # runs a quick self-test with sample prompts
or:
    from model_setup import generate_text, generate_text_beam
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# --------------------------------------------------------------------------
# Model / tokenizer / device setup
# --------------------------------------------------------------------------
MODEL_NAME = "gpt2"  # use "distilgpt2" if the download is slow

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# GPT-2 has no pad token; reuse the end-of-sequence token to avoid warnings.
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

# Pick the best available device: Apple Silicon (MPS) > NVIDIA (CUDA) > CPU.
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = model.to(device)
model.eval()


# --------------------------------------------------------------------------
# Sampling-based generation (probabilistic)
# --------------------------------------------------------------------------
def generate_text(prompt, temperature=0.7, top_k=0, top_p=1.0,
                  max_length=150, min_length=50, num_beams=1, seed=None):
    """
    Generate text with adjustable decoding parameters.

    If num_beams == 1 (default) the function uses SAMPLING (do_sample=True).
    If num_beams > 1 it switches to deterministic BEAM SEARCH
    (do_sample=False); temperature/top_k/top_p are then ignored.

    Args:
        prompt (str): Text the model should continue.
        temperature (float): Randomness of token choice.
            Low (0.2-0.5)  -> focused, predictable, factual.
            Mid (0.6-0.8)  -> balanced, natural-sounding.
            High (0.9-1.5) -> creative, varied, risk of incoherence.
            Must be > 0 (values <0.1 become extremely repetitive).
        top_k (int): Keep only the k most likely tokens at each step.
            0 disables top-k. A fixed-size vocabulary cut-off
            (10 = narrow, 50 = balanced, 100 = loose).
        top_p (float): Nucleus sampling. Keep the smallest set of tokens whose
            cumulative probability >= top_p. 1.0 disables it. The set size
            adapts to the model's confidence (0.9 is a good default).
        max_length (int): Maximum total length in tokens (prompt + output).
        min_length (int): Minimum total length in tokens (avoids tiny outputs).
        num_beams (int): Number of beams. 1 = sampling, >1 = beam search.
        seed (int | None): Optional random seed for reproducible sampling.

    Returns:
        str: The prompt followed by the generated continuation.
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0 (use num_beams>1 for deterministic output)")

    if seed is not None:
        torch.manual_seed(seed)

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    gen_kwargs = dict(
        max_length=max_length,
        min_length=min_length,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=2,  # blocks repeated bigrams (reduces loops)
    )

    if num_beams > 1:
        # Deterministic: explores several candidate sequences in parallel.
        gen_kwargs.update(num_beams=num_beams, do_sample=False, early_stopping=True)
    else:
        gen_kwargs.update(
            do_sample=True,                       # required for temp/top-k/top-p
            temperature=temperature,
            top_k=top_k if top_k > 0 else 0,      # 0 disables top-k
            top_p=top_p,
        )

    with torch.no_grad():
        output = model.generate(input_ids, **gen_kwargs)

    return tokenizer.decode(output[0], skip_special_tokens=True)


# --------------------------------------------------------------------------
# Beam search generation (deterministic)
# --------------------------------------------------------------------------
def generate_text_beam(prompt, num_beams=3, max_length=150, min_length=50):
    """
    Deterministic generation with beam search.

    Args:
        prompt (str): Text the model should continue.
        num_beams (int): Number of beams (3 and 5 are used in the lab).
            Larger values search more candidate sequences, usually giving
            more coherent text at the cost of speed and variety.
        max_length (int): Maximum total length in tokens.
        min_length (int): Minimum total length in tokens.

    Returns:
        str: The prompt followed by the generated continuation. The same input
             always gives the same output.
    """
    return generate_text(prompt, num_beams=max(2, num_beams),
                         max_length=max_length, min_length=min_length)


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Model: {MODEL_NAME} | Device: {device}\n")

    tests = [
        ("Sampling, low temp",  "This cloud-based data analytics platform offers real-time processing capabilities with",
         dict(temperature=0.3, top_p=0.9)),
        ("Sampling, top-k",     "I'm having trouble logging into my account.",
         dict(temperature=0.7, top_k=50)),
        ("Sampling, high temp", "Picture this: You wake up one morning to discover that time travel has just been invented. The first thing you'd want to do is",
         dict(temperature=0.9, top_p=0.95)),
        ("Beam search (3)",     "The future of artificial intelligence in healthcare is",
         dict(num_beams=3)),
    ]
    for name, prompt, params in tests:
        out = generate_text(prompt, seed=42, **params)
        print(f"--- {name} {params}\n{out}\n")

    print("Self-test finished without errors.")
