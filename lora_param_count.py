"""Exact LoRA parameter arithmetic for Llama-2-7B (hidden 4096, 32 layers, q_proj + v_proj).
No GPU or download needed. Each adapted Linear(in,out) adds r*(in+out) params."""
BASE = 6_738_415_616
HID, LAYERS, MODULES = 4096, 32, 2
for r in (4, 8, 16, 32):
    t = r * (HID + HID) * MODULES * LAYERS
    allp = BASE + t
    print(f"r={r:>2}: trainable={t:>11,}  all={allp:,}  trainable%={100*t/allp:.4f}")
full, lora = HID * HID, 8 * (HID + HID)
print(f"\nOne 4096x4096 matrix: {full:,} params vs LoRA r=8: {lora:,} -> {full/lora:.0f}x smaller")
