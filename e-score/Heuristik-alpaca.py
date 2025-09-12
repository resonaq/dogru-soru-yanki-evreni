# generate_alpaca_100.py  (yaratıcı/yorum gerektiren prompt filtreli)
from datasets import load_dataset
import pandas as pd, re, random
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# === 1) Alpaca yükle
ds = load_dataset("yahma/alpaca-cleaned", split="train")

# === 2) Heuristik: yaratıcı/yorum gerektiren promptları seç
PATTERN = re.compile(
    r"\b(write|imagine|story|poem|dialogue|monologue|metaphor|perspective|"
    r"opinion|argue|discuss|debate|explain why|explain.*perspective)\b",
    flags=re.I
)

creative_idxs = [i for i, ex in enumerate(ds) if PATTERN.search(ex["instruction"] or "")]
random.seed(42)
random.shuffle(creative_idxs)

target_n = 100
picked = creative_idxs[:target_n]
if len(picked) < target_n:
    # yaratıcı yoksa rastgele tamamla (tekrarları önle)
    all_idxs = list(range(len(ds)))
    random.shuffle(all_idxs)
    for ix in all_idxs:
        if ix not in picked:
            picked.append(ix)
        if len(picked) == target_n:
            break

sample = ds.select(picked)

# === 3) Model (küçük modelle başla)
model_id = "gpt2"  # daha hızlı istersen: "distilgpt2"
tok = AutoTokenizer.from_pretrained(model_id)
mdl = AutoModelForCausalLM.from_pretrained(model_id)
mdl.eval()

def generate_text(prompt, max_new_tokens=60, temperature=0.8, top_p=0.9):
    enc = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = mdl.generate(
            **enc, max_new_tokens=max_new_tokens,
            do_sample=True, temperature=temperature, top_p=top_p
        )
    return tok.decode(out[0], skip_special_tokens=True)

# === 4) prompt/reference/generation oluştur
rows = []
for ex in sample:
    prompt = ex["instruction"] + (" " + ex["input"] if ex["input"] else "")
    reference = ex["output"]
    generation = generate_text(prompt)
    rows.append({"prompt": prompt, "reference": reference, "generation": generation})

df = pd.DataFrame(rows)
df.to_csv("alpaca_creative_100_with_generations.csv", index=False)
print("Saved: alpaca_creative_100_with_generations.csv (", len(df), "rows )")
