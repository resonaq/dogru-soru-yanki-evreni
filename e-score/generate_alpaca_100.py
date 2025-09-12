# generate_alpaca_100.py  — Alpaca’dan yaratıcı/yorum gerektiren 100 prompt + model çıktısı
from datasets import load_dataset
import pandas as pd, re, random
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# === 1) Alpaca datasetini yükle
ds = load_dataset("yahma/alpaca-cleaned", split="train")

# === 2) Heuristik filtre: yaratıcı / açık uçlu / yorum isteyen promptlar
PATTERN = re.compile(
    r"\b(write|imagine|story|poem|dialogue|monologue|metaphor|perspective|"
    r"opinion|argue|discuss|debate|explain why|roleplay|narrative|diary)\b",
    flags=re.I
)

creative_idxs = [i for i, ex in enumerate(ds) if PATTERN.search(ex.get("instruction") or "")]
random.seed(42)
random.shuffle(creative_idxs)

target_n = 100
picked = creative_idxs[:target_n]

# Yeterli yaratıcı örnek yoksa rastgele tamamla (tekrarları önle)
if len(picked) < target_n:
    all_idxs = list(range(len(ds)))
    random.shuffle(all_idxs)
    for ix in all_idxs:
        if ix not in picked:
            picked.append(ix)
        if len(picked) == target_n:
            break

sample = ds.select(picked)

# === 3) Küçük bir modelle generation (hız için distilgpt2 önerilir)
model_id = "distilgpt2"   # istersen "gpt2" yapabilirsin
tok = AutoTokenizer.from_pretrained(model_id)
mdl = AutoModelForCausalLM.from_pretrained(model_id)
mdl.eval()

def generate_text(prompt, max_new_tokens=80, temperature=0.8, top_p=0.9):
    enc = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = mdl.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p
        )
    return tok.decode(out[0], skip_special_tokens=True)

# === 4) prompt / reference / generation sütunlarını oluştur
rows = []
for ex in sample:
    # instruction + varsa input → prompt
    instruction = ex.get("instruction") or ""
    inp = ex.get("input") or ""
    prompt = instruction + ((" " + inp) if inp else "")
    reference = ex.get("output") or ""
    generation = generate_text(prompt)
    rows.append({"prompt": prompt, "reference": reference, "generation": generation})

df = pd.DataFrame(rows)
out_path = "alpaca_creative_100_with_generations.csv"
df.to_csv(out_path, index=False)
print(f"Saved: {out_path}  ({len(df)} rows)")
