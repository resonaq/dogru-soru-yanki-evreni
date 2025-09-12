#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
E-Score + Uncertainty Experiment from CSV
-----------------------------------------
Input CSV must have:
  - prompt
  - reference
Optional:
  - generation

Outputs:
  - e_score_uncertainty_results.csv with per-sample metrics:
    BLEU, ROUGE_L, H_token, Sharpness, E, MI, VR, ECE, NLL, Brier, etc.

Usage:
  pip install --upgrade transformers datasets torch sacrebleu
  python e_score_uncertainty_from_csv.py --csv sample_prompts_with_generations.csv \
    --model_id gpt2 --num_candidates 5 --max_new_tokens 64 --temperature 0.7 --top_p 0.9 --mc_dropout
"""

import argparse, math, random
from typing import List, Tuple
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

try:
    import sacrebleu
    HAVE_SACREBLEU = True
except Exception:
    HAVE_SACREBLEU = False

# -------------------- Text metrics (BLEU simple, ROUGE-L) --------------------
def _ngrams(tokens, n):
    return list(zip(*[tokens[i:] for i in range(n)]))

def sentence_bleu_simple(reference: str, candidate: str, max_n=4, smooth=1.0) -> float:
    ref = reference.lower().split()
    cand = candidate.lower().split()
    if len(cand) == 0:
        return 0.0
    precisions = []
    for n in range(1, max_n+1):
        ref_ngrams = _ngrams(ref, n)
        cand_ngrams = _ngrams(cand, n)
        if len(cand_ngrams) == 0:
            precisions.append(0.0)
            continue
        ref_counts = {}
        for g in ref_ngrams:
            ref_counts[g] = ref_counts.get(g, 0) + 1
        match = 0
        for g in cand_ngrams:
            if ref_counts.get(g, 0) > 0:
                match += 1
                ref_counts[g] -= 1
        p = (match + smooth) / (len(cand_ngrams) + smooth)
        precisions.append(p)
    gm = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
    r = len(ref); c = len(cand)
    bp = 1.0 if c > r else math.exp(1 - r / max(c, 1))
    return bp * gm

def rouge_l_recall(reference: str, candidate: str) -> float:
    ref = reference.lower().split()
    cand = candidate.lower().split()
    m, n = len(ref), len(cand)
    if m == 0:
        return 0.0
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m):
        for j in range(n):
            if ref[i] == cand[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    lcs = dp[m][n]
    return lcs / m

# -------------------- Core metrics (entropy, sharpness, MI, etc.) --------------------
def normalized_entropy_from_scores(step_scores: List[torch.Tensor]) -> float:
    ents = []
    for s in step_scores:
        p = torch.softmax(s, dim=-1).clamp_min(1e-12)
        H = -(p * p.log()).sum().item()
        ents.append(H / math.log(p.size(-1)))
    return float(np.mean(ents)) if ents else 0.0

def prob_sharpness_from_scores(step_scores):
    """
    Yedek keskinlik: attention yoksa olasılık yoğunluğundan (∑p^2) hesapla.
    1/N..1 arası → 0..1 normalizasyonu.
    """
    if not step_scores:
        return 0.0
    vals = []
    for s in step_scores:
        p = torch.softmax(s, dim=-1).clamp_min(1e-12)
        pk = (p * p).sum().item()  # ∑ p^2
        N = p.numel()
        norm = (pk - 1.0/N) / (1.0 - 1.0/N)
        vals.append(max(0.0, min(1.0, norm)))
    return float(np.mean(vals))

def attention_sharpness_from_attentions(attentions, step_scores=None):
    """
    Attentions None içerebilir; yoksa prob_sharpness_from_scores'a düş.
    """
    if not attentions:
        return prob_sharpness_from_scores(step_scores)

    layerwise = []
    for step_att in attentions:            # her adım
        if step_att is None:
            continue
        step_vals = []
        for A in step_att:                 # her katman
            if A is None:
                continue
            # A: [B,H,T,T]
            A = A[0]                       # batch 0 → [H,T,T]
            A = A / A.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            peaked = (A**2).sum(dim=-1)   # [H,T]
            N = A.size(-1)
            norm = (peaked - 1.0/N) / (1.0 - 1.0/N)
            step_vals.append(norm.clamp(0,1).mean().item())
        if step_vals:
            layerwise.append(float(np.mean(step_vals)))

    if layerwise:
        return float(np.mean(layerwise))
    # attentionlardan hesaplayamadıysak → yedek
    return prob_sharpness_from_scores(step_scores)


def enable_mc_dropout(model: torch.nn.Module):
    for m in model.modules():
        if m.__class__.__name__.lower().startswith("dropout"):
            m.train()
    return model

def generate_once(model, tok, prompt: str, max_new_tokens=64, temperature=0.7, top_p=0.9, seed=42, mc_dropout=False):
    set_seed(seed)
    if mc_dropout:
        enable_mc_dropout(model)
    else:
        model.eval()
    enc = tok(prompt, return_tensors="pt")
    #model.config.output_attentions = True
    #model.config.use_cache = False   #bazı sürümlerde attn dönebilmesi için
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            output_scores=True,
            output_attentions=True,
            return_dict_in_generate=True,
        )
    text = tok.decode(out.sequences[0], skip_special_tokens=True)
    step_scores = [s[0] for s in out.scores] if hasattr(out, "scores") else []
    atts = out.attentions if hasattr(out, "attentions") else []
    H_tok = normalized_entropy_from_scores(step_scores)
    Sharp = attention_sharpness_from_attentions(atts, step_scores=step_scores)
    E = H_tok * (Sharp**2)
    # top-1 probs for ECE/VR
    top1_probs, top1_ids = [], []
    for s in step_scores:
        p = torch.softmax(s, dim=-1)
        val, idx = torch.max(p, dim=-1)
        top1_probs.append(val.item())
        top1_ids.append(idx.item())
    return {"text": text, "H_token": H_tok, "Sharpness": Sharp, "E": E,
            "step_scores": step_scores, "top1_probs": top1_probs, "top1_ids": top1_ids}

def mutual_information(score_samples: List[List[torch.Tensor]]) -> float:
    if not score_samples:
        return 0.0
    min_T = min(len(s) for s in score_samples)
    if min_T == 0:
        return 0.0
    H_mean, mean_H = 0.0, 0.0
    for t in range(min_T):
        Ps = []
        for k in range(len(score_samples)):
            p = torch.softmax(score_samples[k][t], dim=-1).clamp_min(1e-12)
            Ps.append(p.unsqueeze(0))
        P_mean = torch.cat(Ps, dim=0).mean(dim=0)  # [V]
        H_mean += float(-(P_mean * P_mean.log()).sum().item())
        Hs = []
        for k in range(len(score_samples)):
            q = torch.softmax(score_samples[k][t], dim=-1).clamp_min(1e-12)
            Hs.append(float(-(q * q.log()).sum().item()))
        mean_H += float(np.mean(Hs))
    H_mean /= min_T; mean_H /= min_T
    V = score_samples[0][0].numel()
    return (H_mean - mean_H) / math.log(V)

def variation_ratio(all_top1_ids: List[List[int]]) -> float:
    if not all_top1_ids or min(len(x) for x in all_top1_ids) == 0:
        return 0.0
    first_tokens = [x[0] for x in all_top1_ids]
    from collections import Counter
    c = Counter(first_tokens)
    most = c.most_common(1)[0][1]
    K = len(first_tokens)
    return 1.0 - (most / K)

def ece_score(confidences: List[float], corrects: List[int], num_bins=10) -> float:
    if len(confidences) == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, num_bins+1)
    ece = 0.0
    for b in range(num_bins):
        lo, hi = bins[b], bins[b+1]
        idx = [i for i, c in enumerate(confidences) if (c > lo and c <= hi) or (c==0.0 and lo==0.0)]
        if not idx:
            continue
        acc = np.mean([corrects[i] for i in idx])
        conf = np.mean([confidences[i] for i in idx])
        ece += (len(idx)/len(confidences)) * abs(acc - conf)
    return float(ece)

def nll_and_brier_on_reference(model, tok, prompt: str, reference: str):
    inputs = tok(prompt, return_tensors="pt")
    ref_ids = tok(reference, return_tensors="pt")["input_ids"][0]
    input_ids = torch.cat([inputs["input_ids"][0], ref_ids], dim=0).unsqueeze(0)
    labels = input_ids.clone()
    prompt_len = inputs["input_ids"].size(1)
    labels[0, :prompt_len] = -100
    with torch.no_grad():
        out = model(input_ids=input_ids, labels=labels)
    nll = out.loss.item()
    logits = out.logits[0, prompt_len-1:-1, :]
    probs = torch.softmax(logits, dim=-1)
    true = ref_ids
    brier = 0.0; cnt=0
    for t in range(min(probs.size(0), true.size(0))):
        p = probs[t]
        y = torch.zeros_like(p); y[true[t]] = 1.0
        brier += float(((p - y)**2).sum().item()); cnt += 1
    brier = brier / max(cnt,1)
    return nll, brier, probs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--model_id", type=str, default="gpt2")
    ap.add_argument("--num_candidates", type=int, default=5)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mc_dropout", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    df = pd.read_csv(args.csv)
    if not {"prompt","reference"}.issubset(df.columns):
        raise ValueError("CSV must contain 'prompt' and 'reference' columns.")

    tok = AutoTokenizer.from_pretrained(args.model_id)
    mdl = AutoModelForCausalLM.from_pretrained(args.model_id)
    
    if hasattr(mdl.config, "attn_implementation"):
        mdl.config.attn_implementation = "eager"
        
    mdl.eval()

    rows = []
    for i, row in df.iterrows():
        prompt = str(row["prompt"]); reference = str(row["reference"])
        cand_texts, cand_scores, cand_top1_ids = [], [], []
        Hs, Ss, Es = [], [], []

        # K candidate runs (for MI/VR)
        for k in range(args.num_candidates):
            res = generate_once(
                mdl, tok, prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature, top_p=args.top_p,
                seed=args.seed + i*args.num_candidates + k, mc_dropout=args.mc_dropout
            )
            cand_texts.append(res["text"])
            Hs.append(res["H_token"]); Ss.append(res["Sharpness"]); Es.append(res["E"])
            cand_scores.append(res["step_scores"])
            cand_top1_ids.append(res["top1_ids"])

        # MI & VR
        MI = mutual_information(cand_scores)
        VR = variation_ratio(cand_top1_ids)

        # Choose generation: use provided if exists, else first candidate
        chosen = row.get("generation", None)
        if not isinstance(chosen, str) or chosen.strip() == "":
            chosen = cand_texts[0]

        # BLEU / ROUGE-L
        if HAVE_SACREBLEU:
            bleu = sacrebleu.sentence_bleu(chosen, [reference]).score / 100.0
        else:
            bleu = sentence_bleu_simple(reference, chosen)
        rougeL = rouge_l_recall(reference, chosen)

        # Calibration on reference (teacher forcing)
        NLL, Brier, probs = nll_and_brier_on_reference(mdl, tok, prompt, reference)
        confs = probs.max(dim=-1).values.tolist()
        preds = probs.argmax(dim=-1)
        ref_ids = tok(reference, return_tensors="pt")["input_ids"][0]
        corrects = (preds == ref_ids[:len(preds)]).int().tolist()
        ECE = ece_score(confs, corrects, num_bins=10)

        rows.append({
            "prompt": prompt,
            "reference": reference,
            "generation": chosen,
            "BLEU": round(float(bleu), 3),
            "ROUGE_L": round(float(rougeL), 3),
            "H_token_mean": round(float(np.mean(Hs)), 3),
            "Sharpness_mean": round(float(np.mean(Ss)), 3),
            "E_mean": round(float(np.mean(Es)), 3),
            "MI": round(float(MI), 3),
            "VR": round(float(VR), 3),
            "ECE": round(float(ECE), 3),
            "NLL": round(float(NLL), 3),
            "Brier": round(float(Brier), 3),
            "cand_count": int(args.num_candidates),
            "gen_len_char": len(chosen)
        })

    out = pd.DataFrame(rows)
    out.to_csv("e_score_uncertainty_results.csv", index=False)
    print("Saved: e_score_uncertainty_results.csv")
    print(out.head())

if __name__ == "__main__":
    main()
