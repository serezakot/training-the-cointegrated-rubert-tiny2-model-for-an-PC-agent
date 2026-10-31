#!/usr/bin/env python3
"""First-layer parser training.

Notes (2026-10-29): label smoothing - the model stops being "confident in every mistake", so the thresholds in
ml_spec.decide really catch doubtful phrases; after every epoch the accuracy on your own exam (ml_exam.py +
ml_mine.tsv) is printed; the strict model is written to ml_model_strict, the final one to ml_model.

Install (once):
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install transformers

Honest evaluation on unseen sites and unseen sentence structures (do this first):
    python3 ml_gen.py --mode strict --n 60000
    python3 ml_train.py --mode strict --epochs 4
Final model on all the data:
    python3 ml_gen.py --mode final --n 90000
    python3 ml_train.py --mode final --epochs 4

No GPU is needed: the model is small (~29M parameters) and a modern laptop CPU is enough.
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

import ml_model
from ml_exam import EXAM
from ml_spec import BASE_MODEL, INTENTS, LANGS, MODEL_DIR, TAGS, assemble, norm, read_tsv, tag_phrase

HERE = Path(__file__).resolve().parent


def read(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def batches(rows, size, shuffle, rng):
    idx = list(range(len(rows)))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), size):
        yield [rows[j] for j in idx[i:i + size]]


def make_targets(tok, batch):
    words = [r["words"] for r in batch]
    enc = ml_model.tokenize(tok, words)
    tag_y = torch.full(enc["input_ids"].shape, -100, dtype=torch.long)
    for i, r in enumerate(batch):
        if r["intent"] == "llm":
            continue                                   # words of llm phrases are not tagged
        for w, pos in ml_model.first_tokens(enc, i).items():
            tag_y[i, pos] = TAGS.index(r["tags"][w])
    intent_y = torch.tensor([INTENTS.index(r["intent"]) for r in batch])
    lang_y = torch.tensor([LANGS.index(r["lang"]) for r in batch])
    return enc, intent_y, lang_y, tag_y


def loss_fn(outputs, intent_y, lang_y, tag_y, smooth=0.0):
    il, ll, tl = outputs
    loss = F.cross_entropy(il, intent_y, label_smoothing=smooth)
    translate = intent_y == INTENTS.index("translate")
    if translate.any():
        loss = loss + 0.5 * F.cross_entropy(ll[translate], lang_y[translate], label_smoothing=smooth)
    if (tag_y != -100).any():
        loss = loss + F.cross_entropy(tl.reshape(-1, len(TAGS)), tag_y.reshape(-1), ignore_index=-100,
                                      label_smoothing=smooth)
    return loss


def default_out(mode: str) -> str:
    """final -> ml_model (the working model), strict -> ml_model_strict (honest evaluation only, never used in production)."""
    return MODEL_DIR if mode == "final" else MODEL_DIR + "_strict"


def exam_rows():
    """Your exam (ml_exam.py + ml_mine.tsv) as rows for evaluate(): it never enters the training set."""
    rows = []
    for phrase, intent, site, query, text, lang in [tuple(e) for e in EXAM] + read_tsv(HERE / "ml_mine.tsv"):
        words, _ = tag_phrase(phrase, site, query, text)
        rows.append({"words": words, "intent": intent, "site": site, "query": query, "text": text,
                     "lang": lang if intent == "translate" else "none"})
    return rows


def evaluate(model, tok, rows, batch_size=256):
    """Accuracy: intent; exact match (intent + site/query/text + language)."""
    model.eval()
    n = len(rows)
    intent_ok = exact_ok = 0
    errors = []
    for chunk in (rows[i:i + batch_size] for i in range(0, n, batch_size)):
        preds = ml_model.predict(model, tok, [r["words"] for r in chunk])
        for r, p in zip(chunk, preds):
            intent = INTENTS[max(range(len(INTENTS)), key=lambda k: p["intent_p"][k])]
            lang = LANGS[max(range(len(LANGS)), key=lambda k: p["lang_p"][k])]
            got = assemble(r["words"], p["tags"]) if intent != "llm" else {"site": "", "query": "", "text": ""}
            ok_intent = intent == r["intent"]
            intent_ok += ok_intent
            if ok_intent and intent == "llm":
                ok = True
            elif ok_intent:
                ok = (norm(got["site"]) == norm(r["site"]) and norm(got["query"]) == norm(r["query"])
                      and norm(got["text"]) == norm(r["text"])
                      and (intent != "translate" or lang == r["lang"]))
            else:
                ok = False
            exact_ok += ok
            if not ok and len(errors) < 12:
                errors.append((" ".join(r["words"]), r["intent"], intent, r["site"], got["site"],
                               r["query"], got["query"], r["text"], got["text"], r["lang"], lang))
    return {"n": n, "intent": intent_ok / n, "exact": exact_ok / n, "errors": errors}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(HERE / "data"))
    ap.add_argument("--mode", choices=["strict", "final"], default="strict")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--base", default=BASE_MODEL)
    ap.add_argument("--out", default=None, help="model folder (default: final -> ml_model, strict -> ml_model_strict)")
    ap.add_argument("--smooth", type=float, default=0.1, help="label smoothing; 0 disables it")
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 8)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    out_dir = args.out or default_out(args.mode)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    data = Path(args.data)
    if args.mode == "strict":
        train = read(data / "train.jsonl")
        vals = {"unseen sites": read(data / "val_sites.jsonl"),
                "unseen constructions": read(data / "val_templates.jsonl"),
                "both unseen": read(data / "val_both.jsonl")}
    else:
        train = read(data / "train_final.jsonl")
        vals = {"control (same distribution)": read(data / "val_final.jsonl")}
    vals["YOUR EXAM (not in training)"] = exam_rows()

    tok = AutoTokenizer.from_pretrained(args.base)
    model = ml_model.Parser(args.base)
    steps = args.epochs * ((len(train) + args.batch - 1) // args.batch)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    print(f"examples: {len(train)}, steps: {steps}, threads: {args.threads}, label smoothing: {args.smooth}")

    started = time.time()
    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for batch in batches(train, args.batch, True, rng):
            enc, intent_y, lang_y, tag_y = make_targets(tok, batch)
            loss = loss_fn(model(enc["input_ids"], enc["attention_mask"]), intent_y, lang_y, tag_y, args.smooth)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            running += loss.item()
            step += 1
            if step % 100 == 0:
                done = step / steps
                eta = (time.time() - started) * (1 - done) / done
                print(f"  step {step}/{steps}  loss {running / 100:.4f}  ~{eta / 60:.0f} min left", flush=True)
                running = 0.0
        print(f"== epoch {epoch} finished, {(time.time() - started) / 60:.1f} min")
        for name, rows in vals.items():
            m = evaluate(model, tok, rows)
            print(f"   {name}: intent {m['intent']:.2%}, exact match {m['exact']:.2%}  (n={m['n']})")
    ml_model.save(model, tok, out_dir)
    print("model saved:", Path(out_dir).resolve())
    for name, rows in vals.items():
        m = evaluate(model, tok, rows)
        for e in m["errors"][:6]:
            print(f"   [{name}] {e[0]!r}\n      expected: {e[1]} site={e[3]!r} query={e[5]!r} text={e[7]!r} lang={e[9]}"
                  f"\n      model: {e[2]} site={e[4]!r} query={e[6]!r} text={e[8]!r} lang={e[10]}")


if __name__ == "__main__":
    main()
