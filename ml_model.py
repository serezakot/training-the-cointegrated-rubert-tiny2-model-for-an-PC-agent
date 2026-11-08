"""First-layer model: rubert-tiny2 + three heads (intent, translation language, per-word tag)."""
import os
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from ml_spec import INTENTS, LANGS, MAX_LEN, TAGS, _QUOTES


class Parser(nn.Module):
    def __init__(self, base: str):
        super().__init__()
        self.enc = AutoModel.from_pretrained(base)
        hidden = self.enc.config.hidden_size
        self.drop = nn.Dropout(0.1)
        self.intent = nn.Linear(hidden, len(INTENTS))
        self.lang = nn.Linear(hidden, len(LANGS))
        self.tag = nn.Linear(hidden, len(TAGS))

    def forward(self, input_ids, attention_mask):
        hidden = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        hidden = self.drop(hidden)
        return self.intent(hidden[:, 0]), self.lang(hidden[:, 0]), self.tag(hidden)


def tokenize(tok, batch_words):
    """List of word lists -> tensors (+ the encoding itself, so we know which word each sub-token belongs to).

    Слова подаются модели в нижнем регистре и без кавычек («...», "..."): токенизатор регистрозависим, а кавычки иначе
    сдвигают first_tokens() на символ « вместо самого слова. Ответ собирается из ИСХОДНЫХ слов."""
    low = [[(w.translate(_QUOTES).lower().replace("ё", "е") or w.lower()) for w in words] for words in batch_words]
    return tok(low, is_split_into_words=True, truncation=True, max_length=MAX_LEN,
               padding=True, return_tensors="pt")


def first_tokens(enc, i: int) -> dict:
    """{word index: position of its first sub-token} for the i-th example of the batch."""
    first = {}
    for pos, wid in enumerate(enc.word_ids(i)):
        if wid is not None and wid not in first:
            first[wid] = pos
    return first


def save(model: Parser, tok, path) -> None:
    path = Path(path)
    os.makedirs(path, exist_ok=True)
    model.enc.save_pretrained(path)
    tok.save_pretrained(path)
    heads = {k: v for k, v in model.state_dict().items() if not k.startswith("enc.")}
    torch.save(heads, path / "heads.pt")


def load(path):
    path = Path(path)
    tok = AutoTokenizer.from_pretrained(path)
    model = Parser(str(path))
    heads = torch.load(path / "heads.pt", map_location="cpu")
    result = model.load_state_dict(heads, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(f"unexpected keys in heads.pt: {result.unexpected_keys}")
    model.eval()
    return model, tok


@torch.inference_mode()
def predict(model: Parser, tok, batch_words):
    """For every phrase: intent and language probabilities, plus the tag of each word and its probability."""
    enc = tokenize(tok, batch_words)
    il, ll, tl = model(enc["input_ids"], enc["attention_mask"])
    ip, lp, tp = il.softmax(-1), ll.softmax(-1), tl.softmax(-1)
    out = []
    for i, words in enumerate(batch_words):
        first = first_tokens(enc, i)
        tags, tag_p = [], []
        for w in range(len(words)):
            if w in first:
                p = tp[i, first[w]]
                best = int(p.argmax())
                tags.append(TAGS[best])
                tag_p.append(float(p[best]))
            else:                      # the word did not fit into MAX_LEN
                tags.append("O")
                tag_p.append(0.0)
        out.append({"intent_p": ip[i].tolist(), "lang_p": lp[i].tolist(), "tags": tags, "tag_p": tag_p,
                    "truncated": len(first) < len(words)})
    return out
