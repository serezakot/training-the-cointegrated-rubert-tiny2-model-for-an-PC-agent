# Voice Command Router

A tiny, fast, locally-run **first-layer intent parser** for a Russian-language voice/PC assistant.

It reads one spoken command (as text from speech recognition) and, in about 10-30 ms on a laptop CPU, decides:

1. **what the user wants**: open/search in a browser, translate something, or "everything else";
2. **which words of the phrase carry the data**: which site, what to search for, what text to translate;
3. **which language to translate into**.

Anything it is not sure about is handed to a big LLM instead of being executed blindly.

```
 speech recognition ──► "открой ютуб и найди как приготовить пирог"
                                      │
                                      ▼
                     ┌──────────────────────────────┐
                     │  first layer (this project)  │   ~29M parameters, CPU, ~20 ms
                     │  rubert-tiny2 + 3 small heads│
                     └──────────────┬───────────────┘
        confident?                  │                    not confident / "everything else"
   ┌──────────────┬────────────────┴───────────┐                  │
   ▼              ▼                            ▼                  ▼
 browser       translate                   (llm intent)      big LLM / agent
 site=ютуб     text=…  lang=de                               (slow, smart, expensive)
 query=как приготовить пирог
```

> The commands themselves are **Russian** (that is what the model is trained on), so all data, examples and
> labelling rules are in Russian. Code, comments and this documentation are in English.

---

## Why this exists

An assistant that sends *every* phrase to a large language model is slow, costs money per call, needs the network,
and sends everything you say to a third party. But most commands are tiny and repetitive:
*"открой ютуб"*, *"найди рецепт борща на ютубе"*, *"переведи на немецкий где вокзал"*.

| Option | Problem |
|---|---|
| Always ask a big LLM | seconds of latency, cost per call, needs internet, privacy |
| Regular expressions / keyword rules | speech recognition output is messy (typos, no punctuation, any case, many phrasings); every new site or sentence structure needs a new rule |
| **A small trained tagger + a safety net** | fast, offline, cheap, generalises to unseen sites and phrasings, and can say "I'm not sure" |

Design decisions that follow from that:

* **Tag the words, don't generate text.** The model only puts a label on each word of the original phrase
  (`O`, `SITE`, `QUERY`, `TEXT`); `site`/`query`/`text` are then assembled from those very words.
  It cannot invent or alter anything, so there are no hallucinations and the output is deterministic.
* **`llm` is the default class.** If a phrase has no browser/translate command, it is not the browser's job.
* **A safety net instead of blind trust.** If the intent probability or any word's probability is below a
  threshold, or the tagging looks impossible (e.g. a site name torn apart by other words), the phrase goes to the big LLM.
  A mistake becomes a *slower but correct* answer. The metric that matters is therefore the **silent error rate**
  (the model is confident and wrong), not plain accuracy.
* **Synthetic data + real phrases + an honest exam.** Labelled data comes from templates (so labels cannot disagree
  with the phrase), plus hand-labelled real phrases that are multiplied during training. Evaluation is done on phrases,
  sites and sentence structures the model has never seen.
* **The recipe is the product, not the weights.** Weights are reproducible from code + data in about an hour on a CPU,
  so they are not stored in git.

---

## What it understands

Three intents, each with its pieces:

| Intent | When | Pieces |
|---|---|---|
| `browser` | a browser command: открой / запусти / зайди / включи / поставь / найди ... **and** a place to search | `site` (if named), `query` (if said) |
| `translate` | переведи ..., как будет X по-английски, как сказать X на немецком | `text` (verbatim), `lang` (30 languages or `none`) |
| `llm` | everything else: questions, small talk, home/appliance commands, computer system tasks | nothing: the whole phrase goes on |

Expected routing (these examples are the rules; they are enforced by the tests):

| Phrase | Result |
|---|---|
| `открой ютуб и найди как приготовить пирог` | browser, site=`ютуб`, query=`как приготовить пирог` |
| `поищи в гугле формулу площади круга` | browser, no site (Google), query=`формулу площади круга` |
| `найди формулу площади круга` | **llm**: no place to search is named (rule 11) |
| `включи рок` | browser, no site, query=`рок` (media is not a search, rule 6) |
| `где вокзал переведи на немецкий` | translate, text=`где вокзал`, lang=`de` |
| `что значит слово gorgeous на русском` | translate, text=`gorgeous`, lang=`ru` (rule 14) |
| `что такое гитхаб` | llm: a question, even though it names a site |
| `включи свет в комнате` | llm: home command (rule 12) |
| `открой калькулятор` | llm: computer system task (rule 12) |
| `на вольфрам альфа реши интеграл` | browser, site=`вольфрам альфа`, query=`интеграл`, verb is `O` (rule 15) |

The full labelling contract (15 numbered rules) is the docstring of [`ml_spec.py`](ml_spec.py). It is the single source of
truth: the data generator, the real-phrase files and the tests all follow it.

---

## How it works

**Model** ([`ml_model.py`](ml_model.py)): [`cointegrated/rubert-tiny2`](https://huggingface.co/cointegrated/rubert-tiny2)
(BERT, 3 layers, hidden size 312, vocabulary 83,828, ~29M parameters) with three small linear heads:

| Head | Reads | Output |
|---|---|---|
| intent | the `[CLS]` vector (the whole phrase) | `browser` / `translate` / `llm` |
| language | the `[CLS]` vector | one of 30 codes (`none`, `en`, `ru`, `de`, ...) |
| word tag | every token | `O` / `SITE` / `QUERY` / `TEXT` (the label of a word is read from its first sub-token) |

**Decision** ([`ml_spec.py`](ml_spec.py), `decide()`): turns the model output into a route, assembles
`site`/`query`/`text` from tagged words, and applies the safety net. A phrase goes to the big LLM (`fallback=True`) when:
intent probability < `THRESHOLD_INTENT` (0.90), any word's probability < `THRESHOLD_SLOT` (0.75), the tagging is ragged,
a browser phrase has no place to search, a device command is left inside a query, a command has no site/query/text,
or the phrase is too long. `decide()` is a pure function, tested without any model.

**Training** ([`ml_train.py`](ml_train.py)): AdamW (lr 1e-4, weight decay 0.01), 6 % warm-up then linear decay, batch 64,
4 epochs, gradient clipping 1.0, label smoothing 0.1 (so confidence is meaningful and the thresholds actually work).
Loss = intent cross-entropy + 0.5 x language cross-entropy (translate phrases only) + word-tag cross-entropy (browser/translate phrases).
No GPU is needed; measured: about 42 minutes for 103k examples x 4 epochs on a 16-thread laptop CPU.

---

## Data

| Source | File | Role |
|---|---|---|
| Template generator | [`ml_gen.py`](ml_gen.py) + vocabularies [`ml_vocab.py`](ml_vocab.py), [`ml_home.py`](ml_home.py) | 34 templates (+ typos, case, fillers like "джарвис", "пожалуйста", trailing punctuation). Labels come from the construction of the template |
| Your real phrases | [`ml_real.tsv`](ml_real.tsv) | hand-labelled phrases used for training: copied as they are **and** augmented (other sites/queries are substituted, the structure stays yours) |
| Your exam | [`ml_mine.tsv`](ml_mine.tsv) | hand-labelled phrases that **never** enter training |
| Generated exam | [`ml_exam.py`](ml_exam.py) | 265 more exam phrases written by an AI assistant (an optimistic exam) |

Current size (default settings `--n 90000 --real-copies 8`): 90,000 template examples + 1,580 real phrases
(659 browser / 754 llm / 167 translate) expanded to ~41.6k examples = **~131.6k training examples, ~8.2k steps**.
The exam is 591 own phrases + 265 generated ones.

`ml_real.tsv` / `ml_mine.tsv` format: one phrase per line, TAB-separated, `#` starts a comment:

```
phrase <TAB> intent <TAB> site <TAB> query <TAB> text <TAB> language
найди в гугле рецепт борща	browser		рецепт борща		none
где вокзал переведи на немецкий	translate			где вокзал	de
что такое докер	llm				none
```

`site`, `query` and `text` must be **consecutive words copied from the phrase** (case, ё/е and edge punctuation are ignored).
The tests verify this for every row.

### Two training modes: `strict` and `final`

Training on everything and then testing on the same distribution gives a flattering but meaningless number. So:

* **`strict`**: ~15 % of the vocabulary (sites, query topics) and a few whole sentence templates (`HOLDOUT_TEMPLATES`) are
  *held out* of training. Evaluation is then done on **unseen sites**, **unseen sentence structures** and **both**.
  The model is written to `ml_model_strict` and is for measuring only.
* **`final`**: everything is used. This is the model you run. Its own validation set is *not* an honest test; the honest
  tests are your exam (`ml_mine.tsv`, `ml_exam.py`) and new phrases collected from real use.

First experiment (older rule set: 60k generated examples + 90 hand-labelled phrases, strict mode, 4 epochs):

| Evaluation set (3,000 phrases each) | intent correct | everything correct |
|---|---|---|
| unseen sites | 99.90 % | 97.93 % |
| unseen sentence structures | 99.87 % | 60.00 % |
| both unseen | 99.67 % | 55.60 % |

The intent is almost always right; what breaks is the *tagging* when a sentence structure was never seen.
That is why the project keeps adding structures (new templates and real phrases) rather than just more of the same data,
and why the safety net exists. Numbers for the current rule set: run the exam yourself (see below).

---

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu     # CPU-only wheel
pip install -r requirements.txt

# 1. Honest evaluation first (unseen sites / structures)
python3 ml_gen.py --mode strict --n 60000
python3 ml_train.py --mode strict --epochs 4          # -> ml_model_strict/
python3 test_ml.py --mine --model ml_model_strict

# 2. The model you actually run
python3 ml_gen.py --mode final --n 90000 --real-copies 8
python3 ml_train.py --mode final --epochs 4           # -> ml_model/  (about an hour on a laptop CPU)

# 3. Check it
python3 -m pytest test_ml.py -q                       # tests + the exam, one phrase per test
python3 test_ml.py --mine                             # exam summary incl. your phrases (silent errors!)
python3 test_ml.py --sweep                            # sweep confidence thresholds: correct / to LLM / errors

# 4. Use it
python3 ml_infer.py "открой ютуб и найди как приготовить пирог"
python3 ml_infer.py                                   # interactive prompt
```

In code:

```python
import ml_infer
result = await ml_infer.parse("переведи мне на английский как дела")
result["route"]      # "browser" | "translate" | "llm"
result["site"], result["query"]       # browser
result["text"], result["lang"]        # translate
result["fallback"]   # True -> hand the phrase to the big LLM
```

### Metrics

For every exam phrase the verdict is one of:

* **ok**: route and every piece are correct;
* **safe**: the model itself sent the phrase to the big LLM (slower, not wrong);
* **wrong**: a **silent error**, the model was confident and did the wrong thing. This is the number to drive down.

---

## The improvement loop

The best source of new training phrases is what you really say to the assistant.

```bash
export ML_PARSE_LOG=1                 # off by default; every phrase + decision goes to parse_log.jsonl
# ... use the assistant for a while ...
python3 ml_log.py > ml_review.tsv     # doubtful phrases first, in ml_real.tsv format
# fix the wrong rows by hand, then paste them into ml_real.tsv (training) or ml_mine.tsv (exam)
python3 -m pytest test_ml.py -q       # the data tests check labels, duplicates and the labelling rules
python3 ml_gen.py --mode final --n 90000 --real-copies 8 && python3 ml_train.py --mode final
python3 test_ml.py --sweep            # re-tune the thresholds if needed (two lines in ml_spec.py, no retraining)
```

Rules of thumb that proved useful:

* An exam phrase must never also be a training phrase (a test enforces this); keep some phrases **only** in `ml_mine.tsv`.
* Add data where the exam shows *clusters* of errors (a sentence structure, a family of devices), not blindly.
* When a rule changes, change it in `ml_spec.py`, the templates, the real phrases **and** the exam together;
  otherwise the tests (correctly) disagree with the model.

---

## Repository layout

| File | What it is |
|---|---|
| `ml_spec.py` | the labelling contract (rules 1-15), constants, thresholds, `decide()`, helpers for TSV and tagging |
| `ml_vocab.py` | vocabularies for the generator (sites, verbs, query pools, languages, translation texts) |
| `ml_home.py` | home-appliance, computer-system and media phrase generators + the "where to search" rule checks |
| `ml_gen.py` | training-data generator (strict / final), augmentation of real phrases, noise |
| `ml_model.py` | model: rubert-tiny2 + three heads, tokenisation, save / load / predict |
| `ml_train.py` | training loop, label smoothing, per-epoch evaluation on the exam |
| `ml_infer.py` | inference engine and CLI |
| `ml_log.py` | optional parse log and review tool |
| `ml_real.tsv` | hand-labelled training phrases |
| `ml_mine.tsv` | hand-labelled exam phrases |
| `ml_exam.py` | generated exam phrases |
| `test_ml.py` | tests and exam runner |

Not in git (see `.gitignore`): `data/` (generated), `ml_model*/` (weights), `parse_log.jsonl` and `ml_review.tsv` (private: what you said).

---

## Tests

* **Part 1 (no trained model needed; only the two training tests also need torch and are skipped without it):** vocabularies, deterministic generator, label/phrase consistency,
  honest holdout, no exam leakage into training, the labelling rules (11-15), the safety net (`decide()`), the parse log,
  the threshold table, a tiny end-to-end training run.
* **Part 2 (needs `ml_model/` and torch):** the exam, one phrase per test. "Known hard" phrases are marked `xfail`
  so that a *new* failure stands out.

---

## Limitations

* Russian only. The vocabularies, rules and examples are Russian (some Latin-script site names and English words occur).
* Some phrases are inherently ambiguous without world knowledge, e.g. `включи <unfamiliar two-word name>` can be a site,
  a song or a device. Those are the cases the safety net and more training data are for.
* Computer system tasks and home commands currently both route to `llm`; the second layer decides what to do with them.
  Splitting them into separate intents is possible (the data is already organised by category) but not done yet.
* The labelling rules are opinionated (e.g. "find X" without a place to search goes to the LLM). They are explicit in
  `ml_spec.py` so you can change them, but then data, exam and tests must change together.

## Privacy

The model runs locally. The repository contains only labelled example phrases. Do not commit `parse_log.jsonl` or
`ml_review.tsv`: they contain what was actually said to the assistant. Check `ml_real.tsv` / `ml_mine.tsv` before
publishing a fork if you add your own phrases.

## License

Not specified yet. Add a `LICENSE` file before publishing if you want others to be able to reuse the code.
