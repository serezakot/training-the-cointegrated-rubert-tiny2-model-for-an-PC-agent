#!/usr/bin/env python3
"""Tests for the first-layer parser.

Part 1 (no model, always runs): vocabularies, generator, tagging, safety net (decide), exam honesty.
Part 2 (needs a trained model in ml_model/ and torch): the exam - one phrase per test.

Usage:
    python3 -m pytest test_ml.py -q          # everything
    python3 test_ml.py                       # only the exam summary (after training)
    python3 test_ml.py --mine                # the same + your phrases from ml_mine.tsv
    python3 test_ml.py --mine --model ml_model_strict   # check another model (default: ml_model)
    python3 test_ml.py --sweep               # sweep the confidence thresholds: "threshold -> correct / to LLM / errors"

Your own phrases: ml_mine.tsv (exam) and ml_real.tsv (training samples), one phrase per line, TAB-separated fields:
    phrase  intent(browser/translate/llm)  site  query  text  language(en/de/.../none)
Leave empty fields empty. This is the real exam: the assistant has never seen these phrases.
"""
import sys
from collections import Counter
from pathlib import Path

import pytest

import ml_gen
from ml_exam import EXAM
from ml_spec import INTENTS, LANGS, MODEL_DIR, assemble, decide, norm, read_tsv, tag_phrase

HERE = Path(__file__).parent


def load_mine():
    return read_tsv(HERE / "ml_mine.tsv")


# ------------------------------------------------------------------ part 1: no model
def test_exam_is_self_consistent():
    seen = set()
    for phrase, intent, site, query, text, lang in EXAM:
        key = norm(phrase)
        assert key not in seen, f"duplicate in the exam: {phrase!r}"
        seen.add(key)
        assert intent in INTENTS and lang in LANGS, phrase
        for part in (site, query, text):
            assert norm(part) in key, f"{part!r} does not occur in the phrase {phrase!r}"
        if intent == "browser":
            assert site or query, phrase
            assert not text, phrase
        elif intent == "translate":
            assert text and not site and not query, phrase
        else:
            assert not (site or query or text) and lang == "none", phrase


def test_generator_tags_match_phrase():
    rows = ml_gen.generate(4000, 11, "all")
    assert len(rows) == 4000
    for r in rows:
        assert len(r["words"]) == len(r["tags"]) and r["words"]
        got = assemble(r["words"], r["tags"])
        if r["intent"] == "browser":
            assert got["site"] == r["site"] and got["query"] == r["query"] and (r["site"] or r["query"]), r
            assert r["lang"] == "none"
        elif r["intent"] == "translate":
            assert got["text"] == r["text"] and r["text"] and r["lang"] in LANGS, r
        else:
            assert set(r["tags"]) == {"O"} and r["lang"] == "none"
    kinds = Counter(r["intent"] for r in rows)
    assert all(kinds[i] > 400 for i in INTENTS), kinds
    assert len({r["lang"] for r in rows if r["intent"] == "translate"}) >= 25


def test_generator_is_deterministic():
    assert ml_gen.generate(300, 5, "all") == ml_gen.generate(300, 5, "all")
    assert ml_gen.generate(300, 5, "all") != ml_gen.generate(300, 6, "all")


def test_exam_does_not_leak_into_training():
    exam = frozenset(norm(e[0]) for e in EXAM)
    rows = ml_gen.generate(8000, 3, "all", exam=exam)
    assert not any(norm(" ".join(r["words"])) in exam for r in rows)


def test_unseen_sites_and_templates_are_really_held_out():
    train, unseen = set(ml_gen._split(ml_gen.V.SITES_REAL, "train")), set(ml_gen._split(ml_gen.V.SITES_REAL, "unseen"))
    assert train and unseen and not (train & unseen)
    keep = {t[0] for t in ml_gen.TEMPLATES} - ml_gen.HOLDOUT_TEMPLATES
    names = {r["tpl"] for r in ml_gen.generate(3000, 4, "train", keep)}
    assert not (names & ml_gen.HOLDOUT_TEMPLATES)
    assert {r["tpl"] for r in ml_gen.generate(500, 4, "train", ml_gen.HOLDOUT_TEMPLATES)} <= ml_gen.HOLDOUT_TEMPLATES


@pytest.mark.parametrize("name,case,want", [
    ("ютуб", "prep", "ютубе"), ("википедия", "prep", "википедии"), ("википедия", "acc", "википедию"),
    ("одноклассники", "prep", "одноклассниках"), ("яндекс карты", "prep", "яндекс картах"),
    ("вк", "prep", "вк"), ("github", "prep", "github"), ("озон", "gen", "озона"), ("ютуб", "instr", "ютубом"),
])
def test_site_name_declension(name, case, want):
    assert ml_gen.inflect(name, case) == want


def test_query_templates_fill_completely():
    rows = ml_gen.generate(3000, 8, "all")
    for r in rows:
        assert "{" not in " ".join(r["words"]) and "}" not in " ".join(r["words"]), r


def test_your_phrases_are_tagged_correctly_and_unique():
    real, mine = read_tsv(HERE / "ml_real.tsv"), read_tsv(HERE / "ml_mine.tsv")
    assert real and mine
    keys = [norm(r[0]) for r in real + mine]
    assert len(keys) == len(set(keys)), "the same phrase in two files, or a duplicate"
    for phrase, intent, site, query, text, lang in real + mine:
        assert intent in INTENTS and lang in LANGS, phrase
        words, tags = tag_phrase(phrase, site, query, text)          # ValueError if the piece is not in the phrase
        got = assemble(words, tags)
        assert (norm(got["site"]), norm(got["query"])) == (norm(site), norm(query)), phrase
        assert intent == "llm" or site or query or text, phrase      # llm phrases have no slots


def test_your_exam_phrases_do_not_enter_training():
    exam = ml_gen._exam_keys()
    mine = {norm(r[0]) for r in read_tsv(HERE / "ml_mine.tsv")}
    assert mine <= exam
    rows = ml_gen.generate(6000, 12, "all", exam=exam)
    pools = ml_gen.Pools("all")
    import random
    rng = random.Random(3)
    for r in ml_gen.real_rows():
        for _ in range(30):
            a = ml_gen.augment(r, rng, pools)
            if a and norm(" ".join(a["words"])) in exam:
                continue                     # such rows are filtered out when the training set is built
    assert not any(norm(" ".join(r["words"])) in exam for r in rows)


def test_augmentation_keeps_tagging():
    import random
    rng, pools = random.Random(9), ml_gen.Pools("all")
    made = 0
    for r in ml_gen.real_rows():
        if r["intent"] == "llm":
            continue                         # llm phrases are not augmented, they are added as they are
        for _ in range(15):
            a = ml_gen.augment(r, rng, pools)
            assert a and len(a["words"]) == len(a["tags"])
            got = assemble(a["words"], a["tags"])
            assert (got["site"], got["query"]) == (a["site"], a["query"]), a
            assert bool(a["site"]) == bool(r["site"]) and bool(a["query"]) == bool(r["query"]), (r, a)
            made += 1
    assert made > 500


def test_real_constructions_exist_in_generator():
    names = {r["tpl"] for r in ml_gen.generate(4000, 2, "all")}
    for need in ("browser.open_q_on_site", "browser.open_on_site_q", "browser.open_site_q_noverb",
                 "browser.open_site_suffix"):
        assert need in names, need


def test_yandex_music_is_one_site_not_site_plus_query():
    rows = ml_gen.generate(6000, 17, "all")
    for r in rows:
        if r["tpl"] == "browser.open_site_q_noverb" and r["site"] == "яндекс":
            assert not r["query"].startswith(("музык", "почт")), r


def _pred(intent, conf, tags, tag_p=None, lang="none", lang_conf=0.9, truncated=False):
    ip = [0.0] * len(INTENTS)
    ip[INTENTS.index(intent)] = conf
    ip[(INTENTS.index(intent) + 1) % len(INTENTS)] = 1 - conf
    lp = [0.0] * len(LANGS)
    lp[LANGS.index(lang)] = lang_conf
    lp[0] += 1 - lang_conf
    return {"intent_p": ip, "lang_p": lp, "tags": tags, "tag_p": tag_p or [0.99] * len(tags), "truncated": truncated}


def test_decide_normal_case_and_piece_assembly():
    words = "открой ютуб и найди как приготовить пирог".split()
    tags = ["O", "SITE", "O", "O", "QUERY", "QUERY", "QUERY"]
    r = decide("x", words, _pred("browser", 0.97, tags))
    assert (r["route"], r["site"], r["query"], r["fallback"]) == ("browser", "ютуб", "как приготовить пирог", False)


def test_decide_gap_in_query_is_filled():
    words = "найди чехол для телефона на озоне".split()
    tags = ["O", "QUERY", "O", "QUERY", "O", "SITE"]
    r = decide("x", words, _pred("browser", 0.97, tags))
    assert (r["site"], r["query"]) == ("озоне", "чехол для телефона")


def test_decide_translation_and_language():
    words = "переведи на английский как дела".split()
    tags = ["O", "O", "O", "TEXT", "TEXT"]
    r = decide("x", words, _pred("translate", 0.95, tags, lang="en"))
    assert (r["route"], r["text"], r["lang"]) == ("translate", "как дела", "en")
    unsure = decide("x", words, _pred("translate", 0.95, tags, lang="en", lang_conf=0.3))
    assert unsure["lang"] == "none"


@pytest.mark.parametrize("pred,why", [
    (_pred("browser", 0.6, ["O", "SITE"]), "intent"),
    (_pred("browser", 0.99, ["O", "SITE"], tag_p=[0.99, 0.4]), "tagging"),
    (_pred("browser", 0.99, ["O", "O"]), "was found"),
    (_pred("translate", 0.99, ["O", "O"]), "was found"),
    (_pred("browser", 0.99, ["O", "SITE"], truncated=True), "too long"),
])
def test_decide_doubt_hands_phrase_to_big_llm(pred, why):
    r = decide("открой ютуб", ["открой", "ютуб"], pred)
    assert r["route"] == "llm" and r["fallback"] and why in r["reason"]
    assert not (r["site"] or r["query"] or r["text"])


def test_decide_llm_extracts_nothing():
    r = decide("кто такой павел", ["кто", "такой", "павел"], _pred("llm", 0.99, ["O", "O", "O"]))
    assert r["route"] == "llm" and not r["fallback"] and not (r["site"] or r["query"] or r["text"])


# ------------------------------------------------------------------ part 2: with a model
def _model_dir() -> str:
    """Which model to check: --model FOLDER (e.g. ml_model_strict), or the ML_MODEL_DIR variable, otherwise the working ml_model."""
    import os
    if "--model" in sys.argv and sys.argv.index("--model") + 1 < len(sys.argv):
        return sys.argv[sys.argv.index("--model") + 1]
    return os.environ.get("ML_MODEL_DIR") or MODEL_DIR


def _model_ready():
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return (HERE / _model_dir() / "heads.pt").exists()


_engine = None


def engine():
    global _engine
    if _engine is None:
        import ml_infer
        _engine = ml_infer.Engine(str(HERE / _model_dir()))
    return _engine


def verdict(result, case):
    """ok - everything is correct; safe - the model itself handed the phrase to the big LLM (slow, but not a mistake);
    wrong - it silently did the wrong thing (this is the real error)."""
    phrase, intent, site, query, text, lang = case
    if result["route"] == intent:
        if intent == "llm":
            return "ok"
        same = (norm(result["site"]) == norm(site) and norm(result["query"]) == norm(query)
                and norm(result["text"]) == norm(text) and (intent != "translate" or result["lang"] == lang))
        return "ok" if same else "wrong"
    return "safe" if result["fallback"] else "wrong"


def _describe(result, case):
    phrase, intent, site, query, text, lang = case
    return (f"\n  phrase  : {phrase!r}\n  expected: {intent} site={site!r} query={query!r} text={text!r} lang={lang}"
            f"\n  got     : {result['route']} site={result['site']!r} query={result['query']!r} "
            f"text={result['text']!r} lang={result['lang']}"
            + (f"  [-> LLM: {result['reason']}]" if result["fallback"] else ""))


MINE = load_mine()
CASES = [("exam",) + tuple(c) for c in EXAM] + [("mine",) + tuple(c) for c in MINE]


_UNSEEN = "the word is deliberately absent from training (honest exam): without knowing it, it cannot be told apart from a site/music"
KNOWN_HARD = {
    "включи спид тест": "\"включи\" + an unfamiliar two-word name: a site or music - inherently ambiguous",
    "открой веб архив": _UNSEEN, "открой стек эксчейндж": _UNSEEN, "запусти sublime text": _UNSEEN,
    "запусти gimp": _UNSEEN, "запусти wireshark": _UNSEEN, "открой рольставни": _UNSEEN,
    "включи хлебопечку": _UNSEEN, "включи кормушку для кота": _UNSEEN, "включи электрокамин": _UNSEEN,
    "включи мороженицу": _UNSEEN, "включи гирлянду": _UNSEEN, "поставь скайфолл": _UNSEEN,
    "поищи рецепт плова на дзене": "\"дзене\" is a rare form of a site name; the model hands it to the llm",
}
# This is the "ceiling" of an honest exam: the test marks them xfail so that a NEW failure stands out immediately.
# Such words can be learned by adding them to ml_real.tsv (this is what is done for the working model).


@pytest.mark.parametrize("c", CASES, ids=[f"{c[0]}: {c[1]}" for c in CASES])
def test_model_exam(c):
    if not _model_ready():
        pytest.skip("no trained model (ml_model/) or no torch")
    case = c[1:]
    result = engine().predict_one(case[0])
    v = verdict(result, case)
    if v == "wrong" and case[0] in KNOWN_HARD:
        pytest.xfail("known difficulty: " + KNOWN_HARD[case[0]])
    assert v != "wrong", _describe(result, case)      # "to LLM" (safe) is acceptable: that is how the safety net is meant to work


def summary(include_mine: bool) -> None:
    eng = engine()
    groups = {"exam (old + new)": list(EXAM)}
    if include_mine and MINE:
        groups["your phrases"] = MINE
    for name, cases in groups.items():
        count = Counter()
        wrong = []
        by_intent = {i: Counter() for i in INTENTS}
        for case in cases:
            res = eng.predict_one(case[0])
            v = verdict(res, case)
            count[v] += 1
            by_intent[case[1]][v] += 1
            if v != "ok":
                wrong.append((v, case, res))
        n = len(cases)
        print(f"\n=== {name}: {n} phrases ===")
        print(f"correct at once: {count['ok'] / n:.1%}   handed to the big LLM (safe): {count['safe'] / n:.1%}"
              f"   silent errors: {count['wrong'] / n:.1%}")
        for intent, c in by_intent.items():
            total = sum(c.values())
            if total:
                print(f"  {intent:9}: correct {c['ok']}/{total}, LLM {c['safe']}, errors {c['wrong']}")
        for v, case, res in sorted(wrong, key=lambda x: x[0] != "wrong")[:40]:
            print(f"[{'ERROR' if v == 'wrong' else 'to LLM'}]" + _describe(res, case))


def thresholds_table(cases, preds, t_intents=(0.70, 0.80, 0.90), t_slots=(0.55, 0.65, 0.75, 0.85, 0.90)):
    """The same predictions at different thresholds: [(t_intent, t_slot, correct, to LLM, errors)]. A pure function (no model)."""
    out = []
    for ti in t_intents:
        for ts in t_slots:
            count = Counter()
            for case, pred in zip(cases, preds):
                words = case[0].split()
                count[verdict(decide(case[0], words, pred, t_intent=ti, t_slot=ts), case)] += 1
            out.append((ti, ts, count["ok"], count["safe"], count["wrong"]))
    return out


def sweep() -> None:
    """Sweep the thresholds on the exam + your phrases: a silent error counts three times worse than one extra LLM call."""
    import ml_model
    eng = engine()
    cases = list(EXAM) + MINE
    preds = []
    for i in range(0, len(cases), 128):
        preds += ml_model.predict(eng.model, eng.tok, [c[0].split() for c in cases[i:i + 128]])
    n = len(cases)
    table = thresholds_table(cases, preds)
    print(f"\n=== threshold sweep, {n} phrases ===\nINTENT  SLOT   correct  to LLM  errors")
    for ti, ts, ok, safe, wrong in table:
        print(f"{ti:.2f}    {ts:.2f}  {ok / n:6.1%}  {safe / n:6.1%}  {wrong / n:6.1%}")
    ti, ts, ok, safe, wrong = max(table, key=lambda r: (r[2] - 3 * r[4], -r[0] - r[1]))
    print(f"\nBest trade-off: THRESHOLD_INTENT = {ti}, THRESHOLD_SLOT = {ts}  "
          f"(correct {ok / n:.1%}, to LLM {safe / n:.1%}, errors {wrong / n:.1%}).\n"
          f"You can change them in ml_spec.py, the two THRESHOLD_* lines. No retraining is needed.")


if __name__ == "__main__":
    if not _model_ready():
        sys.exit("No trained model: run python3 ml_gen.py && python3 ml_train.py first")
    if "--sweep" in sys.argv:
        sweep()
    else:
        summary("--mine" in sys.argv)


# ---------------------------------------------------------------- the "where to search" rule and household commands
import ml_home as H

SYSTEM_STEMS = ("калькулятор", "папк", "файл", "программ", "скриншот", "компьютер", "ноутбук", "диспетчер", "терминал",
                "буфер обмена", "корзин", "антивирус", "драйвер")
HELD_OUT_DEVICES = ("гирлянд", "увлажнител", "проектор", "хлебопеч", "рольставн", "тостер", "ароматизатор", "кормушку для",
                    "вытяжк", "йогуртниц", "аквариум", "электрокамин", "диффузор", "зарядк", "морожениц")


def test_search_without_place_rule_data_is_clean():
    """browser without a site must have a "place to search" (google/internet/web/search/browser) or "включи/поставь";
    llm must not start with "найди..." and at the same time name a place to search."""
    rows = read_tsv(HERE / "ml_real.tsv") + read_tsv(HERE / "ml_mine.tsv") + [tuple(e) for e in EXAM]
    bad = [(r[0], H.rule_violation(r[0], r[1], r[2], r[3])) for r in rows if H.rule_violation(r[0], r[1], r[2], r[3])]
    assert not bad, bad[:10]


def test_generator_never_makes_search_without_place_a_browser_task():
    rows = ml_gen.generate(6000, 5, "all", exam=frozenset())
    kinds = Counter(r["tpl"] for r in rows)
    assert kinds["llm.search_nowhere"] > 100 and kinds["llm.home"] > 300 and kinds["browser.play_q_web"] > 50, kinds
    assert "browser.find_q_web" not in kinds
    for r in rows:
        if r["tpl"] == "llm.search_nowhere":
            assert r["intent"] == "llm" and not r["site"] and not r["query"], r
        if r["tpl"] == "browser.play_q_web":
            assert r["intent"] == "browser" and r["query"] and not r["site"], r


def test_household_commands_have_no_system_tasks_or_held_out_devices():
    import random
    rng = random.Random(11)
    seen = {H.home_phrase(rng) for _ in range(20000)}
    assert len(seen) > 3000, len(seen)
    for ph in seen:
        low = ph.lower()
        assert not any(s in low for s in SYSTEM_STEMS), ph
        assert not any(h in low for h in HELD_OUT_DEVICES), ph
        assert not H.starts_with_search(ph), ph
    for r in read_tsv(HERE / "ml_real.tsv"):
        assert not any(h in r[0].lower() for h in HELD_OUT_DEVICES), r[0]


HELD_OUT_SYSTEM = ("диспетчер устройств", "ножниц", "просмотр событий", "монитор ресурсов", "лупу", "запусти службы",
                   "sublime", "настройки обновлений", "настройки клавиатуры", "настройки мыши", "настройки языка",
                   "параметры экрана", "параметры питания", "мои документы", "семейными", "чертеж", "квитанц",
                   "редактор реестра", "диспетчер дисков", "gimp", "wireshark", "настройки даты", "настройки печати")
SYSTEM_APP_SITES = {"калькулятор", "блокнот", "проводник", "терминал", "диспетчер задач", "paint", "пейнт",
                    "панель управления", "настройки", "командную строку", "командная строка"}


def test_system_tasks_are_llm_and_never_sites():
    import random
    rng = random.Random(13)
    seen = {H.system_phrase(rng) for _ in range(20000)}
    assert len(seen) > 800, len(seen)
    for ph in seen:
        low = ph.lower()
        assert not any(h in low for h in HELD_OUT_SYSTEM), ph
        assert not H.has_where(ph) and not H.starts_with_search(ph), ph
    rows = read_tsv(HERE / "ml_real.tsv") + read_tsv(HERE / "ml_mine.tsv") + [tuple(e) for e in EXAM]
    for r in rows:
        if r[1] == "browser":
            assert norm(r[2]) not in SYSTEM_APP_SITES, r        # "калькулятор" is never a site
    for r in read_tsv(HERE / "ml_real.tsv"):
        assert not any(h in r[0].lower() for h in HELD_OUT_SYSTEM), r[0]
    gen = ml_gen.generate(6000, 7, "all", exam=frozenset())
    assert sum(1 for r in gen if r["tpl"] == "llm.system") > 150
    assert all(r["intent"] == "llm" for r in gen if r["tpl"] == "llm.system")
    assert not any(norm(r["site"]) in SYSTEM_APP_SITES for r in gen if r["intent"] == "browser")


# ================================================================== 2026-10-29: new checks (no model)
import ml_spec
import ml_vocab as V

FRESH_MARK = "FRESH EXAM"


def fresh_exam():
    """Rows of ml_mine.tsv after the "FRESH EXAM" marker: their words must not appear in training."""
    rows, on = [], False
    for line in (HERE / "ml_mine.tsv").read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            on = on or FRESH_MARK in line
            continue
        if on and line.strip():
            p = line.split("\t") + [""] * 6
            rows.append(tuple(x.strip() for x in p[:5]) + ((p[5].strip() or "none"),))
    return rows


def test_fresh_exam_exists_and_does_not_overlap_training():
    fresh = fresh_exam()
    assert len(fresh) >= 50
    train_keys = {norm(r[0]) for r in read_tsv(HERE / "ml_real.tsv")}
    assert not ({norm(r[0]) for r in fresh} & train_keys)
    sites = {norm(x) for x in ml_gen.all_sites()}
    named = [r[2] for r in fresh if r[1] == "browser" and r[2]]
    seen = [x for x in named if norm(x) in sites]
    # some fresh rows are deliberately about already known sites (a check of ordinary behaviour), but most are about new ones
    assert len(seen) <= 0.25 * len(named), f"too many sites of the fresh exam are already in training: {seen}"


def test_invented_sites_can_have_several_words():
    import random
    rng = random.Random(1)
    names = [ml_gen.fake_site(rng) for _ in range(3000)]
    multi = [n for n in names if " " in n]
    assert 0.2 < len(multi) / len(names) < 0.6, len(multi)
    assert all(len(n.split()) == 2 for n in multi)
    assert not any(b in n for n in names for b in ml_gen._BANNED)


def test_site_vocabulary_grows_from_your_phrases():
    extra = ml_gen.real_sites()
    assert len(extra) > 100 and any(" " in s for s in extra)
    assert not any(s.split()[0].casefold() in {"на", "в", "через"} for s in extra)
    assert len(ml_gen.all_sites()) == len(V.SITES_REAL) + len(extra)


def test_please_in_translation_rule_13():
    rows = [r for r in ml_gen.generate(12000, 21, "all", exam=frozenset()) if r["intent"] == "translate"]
    inside = 0
    for r in rows:
        w, t = r["words"], r["tags"]
        for i, x in enumerate(w):
            if x.strip(",.").casefold() not in ("пожалуйста", "плиз"):
                continue
            inside += t[i] == "TEXT"
            assert not (t[i] == "O" and i > 0 and t[i - 1] == "TEXT"), r      # a polite word after the text is part of the text
    assert inside > 100, inside


def test_please_after_text_in_your_data_is_text():
    for phrase, intent, site, query, text, lang in read_tsv(HERE / "ml_real.tsv") + read_tsv(HERE / "ml_mine.tsv") + [tuple(e) for e in EXAM]:
        if intent != "translate":
            continue
        words, tags = tag_phrase(phrase, site, query, text)
        for i, w in enumerate(words):
            if w.casefold().strip(",.") in ("пожалуйста", "плиз") and tags[i] == "O" and i > 0:
                assert tags[i - 1] != "TEXT", phrase


def test_po_bolgarski_inside_translated_text_is_text():
    rows = [r for r in ml_gen.generate(12000, 25, "all", exam=frozenset()) if r["intent"] == "translate"]
    hits = [r for r in rows if any(w == "по" and t == "TEXT" for w, t in zip(r["words"], r["tags"]))]
    assert len(hits) > 50, len(hits)


def test_what_does_it_mean_rule_14():
    rows = ml_gen.generate(15000, 22, "all", exam=frozenset())
    m_llm = [r for r in rows if r["tpl"] == "llm.meaning"]
    m_tr = [r for r in rows if r["tpl"] == "translate.meaning"]
    assert len(m_llm) > 50 and len(m_tr) > 50, (len(m_llm), len(m_tr))
    assert all(r["intent"] == "llm" and not r["lang"] != "none" for r in m_llm)
    assert all(r["intent"] == "translate" and r["lang"] != "none" and r["text"] for r in m_tr)


def test_action_verbs_between_site_and_query_rule_15():
    rows = ml_gen.generate(15000, 23, "all", exam=frozenset())
    acts = set(V.ACTS)
    site_rows = [r for r in rows if r["tpl"] == "browser.site_act_q"]
    assert len(site_rows) > 150, len(site_rows)
    for r in site_rows:
        assert r["intent"] == "browser" and r["site"] and r["query"], r
        for w, t in zip(r["words"], r["tags"]):
            if w.casefold().strip(".,") in acts:
                assert t == "O", r
    llm_rows = [r for r in rows if r["tpl"] == "llm.act"]
    assert len(llm_rows) > 50 and all(r["intent"] == "llm" and set(r["tags"]) == {"O"} for r in llm_rows)


def test_reference_tagging_is_never_ragged():
    rows = ml_gen.generate(8000, 24, "all", exam=frozenset()) + ml_gen.real_rows()
    for r in rows:
        assert not ml_spec.fragmented(r["tags"]), r
    for phrase, intent, site, query, text, lang in read_tsv(HERE / "ml_mine.tsv") + [tuple(e) for e in EXAM]:
        words, tags = tag_phrase(phrase, site, query, text)
        assert not ml_spec.fragmented(tags), phrase


def test_decide_ragged_tagging_goes_to_llm():
    words = "линкедин поиск рекрутеров в айти".split()
    r = decide("x", words, _pred("browser", 0.99, ["SITE", "O", "QUERY", "O", "SITE"]))
    assert r["route"] == "llm" and r["fallback"] and "ragged" in r["reason"]
    assert not (r["site"] or r["query"] or r["text"])


def test_decide_search_without_place_goes_to_llm_rule_11():
    words = "найди рецепт пиццы".split()
    r = decide("найди рецепт пиццы", words, _pred("browser", 0.99, ["O", "QUERY", "QUERY"]))
    assert r["route"] == "llm" and r["fallback"] and "rule 11" in r["reason"]


@pytest.mark.parametrize("phrase,tags", [
    ("найди в гугле рецепт пиццы", ["O", "O", "O", "QUERY", "QUERY"]),
    ("погугли рецепт пиццы", ["O", "QUERY", "QUERY"]),
    ("включи рок", ["O", "QUERY"]),
    ("хочу послушать queen", ["O", "O", "QUERY"]),
    ("вбей в поисковик курс доллара", ["O", "O", "O", "QUERY", "QUERY"]),
    ("поищи в интернете как варить рис", ["O", "O", "O", "QUERY", "QUERY", "QUERY"]),
    ("загулги как варить рис", ["O", "QUERY", "QUERY", "QUERY"]),
])
def test_decide_search_with_place_or_play_passes(phrase, tags):
    r = decide(phrase, phrase.split(), _pred("browser", 0.99, tags))
    assert r["route"] == "browser" and not r["fallback"], r


def test_decide_command_inside_query_goes_to_llm():
    words = "у меня руки заняты включи чайник".split()
    r = decide("у меня руки заняты включи чайник", words, _pred("browser", 0.99, ["O", "QUERY", "QUERY", "QUERY", "QUERY", "QUERY"]))
    assert r["route"] == "llm" and r["fallback"] and "command" in r["reason"]
    ok = decide("включи рок", ["включи", "рок"], _pred("browser", 0.99, ["O", "QUERY"]))
    assert ok["route"] == "browser" and not ok["fallback"]


def test_no_your_phrase_has_command_verb_in_siteless_query():
    rows = read_tsv(HERE / "ml_real.tsv") + read_tsv(HERE / "ml_mine.tsv") + [tuple(e) for e in EXAM]
    for phrase, intent, site, query, text, lang in rows:
        if intent == "browser" and not site:
            assert not ml_spec._COMMAND_IN_QUERY.search(query), phrase


def test_model_path_is_absolute():
    assert Path(MODEL_DIR).is_absolute() and Path(MODEL_DIR).name == "ml_model"


def test_parse_log(tmp_path, monkeypatch):
    import ml_log
    monkeypatch.delenv("ML_PARSE_LOG", raising=False)
    assert ml_log.log_path() is None and ml_log.log_result({"phrase": "x"}) is False
    p = tmp_path / "log.jsonl"
    monkeypatch.setenv("ML_PARSE_LOG", str(p))
    ok = {"phrase": "открой ютуб", "route": "browser", "site": "ютуб", "query": "", "text": "", "lang": "none",
          "confidence": 0.97, "fallback": False, "reason": ""}
    doubt = {"phrase": "включи хлебопечку", "route": "llm", "site": "", "query": "", "text": "", "lang": "none",
             "confidence": 0.6, "fallback": True, "reason": "not sure about the intent"}
    for r in (ok, doubt, ok):
        assert ml_log.log_result(r)
    rows = ml_log.read_log(p)
    assert len(rows) == 3
    only = ml_log.to_tsv(rows, only_doubtful=True)
    assert len(only) == 1 and only[0].startswith("# review") and "хлебопечку" in only[0]     # the repeated "открой ютуб" is collapsed
    full = ml_log.to_tsv(rows, only_doubtful=False)
    assert len(full) == 2
    out = tmp_path / "review.tsv"
    out.write_text("\n".join(full) + "\n", encoding="utf-8")
    parsed = read_tsv(str(out))
    assert [r[0] for r in parsed] == ["включи хлебопечку", "открой ютуб"] and parsed[1][2] == "ютуб"
    monkeypatch.setenv("ML_PARSE_LOG", "0")
    assert ml_log.log_path() is None


def test_threshold_table():
    cases = [("открой ютуб", "browser", "ютуб", "", "", "none"),
             ("кто такой павел", "llm", "", "", "", "none"),
             ("открой ютуб", "browser", "ютуб", "", "", "none")]
    good = _pred("browser", 0.99, ["O", "SITE"])
    wrong = _pred("browser", 0.85, ["O", "SITE"], tag_p=[0.99, 0.6])     # in fact this is llm
    shaky = _pred("browser", 0.99, ["O", "SITE"], tag_p=[0.99, 0.6])     # correct, but unsure
    table = thresholds_table(cases, [good, wrong, shaky], t_intents=(0.8,), t_slots=(0.55, 0.9))
    assert table[0][2:] == (2, 0, 1)         # low threshold: the phrase about Pavel silently goes to the browser -> error
    assert table[1][2:] == (2, 1, 0)         # high threshold: it safely goes to llm, and the unsure one becomes "safe"


def test_training_exam_and_loss_function():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import ml_train
    assert len(ml_train.exam_rows()) == len(EXAM) + len(MINE)
    assert ml_train.default_out("final") == MODEL_DIR and ml_train.default_out("strict") == MODEL_DIR + "_strict"
    il = torch.full((2, 3), -10.0)
    il[0, 0] = 10
    il[1, 1] = 10
    ll = torch.zeros(2, len(LANGS))
    tl = torch.full((2, 3, 4), -10.0)
    tl[:, :, 0] = 10
    intent_y, lang_y, tag_y = torch.tensor([0, 1]), torch.tensor([0, 0]), torch.zeros(2, 3, dtype=torch.long)
    sharp = ml_train.loss_fn((il, ll, tl), intent_y, lang_y, tag_y, 0.0)
    smooth = ml_train.loss_fn((il, ll, tl), intent_y, lang_y, tag_y, 0.1)
    assert torch.isfinite(sharp) and torch.isfinite(smooth) and smooth > sharp + 0.5


def test_full_training_run_on_tiny_model(tmp_path, monkeypatch):
    """Runs the whole path: data -> ml_train.main -> save -> load -> ml_infer. Without downloading rubert."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    tokenizers = pytest.importorskip("tokenizers")
    import json
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors, trainers
    import ml_infer
    import ml_train
    rows = ml_gen.generate(480, 31, "all", exam=ml_gen._exam_keys())
    data = tmp_path / "data"
    data.mkdir()
    (data / "train_final.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows[:400]), encoding="utf-8")
    (data / "val_final.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows[400:]), encoding="utf-8")
    texts = [" ".join(r["words"]) for r in rows] + [" ".join(c[0].split()) for c in list(EXAM) + MINE]
    tk = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tk.normalizer = normalizers.BertNormalizer(lowercase=True)
    tk.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tk.train_from_iterator(texts, trainers.WordPieceTrainer(vocab_size=1500, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]))
    tk.post_processor = processors.TemplateProcessing(single="[CLS] $A [SEP]", special_tokens=[("[CLS]", tk.token_to_id("[CLS]")), ("[SEP]", tk.token_to_id("[SEP]"))])
    tok = transformers.PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")
    base = tmp_path / "tiny"
    tok.save_pretrained(base)
    cfg = transformers.BertConfig(vocab_size=tk.get_vocab_size(), hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64, max_position_embeddings=128)
    transformers.BertModel(cfg).save_pretrained(base)
    out = tmp_path / "model"
    monkeypatch.setattr(sys, "argv", ["ml_train.py", "--mode", "final", "--data", str(data), "--base", str(base), "--out", str(out),
                                      "--epochs", "1", "--batch", "32", "--threads", "1"])
    ml_train.main()
    assert (out / "heads.pt").exists()
    eng = ml_infer.Engine(str(out), threads=1)
    r = eng.predict_one("открой ютуб и найди котиков")
    assert r["route"] in INTENTS and "confidence" in r


def test_your_training_phrases_do_not_overlap_exam():
    exam = {norm(r[0]) for r in EXAM} | {norm(r[0]) for r in read_tsv(HERE / "ml_mine.tsv")}
    common = sorted(exam & {norm(r[0]) for r in read_tsv(HERE / "ml_real.tsv")})
    assert not common, f"these phrases are in both training (ml_real.tsv) and the exam: {common}"
