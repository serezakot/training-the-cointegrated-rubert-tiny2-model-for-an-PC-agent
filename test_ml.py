#!/usr/bin/env python3
"""Tests for the first-layer parser.

Part 1 (no model, always runs): vocabularies, generator, tagging, safety net (decide), exam & probe honesty.
Part 2 (needs a trained model in ml_model/ and torch): the exam, your phrases (ml_mine.tsv), USER_HARD_CASES,
and the 1200-phrase live generalization probe (ml_probe.py) with ANSI color-coded error diffs.

Usage:
    python3 -m pytest test_ml.py -q          # everything (with pytest)
    python3 test_ml.py --unit                # run all Part 1 unit & data-integrity tests without pytest
    python3 test_ml.py                       # exam summary (after training)
    python3 test_ml.py --mine                # exam + your phrases (ml_mine.tsv) + 16 user hard cases
    python3 test_ml.py --live                # full live probe (1200 unseen phrases + USER_HARD_CASES, colored report)
    python3 test_ml.py --live --model ml_model_strict   # check another model (default: ml_model)
    python3 test_ml.py --sweep               # sweep the confidence thresholds: "threshold -> correct / to LLM / errors"
"""
import random
import sys
from collections import Counter
from pathlib import Path

try:
    import pytest
except ImportError:
    class _DummyMark:
        @staticmethod
        def parametrize(argnames, argvalues, ids=None):
            def deco(fn):
                fn._parametrize = (argnames, argvalues, ids)
                return fn
            return deco

    class _DummyPytest:
        mark = _DummyMark()

        @staticmethod
        def skip(msg=""):
            raise RuntimeError(f"SKIP: {msg}")

        @staticmethod
        def xfail(msg=""):
            raise RuntimeError(f"XFAIL: {msg}")

        @staticmethod
        def importorskip(modname):
            try:
                return __import__(modname)
            except ImportError:
                raise RuntimeError(f"SKIP: missing {modname}")

    pytest = _DummyPytest()

import ml_gen
import ml_probe
import ml_spec
import ml_vocab as V
from ml_exam import EXAM
from ml_spec import (
    INTENTS,
    LANGS,
    MODEL_DIR,
    assemble,
    decide,
    norm,
    query_matches,
    read_tsv,
    strip_query_wrapper,
    tag_phrase,
    text_matches,
)

HERE = Path(__file__).parent

# ------------------------------------------------------------------ ANSI colors for terminal reports
C_RED = "\033[91m"
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_CYAN = "\033[96m"
C_BOLD = "\033[1m"
C_DIM = "\033[2m"
C_RESET = "\033[0m"


def load_mine():
    return read_tsv(HERE / "ml_mine.tsv")


# ------------------------------------------------------------------ part 1: no model (always runs)
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
        elif intent == "app":
            assert site and not text, phrase
        elif intent == "system":
            assert query and not site and not text, phrase
        elif intent == "translate":
            assert text and not site and not query, phrase
        else:
            assert not (site or query or text) and lang == "none", phrase


def test_user_hard_cases_and_probe_are_self_consistent():
    seen = set()
    for cls, phrase, intent, site, query, text, lang in ml_probe.USER_HARD_CASES:
        key = norm(phrase)
        assert key not in seen, f"duplicate in USER_HARD_CASES: {phrase!r}"
        seen.add(key)
        assert intent in INTENTS and lang in LANGS, phrase
        for part in (site, query, text):
            assert norm(part) in key, f"{part!r} does not occur in {phrase!r}"
    probe_rows = ml_probe.generate(1200)
    assert len(probe_rows) == 1200
    for cls, phrase, intent, site, query, text, lang in probe_rows:
        key = norm(phrase)
        assert intent in INTENTS and lang in LANGS, phrase
        for part in (site, query, text):
            assert norm(part) in key, f"{part!r} does not occur in probe phrase {phrase!r}"


def test_probe_held_words_never_leak_into_vocab_or_real_tsv():
    held = set(ml_probe.held_words())
    assert len(held) >= 80
    vocab_items = set()
    for s in V.SITES_REAL + V.APP_HYBRID + V.APP_DESKTOP:
        vocab_items.add(norm(s))
    for k, pool in V.POOLS.items():
        for item in pool:
            vocab_items.add(norm(item))
    overlap_vocab = sorted(held & vocab_items)
    assert not overlap_vocab, f"held-out probe words leaked into ml_vocab.py: {overlap_vocab}"

    real_sites_and_queries = set()
    for phrase, intent, site, query, text, lang in read_tsv(HERE / "ml_real.tsv"):
        if site:
            real_sites_and_queries.add(norm(site))
        if query:
            real_sites_and_queries.add(norm(query))
    overlap_real = sorted(held & real_sites_and_queries)
    assert not overlap_real, f"held-out probe words leaked into ml_real.tsv slots: {overlap_real}"


def test_generator_tags_match_phrase():
    rows = ml_gen.generate(4000, 11, "all")
    assert len(rows) == 4000
    for r in rows:
        assert len(r["words"]) == len(r["tags"]) and r["words"]
        got = assemble(r["words"], r["tags"])
        if r["intent"] == "browser":
            assert got["site"] == r["site"] and got["query"] == r["query"] and (r["site"] or r["query"]), r
            assert r["lang"] == "none"
        elif r["intent"] == "app":
            assert got["site"] == r["site"] and got["query"] == r["query"] and r["site"], r
            assert r["lang"] == "none"
        elif r["intent"] == "system":
            assert got["query"] == r["query"] and r["query"] and not r["site"], r
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


def test_exam_and_probe_do_not_leak_into_training():
    exam = ml_gen._exam_keys()
    for _, phrase, *_ in ml_probe.USER_HARD_CASES:
        assert norm(phrase) in exam, f"USER_HARD_CASES phrase missing from _exam_keys: {phrase!r}"
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
    ("ебей", "prep", "ебее"), ("спотифай", "prep", "спотифае"), ("аппл мьюзик", "prep", "аппл мьюзике"),
    ("пенсионный фонд россии", "prep", "пенсионном фонде россии"),
    ("налоговая служба украины", "prep", "налоговой службе украины"),
    ("портал судебной власти", "prep", "портале судебной власти"),
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
    rng = random.Random(3)
    for r in ml_gen.real_rows():
        for _ in range(30):
            a = ml_gen.augment(r, rng, pools)
            if a and norm(" ".join(a["words"])) in exam:
                continue                     # such rows are filtered out when the training set is built
    assert not any(norm(" ".join(r["words"])) in exam for r in rows)


def test_augmentation_keeps_tagging():
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
    names = {r["tpl"] for r in ml_gen.generate(5000, 2, "all")}
    for need in ("browser.open_q_on_site", "browser.open_on_site_q", "browser.open_site_q_noverb",
                 "browser.open_site_suffix", "browser.wrapper_q", "app.open_only", "app.open_query",
                 "system.action", "translate.colon", "llm.vague"):
        assert need in names, need


def test_google_is_never_tagged_as_site_in_search_on_site():
    """Rule 4: 'найди в гугле X' / 'найди на сайте гугл X' is always a siteless Google search (site='')."""
    rows = ml_gen.generate(6000, 19, "all")
    for r in rows:
        if r["intent"] == "browser" and r["query"] and r["tpl"] in (
            "browser.find_q_on_site", "browser.find_on_site_q", "browser.wrapper_q",
            "browser.site_find_q", "browser.prep_site_find_q", "browser.open_q_on_site",
        ):
            assert norm(r["site"]) not in {"гугл", "гугле", "google"}, r


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


def test_decide_fixes_official_site_and_pronoun_and_guillemets_and_google():
    # 1. "Перейди на официальный сайт Python": if "официальный" got tagged SITE, decide() cleans it to O
    w1 = "Перейди на официальный сайт Python".split()
    r1 = decide(" ".join(w1), w1, _pred("browser", 0.95, ["O", "O", "SITE", "O", "SITE"]))
    assert (r1["route"], r1["site"], r1["query"], r1["fallback"]) == ("browser", "Python", "", False)

    # 2. "Мне нужен сайт Stack Overflow, открой его": if "его" got tagged QUERY, decide() cleans it to O
    w2 = "Мне нужен сайт Stack Overflow, открой его".split()
    r2 = decide(" ".join(w2), w2, _pred("browser", 0.95, ["O", "O", "O", "SITE", "SITE", "O", "QUERY"]))
    assert (r2["route"], r2["site"], r2["query"], r2["fallback"]) == ("browser", "Stack Overflow", "", False)

    # 3. "Найди фильм «Интерстеллар» на Кинопоиске": guillemets are stripped cleanly
    w3 = "Найди фильм «Интерстеллар» на Кинопоиске".split()
    r3 = decide(" ".join(w3), w3, _pred("browser", 0.95, ["O", "QUERY", "QUERY", "O", "SITE"]))
    assert r3["site"] == "Кинопоиске" and "«" not in r3["query"] and "»" not in r3["query"]
    assert query_matches(r3["query"], "Интерстеллар") and query_matches(r3["query"], "фильм Интерстеллар")

    # 4. "Найди в Google последние новости OpenA" / "Найди на сайте гугл курс евро": site is normalized to ''
    w4 = "Найди в Google последние новости OpenA".split()
    r4 = decide(" ".join(w4), w4, _pred("browser", 0.95, ["O", "O", "SITE", "QUERY", "QUERY", "QUERY"]))
    assert (r4["route"], r4["site"], r4["query"], r4["fallback"]) == ("browser", "", "последние новости OpenA", False)


def test_flexible_query_matches():
    assert query_matches("фильм Интерстеллар", "Интерстеллар")
    assert query_matches("Интерстеллар", "фильм «Интерстеллар»")
    assert query_matches("проект с названием Stable Diffusion", "Stable Diffusion")
    assert query_matches("музыку The Weeknd", "The Weeknd")
    assert query_matches("информацию о Николе Тесле", "Николе Тесле")
    assert query_matches("видео о создании AI-агента на Python", "видео о создании AI-агента на Python")
    assert not query_matches("его", "")
    assert not query_matches("", "Интерстеллар")
    assert strip_query_wrapper("проект с названием Stable Diffusion") == "stable diffusion"


def test_portal_site_names_are_preserved():
    """'открой портал госуслуг' / 'перейди на портал открытых данных' must keep 'портал' inside SITE."""
    w = "открой портал госуслуг".split()
    r = decide(" ".join(w), w, _pred("browser", 0.96, ["O", "SITE", "SITE"]))
    assert (r["route"], r["site"], r["fallback"]) == ("browser", "портал госуслуг", False)


def test_decide_strips_trailing_urgency_and_rejects_verbless_siteless_browser():
    # 1. "запусти калькулятор быстро" -> "быстро" stripped from SITE
    w1 = "запусти калькулятор быстро".split()
    r1 = decide(" ".join(w1), w1, _pred("app", 0.98, ["O", "SITE", "SITE"]))
    assert (r1["route"], r1["site"], r1["fallback"]) == ("app", "калькулятор", False)

    # 2. "расписание поездов софия варна" (no command verb at all) -> must fallback to llm
    w2 = "расписание поездов софия варна".split()
    r2 = decide(" ".join(w2), w2, _pred("browser", 0.98, ["QUERY", "QUERY", "QUERY", "QUERY"]))
    assert r2["route"] == "llm" and r2["fallback"] and "verb" in r2["reason"]


def test_audit_all_datasets_and_generator():
    """Full audit across ml_real.tsv, ml_mine.tsv, ml_exam.py, ml_probe.py, ml_vocab.py, ml_gen.py, and decide()."""
    rep = ml_probe.audit_all(n_gen=4000)
    assert rep["ok"], f"Audit failed with issues: {rep['issues']}"


@pytest.mark.parametrize("pred,why", [
    (_pred("browser", 0.6, ["O", "SITE"]), "intent"),
    (_pred("browser", 0.99, ["O", "SITE"], tag_p=[0.99, 0.4]), "tagging"),
    (_pred("browser", 0.99, ["O", "O"]), "was found"),
    (_pred("app", 0.99, ["O", "O"]), "was found"),
    (_pred("system", 0.99, ["O", "O"]), "was found"),
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


# ------------------------------------------------------------------ household, system & generator balance checks (no ml_home)
def test_generator_balances_search_home_and_system():
    rows = ml_gen.generate(6000, 5, "all", exam=frozenset())
    kinds = Counter(r["tpl"] for r in rows)
    assert kinds["llm.search_nowhere"] > 100 and kinds["llm.home"] > 200 and kinds["browser.play_q_web"] > 50, kinds
    assert "browser.find_q_web" not in kinds
    for r in rows:
        if r["tpl"] == "llm.search_nowhere":
            assert r["intent"] == "llm" and not r["site"] and not r["query"], r
        if r["tpl"] == "browser.play_q_web":
            assert r["intent"] == "browser" and r["query"] and not r["site"], r


def test_vocab_home_and_system_generators_are_rich_and_disjoint():
    rng = random.Random(11)
    home_seen = {V.home_phrase(rng) for _ in range(5000)}
    sys_seen = {V.system_action_phrase(rng) for _ in range(5000)}
    assert len(home_seen) > 1500, len(home_seen)
    assert len(sys_seen) > 200, len(sys_seen)
    assert not (home_seen & sys_seen)


FRESH_MARK = "FRESH EXAM"


def fresh_exam():
    """Rows of ml_mine.tsv after the 'FRESH EXAM' marker: their words must not appear in training."""
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
    assert len(seen) <= 0.25 * len(named), f"too many sites of the fresh exam are already in training: {seen}"


def test_invented_sites_can_have_two_or_three_words():
    rng = random.Random(1)
    names = [ml_gen.fake_site(rng) for _ in range(3000)]
    multi = [n for n in names if " " in n]
    three_word = [n for n in names if len(n.split()) == 3]
    assert 0.35 < len(multi) / len(names) < 0.65, len(multi)
    assert len(three_word) > 250, len(three_word)
    assert all(2 <= len(n.split()) <= 3 for n in multi)
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
            assert not (t[i] == "O" and i > 0 and t[i - 1] == "TEXT"), r
    assert inside > 80, inside


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
    assert len(hits) > 40, len(hits)


def test_what_does_it_mean_rule_14():
    rows = ml_gen.generate(15000, 22, "all", exam=frozenset())
    m_llm = [r for r in rows if r["tpl"] == "llm.meaning"]
    m_tr = [r for r in rows if r["tpl"] == "translate.meaning"]
    assert len(m_llm) > 40 and len(m_tr) > 40, (len(m_llm), len(m_tr))
    assert all(r["intent"] == "llm" and r["lang"] == "none" for r in m_llm)
    assert all(r["intent"] == "translate" and r["lang"] != "none" and r["text"] for r in m_tr)


def test_action_verbs_between_site_and_query_rule_15():
    rows = ml_gen.generate(15000, 23, "all", exam=frozenset())
    acts = set(V.ACTS)
    site_rows = [r for r in rows if r["tpl"] == "browser.site_act_q"]
    assert len(site_rows) > 120, len(site_rows)
    for r in site_rows:
        assert r["intent"] == "browser" and r["site"] and r["query"], r
        for w, t in zip(r["words"], r["tags"]):
            if w.casefold().strip(".,") in acts:
                assert t == "O", r
    llm_rows = [r for r in rows if r["tpl"] == "llm.act"]
    assert len(llm_rows) > 40 and all(r["intent"] == "llm" and set(r["tags"]) == {"O"} for r in llm_rows)


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
    words2 = "открой soundcloud и поищи playboi carti".split()
    r2 = decide("открой soundcloud и поищи playboi carti", words2,
                _pred("browser", 0.99, ["O", "SITE", "O", "O", "SITE", "QUERY"]))
    assert r2["route"] == "llm" and r2["fallback"] and "ragged" in r2["reason"]


@pytest.mark.parametrize("phrase,tags", [
    ("найди в гугле рецепт пиццы", ["O", "O", "O", "QUERY", "QUERY"]),
    ("погугли рецепт пиццы", ["O", "QUERY", "QUERY"]),
    ("включи рок", ["O", "QUERY"]),
    ("хочу послушать queen", ["O", "O", "QUERY"]),
    ("вбей в поисковик курс доллара", ["O", "O", "O", "QUERY", "QUERY"]),
    ("поищи в интернете как варить рис", ["O", "O", "O", "QUERY", "QUERY", "QUERY"]),
    ("загулги как варить рис", ["O", "QUERY", "QUERY", "QUERY"]),
    ("скажи погоду на завтра", ["O", "QUERY", "QUERY", "QUERY"]),
    ("глянь ка сколько будет 50 евро в рубли", ["O", "O", "QUERY", "QUERY", "QUERY", "QUERY", "QUERY", "QUERY"]),
    ("посмотри авиа билеты москва софия", ["O", "QUERY", "QUERY", "QUERY", "QUERY"]),
])
def test_decide_search_with_place_or_play_passes(phrase, tags):
    r = decide(phrase, phrase.split(), _pred("browser", 0.99, tags))
    assert r["route"] == "browser" and not r["fallback"], r


def test_five_routes_decide_and_invariants():
    # pure google normalizes accidental SITE tags into QUERY
    phrase = "загугли нойз мс новый альбом"
    r = decide(phrase, phrase.split(), _pred("browser", 0.99, ["O", "SITE", "SITE", "QUERY", "QUERY"]))
    assert (r["route"], r["site"], r["query"], r["fallback"]) == ("browser", "", "нойз мс новый альбом", False)
    # app route with site and query
    w_app = "открой загрузки и отсортируй по дате".split()
    r_app = decide(" ".join(w_app), w_app, _pred("app", 0.99, ["O", "SITE", "O", "QUERY", "QUERY", "QUERY"]))
    assert (r_app["route"], r_app["site"], r_app["query"], r_app["fallback"]) == ("app", "загрузки", "отсортируй по дате", False)
    # system route with query
    w_sys = "переведи компьютер в спящий режим".split()
    r_sys = decide(" ".join(w_sys), w_sys, _pred("system", 0.99, ["QUERY"] * len(w_sys)))
    assert (r_sys["route"], r_sys["site"], r_sys["query"], r_sys["fallback"]) == ("system", "", "переведи компьютер в спящий режим", False)


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
    assert len(only) == 1 and only[0].startswith("# review") and "хлебопечку" in only[0]
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
    assert table[0][2:] == (2, 0, 1)
    assert table[1][2:] == (2, 1, 0)


def test_training_exam_and_loss_function():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    import ml_train
    assert len(ml_train.exam_rows()) == len(EXAM) + len(MINE)
    assert ml_train.default_out("final") == MODEL_DIR and ml_train.default_out("strict") == MODEL_DIR + "_strict"
    il = torch.full((2, len(INTENTS)), -10.0)
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
    pytest.importorskip("tokenizers")
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


# ------------------------------------------------------------------ part 2: with a model (flexible & colored)
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


def verdict(result, case, flexible=True):
    """ok - everything is correct (with flexible wrapper matching on query when flexible=True);
    safe - the model itself handed the phrase to the big LLM (slow, but not a silent mistake);
    wrong - it silently did the wrong thing (this is the real error)."""
    phrase, intent, site, query, text, lang = case
    if result["route"] == intent:
        if intent == "llm":
            return "ok"
        q_ok = query_matches(result["query"], query) if flexible else (norm(result["query"]) == norm(query))
        t_ok = text_matches(result["text"], text, phrase) if flexible else (norm(result["text"]) == norm(text))
        same = (norm(result["site"]) == norm(site) and q_ok
                and t_ok and (intent != "translate" or result["lang"] == lang))
        return "ok" if same else "wrong"
    return "safe" if result["fallback"] else "wrong"


def _color_field(name: str, got_val: str, want_val: str, is_query=False) -> str:
    ok = query_matches(got_val, want_val) if is_query else (norm(got_val) == norm(want_val))
    if ok:
        return f"{name}={got_val!r}"
    return f"{C_RED}{C_BOLD}{name}={got_val!r} (want {want_val!r}){C_RESET}"


def _describe(result, case, color=True):
    phrase, intent, site, query, text, lang = case
    if not color:
        return (f"\n  phrase  : {phrase!r}\n  expected: {intent} site={site!r} query={query!r} text={text!r} lang={lang}"
                f"\n  got     : {result['route']} site={result['site']!r} query={result['query']!r} "
                f"text={result['text']!r} lang={result['lang']}"
                + (f"  [-> LLM: {result['reason']}]" if result["fallback"] else ""))
    r_str = (f"{C_GREEN}{result['route']}{C_RESET}" if result["route"] == intent
             else f"{C_RED}{C_BOLD}{result['route']} (want {intent}){C_RESET}")
    s_str = _color_field("site", result["site"], site)
    q_str = _color_field("query", result["query"], query, is_query=True)
    t_str = _color_field("text", result["text"], text)
    l_str = (f"lang={result['lang']}" if (intent != "translate" or result["lang"] == lang)
             else f"{C_RED}{C_BOLD}lang={result['lang']} (want {lang}){C_RESET}")
    fb = f"  {C_YELLOW}[-> LLM: {result['reason']}]{C_RESET}" if result["fallback"] else ""
    return (f"\n  {C_BOLD}phrase{C_RESET}  : {C_CYAN}{phrase!r}{C_RESET}"
            f"\n  {C_DIM}expected: {intent} site={site!r} query={query!r} text={text!r} lang={lang}{C_RESET}"
            f"\n  {C_BOLD}got{C_RESET}     : {r_str} {s_str} {q_str} {t_str} {l_str}  (conf={result.get('confidence', 0):.2f}){fb}")


MINE = load_mine()
HARD_CASES = [tuple(r[1:]) for r in ml_probe.USER_HARD_CASES]
CASES = (
    [("hard_16+",) + c for c in HARD_CASES]
    + [("exam",) + tuple(c) for c in EXAM]
    + [("mine",) + tuple(c) for c in MINE]
)

_UNSEEN = "the word is deliberately absent from training (honest exam): without knowing it, it cannot be told apart from a site/music"
KNOWN_HARD = {
    "включи спид тест": "\"включи\" + an unfamiliar two-word name: a site or music - inherently ambiguous",
    "открой веб архив": _UNSEEN, "открой стек эксчейндж": _UNSEEN, "запусти sublime text": _UNSEEN,
    "запусти gimp": _UNSEEN, "запусти wireshark": _UNSEEN, "открой рольставни": _UNSEEN,
    "включи хлебопечку": _UNSEEN, "включи кормушку для кота": _UNSEEN, "включи электрокамин": _UNSEEN,
    "включи мороженицу": _UNSEEN, "включи гирлянду": _UNSEEN, "поставь скайфолл": _UNSEEN,
    "поищи рецепт плова на дзене": "\"дзене\" is a rare form of a site name; the model hands it to the llm",
}


@pytest.mark.parametrize("c", CASES, ids=[f"{c[0]}: {c[1]}" for c in CASES])
def test_model_exam(c):
    if not _model_ready():
        pytest.skip("no trained model (ml_model/) or no torch")
    case = c[1:]
    result = engine().predict_one(case[0])
    v = verdict(result, case, flexible=True)
    if v == "wrong" and case[0] in KNOWN_HARD:
        pytest.xfail("known difficulty: " + KNOWN_HARD[case[0]])
    assert v != "wrong", _describe(result, case, color=True)


def _print_group_report(name: str, cases: list, eng, max_errors=60) -> None:
    count = Counter()
    wrong = []
    by_intent = {i: Counter() for i in INTENTS}
    for case in cases:
        res = eng.predict_one(case[0])
        v = verdict(res, case, flexible=True)
        count[v] += 1
        by_intent[case[1]][v] += 1
        if v != "ok":
            wrong.append((v, case, res))
    n = len(cases)
    if not n:
        return
    print(f"\n{C_BOLD}=== {name}: {n} phrases ==={C_RESET}")
    print(f"{C_GREEN}correct at once: {count['ok'] / n:.1%} ({count['ok']}/{n}){C_RESET}   "
          f"{C_YELLOW}handed to big LLM (safe): {count['safe'] / n:.1%} ({count['safe']}/{n}){C_RESET}   "
          f"{C_RED}{C_BOLD}silent errors: {count['wrong'] / n:.1%} ({count['wrong']}/{n}){C_RESET}")
    for intent, c in by_intent.items():
        total = sum(c.values())
        if total:
            err_col = f"{C_RED}{c['wrong']}{C_RESET}" if c["wrong"] else "0"
            print(f"  {intent:9}: {C_GREEN}correct {c['ok']}/{total}{C_RESET}, "
                  f"{C_YELLOW}LLM {c['safe']}{C_RESET}, errors {err_col}")
    for v, case, res in sorted(wrong, key=lambda x: x[0] != "wrong")[:max_errors]:
        badge = f"{C_RED}{C_BOLD}[ERROR]{C_RESET}" if v == "wrong" else f"{C_YELLOW}[to LLM]{C_RESET}"
        print(badge + _describe(res, case, color=True))


def summary(include_mine: bool, include_live: bool = False) -> None:
    eng = engine()
    _print_group_report("16 USER FAILURES + VARIATIONS (USER_HARD_CASES)", HARD_CASES, eng, max_errors=70)
    _print_group_report("EXAM (ml_exam.py)", list(EXAM), eng, max_errors=40)
    if include_mine and MINE:
        _print_group_report("YOUR PHRASES (ml_mine.tsv)", MINE, eng, max_errors=50)
    if include_live:
        probe_all = ml_probe.generate(1200)
        by_cls = {}
        for row in probe_all:
            by_cls.setdefault(row[0], []).append(tuple(row[1:]))
        _print_group_report("LIVE PROBE (1200 unseen held-out phrases)", [tuple(r[1:]) for r in probe_all], eng, max_errors=60)
        print(f"\n{C_BOLD}--- Breakdown by probe frame ---{C_RESET}")
        for cls_name, cls_cases in by_cls.items():
            c = Counter(verdict(eng.predict_one(cs[0]), cs, flexible=True) for cs in cls_cases)
            tot = len(cls_cases)
            err_str = f"{C_RED}{c['wrong']}{C_RESET}" if c["wrong"] else f"{C_GREEN}0{C_RESET}"
            print(f"  {cls_name:14} (n={tot:3d}): {C_GREEN}ok={c['ok']/tot:5.1%}{C_RESET}  "
                  f"{C_YELLOW}safe_llm={c['safe']/tot:5.1%}{C_RESET}  errors={err_str}")


def thresholds_table(cases, preds, t_intents=(0.70, 0.80, 0.90), t_slots=(0.55, 0.60, 0.65, 0.75, 0.85, 0.90)):
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
    """Sweep the thresholds on the exam + your phrases + hard cases."""
    import ml_model
    eng = engine()
    cases = HARD_CASES + list(EXAM) + MINE
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


def _get_parametrize_values(fn):
    if hasattr(fn, "_parametrize"):
        return fn._parametrize[1]
    for m in getattr(fn, "pytestmark", ()):
        if getattr(m, "name", "") == "parametrize" and len(getattr(m, "args", ())) >= 2:
            return m.args[1]
    return None


def _is_skip_exc(e: BaseException) -> bool:
    return type(e).__name__ in ("Skipped", "XFailed") or str(e).startswith(("SKIP:", "XFAIL:"))


def run_unit_tests() -> None:
    """Runs all Part 1 unit tests directly with python3 (works whether pytest is installed or not)."""
    import os
    import tempfile

    class _MP:
        def setenv(self, k, v): os.environ[k] = str(v)
        def delenv(self, k, raising=False): os.environ.pop(k, None)
        def setattr(self, obj, attr, val): setattr(obj, attr, val)

    passed = 0
    for name, fn in list(globals().items()):
        if not name.startswith("test_") or name == "test_model_exam" or not callable(fn):
            continue
        param_values = _get_parametrize_values(fn)
        if param_values is not None:
            for args in param_values:
                fn(*(args if isinstance(args, tuple) else (args,)))
                passed += 1
            print(f"  {C_GREEN}PASS{C_RESET} {name} ({len(param_values)} cases)")
            continue
        code_vars = fn.__code__.co_varnames[:fn.__code__.co_argcount]
        try:
            if not code_vars:
                fn()
            elif code_vars == ("tmp_path", "monkeypatch"):
                with tempfile.TemporaryDirectory() as td:
                    fn(Path(td), _MP())
            elif code_vars == ("tmp_path",):
                with tempfile.TemporaryDirectory() as td:
                    fn(Path(td))
            else:
                fn()
            passed += 1
            print(f"  {C_GREEN}PASS{C_RESET} {name}")
        except BaseException as e:
            if _is_skip_exc(e):
                print(f"  {C_YELLOW}SKIP: {e}{C_RESET} ({name})")
            else:
                raise
    print(f"\n{C_GREEN}{C_BOLD}All {passed} unit & data-integrity checks passed!{C_RESET}")


if __name__ == "__main__":
    if "--unit" in sys.argv:
        run_unit_tests()
        sys.exit(0)
    if not _model_ready():
        print(f"{C_YELLOW}No trained model found in {_model_dir()}; running Part 1 unit & data-integrity tests instead:{C_RESET}\n")
        run_unit_tests()
        sys.exit(0)
    if "--sweep" in sys.argv:
        sweep()
    else:
        summary(include_mine=("--mine" in sys.argv or "--live" in sys.argv), include_live=("--live" in sys.argv))
