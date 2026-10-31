#!/usr/bin/env python3
"""Training-data generator for the first-layer parser (no torch needed).

Every example: the words of a phrase, a tag for every word (O/SITE/QUERY/TEXT), the intent (browser/translate/llm)
and the target language of a translation. Tags are assigned by construction of the template, so they cannot
disagree with the phrase.

How the data protects itself from self-deception
  * Vocabularies (sites, query topics) are split deterministically: ~15% of the words go to the "unseen" part
    and never reach training (strict mode) -> evaluation on sites and queries the model has never seen.
  * A few templates (HOLDOUT_TEMPLATES) are also removed from training in strict mode -> evaluation
    on sentence structures the model has never seen.
  * Exam phrases (ml_exam.py) never enter training in any form (exact match).

Usage:
    python3 ml_gen.py --out data --mode strict --n 60000     # for honest evaluation
    python3 ml_gen.py --out data --mode final  --n 90000     # final training (all vocabulary is used)
    python3 ml_gen.py --show 40                              # eyeball 40 random examples
"""
import argparse
import json
import random
import re
import zlib
from pathlib import Path

import ml_home as H
import ml_vocab as V
from ml_spec import INTENTS, LANGS, assemble, norm, read_tsv, tag_phrase

UNSEEN_PERCENT = 15
HOLDOUT_TEMPLATES = {"browser.site_find_q", "browser.prep_site_find_q", "translate.text_then_cmd", "llm.cmp"}
INTENT_WEIGHTS = {"browser": 0.46, "translate": 0.17, "llm": 0.37}
QUESTION_STARTS = ("как ", "что ", "кто ", "какая ", "какой ", "где ", "почему ", "сколько ", "когда ", "в чём ")


# ------------------------------------------------------------------ vocabularies and declension
def _hash(item: str) -> int:
    return zlib.crc32(item.encode("utf-8")) % 100


def _split(items, which):
    if which == "all":
        return list(items)
    want_unseen = which == "unseen"
    out = [x for x in items if (_hash(x) < UNSEEN_PERCENT) == want_unseen]
    return out or list(items)


_CONS = "бвгджзклмнпрстфхцчшщ"
_ROWS = ["йцукенгшщзхъ", "фывапролджэ", "ячсмитьбю"]
_NEIGH = {}
for _row in _ROWS:
    for _i, _ch in enumerate(_row):
        _NEIGH[_ch] = "".join(c for c in _row[max(0, _i - 1):_i + 2] if c != _ch) or _ch
_BANNED = ("хуй", "хуе", "пизд", "еба", "бля", "сос", "сра", "жоп", "муд", "гов", "хер", "пид")
_SYL = "ка ко ку ки ра ро ре ри ту то та ти ло ла ли ма мо ми на но ни ва во ви се со си де до ди бо ба бу зо за зи гу го га пи по па фи фо ля лю".split()
_SYL_LAT = "zo ra vi ko lu ta mi no se da ex ly qu ne ri bo ka fu".split()


def inflect(name: str, case: str) -> str:
    """Case of a site name (nom/prep/acc/gen/instr). Unknown and indeclinable names are left as they are."""
    if case == "nom" or name in V.INDECL or not re.search("[а-яё]", name):
        return name
    if " " in name:
        head, last = name.rsplit(" ", 1)
        if case == "prep" and last in V.SECOND_PREP:
            return head + " " + V.SECOND_PREP[last]
        if case == "acc" and last in V.SECOND_ACC:
            return head + " " + V.SECOND_ACC[last]
        return name
    w = name
    if case == "acc":
        if w.endswith("ия"):
            return w[:-2] + "ию"
        if w.endswith("а"):
            return w[:-1] + "у"
        if w.endswith("я"):
            return w[:-1] + "ю"
        return w
    if case == "prep":
        if w.endswith("ия"):
            return w[:-2] + "ии"
        if w[-1] in "ая":
            return w[:-1] + "е"
        if w.endswith(("ки", "ги", "хи")):
            return w[:-1] + "ах"
        if w[-1] in _CONS:
            return w + "е"
        return w
    if case == "gen":
        if w.endswith("ия"):
            return w[:-2] + "ии"
        if w.endswith("а"):
            return w[:-1] + "ы"
        if w[-1] in _CONS:
            return w + "а"
        return w
    if case == "instr":
        if w[-1] in _CONS:
            return w + "ом"
        return w
    return w


def _fake_word(rng: random.Random) -> str:
    for _ in range(20):
        if rng.random() < 0.6:
            w = "".join(rng.choice(_SYL) for _ in range(rng.choice([2, 3, 3, 4])))
            if rng.random() < 0.45:
                w += rng.choice(["хаб", "бокс", "лайн", "маркет", "тека", "нет", "ру", "тв", "про"])
        else:
            w = "".join(rng.choice(_SYL_LAT) for _ in range(rng.choice([2, 3])))
            if rng.random() < 0.5:
                w += rng.choice(["hub", "box", "ly", "tube", "flix", "soft", "net", "app"])
        if not any(b in w for b in _BANNED):
            return w
    return "кинобокс"


def fake_site(rng: random.Random) -> str:
    """An invented name: the model must recognise a site by its position in the phrase, not by a dictionary.
    Can be one word ("зоратека") or two ("нова хаб", "кедр маркет", "лумен бук"): an unfamiliar site
    after "открой / запусти / зайди на" is always one whole SITE, however many words it has."""
    if rng.random() < 0.60:
        return _fake_word(rng)
    for _ in range(30):
        if rng.random() < 0.15:
            name = _fake_word(rng) + " " + _fake_word(rng)
        else:
            first = rng.choice(V.FAKE_FIRST) if rng.random() < 0.5 else _fake_word(rng)
            lat = re.search("[a-z]", first) is not None
            tails = [t for t in V.FAKE_TAIL if (re.search("[a-z]", t) is not None) == lat]
            name = first + " " + rng.choice(tails)
        if not any(b in name for b in _BANNED):
            return name
    return "кино бокс"


_HARD_OPEN = {"открой", "запусти", "зайди", "перейди", "открывай", "запускай", "включи", "включай", "открыть", "запустить"}
_ALL_SITES = None


def real_sites(path=None) -> list:
    """Site names from your "открой X" phrases (ml_real.tsv): this is how "гит хаб", "пай пай" etc. get into the vocabulary."""
    path = path or REAL_PATH
    out, seen = [], {norm(x) for x in V.SITES_REAL}
    for phrase, intent, site, query, text, lang in read_tsv(path):
        words = phrase.split()
        if intent != "browser" or query or not site or not words or words[0].casefold() not in _HARD_OPEN:
            continue
        if site.split()[0].casefold() in {"на", "в", "через"} or norm(site) in seen:
            continue
        seen.add(norm(site))
        out.append(site)
    return out


def all_sites() -> list:
    global _ALL_SITES
    if _ALL_SITES is None:
        _ALL_SITES = list(V.SITES_REAL) + real_sites()
    return _ALL_SITES


class Pools:
    def __init__(self, which: str):
        self.which = which
        self.p = {k: _split(v, which) for k, v in V.POOLS.items()}
        self.t = dict(V.TUPLE_POOLS)
        self.sites = _split(all_sites(), which)

    def site(self, rng, real_only=False) -> str:
        if not real_only and rng.random() < 0.28:
            return fake_site(rng)
        return rng.choice(self.sites)

    def draw(self, rng, key: str):
        base = key.rstrip("0123456789")
        if base == "site":
            return self.site(rng)
        if base == "n":
            return str(rng.randint(2, 99))
        if base in self.t:
            return rng.choice(self.t[base])
        return rng.choice(self.p[base])


_PH = re.compile(r"\{([a-z]+[0-9]?)(?:_(nom|prep|acc|gen|instr))?\}")


def fill(rng, pools: Pools, pattern: str) -> str:
    cache = {}

    def rep(m):
        key, case = m.group(1), m.group(2) or "nom"
        if key not in cache:
            cache[key] = pools.draw(rng, key)
        v = cache[key]
        if isinstance(v, tuple):
            return v[V.CASES.index(case)] if case in V.CASES else v[0]
        if key.startswith("site"):
            return inflect(v, case)
        return v

    return _PH.sub(rep, pattern)


def make_query(rng, pools: Pools, questions=True, hard=False) -> str:
    if hard:
        return fill(rng, pools, rng.choice(V.QUERY_HARD))
    if questions and rng.random() < 0.34:
        q = fill(rng, pools, rng.choice(V.QUERY_QUEST))
    else:
        q = fill(rng, pools, rng.choice(V.QUERY_NOUN))
        if rng.random() < 0.10 and "за " not in q:
            q += " за " + rng.choice(pools.p["time"])
    return q


def site_form(rng, pools: Pools, kind: str, real_only=False) -> str:
    """A site name in the required form: kind = open (nominative/accusative) or prep ("на X")."""
    name = pools.site(rng, real_only)
    if kind == "open":
        return inflect(name, rng.choice(["nom", "nom", "acc"]))
    return inflect(name, "prep") if rng.random() < 0.78 else name


# ------------------------------------------------------------------ templates
def O(t): return (t, "O")
def S(t): return (t, "SITE")
def Q(t): return (t, "QUERY")
def T(t): return (t, "TEXT")


def tpl_open_only(rng, P):
    soft = rng.random() < 0.2
    return [O(rng.choice(V.OPEN_SOFT if soft else V.OPEN)), S(site_form(rng, P, "open", soft))], "none"


def tpl_open_find_and(rng, P):
    return [O(rng.choice(V.OPEN)), S(site_form(rng, P, "open")), O(rng.choice(V.AND_VERBS)),
            Q(make_query(rng, P))], "none"


def tpl_open_find_noand(rng, P):
    site = site_form(rng, P, "open")
    if rng.random() < 0.3:
        site += ","
    return [O(rng.choice(V.OPEN)), S(site), O(rng.choice(V.FIND)), Q(make_query(rng, P))], "none"


def tpl_find_q_on_site(rng, P):
    prep = rng.choices(V.SITE_PREPS, weights=[45, 40, 6, 5, 4])[0]
    return [O(rng.choice(V.FIND)), Q(make_query(rng, P, questions=False)), O(prep),
            S(site_form(rng, P, "prep"))], "none"


def tpl_find_on_site_q(rng, P):
    prep = rng.choices(V.SITE_PREPS, weights=[45, 40, 6, 5, 4])[0]
    return [O(rng.choice(V.FIND)), O(prep), S(site_form(rng, P, "prep")), Q(make_query(rng, P))], "none"


def tpl_site_find_q(rng, P):
    return [S(site_form(rng, P, "open")), O(rng.choice(V.FIND)), Q(make_query(rng, P))], "none"


def tpl_prep_site_find_q(rng, P):
    return [O(rng.choice(["в", "на"])), S(site_form(rng, P, "prep")), O(rng.choice(V.FIND)),
            Q(make_query(rng, P))], "none"


def _conflicts(site: str, query: str) -> bool:
    """"включи яндекс музыку": one two-word site, not the site "яндекс" + the query "музыку"."""
    first = site.split()[0] if site else ""
    for name in all_sites():
        parts = name.split()
        if len(parts) > 1 and parts[0] == first and len(site.split()) == 1:
            tails = {parts[-1], inflect(name, "acc").split()[-1], inflect(name, "prep").split()[-1]}
            if query.split()[0] in tails:
                return True
    return False


def tpl_open_q_on_site(rng, P):
    prep = rng.choices(V.SITE_PREPS, weights=[55, 35, 5, 3, 2])[0]
    return [O(rng.choice(V.OPEN + V.OPEN_SOFT)), Q(make_query(rng, P, questions=False)), O(prep),
            S(site_form(rng, P, "prep"))], "none"


def tpl_open_on_site_q(rng, P):
    return [O(rng.choice(V.OPEN)), O(rng.choice(["на", "в"])), S(site_form(rng, P, "prep")),
            Q(make_query(rng, P, questions=False))], "none"


def tpl_open_site_q_noverb(rng, P):
    """"включи мне ютуб канал exile", "открой leet code задачи": a site and a query without the word "найди"."""
    for _ in range(10):
        site = site_form(rng, P, "open", real_only=True)
        query = make_query(rng, P, questions=False)
        if not _conflicts(site, query):
            return [O(rng.choice(V.OPEN + V.OPEN_SOFT)), S(site), Q(query)], "none"
    return [O(rng.choice(V.OPEN)), S(site_form(rng, P, "open"))], "none"


def tpl_open_site_suffix(rng, P):
    return [O(rng.choice(V.OPEN)), S(site_form(rng, P, "open")), O(rng.choice(V.BROWSER_SUFFIX))], "none"


def tpl_site_act_q(rng, P):
    """"на вольфраме реши уравнение", "открой гугл и проверь пинг", "построй график на десмосе": the action verb is O."""
    act = rng.choice(list(V.ACTS))
    q = rng.choice(V.ACTS[act])
    prep = rng.choices(["на", "в", "через"], weights=[50, 45, 5])[0]
    form = rng.randrange(5)
    if form == 0:
        return [O(prep), S(site_form(rng, P, "prep")), O(act), Q(q)], "none"
    if form == 1:
        return [S(site_form(rng, P, "open")), O(act), Q(q)], "none"
    if form == 2:
        return [O(rng.choice(V.OPEN)), S(site_form(rng, P, "open")), O("и " + act), Q(q)], "none"
    if form == 3:
        return [O(act), Q(q), O(prep), S(site_form(rng, P, "prep"))], "none"
    return [O(rng.choice(V.OPEN)), S(site_form(rng, P, "open")), O(act), Q(q)], "none"


def tpl_llm_act(rng, P):
    """"реши уравнение", "закажи пиццу" without a site and without a place to search -> llm (the browser cannot solve this by itself)."""
    act = rng.choice(list(V.ACTS))
    return [O(f"{act} {rng.choice(['мне ', '', '']) }{rng.choice(V.ACTS[act])}".replace("  ", " "))], "none"


def tpl_llm_meaning(rng, P):
    """"что значит слово X" without a language is an explanation request (llm), not a translation (rule 14)."""
    word = rng.choice(V.TR_WORDS) if rng.random() < 0.25 else rng.choice(P.p["foreignw"])
    return [O(f"{rng.choice(V.MEANING_CMD)} {word}")], "none"


SEARCH_VERBS = [v for v in V.FIND if v not in H.PLAY_FIND] + H.SEARCH_EXTRA


def tpl_play_q_web(rng, P):
    """"включи рок", "поставь дюну", "хочу посмотреть фильм про пиратов": media without a platform -> Google search."""
    if rng.random() < 0.55:
        verb, pattern = rng.choice(H.LISTEN_VERBS), rng.choice(H.PLAY_LISTEN)
    else:
        verb, pattern = rng.choice(H.WATCH_VERBS), rng.choice(H.PLAY_WATCH)
    return [O(verb), Q(fill(rng, P, pattern))], "none"


def tpl_search_nowhere(rng, P):
    """"найди X" WITHOUT a place to search (no site, no google, no internet) -> llm (rule 11)."""
    hard = rng.random() < 0.22
    return [O(rng.choice(SEARCH_VERBS)), O(make_query(rng, P, hard=hard))], "none"


def tpl_llm_home(rng, P):
    return [O(H.home_phrase(rng))], "none"


def tpl_llm_system(rng, P):
    """Computer system tasks (calculator, folders, files, windows, screenshot, shutdown) -> llm, not the browser."""
    return [O(H.system_phrase(rng))], "none"


def tpl_llm_howto(rng, P):
    return [O(rng.choice(H.HOWTO))], "none"


def tpl_google_q(rng, P):
    hard = rng.random() < 0.12
    return [O(rng.choice(V.GOOGLE)), Q(make_query(rng, P, hard=hard))], "none"


def _lang_phrase(rng, lang: str, style: str) -> str:
    acc, prep, adv = V.LANG_FORMS[lang]
    if style == "to":
        return rng.choice([f"на {acc}", f"на {acc}", f"на {prep}", f"на {acc} язык"])
    return rng.choice([f"на {prep}", adv, adv.replace("по-", "по ") if adv.startswith("по-") else adv,
                       f"на {prep} языке"])


def _tr_text(rng, P, lang: str) -> str:
    r = rng.random()
    if r < 0.40:
        return rng.choice(V.TR_PHRASES)
    if r < 0.46:
        return rng.choice(V.TR_WORDS)
    if r < 0.56:
        return rng.choice(V.TR_FOREIGN)
    if r < 0.64:
        return rng.choice(V.TR_COMMANDISH)
    if r < 0.72:
        return rng.choice(V.TR_POLITE)            # "два чая пожалуйста": "пожалуйста" after the text is part of the text (rule 13)
    if r < 0.78:
        return _speak_phrase(rng)                 # "я не говорю по болгарски": "по X" inside the text is TEXT
    return make_query(rng, P)


def _speak_phrase(rng) -> str:
    lang = rng.choice([k for k in LANGS[1:] if V.LANG_FORMS[k][2].startswith("по-")])
    adv = V.LANG_FORMS[lang][2]
    if rng.random() < 0.6:
        adv = adv.replace("по-", "по ")
    return rng.choice([f"я не говорю {adv}", f"вы говорите {adv}", f"ты понимаешь {adv}", f"он говорит {adv}",
                       f"говорите ли вы {adv}", f"я хорошо говорю {adv}", f"здесь говорят {adv}", f"мы разговариваем {adv}"])


def _tr_lang(rng) -> str:
    if rng.random() < 0.30:
        return "en"
    return rng.choice(LANGS[1:])


def tpl_tr_cmd_lang_text(rng, P):
    lang = _tr_lang(rng)
    return [O(rng.choice(V.TR_CMD)), O(_lang_phrase(rng, lang, "to")), T(_tr_text(rng, P, lang))], lang


def tpl_tr_cmd_text_lang(rng, P):
    lang = _tr_lang(rng)
    parts = [O(rng.choice(V.TR_CMD)), T(_tr_text(rng, P, lang))]
    if rng.random() < 0.2:
        parts.append(O("с русского"))
    parts.append(O(_lang_phrase(rng, lang, "to")))
    return parts, lang


def tpl_tr_word_prefix(rng, P):
    lang = _tr_lang(rng)
    noun = rng.choice(["слово", "фразу", "предложение", "текст", "это", "выражение"])
    if rng.random() < 0.5:
        return [O(rng.choice(V.TR_CMD)), O(noun), T(_tr_text(rng, P, lang)), O(_lang_phrase(rng, lang, "to"))], lang
    return [O(rng.choice(V.TR_CMD)), O(_lang_phrase(rng, lang, "to")), O(noun), T(_tr_text(rng, P, lang))], lang


def tpl_tr_how_to_say(rng, P):
    lang = _tr_lang(rng)
    verb = rng.choice(["как будет", "как сказать", "как будет по русски", "как переводится", "как звучит"])
    return [O(verb), T(_tr_text(rng, P, lang)), O(_lang_phrase(rng, lang, "adv"))], lang


def tpl_tr_how_adv(rng, P):
    lang = _tr_lang(rng)
    adv = V.LANG_FORMS[lang][2]
    return [O("как"), O(adv.replace("по-", "по ") if adv.startswith("по-") and rng.random() < 0.3 else adv),
            T(_tr_text(rng, P, lang))], lang


def tpl_tr_bare(rng, P):
    return [O(rng.choice(V.TR_CMD)), T(_tr_text(rng, P, "none"))], "none"


def tpl_tr_colon(rng, P):
    lang = _tr_lang(rng)
    return [O(rng.choice(V.TR_CMD)), O(_lang_phrase(rng, lang, "to") + ":"), T(_tr_text(rng, P, lang))], lang


def tpl_tr_via_translator(rng, P):
    lang = _tr_lang(rng)
    pre = rng.choice(["открой переводчик и", "запусти переводчик и", "в переводчике", "через переводчик",
                      "открой гугл переводчик и"])
    return [O(pre), O(rng.choice(V.TR_CMD)), O(_lang_phrase(rng, lang, "to")), T(_tr_text(rng, P, lang))], lang


def tpl_tr_text_then_cmd(rng, P):
    lang = _tr_lang(rng)
    return [T(_tr_text(rng, P, lang)), O(rng.choice(V.TR_CMD)), O(_lang_phrase(rng, lang, "to"))], lang


def tpl_tr_meaning(rng, P):
    """"что значит слово X на русском": the language is named, so it is a translation (rule 14)."""
    lang = "ru" if rng.random() < 0.75 else rng.choice(LANGS[1:])
    word = rng.choice(P.p["foreignw"]) if lang == "ru" else rng.choice(V.TR_WORDS)
    return [O(rng.choice(V.MEANING_CMD)), T(word), O(_lang_phrase(rng, lang, "adv"))], lang


def tpl_llm_question(rng, P):
    return [O(fill(rng, P, rng.choice(V.QUERY_QUEST)))], "none"


def tpl_llm_hard(rng, P):
    return [O(make_query(rng, P, hard=True))], "none"


def tpl_llm_only(rng, P):
    pattern = rng.choice(V.LLM_ONLY)
    return [O(fill(rng, P, pattern))], "none"


def tpl_llm_cmp(rng, P):
    pre = rng.choice(["в чём разница между", "сравни", "что лучше", "объясни разницу между", "чем отличается"])
    return [O(f"{pre} {rng.choice(P.p['cmp'])}")], "none"


TEMPLATES = [
    # (name, intent, weight, function)
    ("browser.open_only", "browser", 7, tpl_open_only),
    ("browser.open_find_and", "browser", 12, tpl_open_find_and),
    ("browser.open_find_noand", "browser", 4, tpl_open_find_noand),
    ("browser.find_q_on_site", "browser", 10, tpl_find_q_on_site),
    ("browser.find_on_site_q", "browser", 8, tpl_find_on_site_q),
    ("browser.site_find_q", "browser", 3, tpl_site_find_q),
    ("browser.prep_site_find_q", "browser", 3, tpl_prep_site_find_q),
    ("browser.open_q_on_site", "browser", 7, tpl_open_q_on_site),
    ("browser.open_on_site_q", "browser", 4, tpl_open_on_site_q),
    ("browser.open_site_q_noverb", "browser", 6, tpl_open_site_q_noverb),
    ("browser.open_site_suffix", "browser", 1.5, tpl_open_site_suffix),
    ("browser.play_q_web", "browser", 5, tpl_play_q_web),
    ("browser.google_q", "browser", 8, tpl_google_q),
    ("browser.site_act_q", "browser", 6, tpl_site_act_q),
    ("translate.cmd_lang_text", "translate", 10, tpl_tr_cmd_lang_text),
    ("translate.cmd_text_lang", "translate", 8, tpl_tr_cmd_text_lang),
    ("translate.word_prefix", "translate", 5, tpl_tr_word_prefix),
    ("translate.how_to_say", "translate", 5, tpl_tr_how_to_say),
    ("translate.how_adv", "translate", 3, tpl_tr_how_adv),
    ("translate.bare", "translate", 3, tpl_tr_bare),
    ("translate.colon", "translate", 2, tpl_tr_colon),
    ("translate.via_translator", "translate", 3, tpl_tr_via_translator),
    ("translate.text_then_cmd", "translate", 3, tpl_tr_text_then_cmd),
    ("translate.meaning", "translate", 3, tpl_tr_meaning),
    ("llm.question", "llm", 8, tpl_llm_question),
    ("llm.hard", "llm", 6, tpl_llm_hard),
    ("llm.only", "llm", 14, tpl_llm_only),
    ("llm.cmp", "llm", 2, tpl_llm_cmp),
    ("llm.search_nowhere", "llm", 9, tpl_search_nowhere),
    ("llm.home", "llm", 16, tpl_llm_home),
    ("llm.howto", "llm", 4, tpl_llm_howto),
    ("llm.act", "llm", 4, tpl_llm_act),
    ("llm.meaning", "llm", 3, tpl_llm_meaning),
    ("llm.system", "llm", 10, tpl_llm_system),
]


# ------------------------------------------------------------------ noise: speech fillers, typos, case
def typo(rng, w: str) -> str:
    if len(w) < 4 or any(ch.isdigit() for ch in w) or not w[1:-1]:
        return w
    i = rng.randrange(1, len(w) - 1)
    op = rng.random()
    if op < 0.35:
        return w[:i] + w[i + 1:]
    if op < 0.6:
        return w[:i] + w[i] + w[i:]
    if op < 0.85:
        return w[:i - 1] + w[i] + w[i - 1] + w[i + 1:]
    return w[:i] + rng.choice(_NEIGH.get(w[i], w[i])) + w[i + 1:]


TYPO_RATE = {"O": 0.03, "SITE": 0.045, "QUERY": 0.025, "TEXT": 0.02}


def _polite(text: str) -> bool:
    return any(w.strip(",.") in ("пожалуйста", "плиз") for w in text.split())


def finalize(rng, parts, intent, lang, tpl_name):
    parts = list(parts)
    first_tag, last_tag = parts[0][1], parts[-1][1]
    if rng.random() < 0.22:
        pre = rng.choice(V.PRE)
        if intent == "translate" and first_tag == "TEXT":
            pre = rng.choice([p for p in V.PRE if not _polite(p)])    # "пожалуйста" before the text is ambiguous, so we do not generate it
        parts.insert(0, O(pre))
    if rng.random() < 0.12:
        if intent == "translate" and last_tag == "TEXT":
            # rule 13: "пожалуйста" after the translated text is part of the text; no other tails are added here
            if rng.random() < 0.6:
                parts.append(T(rng.choice(["пожалуйста", "плиз"])))
        else:
            parts.append(O(rng.choice(V.POST)))
    words, tags = [], []
    for text, tag in parts:
        for w in text.split():
            words.append(w)
            tags.append("O" if intent == "llm" else tag)
    # typos
    words = [typo(rng, w) if rng.random() < TYPO_RATE[t] else w for w, t in zip(words, tags)]
    # letter case (ASR usually gives lower case, but not always)
    r = rng.random()
    if r < 0.07:
        words[0] = words[0][:1].upper() + words[0][1:]
    elif r < 0.12:
        words = [w.upper() for w in words]
    if rng.random() < 0.5:
        words = [w.replace("ё", "е") for w in words]
    # trailing punctuation
    if rng.random() < 0.08:
        words[-1] += "."
    elif intent == "llm" and rng.random() < 0.25:
        words[-1] += "?"
    gold = assemble(words, tags) if intent != "llm" else {"site": "", "query": "", "text": ""}
    if intent == "browser":
        gold["text"] = ""
    return {"words": words, "tags": tags, "intent": intent, "lang": lang if intent == "translate" else "none",
            "tpl": tpl_name, "site": gold["site"], "query": gold["query"], "text": gold["text"]}


def generate(n, seed, which, templates=None, exam=frozenset()):
    """n examples. `templates` is the set of allowed templates (None = all)."""
    rng = random.Random(seed)
    pools = Pools(which)
    by_intent = {i: [t for t in TEMPLATES if t[1] == i and (templates is None or t[0] in templates)]
                 for i in INTENTS}
    intents = [i for i in INTENTS if by_intent[i]]
    weights = [INTENT_WEIGHTS[i] for i in intents]
    seen, out, guard = set(), [], 0
    while len(out) < n and guard < n * 30:
        guard += 1
        intent = rng.choices(intents, weights=weights)[0]
        pool = by_intent[intent]
        name, _, _, fn = rng.choices(pool, weights=[t[2] for t in pool])[0]
        parts, lang = fn(rng, pools)
        sample = finalize(rng, parts, intent, lang, name)
        key = norm(" ".join(sample["words"]))
        if not key or key in seen or key in exam:
            continue
        seen.add(key)
        out.append(sample)
    return out


REAL_PATH = Path(__file__).with_name("ml_real.tsv")
MINE_PATH = Path(__file__).with_name("ml_mine.tsv")
_SITE_PREPS = {"на", "в", "через"}
_SOFT = {"включи", "включай", "поставь"}


def real_rows(path=REAL_PATH):
    """Your real phrases (ml_real.tsv) -> training examples as they are."""
    out = []
    for phrase, intent, site, query, text, lang in read_tsv(path):
        words, tags = tag_phrase(phrase, site, query, text)
        if intent == "llm":
            tags = ["O"] * len(words)
        out.append({"words": words, "tags": tags, "intent": intent, "lang": lang if intent == "translate" else "none",
                    "tpl": "real", "site": site, "query": query, "text": text})
    return out


def augment(row, rng, pools):
    """Takes one of your real phrases and substitutes other sites and queries: the structure is yours, the filling is new."""
    if row["intent"] == "llm":
        return None
    groups = []
    for w, t in zip(row["words"], row["tags"]):
        if groups and groups[-1][1] == t:
            groups[-1][0].append(w)
        else:
            groups.append(([w], t))
    parts, prev_o = [], ""
    soft = any(w.casefold() in _SOFT for g, t in groups if t == "O" for w in g)
    for words, tag in groups:
        text = " ".join(words)
        if tag == "SITE":
            kind = "prep" if prev_o.split()[-1:] and prev_o.split()[-1].casefold() in _SITE_PREPS else "open"
            text = site_form(rng, pools, kind, real_only=soft)
        elif tag == "QUERY":
            last_o = prev_o.split()[-1].casefold() if prev_o else ""
            if last_o in V.ACTS:
                text = rng.choice(V.ACTS[last_o])
            else:
                asked = text.casefold().startswith(QUESTION_STARTS)
                text = make_query(rng, pools, questions=asked)
        elif tag == "TEXT":
            text = _tr_text(rng, pools, row["lang"])
        else:
            prev_o = text
        parts.append((text, tag))
    return finalize(rng, parts, row["intent"], row["lang"], "real_aug")


def _write(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _exam_keys():
    try:
        from ml_exam import EXAM
    except ImportError:
        return frozenset()
    mine = [r[0] for r in read_tsv(MINE_PATH)]
    return frozenset(norm(row[0]) for row in EXAM) | frozenset(norm(p) for p in mine)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).with_name("data")))
    ap.add_argument("--mode", choices=["strict", "final"], default="strict")
    ap.add_argument("--n", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--show", type=int, default=0, help="show N random examples and exit")
    ap.add_argument("--real-aug", type=int, default=40, help="how many variants to make from each of your phrases")
    ap.add_argument("--real-copies", type=int, default=3, help="how many times to add your phrase unchanged")
    args = ap.parse_args()
    exam = _exam_keys()
    if args.show:
        for s in generate(args.show, args.seed, "all", exam=exam):
            marked = " ".join(f"{w}/{t}" if t != "O" else w for w, t in zip(s["words"], s["tags"]))
            print(f"[{s['intent']:9}|{s['lang']:4}|{s['tpl']}] {marked}")
        return
    out = Path(args.out)
    real = real_rows()
    aug_pools = Pools("train" if args.mode == "strict" else "all")
    arng = random.Random(args.seed + 100)

    def with_real(rows):
        seen = {norm(" ".join(r["words"])) for r in rows}
        extra = real * args.real_copies
        for r in real:
            for _ in range(args.real_aug):
                a = augment(r, arng, aug_pools)
                key = norm(" ".join(a["words"])) if a else ""
                if a and key and key not in seen and key not in exam:
                    seen.add(key)
                    extra.append(a)
        rows = rows + extra
        random.Random(args.seed).shuffle(rows)
        print(f"  from your phrases: {len(real)} samples -> {len(extra)} training examples")
        return rows

    if args.mode == "strict":
        keep = {t[0] for t in TEMPLATES} - HOLDOUT_TEMPLATES
        n_val = max(2000, args.n // 20)
        _write(out / "train.jsonl", with_real(generate(args.n, args.seed, "train", keep, exam)))
        _write(out / "val_sites.jsonl", generate(n_val, args.seed + 1, "unseen", keep, exam))
        _write(out / "val_templates.jsonl", generate(n_val, args.seed + 2, "train", HOLDOUT_TEMPLATES, exam))
        _write(out / "val_both.jsonl", generate(n_val, args.seed + 3, "unseen", HOLDOUT_TEMPLATES, exam))
    else:
        _write(out / "train_final.jsonl", with_real(generate(args.n, args.seed, "all", None, exam)))
        _write(out / "val_final.jsonl", generate(3000, args.seed + 9, "all", None, exam))
    print("done:", out.resolve())


if __name__ == "__main__":
    main()
