#!/usr/bin/env python3
"""Training-data generator for the first-layer parser (no torch needed).

Every example: the words of a phrase, a tag for every word (O/SITE/QUERY/TEXT), the intent
(browser/app/system/translate/llm) and the target language of a translation. Tags are assigned by
construction of the template, so they cannot disagree with the phrase.

How the data protects itself from self-deception
  * Vocabularies (sites, query topics) are split deterministically: ~15% of the words go to the "unseen" part
    and never reach training (strict mode) -> evaluation on sites and queries the model has never seen.
  * A few templates (HOLDOUT_TEMPLATES) are also removed from training in strict mode -> evaluation
    on sentence structures the model has never seen.
  * Exam and probe phrases (ml_exam.py, ml_mine.tsv, ml_probe.py) never enter training in any form.

Usage:
    python3 ml_gen.py --out data --mode strict --n 60000     # for honest evaluation
    python3 ml_gen.py --out data --mode final  --n 100000    # final training (all vocabulary is used)
    python3 ml_gen.py --show 40                              # eyeball 40 random examples
"""
import argparse
import json
import random
import re
import zlib
from pathlib import Path

import ml_vocab as V
from ml_spec import INTENTS, LANGS, assemble, norm, read_tsv, tag_phrase

UNSEEN_PERCENT = 15
HOLDOUT_TEMPLATES = {"browser.site_find_q", "browser.prep_site_find_q", "translate.text_then_cmd", "llm.cmp"}
INTENT_WEIGHTS = {"browser": 0.36, "app": 0.17, "system": 0.13, "translate": 0.15, "llm": 0.19}
QUESTION_STARTS = ("как ", "что ", "кто ", "какая ", "какой ", "где ", "почему ", "сколько ", "когда ", "в чём ")
_GOOGLE_NAMES = {"гугл", "гугле", "google", "гугл поиск", "google search"}


# ------------------------------------------------------------------ vocabularies and declension
def _hash(item: str) -> int:
    return zlib.crc32(item.encode("utf-8")) % 100


def _split(items, which):
    if which == "all":
        return list(items)
    want_unseen = which == "unseen"
    out = [x for x in items if (_hash(x) < UNSEEN_PERCENT) == want_unseen]
    return out or list(items)


_CONS = "бвгджзйклмнпрстфхцчшщ"
_ROWS = ["йцукенгшщзхъ", "фывапролджэ", "ячсмитьбю"]
_NEIGH = {}
for _row in _ROWS:
    for _i, _ch in enumerate(_row):
        _NEIGH[_ch] = "".join(c for c in _row[max(0, _i - 1):_i + 2] if c != _ch) or _ch
_BANNED = ("хуй", "хуе", "пизд", "бля", "сос", "сра", "жоп", "муд", "гов", "хер", "пид")
_SYL = "ка ко ку ки ра ро ре ри ту то та ти ло ла ли ма мо ми на но ни ва во ви се со си де до ди бо ба бу зо за зи гу го га пи по па фи фо ля лю".split()
_SYL_LAT = "zo ra vi ko lu ta mi no se da ex ly qu ne ri bo ka fu".split()


def _inflect_adj_prep(w: str) -> str:
    if w.endswith(("кий", "гий", "хий")):
        return w[:-2] + "ом"
    if w.endswith(("ий", "жий", "ший", "щий", "чий")):
        return w[:-2] + "ем"
    if w.endswith(("ый", "ой")):
        return w[:-2] + "ом"
    if w.endswith("яя"):
        return w[:-2] + "ей"
    if w.endswith("ая"):
        return w[:-2] + "ой"
    return w


def _inflect_adj_acc(w: str) -> str:
    if w.endswith("яя"):
        return w[:-2] + "юю"
    if w.endswith("ая"):
        return w[:-2] + "ую"
    return w


def inflect(name: str, case: str) -> str:
    """Case of a site or app name (nom/prep/acc/gen/instr). Unknown and indeclinable names are left as they are."""
    if case == "nom" or name in V.INDECL or not re.search("[а-яё]", name):
        return name
    if " " in name:
        parts = name.split()
        if len(parts) == 2:
            w0, w1 = parts
            if case == "prep":
                if w1 in V.SECOND_PREP:
                    return _inflect_adj_prep(w0) + " " + V.SECOND_PREP[w1]
                if w0 in V.SECOND_PREP:
                    return V.SECOND_PREP[w0] + " " + w1
            if case == "acc":
                if w1 in V.SECOND_ACC:
                    return _inflect_adj_acc(w0) + " " + V.SECOND_ACC[w1]
                if w0 in V.SECOND_ACC:
                    return V.SECOND_ACC[w0] + " " + w1
            return name
        if len(parts) >= 3:
            w0, w1 = parts[0], parts[1]
            tail = " ".join(parts[2:])
            if case == "prep":
                if w1 in V.SECOND_PREP:
                    return f"{_inflect_adj_prep(w0)} {V.SECOND_PREP[w1]} {tail}"
                if w0 in V.SECOND_PREP:
                    return f"{V.SECOND_PREP[w0]} {w1} {tail}"
            if case == "acc":
                if w1 in V.SECOND_ACC:
                    return f"{_inflect_adj_acc(w0)} {V.SECOND_ACC[w1]} {tail}"
                if w0 in V.SECOND_ACC:
                    return f"{V.SECOND_ACC[w0]} {w1} {tail}"
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
        if w.endswith("й"):
            return w[:-1] + "е"
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
        if w.endswith("й"):
            return w[:-1] + "я"
        if w.endswith("а"):
            return w[:-1] + "ы"
        if w[-1] in _CONS:
            return w + "а"
        return w
    if case == "instr":
        if w.endswith("й"):
            return w[:-1] + "ем"
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
    Can be one word ("зоратека"), two ("нова хаб", "кедр маркет"), or three ("фонд развития региона",
    "единый портал студентов"): an unfamiliar site after "открой / запусти / зайди на" is one whole SITE."""
    apps = app_norms()
    for _ in range(30):
        r = rng.random()
        if r < 0.52:
            name = _fake_word(rng)
        elif r < 0.84:
            if rng.random() < 0.15:
                name = _fake_word(rng) + " " + _fake_word(rng)
            else:
                first = rng.choice(V.FAKE_FIRST) if rng.random() < 0.5 else _fake_word(rng)
                lat = re.search("[a-z]", first) is not None
                tails = [t for t in V.FAKE_TAIL if (re.search("[a-z]", t) is not None) == lat]
                name = first + " " + rng.choice(tails)
        else:
            first = rng.choice([w for w in V.FAKE_FIRST if not re.search("[a-z]", w)])
            mid = rng.choice(["фонд", "портал", "центр", "реестр", "сервис", "каталог", "институт", "кабинет", "клуб"])
            tail = rng.choice(V.FAKE_GEN_TAIL)
            name = f"{first} {mid} {tail}"
        if not any(b in name for b in _BANNED) and norm(name) not in apps:
            return name
    return "кино бокс"


_HARD_OPEN = {"открой", "запусти", "зайди", "перейди", "открывай", "запускай", "включи", "включай", "открыть", "запустить"}
_WEB_EXPLICIT_MARKERS = ("в вебе", "в браузере", "на сайте", "веб верси", "через браузер", "официальный сайт", "главный сайт", "в хроме", "гугл хром")
_ALL_SITES = None
_APP_NORMS = None


def app_norms() -> set:
    """Normalized names (and declensions) of all desktop/hybrid/system apps and folders so they never leak into website pools."""
    global _APP_NORMS
    if _APP_NORMS is None:
        s = set()
        for a in list(V.APP_HYBRID) + list(V.APP_DESKTOP) + list(V.SYS_APPS) + list(V.SYS_FOLDERS) + list(V.SYS_PLACES):
            s.add(norm(a))
            for c in ("acc", "prep", "gen", "instr"):
                s.add(norm(inflect(a, c)))
        _APP_NORMS = s
    return _APP_NORMS


def real_sites(path=None) -> list:
    """Site names from your "открой X" phrases (ml_real.tsv): this is how "гит хаб", "пай пай" etc. get into the vocabulary.
    Hybrid/desktop apps opened with 'в вебе / в браузере / официальный сайт' are excluded so they don't leak into bare browser templates."""
    path = path or REAL_PATH
    apps = app_norms()
    out, seen = [], {norm(x) for x in V.SITES_REAL}
    for phrase, intent, site, query, text, lang in read_tsv(path):
        words = phrase.split()
        if intent != "browser" or query or not site or not words or words[0].casefold() not in _HARD_OPEN:
            continue
        low = norm(phrase)
        ns = norm(site)
        if ns in apps or any(m in low for m in _WEB_EXPLICIT_MARKERS):
            continue
        if site.split()[0].casefold() in {"на", "в", "через"} or ns in seen:
            continue
        seen.add(ns)
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
        self.core_sites = _split(V.SITES_REAL, which)
        self.soft_sites = _split(V.SOFT_SITES, which)
        self.non_google_sites = [s for s in self.sites if norm(s) not in _GOOGLE_NAMES]
        self.non_google_core = [s for s in self.core_sites if norm(s) not in _GOOGLE_NAMES]
        self.hybrid_apps = _split(V.APP_HYBRID, which)
        self.desktop_apps = _split(V.APP_DESKTOP, which)

    def site(self, rng, real_only=False, allow_google=True, soft_only=False) -> str:
        if soft_only:
            return rng.choice(self.soft_sites)
        if real_only:
            pool = self.core_sites if allow_google else self.non_google_core
            return rng.choice(pool)
        if rng.random() < 0.24:
            return fake_site(rng)
        pool = self.sites if allow_google else self.non_google_sites
        return rng.choice(pool)

    def app_name(self, rng, hybrid_only=False) -> str:
        if hybrid_only or rng.random() < 0.52:
            return rng.choice(self.hybrid_apps)
        return rng.choice(self.desktop_apps)

    def draw(self, rng, key: str):
        base = key.rstrip("0123456789")
        if base == "site":
            return self.site(rng, allow_google=False)
        if base == "n":
            return str(rng.randint(2, 99))
        if base in self.t:
            return rng.choice(self.t[base])
        return rng.choice(self.p[base])


_PH = re.compile(r"\{([a-z0-9_]+)\}")
_CASE_SUFFIXES = {"nom", "prep", "acc", "gen", "instr"}


def fill(rng, pools: Pools, pattern: str) -> str:
    cache = {}

    def rep(m):
        full = m.group(1)
        base_full = full.rstrip("0123456789")
        if full in pools.p or base_full in pools.p:
            key, case = full, "nom"
        elif "_" in full and full.rsplit("_", 1)[1] in _CASE_SUFFIXES:
            key, case = full.rsplit("_", 1)
        else:
            key, case = full, "nom"
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
    if questions and rng.random() < 0.32:
        q = fill(rng, pools, rng.choice(V.QUERY_QUEST))
    else:
        q = fill(rng, pools, rng.choice(V.QUERY_NOUN))
        if rng.random() < 0.08 and "за " not in q and "на " not in q:
            q += " за " + rng.choice(pools.p["time"])
    return q


def site_form(rng, pools: Pools, kind: str, real_only=False, allow_google=True, soft_only=False) -> str:
    """A site name in the required form: kind = open (nominative/accusative) or prep ("на X")."""
    name = pools.site(rng, real_only=real_only, allow_google=allow_google, soft_only=soft_only)
    if kind == "open":
        return inflect(name, rng.choice(["nom", "nom", "acc"]))
    return inflect(name, "prep") if rng.random() < 0.78 else name


def app_form(rng, pools: Pools, kind: str, hybrid_only=False) -> str:
    """An app name in the required form: kind = open or prep."""
    name = pools.app_name(rng, hybrid_only=hybrid_only)
    if kind == "open":
        return inflect(name, rng.choice(["nom", "nom", "acc"]))
    return inflect(name, "prep") if rng.random() < 0.78 else name


# ------------------------------------------------------------------ templates
def O(t): return (t, "O")
def S(t): return (t, "SITE")
def Q(t): return (t, "QUERY")
def T(t): return (t, "TEXT")


def tpl_open_only(rng, P):
    r = rng.random()
    if r < 0.12:
        # Rule 17: anaphoric pronoun after the site ("мне нужен сайт Stack Overflow, открой его") -> pronoun is O
        s = site_form(rng, P, "open")
        if rng.random() < 0.45:
            s += ","
        lead = rng.choice(["мне нужен сайт", "нужен сайт", "мне нужен", "хочу зайти на сайт", "мне нужна страница"])
        tail = rng.choice(["открой его", "запусти его", "открой его пожалуйста", "открой мне его", "запусти её", "открой её"])
        return [O(lead), S(s), O(tail)], "none"
    if r < 0.25:
        # Rule 16: official / main site wrapper ("перейди на официальный сайт Python")
        lead = rng.choice(["перейди на официальный сайт", "открой официальный сайт", "зайди на официальный сайт",
                           "запусти официальный сайт", "перейди на главный сайт", "открой главный сайт",
                           "открой сайт", "перейди на сайт", "зайди на сайт"])
        return [O(lead), S(site_form(rng, P, "open"))], "none"
    if r < 0.37:
        # Verb after the site ("Crunchyroll открой мне", "нова пошта запусти", "ютуб открой")
        tail_verb = rng.choice(["открой", "запусти", "открой мне", "запусти мне", "открой пожалуйста", "открывай"])
        return [S(site_form(rng, P, "open")), O(tail_verb)], "none"
    soft = rng.random() < 0.16
    return [O(rng.choice(V.OPEN_SOFT if soft else V.OPEN)), S(site_form(rng, P, "open", soft_only=soft))], "none"


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
    if rng.random() < 0.28:
        q = rng.choice(P.p["howto"])
    else:
        q = make_query(rng, P, questions=True)
    return [O(rng.choice(V.FIND)), Q(q), O(prep),
            S(site_form(rng, P, "prep", allow_google=False))], "none"


def tpl_find_on_site_q(rng, P):
    prep = rng.choices(V.SITE_PREPS, weights=[42, 42, 6, 6, 4])[0]
    q = rng.choice(P.p["howto"]) if rng.random() < 0.22 else make_query(rng, P, questions=True)
    return [O(rng.choice(V.FIND)), O(prep), S(site_form(rng, P, "prep", allow_google=False)),
            Q(q)], "none"


def tpl_wrapper_query(rng, P):
    """Entity and content-type queries ('найди фильм «Интерстеллар» на Кинопоиске', 'найди на GitHub проект с названием Stable Diffusion',
    'найди на Wikipedia информацию о Николе Тесле'). Explicit 'с/под названием' is O; single-word content prefixes stay in QUERY
    consistently with QUERY_TEMPLATES and ml_real.tsv so the tagger has zero boundary conflict."""
    kind = rng.randrange(6)
    if kind == 0:
        # "найди фильм «Интерстеллар» на Кинопоиске" / "найди сериал под названием Кухня на Кинопоиске"
        title = rng.choice(P.p["title"])
        if rng.random() < 0.35:
            title = f"«{title}»"
        site = site_form(rng, P, "prep", allow_google=False)
        prep = rng.choice(["на", "в", "на сайте"])
        if rng.random() < 0.30:
            w = rng.choice(["сериал под названием", "фильм под названием", "фильм с названием", "сериал с названием"])
            if rng.random() < 0.5:
                return [O(rng.choice(V.FIND)), O(w), Q(title), O(prep), S(site)], "none"
            return [O(rng.choice(V.FIND)), O(prep), S(site), O(w), Q(title)], "none"
        w = rng.choice(["фильм", "сериал", "аниме", "мультфильм", ""])
        q = f"{w} {title}".strip()
        if rng.random() < 0.45:
            return [O(rng.choice(V.FIND)), Q(q), O(prep), S(site)], "none"
        if rng.random() < 0.5:
            return [O(rng.choice(V.FIND)), O(prep), S(site), Q(q)], "none"
        return [O(rng.choice(V.OPEN)), S(site_form(rng, P, "open", allow_google=False)), O(rng.choice(V.AND_VERBS)), Q(q)], "none"
    if kind == 1:
        # "найди на GitHub проект с названием Stable Diffusion" / "найди на гитхабе репозиторий fastapi"
        repo = rng.choice(P.p["repo"] + P.p["pkg"])
        site = site_form(rng, P, "prep", allow_google=False)
        prep = rng.choice(["на", "в", "на сайте"])
        if rng.random() < 0.55:
            w = rng.choice(["проект с названием", "репозиторий с названием", "проект под названием", "репозиторий под названием"])
            if rng.random() < 0.65:
                return [O(rng.choice(V.FIND)), O(prep), S(site), O(w), Q(repo)], "none"
            return [O(rng.choice(V.FIND)), O(w), Q(repo), O(prep), S(site)], "none"
        w = rng.choice(["проект", "репозиторий", "библиотеку", "пакет"])
        q = f"{w} {repo}"
        if rng.random() < 0.55:
            return [O(rng.choice(V.FIND)), O(prep), S(site), Q(q)], "none"
        return [O(rng.choice(V.OPEN)), S(site_form(rng, P, "open", allow_google=False)), O(rng.choice(V.AND_VERBS)), Q(q)], "none"
    if kind == 2:
        # "найди на Wikipedia информацию о Николе Тесле" / "поищи на википедии статью о ..."
        w = rng.choice(["информацию о", "информацию об", "сведения о", "материал о", "статью о", "информацию про", "статью про"])
        target = rng.choice(P.p["person_prep"] + P.p["topic_prep"]) if not w.endswith("про") else rng.choice(P.p["person"] + P.p["topic"])
        q = f"{w} {target}"
        site = site_form(rng, P, "prep", allow_google=False)
        prep = rng.choice(["на", "в", "на сайте"])
        if rng.random() < 0.65:
            return [O(rng.choice(V.FIND)), O(prep), S(site), Q(q)], "none"
        return [O(rng.choice(V.FIND)), Q(q), O(prep), S(site)], "none"
    if kind == 3:
        # "найди на саундклауде музыку The Weeknd" / "открой ютуб и включи песню Queen"
        w = rng.choice(["музыку", "песню", "песни", "трек", "треки", "исполнителя", "группу", "альбом", ""])
        target = rng.choice(P.p["album"]) if w == "альбом" else rng.choice(P.p["artist"])
        q = f"{w} {target}".strip()
        site = site_form(rng, P, "prep", allow_google=False)
        prep = rng.choice(["на", "в"])
        if rng.random() < 0.45:
            return [O(rng.choice(V.OPEN)), S(site_form(rng, P, "open", allow_google=False)), O(rng.choice(V.AND_VERBS)), Q(q)], "none"
        if rng.random() < 0.5:
            return [O(rng.choice(V.FIND)), O(prep), S(site), Q(q)], "none"
        return [O(rng.choice(V.FIND)), Q(q), O(prep), S(site)], "none"
    if kind == 4:
        # "найди на ютубе канал vsauce" / "на гитхабе найди пользователя torvalds" / "на стековерфлоу найди ошибку keyerror"
        sub = rng.randrange(3)
        if sub == 0:
            w, target = rng.choice(["канал", "стримера", "блогера"]), rng.choice(P.p["channel"])
        elif sub == 1:
            w, target = "пользователя", rng.choice(P.p["username"])
        else:
            w, target = "ошибку", rng.choice(P.p["err"])
        q = f"{w} {target}"
        site = site_form(rng, P, "prep", allow_google=False)
        prep = rng.choice(["на", "в"])
        if rng.random() < 0.5:
            return [O(rng.choice(V.FIND)), O(prep), S(site), Q(q)], "none"
        return [O(rng.choice(V.FIND)), Q(q), O(prep), S(site)], "none"
    # kind == 5: siteless play with wrapper ("хочу посмотреть фильм Интерстеллар", "включи песню Nirvana", "поставь сериал Офис")
    w = rng.choice(["фильм", "сериал", "аниме", "песню", "трек", "музыку", "альбом", "канал"])
    if w in ("фильм", "сериал", "аниме"):
        target = rng.choice(P.p["title"])
        verb = rng.choice(V.WATCH_VERBS)
    elif w == "канал":
        target = rng.choice(P.p["channel"])
        verb = rng.choice(V.WATCH_VERBS)
    elif w == "альбом":
        target = rng.choice(P.p["album"])
        verb = rng.choice(V.LISTEN_VERBS)
    else:
        target = rng.choice(P.p["artist"])
        verb = rng.choice(V.LISTEN_VERBS)
    return [O(verb), Q(f"{w} {target}")], "none"


def tpl_site_find_q(rng, P):
    return [S(site_form(rng, P, "open", allow_google=False)), O(rng.choice(V.FIND)), Q(make_query(rng, P))], "none"


def tpl_prep_site_find_q(rng, P):
    return [O(rng.choice(["в", "на"])), S(site_form(rng, P, "prep", allow_google=False)), O(rng.choice(V.FIND)),
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
    prep = rng.choices(V.SITE_PREPS, weights=[48, 42, 4, 4, 2])[0]
    verbs = ["включи", "поставь", "включи мне", "поставь мне", "включай", "врубай", "вруби",
             "открой", "запусти", "открой мне", "запусти мне", "хочу посмотреть", "хочу послушать", "давай посмотрим"]
    q = rng.choice(P.p["title"] + P.p["artist"]) if rng.random() < 0.55 else make_query(rng, P, questions=True)
    return [O(rng.choice(verbs)), Q(q), O(prep),
            S(site_form(rng, P, "prep", allow_google=False))], "none"


def tpl_open_on_site_q(rng, P):
    return [O(rng.choice(V.OPEN_ON_SITE)), O(rng.choice(["на", "в", "на сайте"])),
            S(site_form(rng, P, "prep", allow_google=False)),
            Q(make_query(rng, P, questions=True))], "none"


def tpl_open_site_q_noverb(rng, P):
    """"включи мне ютуб канал exile", "открой leet code задачи": a site and a query without the word "найди"."""
    for _ in range(10):
        site = site_form(rng, P, "open", real_only=True, allow_google=False)
        query = make_query(rng, P, questions=False)
        if not _conflicts(site, query):
            return [O(rng.choice(V.OPEN + V.OPEN_SOFT)), S(site), Q(query)], "none"
    return [O(rng.choice(V.OPEN)), S(site_form(rng, P, "open"))], "none"


def tpl_open_site_suffix(rng, P):
    """Web site or hybrid app explicitly opened in browser/web ('открой телеграм в вебе', 'запусти discord в браузере')."""
    if rng.random() < 0.65:
        target = app_form(rng, P, "open", hybrid_only=True)
        marker = rng.choice(V.WEB_MARKERS)
        if rng.random() < 0.30:
            return [O(rng.choice(V.OPEN_BARE)), S(target), O(marker), O(rng.choice(V.AND_VERBS)),
                    Q(make_query(rng, P, questions=False))], "none"
        return [O(rng.choice(V.OPEN_BARE)), S(target), O(marker)], "none"
    return [O(rng.choice(V.OPEN)), S(site_form(rng, P, "open")), O(rng.choice(V.BROWSER_SUFFIX))], "none"


def tpl_site_act_q(rng, P):
    """"на вольфраме реши уравнение", "открой гугл и проверь пинг", "построй график на десмосе": the action verb is O."""
    act = rng.choice(list(V.ACTS))
    q = rng.choice(V.ACTS[act])
    prep = rng.choices(["на", "в", "через"], weights=[50, 45, 5])[0]
    form = rng.randrange(5)
    if form == 0:
        return [O(prep), S(site_form(rng, P, "prep", allow_google=False)), O(act), Q(q)], "none"
    if form == 1:
        return [S(site_form(rng, P, "open", allow_google=False)), O(act), Q(q)], "none"
    if form == 2:
        return [O(rng.choice(V.OPEN)), S(site_form(rng, P, "open", allow_google=False)), O("и " + act), Q(q)], "none"
    if form == 3:
        return [O(act), Q(q), O(prep), S(site_form(rng, P, "prep", allow_google=False))], "none"
    return [O(rng.choice(V.OPEN)), S(site_form(rng, P, "open", allow_google=False)), O(act), Q(q)], "none"


def make_light_query(rng, P: Pools) -> str:
    kind = rng.randrange(4)
    if kind == 0:
        city = rng.choice(P.t["city"])[1]
        return rng.choice(["погоду", "погоду на завтра", "погоду на выходные", f"погоду в {city}",
                           f"погоду в {city} на завтра", "какая погода будет завтра"])
    if kind == 1:
        cur1 = rng.choice(["евро", "долларов", "доллара", "рублей", "фунтов", "юаней"])
        target = rng.choice(["рубли", "доллары", "евро", "тенге"])
        n = rng.choice([10, 20, 50, 100, 200, 500])
        return rng.choice([f"сколько будет {n} {cur1} в {target}", f"{n} {cur1} в {target}",
                           f"курс {rng.choice(P.p['currency'])}", "курс валют на сегодня",
                           "что там по курсу доллара", "какой сегодня курс евро"])
    if kind == 2:
        c1, c2 = rng.sample(P.t["city"], 2)
        return rng.choice([f"авиа билеты {c1[0]} {c2[0]}", f"авиабилеты {c1[0]} {c2[0]}",
                           f"расписание поездов {c1[0]} {c2[0]}", "расписание автобусов",
                           f"расписание электричек до {c2[3]}", f"самый дешёвый билет в {c2[2]}"])
    return rng.choice([f"цену на {rng.choice(P.p['gadget'])}", "цены на видеокарты", "цену на айфон"])


def tpl_light_search(rng, P):
    """Lightweight factual lookups without 'в гугле' -> browser with site='' and query."""
    verb = rng.choice(["скажи", "посмотри", "глянь", "глянь ка", "покажи", "узнай", "найди", "поищи", "покажи мне"])
    q = make_light_query(rng, P)
    return [O(verb), Q(q)], "none"


# ------------------------------------------------------------------ `app` templates
def tpl_app_open_only(rng, P):
    """Open a desktop or hybrid app on PC ('открой калькулятор', 'открой телеграм', 'запусти стим', 'открой claude')."""
    name = app_form(rng, P, "open")
    if rng.random() < 0.12:
        lead = rng.choice(["мне нужно приложение", "нужна программа", "мне нужен"])
        tail = rng.choice(["открой его", "запусти его", "открой её"])
        return [O(lead), S(name), O(tail)], "none"
    cmd = rng.choice(V.SYS_OPEN + ["открой приложение", "запусти программу", "открой программу"])
    if rng.random() < 0.15:
        return [O(cmd), S(name), O("на компьютере")], "none"
    return [O(cmd), S(name)], "none"


def tpl_app_folder_file(rng, P):
    """Open a local folder, drive or file ('открой папку загрузки', 'открой папку с фотографиями', 'открой файл отчёт')."""
    s = rng.randrange(5)
    cmd = rng.choice(["открой", "открой мне", "зайди в", "перейди в", "открой пожалуйста"])
    if s == 0:
        return [O(cmd + " папку"), S(rng.choice(V.SYS_FOLDERS))], "none"
    if s == 1:
        return [O(cmd + " папку с"), S(rng.choice(V.SYS_FOLDERS_WITH))], "none"
    if s == 2:
        return [O("открой"), S(rng.choice(V.SYS_PLACES))], "none"
    if s == 3:
        f = rng.choice(V.SYS_FILES)
        k = rng.choice(V.SYS_FILE_KIND)
        target = f"{f} {k}".strip()
        return [O(rng.choice(["открой файл", "открой документ", "запусти файл"])), S(target)], "none"
    return [O("открой"), S(rng.choice(["последний файл", "последний документ", "презентацию", "загрузки", "документы"]))], "none"


def make_app_query(rng, P: Pools, app_name: str = "") -> tuple:
    """Returns (optional_wrapper_O, query_str) tailored to the hybrid app (Steam -> games, Spotify -> music/artists, etc.)."""
    low = app_name.casefold()
    if any(k in low for k in ("steam", "стим", "epic", "эпик")):
        g = rng.choice(P.p["game"] + ["кооп игры", "кооперативные игры", "стратегии", "шутеры"])
        if g not in ("кооп игры", "кооперативные игры") and rng.random() < 0.25:
            pref = rng.choice(["игру", "игрушку"])
            return "", f"{pref} {g}"
        return "", g
    if any(k in low for k in ("spotify", "спотиф", "спотик")):
        if rng.random() < 0.72:
            a = rng.choice(P.p["artist"])
            pref = rng.choice(["", "", "", "музыку", "песню", "песни", "трек", "исполнителя"])
            return "", f"{pref} {a}".strip()
        return "", rng.choice(["плейлист для работы", "плейлист для бега", "плейлист для сна", "рок хиты",
                               "подкасты про историю", "новый альбом " + rng.choice(P.p["artist"])])
    if any(k in low for k in ("telegram", "телеграм", "телег", "discord", "дискорд", "slack", "слак")):
        return "", rng.choice(["канал с новостями", "чат про питон", "канал про айти", "группу курса", "рабочий чат"])
    if rng.random() < 0.45:
        return "", rng.choice(V.APP_ACTIONS)
    return "", make_query(rng, P, questions=False)


def tpl_app_open_query(rng, P):
    """Open an app or folder and perform an inner query/action ('открой Steam и найди Elden Ring', 'открой Spotify и найди музыку The Weeknd')."""
    s = rng.randrange(6)
    if s == 0:
        folder = rng.choice(["загрузки", "документы", "проекты", "фото", "рабочий стол"])
        act = rng.choice(["отсортируй по дате", "отсортируй по имени", "отсортируй по размеру", "покажи скрытые файлы", "найди старые файлы"])
        prefix = rng.choice(["открой", "открой папку", "зайди в", "зайди в папку"])
        conn = rng.choice(["и", "и там", ""])
        parts = [O(prefix), S(folder)]
        if conn:
            parts.append(O(conn))
        parts.append(Q(act))
        return parts, "none"
    if s == 1:
        return [O(rng.choice(["открой", "запусти"])), S("терминал"),
                Q(rng.choice(["в папке проекта", "от имени администратора"]))], "none"
    app = app_form(rng, P, "open", hybrid_only=True)
    wrapper, q = make_app_query(rng, P, app)
    parts = [O(rng.choice(V.OPEN_BARE)), S(app), O(rng.choice(V.AND_VERBS))]
    if wrapper:
        parts.append(O(wrapper))
    parts.append(Q(q))
    return parts, "none"


def tpl_app_prep_query(rng, P):
    """Query inside a hybrid app without web markers ('в спотифае включи face', 'в стиме найди игру elden ring')."""
    app = app_form(rng, P, "prep", hybrid_only=True)
    prep = rng.choice(["в", "на"])
    wrapper, q = make_app_query(rng, P, app)
    form = rng.randrange(4)
    if form == 0:
        parts = [O(prep), S(app), O(rng.choice(V.FIND))]
        if wrapper:
            parts.append(O(wrapper))
        parts.append(Q(q))
        return parts, "none"
    if form == 1:
        parts = [O(rng.choice(V.FIND)), O(prep), S(app)]
        if wrapper:
            parts.append(O(wrapper))
        parts.append(Q(q))
        return parts, "none"
    if form == 2:
        parts = [O(rng.choice(V.FIND + V.OPEN_SOFT + ["поставь", "поставь мне"]))]
        if wrapper:
            parts.append(O(wrapper))
        parts.extend([Q(q), O(prep), S(app)])
        return parts, "none"
    parts = [S(app_form(rng, P, "open", hybrid_only=True)), O(rng.choice(V.FIND))]
    if wrapper:
        parts.append(O(wrapper))
    parts.append(Q(q))
    return parts, "none"


# ------------------------------------------------------------------ `system` templates
def tpl_system_action(rng, P):
    """OS/PC control or file-system management command -> route='system', query=the action span."""
    return [Q(V.system_action_phrase(rng))], "none"


# ------------------------------------------------------------------ `llm` & `browser` templates
def tpl_llm_act(rng, P):
    """"реши уравнение", "закажи пиццу" without a site and without a place to search -> llm."""
    act = rng.choice(list(V.ACTS))
    return [O(f"{act} {rng.choice(['мне ', '', '']) }{rng.choice(V.ACTS[act])}".replace("  ", " "))], "none"


def tpl_llm_meaning(rng, P):
    """"что значит слово X" without a language is an explanation request (llm), not a translation (rule 14)."""
    word = rng.choice(V.TR_WORDS) if rng.random() < 0.25 else rng.choice(P.p["foreignw"])
    return [O(f"{rng.choice(V.MEANING_CMD)} {word}")], "none"


def tpl_llm_vague(rng, P):
    """Rule 17: vague goal on a site without a concrete query ('давай зайдём на GitHub, мне нужно посмотреть один репозиторий') -> llm."""
    site = site_form(rng, P, "open")
    lead = rng.choice(["давай зайдём на", "зайди на", "открой", "давай откроем", "перейди на", "хочу зайти на"])
    vague = rng.choice([
        "мне нужно посмотреть один репозиторий", "мне надо посмотреть один репозиторий",
        "хочу посмотреть один репозиторий", "мне нужно глянуть один проект",
        "хочу глянуть одну штуку", "надо кое что проверить", "мне нужно найти одну статью",
        "хочу посмотреть кое что", "мне надо проверить одно видео", "нужно проверить один аккаунт",
        "хочу найти там одну вещь", "мне нужно посмотреть там одну страницу",
    ])
    if rng.random() < 0.45:
        site += ","
    return [O(f"{lead} {site} {vague}")], "none"


SEARCH_VERBS = [v for v in V.FIND if v not in V.PLAY_FIND] + V.SEARCH_EXTRA


def tpl_play_q_web(rng, P):
    """"включи рок", "поставь дюну", "хочу посмотреть фильм про пиратов": media without a platform -> Google search."""
    if rng.random() < 0.55:
        verb, pattern = rng.choice(V.LISTEN_VERBS), rng.choice(V.PLAY_LISTEN)
    else:
        verb, pattern = rng.choice(V.WATCH_VERBS), rng.choice(V.PLAY_WATCH)
    return [O(verb), Q(fill(rng, P, pattern))], "none"


_LIGHT_HINTS = ("погод", "курс", "евро", "доллар", "рубл", "билет", "расписани", "цену на", "цены на")


def tpl_search_nowhere(rng, P):
    """"найди X" WITHOUT a place to search (no site, no google, no internet) and outside light search -> llm (rule 11)."""
    r = rng.random()
    if r < 0.16:
        # Contrasts with light search ("самый дешёвый билет") and "найди в <SITE>":
        # "найди дешёвый ноутбук для программирования", "найди в тексте грамматические ошибки", "узнай расписание работы банка"
        neg = rng.choice([
            f"найди дешёвый {rng.choice(P.p['thing'])} для {rng.choice(P.p['activity'])}",
            f"найди дешёвый ноутбук для {rng.choice(['программирования', 'учёбы', 'работы', 'игр'])}",
            f"поищи недорогой {rng.choice(P.p['thing'])}",
            "найди в тексте грамматические ошибки",
            "найди в тексте ошибки",
            "найди ошибки в тексте",
            "найди в коде ошибку",
            "найди в документе опечатки",
            "найди в статье главную мысль",
            "найди в предложении подлежащее и сказуемое",
            "узнай расписание работы банка",
            "посмотри расписание работы почты",
            "узнай график работы поликлиники",
            "найди часы работы магазина",
            "посмотри расписание звонков в школе",
        ])
        return [O(neg)], "none"
    if r < 0.24:
        # Bare noun query without any command verb ("расписание поездов софия варна", "рецепт борща") -> llm
        c1, c2 = rng.sample(P.t["city"], 2)
        bare = rng.choice([
            f"расписание поездов {c1[0]} {c2[0]}",
            f"авиабилеты {c1[0]} {c2[0]}",
            f"рецепт {rng.choice(P.p['foodg'])}",
            f"курс {rng.choice(P.p['currency'])}",
            f"погода в {c1[1]}",
        ])
        return [O(bare)], "none"
    hard = rng.random() < 0.22
    for _ in range(10):
        v = rng.choice(SEARCH_VERBS)
        q = make_query(rng, P, hard=hard)
        if not any(h in q.casefold() for h in _LIGHT_HINTS):
            return [O(v), O(q)], "none"
    return [O("найди"), O("рецепт пиццы")], "none"


def tpl_llm_home(rng, P):
    return [O(V.home_phrase(rng))], "none"


def tpl_llm_howto(rng, P):
    return [O(rng.choice(V.HOWTO))], "none"


def tpl_google_q(rng, P):
    hard = rng.random() < 0.12
    q = make_query(rng, P, hard=hard)
    if rng.random() < 0.24:
        tail_g = rng.choice(["погугли", "загугли", "нагугли", "найди в гугле", "поищи в гугле",
                             "найди в интернете", "поищи в интернете", "пробей в гугле", "погугли мне", "загугли мне"])
        return [Q(q), O(tail_g)], "none"
    return [O(rng.choice(V.GOOGLE)), Q(q)], "none"


def _lang_phrase(rng, lang: str, style: str) -> str:
    if lang in V.LANG_TYPOS and rng.random() < 0.18:
        return rng.choice(V.LANG_TYPOS[lang])
    acc, prep, adv = V.LANG_FORMS[lang]
    if style == "to":
        return rng.choice([f"на {acc}", f"на {acc}", f"на {prep}", f"на {acc} язык"])
    return rng.choice([f"на {prep}", adv, adv.replace("по-", "по ") if adv.startswith("по-") else adv,
                       f"на {prep} языке"])


def _tr_text(rng, P, lang: str) -> str:
    r = rng.random()
    if r < 0.38:
        return rng.choice(V.TR_PHRASES)
    if r < 0.46:
        return rng.choice(V.TR_WORDS)
    if r < 0.55:
        return rng.choice(V.TR_FOREIGN)
    if r < 0.63:
        return rng.choice(V.TR_COMMANDISH)
    if r < 0.73:
        return rng.choice(V.TR_POLITE)            # "два чая пожалуйста": "пожалуйста" after the text is part of the text (rule 13)
    if r < 0.83:
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
    if rng.random() < 0.32:
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
    noun = rng.choice(["слово", "фразу", "предложение", "текст", "выражение"])
    if rng.random() < 0.5:
        return [O(rng.choice(V.TR_CMD)), O(noun), T(_tr_text(rng, P, lang)), O(_lang_phrase(rng, lang, "to"))], lang
    return [O(rng.choice(V.TR_CMD)), O(_lang_phrase(rng, lang, "to")), O(noun), T(_tr_text(rng, P, lang))], lang


def tpl_tr_how_to_say(rng, P):
    lang = _tr_lang(rng)
    verb = rng.choice([
        "как будет", "как сказать", "как будет по русски", "как переводится", "как звучит",
        "переведи как будет", "переведи как сказать", "переведи мне как будет",
        "подскажи как будет", "скажи как будет", "переведи пожалуйста как будет",
    ])
    return [O(verb), T(_tr_text(rng, P, lang)), O(_lang_phrase(rng, lang, "adv"))], lang


def tpl_tr_how_adv(rng, P):
    lang = _tr_lang(rng)
    adv = V.LANG_FORMS[lang][2]
    lead = rng.choice(["как", "переведи как", "подскажи как"])
    return [O(lead), O(adv.replace("по-", "по ") if adv.startswith("по-") and rng.random() < 0.3 else adv),
            T(_tr_text(rng, P, lang))], lang


def tpl_tr_bare(rng, P):
    if rng.random() < 0.35:
        verb = rng.choice(["как переводится", "как переводится слово", "как переводится выражение"])
        word = rng.choice(P.p["foreignw"] + V.TR_FOREIGN)
        return [O(verb), T(word)], "none"
    return [O(rng.choice(V.TR_CMD)), T(_tr_text(rng, P, "none"))], "none"


def tpl_tr_colon(rng, P):
    form = rng.randrange(3)
    if form == 0:
        # "Переведи: я ищу стажировку в области искусственного интеллекта" (no target language)
        cmd = rng.choice(["переведи", "переведи мне", "переведите", "нужно перевести"]) + ":"
        return [O(cmd), T(_tr_text(rng, P, "none"))], "none"
    lang = _tr_lang(rng)
    if form == 1:
        cmd = rng.choice(["переведи", "переведи мне", "переведите"]) + ":"
        return [O(cmd), T(_tr_text(rng, P, lang)), O(_lang_phrase(rng, lang, "to"))], lang
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
    ("browser.open_only", "browser", 9, tpl_open_only),
    ("browser.open_find_and", "browser", 11, tpl_open_find_and),
    ("browser.open_find_noand", "browser", 4, tpl_open_find_noand),
    ("browser.find_q_on_site", "browser", 9, tpl_find_q_on_site),
    ("browser.find_on_site_q", "browser", 8, tpl_find_on_site_q),
    ("browser.wrapper_q", "browser", 12, tpl_wrapper_query),
    ("browser.site_find_q", "browser", 3, tpl_site_find_q),
    ("browser.prep_site_find_q", "browser", 3, tpl_prep_site_find_q),
    ("browser.open_q_on_site", "browser", 7, tpl_open_q_on_site),
    ("browser.open_on_site_q", "browser", 5, tpl_open_on_site_q),
    ("browser.open_site_q_noverb", "browser", 5, tpl_open_site_q_noverb),
    ("browser.open_site_suffix", "browser", 3, tpl_open_site_suffix),
    ("browser.play_q_web", "browser", 6, tpl_play_q_web),
    ("browser.google_q", "browser", 10, tpl_google_q),
    ("browser.light_search", "browser", 4, tpl_light_search),
    ("browser.site_act_q", "browser", 7, tpl_site_act_q),
    ("app.open_only", "app", 9, tpl_app_open_only),
    ("app.folder_file", "app", 7, tpl_app_folder_file),
    ("app.open_query", "app", 10, tpl_app_open_query),
    ("app.prep_query", "app", 9, tpl_app_prep_query),
    ("system.action", "system", 20, tpl_system_action),
    ("translate.cmd_lang_text", "translate", 10, tpl_tr_cmd_lang_text),
    ("translate.cmd_text_lang", "translate", 8, tpl_tr_cmd_text_lang),
    ("translate.word_prefix", "translate", 5, tpl_tr_word_prefix),
    ("translate.how_to_say", "translate", 7, tpl_tr_how_to_say),
    ("translate.how_adv", "translate", 3, tpl_tr_how_adv),
    ("translate.bare", "translate", 4, tpl_tr_bare),
    ("translate.colon", "translate", 4, tpl_tr_colon),
    ("translate.via_translator", "translate", 3, tpl_tr_via_translator),
    ("translate.text_then_cmd", "translate", 3, tpl_tr_text_then_cmd),
    ("translate.meaning", "translate", 4, tpl_tr_meaning),
    ("llm.question", "llm", 8, tpl_llm_question),
    ("llm.hard", "llm", 5, tpl_llm_hard),
    ("llm.only", "llm", 11, tpl_llm_only),
    ("llm.cmp", "llm", 2, tpl_llm_cmp),
    ("llm.search_nowhere", "llm", 13, tpl_search_nowhere),
    ("llm.home", "llm", 26, tpl_llm_home),
    ("llm.howto", "llm", 4, tpl_llm_howto),
    ("llm.act", "llm", 4, tpl_llm_act),
    ("llm.meaning", "llm", 4, tpl_llm_meaning),
    ("llm.vague", "llm", 6, tpl_llm_vague),
]


# ------------------------------------------------------------------ noise: speech fillers, typos, case
_NO_TYPO_WORDS = {"сайт", "сайте", "вебе", "браузере", "хроме", "приложение", "приложении", "программу", "папку", "файл"}


def typo(rng, w: str) -> str:
    if len(w) < 4 or any(ch.isdigit() for ch in w) or not w[1:-1] or norm(w) in _NO_TYPO_WORDS:
        return w
    i = rng.randrange(1, len(w) - 1)
    op = rng.random()
    if op < 0.35:
        cand = w[:i] + w[i + 1:]
    elif op < 0.6:
        cand = w[:i] + w[i] + w[i:]
    elif op < 0.85:
        cand = w[:i - 1] + w[i] + w[i - 1] + w[i + 1:]
    else:
        cand = w[:i] + rng.choice(_NEIGH.get(w[i], w[i])) + w[i + 1:]
    if norm(cand) in app_norms() and norm(w) not in app_norms():
        return w
    return cand


TYPO_RATE = {"O": 0.03, "SITE": 0.045, "QUERY": 0.025, "TEXT": 0.02}


def _polite(text: str) -> bool:
    return any(w.strip(",.") in ("пожалуйста", "плиз") for w in text.split())


def finalize(rng, parts, intent, lang, tpl_name):
    parts = list(parts)
    first_tag, last_tag = parts[0][1], parts[-1][1]
    if rng.random() < 0.22:
        pre = rng.choice(V.PRE)
        if intent == "translate" and first_tag == "TEXT":
            pre = rng.choice([p for p in V.PRE if not _polite(p)])
        parts.insert(0, O(pre))
    if rng.random() < 0.12:
        if intent == "translate" and last_tag == "TEXT":
            if rng.random() < 0.6:
                parts.append(T(rng.choice(["пожалуйста", "плиз"])))
        else:
            parts.append(O(rng.choice(V.POST)))
    words, tags = [], []
    cap_site = rng.random() < 0.22
    cap_query = rng.random() < 0.16
    for text, tag in parts:
        for w in text.split():
            eff_tag = "O" if intent == "llm" else tag
            if eff_tag == "SITE" and cap_site:
                w = w[:1].upper() + w[1:]
            elif eff_tag == "QUERY" and cap_query and re.search("[a-zA-Z]", w):
                w = w[:1].upper() + w[1:]
            words.append(w)
            tags.append(eff_tag)
    words = [typo(rng, w) if rng.random() < TYPO_RATE[t] else w for w, t in zip(words, tags)]
    r = rng.random()
    if r < 0.18:
        words[0] = words[0][:1].upper() + words[0][1:]
    elif r < 0.22:
        words = [w.upper() for w in words]
    if rng.random() < 0.5:
        words = [w.replace("ё", "е") for w in words]
    if rng.random() < 0.08:
        words[-1] += "."
    elif intent == "llm" and rng.random() < 0.25:
        words[-1] += "?"
    gold = assemble(words, tags) if intent != "llm" else {"site": "", "query": "", "text": ""}
    if intent in ("browser", "app"):
        gold["text"] = ""
    elif intent == "system":
        gold["site"] = gold["text"] = ""
    elif intent == "translate":
        gold["site"] = gold["query"] = ""
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
    """Takes one of your real phrases and substitutes other sites/apps and queries: the structure is yours, the filling is new."""
    intent = row["intent"]
    if intent == "llm":
        return None
    if intent == "system":
        return finalize(rng, [(V.system_action_phrase(rng), "QUERY")], "system", "none", "real_aug")
    groups = []
    for w, t in zip(row["words"], row["tags"]):
        if groups and groups[-1][1] == t:
            groups[-1][0].append(w)
        else:
            groups.append(([w], t))
    parts, prev_o, chosen_site = [], "", ""
    soft = any(w.casefold() in _SOFT for g, t in groups if t == "O" for w in g) and not row["query"]
    orig_q = row["query"].casefold()
    is_light = intent == "browser" and not row["site"] and any(h in orig_q for h in _LIGHT_HINTS)
    orig_phrase_norm = norm(" ".join(row["words"]))
    has_web_marker = any(m in orig_phrase_norm for m in ("в вебе", "в браузере", "на сайте", "веб верси", "через браузер", "в хроме"))
    for words, tag in groups:
        text = " ".join(words)
        if tag == "SITE":
            kind = "prep" if prev_o.split()[-1:] and prev_o.split()[-1].casefold() in _SITE_PREPS else "open"
            if intent == "app":
                text = app_form(rng, pools, kind, hybrid_only=bool(row["query"]))
            elif has_web_marker and norm(row["site"]) in app_norms() and rng.random() < 0.7:
                text = app_form(rng, pools, kind, hybrid_only=True)
            else:
                text = site_form(rng, pools, kind, soft_only=soft, allow_google=not bool(row["query"]))
            chosen_site = text
        elif tag == "QUERY":
            if intent == "app":
                _, text = make_app_query(rng, pools, chosen_site or row["site"])
            elif is_light:
                text = make_light_query(rng, pools)
            else:
                last_o = prev_o.split()[-1].casefold() if prev_o else ""
                if last_o in V.ACTS:
                    text = rng.choice(V.ACTS[last_o])
                else:
                    asked = text.casefold().startswith(QUESTION_STARTS) or rng.random() < 0.25
                    text = make_query(rng, pools, questions=asked)
        elif tag == "TEXT":
            text = _tr_text(rng, pools, row["lang"])
        else:
            prev_o = text
        parts.append((text, tag))
    return finalize(rng, parts, intent, row["lang"], "real_aug")


def _write(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _exam_keys():
    keys = set()
    try:
        from ml_exam import EXAM
        keys |= {norm(row[0]) for row in EXAM}
    except ImportError:
        pass
    try:
        import ml_probe
        keys |= {norm(row[1]) for row in ml_probe.generate()}
        keys |= {norm(row[1]) for row in ml_probe.USER_HARD_CASES}
    except ImportError:
        pass
    keys |= {norm(r[0]) for r in read_tsv(MINE_PATH)}
    return frozenset(keys)


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

    def with_real(rows, allowed_templates=None):
        seen = {norm(" ".join(r["words"])) for r in rows}
        extra = [r for r in real if norm(" ".join(r["words"])) not in exam] * args.real_copies
        llm_tpls = [t for t in TEMPLATES if t[1] == "llm" and (allowed_templates is None or t[0] in allowed_templates)]
        llm_weights = [t[2] for t in llm_tpls]
        for r in real:
            for _ in range(args.real_aug):
                if r["intent"] == "llm" and llm_tpls:
                    name, _, _, fn = arng.choices(llm_tpls, weights=llm_weights)[0]
                    parts, lang = fn(arng, aug_pools)
                    a = finalize(arng, parts, "llm", lang, name)
                else:
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
        _write(out / "train.jsonl", with_real(generate(args.n, args.seed, "train", keep, exam), keep))
        _write(out / "val_sites.jsonl", generate(n_val, args.seed + 1, "unseen", keep, exam))
        _write(out / "val_templates.jsonl", generate(n_val, args.seed + 2, "train", HOLDOUT_TEMPLATES, exam))
        _write(out / "val_both.jsonl", generate(n_val, args.seed + 3, "unseen", HOLDOUT_TEMPLATES, exam))
    else:
        _write(out / "train_final.jsonl", with_real(generate(args.n, args.seed, "all", None, exam), None))
        _write(out / "val_final.jsonl", generate(3000, args.seed + 9, "all", None, exam))
    print("done:", out.resolve())


if __name__ == "__main__":
    main()
