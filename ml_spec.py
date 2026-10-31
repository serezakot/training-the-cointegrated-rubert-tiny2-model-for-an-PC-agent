"""Contract of the first-layer parser: what it understands and how phrases are labelled (no torch needed).

The first layer of the voice assistant is a small trained model. In ~20 ms it decides what the user wants
and pulls the needed pieces out of the phrase. Anything complicated is handed to the big LLM (second layer).
The commands themselves are Russian (they come from speech recognition); all examples below are Russian on purpose.

THREE INTENTS
  browser   - the phrase contains a browser command: открой / запусти / зайди / перейди / включи /
              поставь / найди / поищи / покажи / посмотри / узнай / погугли / загугли ...
              Pieces: SITE (which site, if named) and QUERY (what to search for, if said).
              No site, but "where to search" is named (google / internet / web / search / browser, or "погугли X")
              -> plain Google search. "Включи / поставь X" (music, film) without a platform is also a Google search.
              Only a site -> open the site. Both -> the site with the query.
  translate - "переведи ...", "как будет X по-английски", "как сказать X на немецком".
              Pieces: TEXT (what to translate, verbatim) and the TARGET language (a separate head: a code from
              LANGS, "none" if no language was named). The source language is not needed: translators detect it.
  llm       - everything else: questions, requests to tell / explain / write / calculate, small talk.
              If there is no browser command, the phrase is not for the browser but for the big LLM.

WORD TAGS (tags): O - noise, SITE, QUERY, TEXT. Words are never changed: the model only puts labels on the
words of the original phrase, and site/query/text are assembled from those very words.

RULES (they can and should be tuned - quality depends on them; everything written here is applied
identically by the data generator)
  1. Commands, politeness, addressing the assistant ("джарвис", "пожалуйста", "ну", "плиз") - O.
  2. A preposition before the site ("на", "в", "через") and the words "сайт", "страница", "приложение" - O.
     Exception: a preposition that is part of the name itself is SITE.
  3. A multi-word site name - all words are SITE ("яндекс карты", "саунд клауд").
     The name is written as spoken: case, typos and language are left untouched.
  4. "Погугли / загугли / нагугли / поищи в гугле / поищи в интернете X" - browser without a site:
     "в гугле", "в интернете", "в сети" - O. "Открой гугл" opens a site: "гугл" is SITE.
  5. QUERY is one continuous piece: everything that describes the search object, including inner prepositions
     ("чехол для телефона", "комнату в софии") and filters ("за час", "новые"). A command ("найди",
     "включи") is never part of QUERY, not even in "и включи radiohead".
  6. "Включи / поставь X" without a named site - QUERY (browser without a site).
     "Открой / запусти / зайди на X" - X is always SITE, even if unfamiliar.
  7. A site name inside the question itself ("найди как отключить рекламу в ютубе", "найди отзывы
     о ютуб премиум") is part of QUERY, not SITE. For "найди <question> на/в <site>" there is no unambiguous
     rule: such phrases are not in the training data, and when in doubt the model hands the phrase to the big LLM.
  8. Questions and requests without a browser command ("кто такой павел первый", "что такое гитхаб",
     "расскажи про чёрные дыры", "как дела") - llm, even if they mention sites.
  9. translate: TEXT is everything that must be translated, verbatim (even if it looks like a command:
     "переведи на английский открой ютуб" -> TEXT "открой ютуб"). The words "слово", "фразу", "на
     английский", "с русского", "по-английски" - O.
 10. If the model is unsure (low probability or odd tagging) - llm.
 11. WHERE TO SEARCH. "Найди / поищи / ищи / покажи / посмотри / глянь / узнай / вбей / сделай запрос X" without
     a place to search (no site, no "в гугле", "в интернете", "в сети", "в поиске", "в браузере") - llm:
     "найди формулу площади круга" -> llm, while "найди в гугле формулу площади круга" and "найди формулу
     площади круга на википедии" -> browser. "Погугли / загугли" already contain the place. A site name inside
     the question itself ("найди как отключить рекламу в ютубе") does not count as a place -> llm.
     "Включи / поставь X" (music, film) is not a search, rule 6 still applies: browser without a site.
 12. Home and appliances (light, curtains, alarm, timer, reminder, call, SMS, TV, climate, kitchen
     appliances, "как..." how-to questions) - llm, even when the command looks like "включи / открой / запусти / поставь".
     Computer system tasks (calculator, notepad, folders, files, windows, screenshot, shutdown and
     restart, running scripts) are also neither browser nor translation -> llm. In the data they are a separate
     category (template llm.system, files system*.txt) and are never mixed with the home category.
 13. "Пожалуйста" / "плиз" in translation. Right after the command ("переведи пожалуйста ...") or before it - O.
     After the text to translate ("переведи на чешский два чая пожалуйста") and inside it ("как сказать помогите
     пожалуйста по-английски") - part of TEXT: the translator gets the phrase together with the polite word.
     A language adverb "по болгарски" inside the translated phrase ("я не говорю по болгарски") is TEXT too; the "по X"
     that names the TARGET language is O.
 14. "Что значит / что означает X". Without a language ("что означает выражение by the way") it is a request to explain: llm.
     With a language ("что значит слово gorgeous на русском", "что значит awkward по-русски") - translate.
 15. Action verbs between the site and the query ("на вольфрам альфа реши интеграл", "в спидтесте проверь пинг",
     "на десмосе построй график") - O, not QUERY. If there is no site and no place to search ("реши уравнение",
     "закажи пиццу", "проверь пинг") - llm. If the tagging is ragged (a SITE / QUERY / TEXT piece is torn apart by words of
     another label) the model made a mistake and the phrase goes to llm (decide() checks this).
"""
import re
from pathlib import Path

import ml_home

INTENTS = ("browser", "translate", "llm")
TAGS = ("O", "SITE", "QUERY", "TEXT")
LANGS = ("none", "en", "ru", "de", "fr", "es", "it", "pt", "pl", "bg", "tr", "uk", "cs", "nl",
         "sv", "el", "ro", "sr", "zh", "ja", "ko", "ar", "hi", "fi", "da", "hu", "he", "vi", "id", "kk")

BASE_MODEL = "cointegrated/rubert-tiny2"
MODEL_DIR = str(Path(__file__).resolve().parent / "ml_model")   # absolute path: independent of the working directory
MAX_LEN = 64
EDGE = " ,.;:!?«»\"'"


def norm(text: str) -> str:
    """Soft normalization for comparison: case, ё/е, spaces, punctuation at the edges."""
    return " ".join(text.casefold().replace("ё", "е").split()).strip(EDGE)


def _span(tags, label):
    """Indices of words tagged `label`; gaps of O between the first and the last one are filled
    (a query and a site name are continuous pieces). If the gap contains another label, it is not filled."""
    idx = [i for i, t in enumerate(tags) if t == label]
    if not idx:
        return []
    first, last = idx[0], idx[-1]
    if any(t not in (label, "O") for t in tags[first:last + 1]):
        return idx
    return list(range(first, last + 1))


def assemble(words, tags) -> dict:
    """Word tags -> {"site", "query", "text"}. Text is taken only from the words of the phrase."""
    def build(label):
        return " ".join(words[i] for i in _span(tags, label)).strip(EDGE)

    return {"site": build("SITE"), "query": build("QUERY"), "text": build("TEXT")}


THRESHOLD_INTENT = 0.90     # below this -> unsure about the intent -> llm   (tuned with test_ml.py --sweep on the final model)
THRESHOLD_SLOT = 0.75       # below this for any word -> unsure about the tagging -> llm
THRESHOLD_LANG = 0.50       # target language below the threshold -> "none"


def fragmented(tags) -> str:
    """Ragged tagging: between the first and the last word of SITE / QUERY / TEXT there is a word of ANOTHER label.
    Returns the label name or ''. This never happens in the training data, so it is a model mistake."""
    for label in ("SITE", "QUERY", "TEXT"):
        idx = [i for i, t in enumerate(tags) if t == label]
        if idx and any(t not in (label, "O") for t in tags[idx[0]:idx[-1] + 1]):
            return label
    return ""


# A command verb inside a search query without a site ("у меня руки заняты включи чайник"): this happens when the model
# took a device command for a media query. Real queries almost never contain such words -> better ask the big LLM.
_COMMAND_IN_QUERY = re.compile(r"(?<!\w)(включи|выключи|включай|выключай|включите|выключите|запусти|запускай|открой|открывай|закрой)(?!\w)", re.I)


def decide(phrase: str, words, pred: dict, t_intent=THRESHOLD_INTENT, t_slot=THRESHOLD_SLOT,
           t_lang=THRESHOLD_LANG) -> dict:
    """Model prediction (probabilities and word tags) -> decision where to route the phrase.

    Safety net: if the model is unsure or the tagging looks odd, the phrase is not executed blindly but
    goes to the big LLM (fallback=True). A pure function - tested without the model.
    """
    best = max(range(len(INTENTS)), key=lambda k: pred["intent_p"][k])
    intent, conf = INTENTS[best], pred["intent_p"][best]
    lbest = max(range(len(LANGS)), key=lambda k: pred["lang_p"][k])
    lang = LANGS[lbest] if pred["lang_p"][lbest] >= t_lang else "none"
    parts = assemble(words, pred["tags"])
    result = {"route": intent, "site": parts["site"], "query": parts["query"], "text": parts["text"],
              "lang": lang if intent == "translate" else "none", "confidence": conf,
              "fallback": False, "reason": "", "phrase": phrase}
    if intent == "browser":
        result["text"] = ""
    elif intent == "translate":
        result["site"] = result["query"] = ""
    else:
        result["site"] = result["query"] = result["text"] = ""
    if intent == "llm":
        return result
    reason = ""
    if pred.get("truncated"):
        reason = "phrase is too long"
    elif conf < t_intent:
        reason = f"not sure about the intent ({conf:.2f})"
    elif min(pred["tag_p"]) < t_slot:
        reason = f"not sure about the word tagging ({min(pred['tag_p']):.2f})"
    elif intent != "llm" and fragmented(pred["tags"]):
        reason = f"ragged tagging ({fragmented(pred['tags'])})"
    elif intent == "browser" and not (result["site"] or result["query"]):
        reason = "there is a command, but no site or query was found"
    elif (intent == "browser" and not result["site"] and result["query"]
          and not (ml_home.has_where(phrase) or ml_home.is_play(phrase))):
        reason = "search without a place to search (rule 11)"
    elif intent == "browser" and not result["site"] and _COMMAND_IN_QUERY.search(result["query"]):
        reason = "a command (включи/открой...) is left inside the query: looks like a device command"
    elif intent == "translate" and not result["text"]:
        reason = "there is a translation request, but no text was found"
    if reason:
        result.update(route="llm", site="", query="", text="", lang="none", fallback=True, reason=reason)
    return result


def _wkey(word: str) -> str:
    return word.casefold().replace("ё", "е").strip(EDGE)


def find_words(words, chunk: str, used=()) -> list:
    """Indices of consecutive words of the phrase equal to `chunk` (case, ё/е and edge punctuation do not matter)."""
    target = [_wkey(w) for w in chunk.split()]
    if not target:
        return []
    keys = [_wkey(w) for w in words]
    for start in range(len(words) - len(target) + 1):
        span = list(range(start, start + len(target)))
        if keys[start:start + len(target)] == target and not any(i in used for i in span):
            return span
    raise ValueError(f"the phrase {' '.join(words)!r} has no consecutive words {chunk!r}")


def tag_phrase(phrase: str, site: str = "", query: str = "", text: str = ""):
    """Phrase + its pieces -> (words, tags). Needed to turn your real phrases into training examples."""
    words = phrase.split()
    tags = ["O"] * len(words)
    used = set()
    for label, chunk in (("SITE", site), ("QUERY", query), ("TEXT", text)):
        for i in find_words(words, chunk, used):
            tags[i] = label
            used.add(i)
    return words, tags


def read_tsv(path):
    """ml_real.tsv / ml_mine.tsv -> [(phrase, intent, site, query, text, language)]. Lines starting with # are skipped."""
    import os
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip() or line.startswith("#"):
                continue
            p = line.split("\t") + [""] * 6
            rows.append((p[0].strip(), p[1].strip(), p[2].strip(), p[3].strip(), p[4].strip(), p[5].strip() or "none"))
    return rows
