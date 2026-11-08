"""Contract of the first-layer parser: what it understands and how phrases are labelled (no torch needed).

The first layer of the voice assistant is a small trained model. In ~3-15 ms on CPU (e.g. AMD Ryzen 7 7840HS)
it decides what the user wants and pulls the needed pieces out of the phrase. Anything complicated is handed
to the big LLM (second layer). The commands themselves are Russian (they come from speech recognition); all
examples below are Russian on purpose.

FIVE INTENTS (ROUTES)
  browser   - web sites, web services, plain Google/web searches, media playback without a named platform,
              and light factual searches (weather, currency rates/conversion, flight/train tickets & schedules).
              Also hybrid apps when explicitly requested in the browser ("открой телеграм в вебе / на сайте / в браузере").
              Pieces: SITE (which site, if named) and QUERY (what to search for / play, if said).
  app       - desktop applications, hybrid apps (telegram, steam, spotify, discord, claude, whatsapp, etc. without
              a web marker), system utilities (calculator, notepad, terminal, settings, explorer/files), local folders,
              drives and files ("открой калькулятор", "открой телеграм", "открой загрузки и отсортируй по дате").
              Pieces: SITE (the application, folder, drive or file name) and QUERY (optional inner query or action).
  system    - lightweight PC / OS control commands ("выключи комп", "переведи в спящий режим", "сделай скриншот",
              "сверни все окна", "очисти корзину", "создай папку для отчётов", "удали файл отчёт").
              Pieces: QUERY (the system action/command span), SITE is empty.
  translate - "переведи ...", "переведи: ...", "переведи как будет X на английском", "как будет X по-английски",
              "как сказать X на немецком", "как переводится X".
              Pieces: TEXT (what to translate, verbatim) and the TARGET language (a separate head: a code from
              LANGS, "none" if no language was named).
  llm       - everything else: complex questions, explanations ("скажи в чём разница между X и Y",
              "скажи погоду на завтра в 12"), smart home and appliances ("включи будильник", "включи свет",
              "включи чайник"), vague goals ("давай зайдём на гитхаб, мне нужно посмотреть один репозиторий"),
              and any phrase where the model is unsure (fallback=True).

WORD TAGS (tags): O - noise, SITE, QUERY, TEXT. Words are never changed: the model only puts labels on the
words of the original phrase, and site/query/text are assembled from those very words.

RULES (applied identically by the data generator and tests)
  1. Commands, politeness, addressing the assistant ("джарвис", "пожалуйста", "ну", "плиз") - O.
  2. A preposition before the site/app ("на", "в", "через") and generic nouns ("сайт", "официальный сайт",
     "главный сайт", "страница", "приложение", "программа", "папка", "файл", "документ") - O.
     Exception: a preposition that is part of the name itself is SITE ("на урок").
  3. A multi-word site or app name (1, 2, or 3+ words) - all words are SITE ("яндекс карты", "саунд клауд",
     "пенсионный фонд украины", "диспетчер задач"). The name is written as spoken.
  4. "Погугли / загугли / нагугли / поищи в гугле / найди в Google / найди на сайте гугл / поищи в интернете X" -
     browser without a site (site=''): "в гугле", "в Google", "на сайте гугл", "в интернете", "в сети" - O.
     "Открой гугл" (or "открой гугл и найди X") opens the site explicitly: "гугл" is SITE.
  5. QUERY is one continuous piece: everything that describes the search object or inner app action, including inner
     prepositions ("чехол для телефона", "термос на литр", "видео о создании AI-агента на Python") and adverbs ("быстро").
     Search/play verbs ("найди", "поищи", "включи", "поставь") are O, while inner app action verbs ("отсортируй по дате")
     and system commands in `system` are part of QUERY.
  6. "Включи / поставь X" (music, artist, film) without a named platform - QUERY (browser without a site).
     "Открой / запусти / зайди на X" - X is SITE (browser for websites, app for desktop/hybrid apps and folders).
  7. A site name inside the question itself ("найди как отключить рекламу в ютубе", "найди отзывы
     о ютуб премиум") is part of QUERY, not SITE.
  8. Complex questions and requests without a browser command ("кто такой павел первый", "что такое гитхаб",
     "в чём разница между list и tuple в питоне", "скажи погоду на завтра в 12") - llm.
     Light factual lookups ("скажи погоду", "скажи погоду на завтра", "глянь ка сколько будет 50 евро в рубли",
     "посмотри авиа билеты москва софия", "узнай курс доллара") - browser with site='' and query.
  9. translate: TEXT is everything that must be translated, verbatim (including leading "это", "а", "как сказать").
     Command wrappers ("переведи", "переведи:", "переведи как будет", "как будет", "слово", "фразу", "на английский",
     "с русского", "по-английски") - O.
 10. If the model is unsure (low probability or ragged tagging) - llm (fallback=True).
 11. WHERE TO SEARCH. General "найди / поищи X" without a place to search and outside light search categories
     (weather, currency, tickets/schedules) - llm.
 12. Home and appliances (light, curtains, alarm, timer, reminder, call, SMS, TV, climate, kitchen appliances) - llm.
     Desktop/hybrid apps and folders go to `app`; OS power/window/file-management commands go to `system`.
 13. "Пожалуйста" / "плиз" in translation: right after the command or before it - O; after or inside the text - TEXT.
 14. "Что значит / что означает X": without a language - llm; with a language - translate.
     "Как переводится X" is always translate (with lang='none' if no target language is named).
 15. Action verbs between a web site and the query ("на вольфрам альфа реши интеграл", "в спидтесте проверь пинг") - O.
     If the tagging is ragged (SITE has any non-SITE gap, or QUERY/TEXT is torn apart by another slot) -> llm.
 16. ОБЁРТКИ перед названием - O, в QUERY только само название:
     - пустые обёртки: "информацию / сведения о X", "проект / репозиторий с названием X", "под названием X",
       "официальный / главный сайт X" ("найди на википедии информацию о николе тесле" -> query="николе тесле");
     - тип контента СРАЗУ перед именем собственным: "фильм «Интерстеллар»", "музыку The Weeknd", "игру Elden Ring"
       -> тип - O, query = имя;
     - тип контента + предлог темы остаётся целиком: "фильм про пиратов", "музыку для сна", "видео о создании AI-агента на Python".
 17. Размытая цель без конкретного запроса ("давай зайдём на гитхаб, мне нужно посмотреть один репозиторий",
     "хочу глянуть кое-что") - llm: объекта поиска нет, решает большая LLM.
     "Мне нужен сайт X, открой его": местоимение "его/её" - O, query пустой.
"""
import re
from pathlib import Path

INTENTS = ("browser", "app", "system", "translate", "llm")
TAGS = ("O", "SITE", "QUERY", "TEXT")
LANGS = ("none", "en", "ru", "de", "fr", "es", "it", "pt", "pl", "bg", "tr", "uk", "cs", "nl",
         "sv", "el", "ro", "sr", "zh", "ja", "ko", "ar", "hi", "fi", "da", "hu", "he", "vi", "id", "kk")

# Base encoder presets (all compatible with PreTrainedTokenizerFast.word_ids() and AutoModel):
# - "tiny"  : cointegrated/rubert-tiny2 (~29M params, 3 layers, 312d, ~3-4 ms on Ryzen 7 7840HS)
# - "small" : intfloat/multilingual-e5-small (~118M params, 12 layers, 384d, ~9-12 ms, strong EN+RU brand/game awareness)
# - "base"  : deepvk/USER-base (~110M params, 12 layers, 768d, ~14-18 ms, high capacity for complex Russian syntax)
MODEL_PRESETS = {
    "tiny": "cointegrated/rubert-tiny2",
    "small": "intfloat/multilingual-e5-small",
    "base": "deepvk/USER-base",
}
BASE_MODEL = MODEL_PRESETS["tiny"]
MODEL_DIR = str(Path(__file__).resolve().parent / "ml_model")   # absolute path: independent of the working directory
MAX_LEN = 64
EDGE = " ,.;:!?«»\"'“”„"
_QUOTES = str.maketrans("", "", "«»\"“”„")


def norm(text: str) -> str:
    """Soft normalization for comparison: case, ё/е, quotes, spaces, punctuation at the edges."""
    cleaned = text.translate(_QUOTES).casefold().replace("ё", "е")
    return " ".join(cleaned.split()).strip(EDGE)



_OPT_QUERY_PREFIX = re.compile(
    r"^(?:(?:проект|репозиторий|фильм|сериал|аниме|книгу|игру|песню|трек|альбом|канал|пакет|библиотеку|группу|статью|видео|музыку)\s+(?:с\s+названием|под\s+названием)\s+"
    r"|(?:информацию|сведения|статью|статьи|материал|материалы|данные)\s+(?:о|об|про)\s+"
    r"|(?:видео|ролик|ролики|клип|клипы|туториал|гайд|урок|уроки|лекцию|лекции|подкаст|подкасты|стрим|стримы)\s+(?:о|об|про|по)\s+"
    r"|(?:фильм|фильмы|кино|сериал|сериалы|аниме|мультфильм|мультики|книгу|книги|игру|игры|песню|песни|трек|треки|музыку|альбом|плейлист|канал|репозиторий|проект|пакет|библиотеку|статью|статьи|видео|ролик|клип|стримера|пользователя|группу)\s+)",
    re.I,
)


def strip_query_wrapper(q: str) -> str:
    """Strips an optional leading content-type or meta wrapper ('фильм X' -> 'X', 'проект с названием X' -> 'X',
    'информацию о X' -> 'X') if a non-empty core remains."""
    n = norm(q)
    if not n:
        return ""
    stripped = _OPT_QUERY_PREFIX.sub("", n).strip(EDGE)
    return stripped if stripped else n


def query_matches(got: str, want: str) -> bool:
    """Flexible query/text comparison: exact match after norm(), or match after stripping optional content-type wrapper."""
    ng, nw = norm(got), norm(want)
    if ng == nw:
        return True
    if not ng or not nw:
        return False
    return strip_query_wrapper(ng) == strip_query_wrapper(nw)


def text_matches(got: str, want: str, phrase: str = "") -> bool:
    """Comparison for translation text: exact match after norm(), or 'мне <X>' vs '<X>' after 'переведи мне'."""
    ng, nw = norm(got), norm(want)
    if ng == nw:
        return True
    if "переведи мне" in norm(phrase):
        if ng.removeprefix("мне ").strip() == nw.removeprefix("мне ").strip():
            return True
    return False


def _span(tags, label):
    """Indices of words tagged `label`; gaps of O between the first and the last one are filled for QUERY/TEXT
    (for SITE, only contiguous SITE tokens are valid). If the gap contains another label, it is not filled."""
    idx = [i for i, t in enumerate(tags) if t == label]
    if not idx:
        return []
    first, last = idx[0], idx[-1]
    if label == "SITE":
        if any(t != "SITE" for t in tags[first:last + 1]):
            return idx
        return list(range(first, last + 1))
    if any(t not in (label, "O") for t in tags[first:last + 1]):
        return idx
    return list(range(first, last + 1))


def assemble(words, tags) -> dict:
    """Word tags -> {"site", "query", "text"}. Text is taken only from the words of the phrase.
    Surrounding or paired guillemets/quotes («Интерстеллар», "Stable Diffusion") are stripped cleanly."""
    def build(label):
        raw = " ".join(words[i] for i in _span(tags, label)).strip(EDGE)
        if label in ("SITE", "QUERY"):
            raw = raw.translate(_QUOTES).strip(EDGE)
        return raw

    return {"site": build("SITE"), "query": build("QUERY"), "text": build("TEXT")}


THRESHOLD_INTENT = 0.80     # calibrated for 5 intents; tuneable via `python3 test_ml.py --sweep`
THRESHOLD_SLOT = 0.60       # below this for any word -> unsure about the tagging -> llm
THRESHOLD_LANG = 0.50       # target language below the threshold -> "none"


def fragmented(tags) -> str:
    """Ragged tagging:
    - For SITE: any non-SITE word (including O) between the first and last SITE word means a torn site span.
    - For QUERY / TEXT: a word of ANOTHER slot label between the first and last word means ragged tagging.
    Returns the label name or ''."""
    for label in ("SITE", "QUERY", "TEXT"):
        idx = [i for i, t in enumerate(tags) if t == label]
        if not idx:
            continue
        span = tags[idx[0]:idx[-1] + 1]
        if label == "SITE":
            if any(t != "SITE" for t in span):
                return label
        elif any(t not in (label, "O") for t in span):
            return label
    return ""


# Глагол-команда внутри поискового запроса без сайта ("у меня руки заняты включи чайник"): модель приняла команду устройству
# за медиа-запрос. Настоящие запросы почти никогда не содержат таких слов -> лучше спросить большую LLM.
_COMMAND_IN_QUERY = re.compile(r"(?<!\w)(включи|выключи|включай|выключай|включите|выключите|запусти|запускай|открой|открывай|закрой)(?!\w)", re.I)
_OPEN_GOOGLE_START = re.compile(r"\b(открой|открывай|открыть|запусти|запускай|запустить|зайди|перейди|включи)\b", re.I)
_PURE_GOOGLE_VERB = re.compile(r"\b(\w*(?:гугли|гулги|гуггли|гугил)\w*|загуглить)\b", re.I)
_SITE_WRAPPER_WORDS = {"официальный", "главный", "сайт", "сайте"}
_SITE_TAIL_FILLERS = {"пожалуйста", "плиз", "быстро", "сейчас", "побыстрее", "скорее"}
_PRONOUN_QUERY_WORDS = {"его", "её", "ее", "их", "этот", "эту", "сайт", "страницу", "приложение"}
_SPEECH_FILLERS = {
    "джарвис", "эй", "слушай", "ну", "типа", "эээ", "короче", "пожалуйста", "плиз",
    "окей", "так", "а", "ладно", "давай", "будь", "добр", "добра", "мне", "нам",
}


def decide(phrase: str, words, pred: dict, t_intent=THRESHOLD_INTENT, t_slot=THRESHOLD_SLOT,
           t_lang=THRESHOLD_LANG) -> dict:
    """Model prediction (probabilities and word tags) -> decision where to route the phrase.

    Safety net: if the model is unsure or the tagging looks odd, the phrase is not executed blindly but
    goes to the big LLM (fallback=True). A pure function - tested without the model.
    Только структурные проверки: никаких статических списков сайтов или бытовых приборов.
    """
    best = max(range(len(INTENTS)), key=lambda k: pred["intent_p"][k])
    intent, conf = INTENTS[best], pred["intent_p"][best]
    lbest = max(range(len(LANGS)), key=lambda k: pred["lang_p"][k])
    lang = LANGS[lbest] if pred["lang_p"][lbest] >= t_lang else "none"
    tags = list(pred["tags"])

    # If the command is a pure Google verb ("загугли X", "погугли X"), any accidental SITE tags belong to QUERY
    if intent == "browser" and _PURE_GOOGLE_VERB.search(phrase):
        tags = ["QUERY" if t == "SITE" else t for t in tags]

    # Rule 16: if generic site-wrapper words ("официальный", "главный", "сайт") before the real SITE were tagged SITE, clear them to O
    if intent in ("browser", "app") and len(words) == len(tags):
        site_idx = [i for i, t in enumerate(tags) if t == "SITE"]
        real_site_idx = [i for i in site_idx if norm(words[i]) not in _SITE_WRAPPER_WORDS]
        if real_site_idx:
            first_real = real_site_idx[0]
            for i in site_idx:
                if i < first_real and norm(words[i]) in _SITE_WRAPPER_WORDS:
                    tags[i] = "O"
        # Strip trailing urgency/politeness words ("запусти калькулятор быстро") from the end of SITE
        site_idx = [i for i, t in enumerate(tags) if t == "SITE"]
        while len(site_idx) > 1 and norm(words[site_idx[-1]]) in _SITE_TAIL_FILLERS:
            tags[site_idx.pop()] = "O"

    # Rule 17: if anaphoric pronoun ("открой его") after a named SITE was tagged QUERY, clear it to O
    if intent in ("browser", "app") and len(words) == len(tags) and "SITE" in tags and "QUERY" in tags:
        q_words = [norm(words[i]) for i, t in enumerate(tags) if t == "QUERY"]
        if q_words and all(w in _PRONOUN_QUERY_WORDS for w in q_words):
            tags = ["O" if t == "QUERY" else t for t in tags]

    parts = assemble(words, tags)

    # Rule 4: "найди в гугл / в google / на сайте гугл X" (without "открой/запусти" before SITE) is a plain Google search (site='')
    first_site = next((i for i, t in enumerate(tags) if t == "SITE"), len(words))
    prefix_before_site = " ".join(words[:first_site]) if first_site <= len(words) else phrase
    if (intent == "browser" and parts["query"]
            and norm(parts["site"]) in {"гугл", "гугле", "google", "гугл поиск", "google search"}
            and not _OPEN_GOOGLE_START.search(prefix_before_site)):
        parts["site"] = ""

    result = {"route": intent, "site": parts["site"], "query": parts["query"], "text": parts["text"],
              "lang": lang if intent == "translate" else "none", "confidence": conf,
              "fallback": False, "reason": "", "phrase": phrase}
    if intent in ("browser", "app"):
        result["text"] = ""
    elif intent == "system":
        result["site"] = result["text"] = ""
    elif intent == "translate":
        result["site"] = result["query"] = ""
    else:
        result["site"] = result["query"] = result["text"] = ""
        if conf < t_intent:
            result.update(fallback=True, reason=f"not sure about the intent ({conf:.2f})")
        return result

    has_cmd_o = any(norm(words[i]) not in _SPEECH_FILLERS for i, t in enumerate(tags) if t == "O" and i < len(words))
    reason = ""
    if pred.get("truncated"):
        reason = "phrase is too long"
    elif conf < t_intent:
        reason = f"not sure about the intent ({conf:.2f})"
    elif min(pred["tag_p"]) < t_slot:
        reason = f"not sure about the word tagging ({min(pred['tag_p']):.2f})"
    elif fragmented(tags):
        reason = f"ragged tagging ({fragmented(tags)})"
    elif intent == "browser" and not (result["site"] or result["query"]):
        reason = "there is a command, but no site or query was found"
    elif intent == "browser" and not result["site"] and not has_cmd_o and not _PURE_GOOGLE_VERB.search(phrase):
        reason = "siteless query without a search or play command verb"
    elif intent == "browser" and not result["site"] and _COMMAND_IN_QUERY.search(result["query"]):
        reason = "a command (включи/открой...) is left inside the query: looks like a device command"
    elif intent == "app" and not result["site"]:
        reason = "there is an app command, but no app or folder was found"
    elif intent == "system" and not result["query"]:
        reason = "there is a system command, but no action was found"
    elif intent == "translate" and not result["text"]:
        reason = "there is a translation request, but no text was found"
    if reason:
        result.update(route="llm", site="", query="", text="", lang="none", fallback=True, reason=reason)
    return result


def _wkey(word: str) -> str:
    return word.translate(_QUOTES).casefold().replace("ё", "е").strip(EDGE)


def find_words(words, chunk: str, used=()) -> list:
    """Indices of consecutive words of the phrase equal to `chunk` (case, ё/е and edge punctuation do not matter)."""
    target = [_wkey(w) for w in chunk.split() if _wkey(w)]
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
