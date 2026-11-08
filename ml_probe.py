#!/usr/bin/env python3
"""Зонд обобщения (живой экзамен): сотни фраз из независимых рамок, отложенной лексики и живых провалов (USER_HARD_CASES).

Зачем: старый экзамен был похож на генератор и не ловил провалы на живой речи. Зонд проверяет, понимает ли модель
СТРУКТУРУ фразы (где сайт, где запрос, где приложение, где перевод, а где размытая цель для LLM), когда сайт / игра /
артист / приложение ей незнакомы или когда конструкция нестандартная.

Правила честности:
  * Слова из HELD_* никогда не попадают в генератор (ml_vocab.py, ml_gen.py) и в ml_real.tsv - это проверяет test_ml.py.
  * Рамки зонда и USER_HARD_CASES пишутся отдельно от шаблонов генератора; фразы зонда исключаются из обучения.

Использование:
    python3 ml_probe.py --show 40          # посмотреть сгенерированные фразы зонда
    python3 ml_probe.py --hard             # показать все фразы из USER_HARD_CASES
    python3 test_ml.py --live              # прогнать модель по зонду и USER_HARD_CASES (цветной отчёт)
"""
import argparse
import random
import re
from pathlib import Path

from ml_spec import INTENTS, LANGS as SPEC_LANGS, decide, norm, query_matches, read_tsv, strip_query_wrapper, tag_phrase, text_matches

# ------------------------------------------------------------------ отложенная лексика (модель её никогда не видит в обучении)
# сайты: (именительный, предложный, винительный)
HELD_SITES = [
    ("мегого", "мегого", "мегого"), ("кинопаб", "кинопабе", "кинопаб"), ("more tv", "more tv", "more tv"),
    ("монобанк", "монобанке", "монобанк"), ("нова пошта", "нова пошта", "нова пошта"), ("приват24", "приват24", "приват24"),
    ("вкусвилл", "вкусвилле", "вкусвилл"), ("дром", "дроме", "дром"), ("зоон", "зооне", "зоон"),
    ("прозорро", "прозорро", "прозорро"), ("ютуб кидс", "ютуб кидс", "ютуб кидс"), ("квизлет", "квизлете", "квизлет"),
    ("crunchyroll", "crunchyroll", "crunchyroll"), ("indie hackers", "indie hackers", "indie hackers"),
    ("rotten tomatoes", "rotten tomatoes", "rotten tomatoes"), ("etsy", "etsy", "etsy"), ("glassdoor", "glassdoor", "glassdoor"),
    ("wolt", "wolt", "wolt"), ("bolt food", "bolt food", "bolt food"), ("archive org", "archive org", "archive org"),
    ("пенсионный фонд украины", "пенсионном фонде украины", "пенсионный фонд украины"),
    ("пенсионый фонд украины", "пенсионом фонде украины", "пенсионый фонд украины"),
    ("налоговая служба украины", "налоговой службе украины", "налоговую службу украины"),
    ("портал открытых данных", "портале открытых данных", "портал открытых данных"),
    ("министерство здравоохранения", "министерстве здравоохранения", "министерство здравоохранения"),
    ("городская библиотека онлайн", "городской библиотеке онлайн", "городскую библиотеку онлайн"),
    ("портал судебной власти", "портале судебной власти", "портал судебной власти"),
    ("реестр судебных решений", "реестре судебных решений", "реестр судебных решений"),
]
# гибридные / десктопные приложения, которых нет в обучении
HELD_APPS = [
    ("obsidian", "obsidian"), ("anki", "anki"), ("krita", "krita"), ("inkscape", "inkscape"),
    ("signal", "signal"), ("thunderbird", "thunderbird"), ("battle net", "battle net"), ("gog galaxy", "gog galaxy"),
    ("ubisoft connect", "ubisoft connect"), ("element", "element"), ("daVinci resolve", "daVinci resolve"),
    ("bitwarden", "bitwarden"), ("anydesk", "anydesk"), ("teamviewer", "teamviewer"),
]
KNOWN_HYBRID = [
    ("steam", "steam"), ("стим", "стиме"), ("spotify", "spotify"), ("спотифай", "спотифае"),
    ("discord", "discord"), ("дискорд", "дискорде"), ("telegram", "telegram"), ("телеграм", "телеграме"),
]
HELD_GAMES = [
    "hades 2", "disco elysium", "outer wilds", "slay the spire", "deep rock galactic", "sea of thieves",
    "hollow knight silksong", "lethal company", "palworld", "manor lords", "helldivers 2", "frostpunk 2",
    "noita", "rain world", "ultrakill", "barotrauma", "kenshi", "inscryption",
]
HELD_ARTISTS = [
    "fred again", "tame impala", "glass animals", "кишлака", "jony", "лсп",
    "the neighbourhood", "mitski", "три дня дождя", "bicep", "bonobo", "moderat", "stromae",
]
HELD_TITLES = [
    "шоу трумана", "бегущего по лезвию", "сёгуна", "разделение", "пацанов", "мандалорца",
    "оппенгеймера", "дюну два", "твин пикс", "настоящего детектива", "андор", "мистера робота",
]
HELD_PRODUCTS = [
    "робот пылесос до двадцати тысяч", "палатку на четверых", "зимние шины r16", "электросамокат xiaomi",
    "механическую клавиатуру", "детское автокресло", "кофемолку жерновую", "рюкзак для ноутбука",
    "увлажнитель воздуха с гигрометром", "кронштейн для двух мониторов", "графический планшет", "павербанк на 20000",
]
HELD_TOPICS = [
    "как работает квантовый компьютер", "рецепт солянки", "обзор rtx 5070", "как настроить nginx",
    "история римской империи", "тренировки для спины", "как ухаживать за орхидеей", "курс по rust для новичков",
    "видео о создании ai-агента на python", "видео о создании телеграм-бота на python",
    "видео о создании парсера на rust", "видео о создании игры на godot",
    "последние новости openai", "последние новости anthropic", "последние новости nvidia",
]
HELD_REPOS = ["comfyui", "vllm", "ruff", "stable diffusion webui", "open interpreter", "auto gpt", "litellm", "dify"]
HELD_PEOPLE = ["николе тесле", "аде лавлейс", "алане тьюринге", "марии кюри", "ричарде фейнмане", "стивене хокинге"]

# ------------------------------------------------------------------ 16 живых провалов пользователя + вариации
# Формат: (категория, фраза, intent, site, query, text, lang)
USER_HARD_CASES = [
    # --- Точные фразы из сессии пользователя ---
    ("user_16", "Перейди на официальный сайт Python", "browser", "Python", "", "", "none"),
    ("user_16", "открой пенсионый фонд украины", "browser", "пенсионый фонд украины", "", "", "none"),
    ("user_16", "открой сайт пенсионый фонд украины", "browser", "пенсионый фонд украины", "", "", "none"),
    ("user_16", "Мне нужен сайт Stack Overflow, открой его", "browser", "Stack Overflow", "", "", "none"),
    ("user_16", "Открой Steam и найди Counter-Strike 2", "app", "Steam", "Counter-Strike 2", "", "none"),
    ("user_16", "Найди в Google последние новости OpenA", "browser", "", "последние новости OpenA", "", "none"),
    ("user_16", "Переведи: я ищу стажировку в области искусственного интеллекта", "translate", "", "", "я ищу стажировку в области искусственного интеллекта", "none"),
    ("user_16", "переведи как будет неудобно на англиском", "translate", "", "", "неудобно", "en"),
    ("user_16", "Включи Игру престолов в Kingo", "browser", "Kingo", "Игру престолов", "", "none"),
    ("user_16", "Найди фильм «Интерстеллар» на Кинопоиске", "browser", "Кинопоиске", "Интерстеллар", "", "none"),
    ("user_16", "Открой Steam и найди Elden Ring", "app", "Steam", "Elden Ring", "", "none"),
    ("user_16", "Найди на GitHub проект с названием Stable Diffusion", "browser", "GitHub", "Stable Diffusion", "", "none"),
    ("user_16", "Найди на YouTube видео о создании AI-агента на Python", "browser", "YouTube", "видео о создании AI-агента на Python", "", "none"),
    ("user_16", "Открой Spotify и найди музыку The Weeknd", "app", "Spotify", "The Weeknd", "", "none"),
    ("user_16", "Найди на Wikipedia информацию о Николе Тесле", "browser", "Wikipedia", "Николе Тесле", "", "none"),
    ("user_16", "Давай зайдём на GitHub, мне нужно посмотреть один репозиторий", "llm", "", "", "", "none"),
    ("user_16", "найди как варить макароны в ютубе", "browser", "ютубе", "как варить макароны", "", "none"),
    ("user_16", "найди игру пристолов в hd rezka", "browser", "hd rezka", "игру пристолов", "", "none"),
    ("user_16", "найди как варить макроны в hd rezka", "browser", "hd rezka", "как варить макроны", "", "none"),
    ("user_16", "включи как закалялся стайл на ютубе", "browser", "ютубе", "как закалялся стайл", "", "none"),

    # --- Вариации 1: официальный / главный сайт + технологии и бренды ---
    ("official_site", "Перейди на официальный сайт Rust", "browser", "Rust", "", "", "none"),
    ("official_site", "Открой официальный сайт Docker", "browser", "Docker", "", "", "none"),
    ("official_site", "Зайди на официальный сайт PyTorch", "browser", "PyTorch", "", "", "none"),
    ("official_site", "Перейди на официальный сайт Blender", "browser", "Blender", "", "", "none"),
    ("official_site", "Открой главный сайт FastAPI", "browser", "FastAPI", "", "", "none"),
    ("official_site", "Запусти официальный сайт Kubernetes", "browser", "Kubernetes", "", "", "none"),

    # --- Вариации 2-3: 3-словные порталы, ведомства и опечатки ---
    ("portal_3w", "открой пенсионный фонд украины", "browser", "пенсионный фонд украины", "", "", "none"),
    ("portal_3w", "зайди на налоговая служба украины", "browser", "налоговая служба украины", "", "", "none"),
    ("portal_3w", "открой сайт министерство здравоохранения", "browser", "министерство здравоохранения", "", "", "none"),
    ("portal_3w", "перейди на портал открытых данных", "browser", "портал открытых данных", "", "", "none"),
    ("portal_3w", "открой городская библиотека онлайн", "browser", "городская библиотека онлайн", "", "", "none"),
    ("portal_3w", "зайди на сайт реестр судебных решений", "browser", "реестр судебных решений", "", "", "none"),
    ("portal_3w", "открой портал судебной власти", "browser", "портал судебной власти", "", "", "none"),

    # --- Вариации 4: анафорические местоимения ("открой его / запусти его") ---
    ("pronoun_open", "Мне нужен сайт GitHub, открой его", "browser", "GitHub", "", "", "none"),
    ("pronoun_open", "Мне нужен сайт Кинопоиск, запусти его", "browser", "Кинопоиск", "", "", "none"),
    ("pronoun_open", "Нужен сайт Habr, открой его пожалуйста", "browser", "Habr", "", "", "none"),
    ("pronoun_open", "Хочу зайти на сайт Coursera, открой его", "browser", "Coursera", "", "", "none"),
    ("pronoun_open", "Мне нужен сайт Мегого открой его", "browser", "Мегого", "", "", "none"),

    # --- Вариации 5, 11, 14: гибридные приложения (Steam / Spotify / Epic) + игры и музыка ---
    ("hybrid_app", "Запусти Steam и найди Baldur's Gate 3", "app", "Steam", "Baldur's Gate 3", "", "none"),
    ("hybrid_app", "Открой Steam и найди Cyberpunk 2077", "app", "Steam", "Cyberpunk 2077", "", "none"),
    ("hybrid_app", "Открой Стим и найди Hades 2", "app", "Стим", "Hades 2", "", "none"),
    ("hybrid_app", "В Стиме найди игру Disco Elysium", "app", "Стиме", "Disco Elysium", "", "none"),
    ("hybrid_app", "Запусти Spotify и найди музыку Daft Punk", "app", "Spotify", "Daft Punk", "", "none"),
    ("hybrid_app", "Открой Спотифай и найди музыку Tame Impala", "app", "Спотифай", "Tame Impala", "", "none"),
    ("hybrid_app", "В Spotify включи трек The Weeknd", "app", "Spotify", "The Weeknd", "", "none"),
    ("hybrid_app", "Открой Epic Games и найди Alan Wake 2", "app", "Epic Games", "Alan Wake 2", "", "none"),

    # --- Вариации 6: поиск в Google / на сайте гугл ---
    ("google_rule", "Найди в Google последние новости Anthropic", "browser", "", "последние новости Anthropic", "", "none"),
    ("google_rule", "Найди в Google последние новости Nvidia", "browser", "", "последние новости Nvidia", "", "none"),
    ("google_rule", "Найди на сайте гугл курс доллара на сегодня", "browser", "", "курс доллара на сегодня", "", "none"),
    ("google_rule", "Поищи на сайте Google расписание электричек", "browser", "", "расписание электричек", "", "none"),
    ("google_rule", "Найди в гугл как настроить nginx", "browser", "", "как настроить nginx", "", "none"),
    ("google_rule", "Поищи в Google обзор RTX 5070", "browser", "", "обзор RTX 5070", "", "none"),

    # --- Вариации 7-8: перевод с двоеточием, "переведи как будет", опечатки в языке ---
    ("translate_hard", "Переведи: мы разрабатываем систему компьютерного зрения", "translate", "", "", "мы разрабатываем систему компьютерного зрения", "none"),
    ("translate_hard", "Переведи: где находится ближайшая станция метро", "translate", "", "", "где находится ближайшая станция метро", "none"),
    ("translate_hard", "Переведи на английский: я ищу стажировку в области искусственного интеллекта", "translate", "", "", "я ищу стажировку в области искусственного интеллекта", "en"),
    ("translate_hard", "переведи как будет вдохновение на англиском", "translate", "", "", "вдохновение", "en"),
    ("translate_hard", "переведи как будет неловко на немецком", "translate", "", "", "неловко", "de"),
    ("translate_hard", "переведи как сказать странно на французком", "translate", "", "", "странно", "fr"),
    ("translate_hard", "переведи как будет дорого на испанском", "translate", "", "", "дорого", "es"),

    # --- Вариации 9-10: кавычки «...», предлог "в" с незнакомым сайтом, склонения названий ---
    ("media_quotes", "Включи Шоу Трумана в Мегого", "browser", "Мегого", "Шоу Трумана", "", "none"),
    ("media_quotes", "Включи Бегущего по лезвию в Кинопабе", "browser", "Кинопабе", "Бегущего по лезвию", "", "none"),
    ("media_quotes", "Поставь Магическую битву в Crunchyroll", "browser", "Crunchyroll", "Магическую битву", "", "none"),
    ("media_quotes", "Найди фильм «Оппенгеймер» на Кинопоиске", "browser", "Кинопоиске", "Оппенгеймер", "", "none"),
    ("media_quotes", "Найди сериал «Разделение» на Кинопоиске", "browser", "Кинопоиске", "Разделение", "", "none"),
    ("media_quotes", "Открой Кинопоиск и найди фильм «Дюна»", "browser", "Кинопоиск", "Дюна", "", "none"),

    # --- Вариации 12, 13, 15: обёртками ("проект с названием", "информацию о") и "видео о ... на Python" ---
    ("wrappers_dev", "Найди на GitHub проект с названием ComfyUI", "browser", "GitHub", "ComfyUI", "", "none"),
    ("wrappers_dev", "Найди на гитхабе репозиторий с названием vllm", "browser", "гитхабе", "vllm", "", "none"),
    ("wrappers_dev", "Найди на YouTube видео о создании телеграм-бота на Python", "browser", "YouTube", "видео о создании телеграм-бота на Python", "", "none"),
    ("wrappers_dev", "Найди на ютубе видео о создании парсера на Rust", "browser", "ютубе", "видео о создании парсера на Rust", "", "none"),
    ("wrappers_dev", "Поищи на YouTube ролик о собаках", "browser", "YouTube", "ролик о собаках", "", "none"),
    ("wrappers_dev", "Найди на Wikipedia информацию об Алане Тьюринге", "browser", "Wikipedia", "Алане Тьюринге", "", "none"),
    ("wrappers_dev", "Найди на Википедии информацию о Марии Кюри", "browser", "Википедии", "Марии Кюри", "", "none"),
    ("wrappers_dev", "Поищи на Wikipedia сведения о Ричарде Фейнмане", "browser", "Wikipedia", "Ричарде Фейнмане", "", "none"),

    # --- Вариации 16: размытая цель ("давай зайдём на ..., мне нужно посмотреть один ...") -> llm ---
    ("vague_llm", "Давай зайдём на YouTube, мне нужно посмотреть одно видео", "llm", "", "", "", "none"),
    ("vague_llm", "Давай зайдём на Хабр, мне надо найти одну статью", "llm", "", "", "", "none"),
    ("vague_llm", "Слушай давай откроем Кинопоиск, хочу глянуть один фильм", "llm", "", "", "", "none"),
    ("vague_llm", "Давай зайдём на Википедию, мне нужно кое-что проверить", "llm", "", "", "", "none"),
]

PREFIX = ["", "", "", "", "слушай", "окей", "джарвис", "будь добр", "эй", "ну", "короче", "так"]
OPEN = ["открой", "запусти", "зайди на", "перейди на", "открой мне", "открой пожалуйста", "можешь открыть",
        "мне нужно открыть", "хочу зайти на", "давай откроем", "открывай", "врубай"]
FIND = ["найди", "поищи", "ищи", "глянь", "посмотри", "найди мне", "поищи мне", "покажи", "вбей", "пробей"]
GOOGLE = ["погугли", "загугли", "найди в гугле", "найди в google", "найди на сайте гугл", "поищи в интернете",
          "найди в интернете", "пробей в гугле", "глянь в интернете", "вбей в гугл", "поищи в сети", "погугли мне"]
PLAY = ["включи", "поставь", "включи мне", "хочу послушать", "давай послушаем", "врубай", "запусти песни"]
LANGS = [("английский", "en", "по английски"), ("немецкий", "de", "по немецки"), ("французский", "fr", "по французски"),
         ("испанский", "es", "по испански"), ("польский", "pl", "по польски"), ("японский", "ja", "по японски"),
         ("турецкий", "tr", "по турецки"), ("итальянский", "it", "по итальянски")]
LANG_TYPO = {"английский": "англиский", "немецкий": "немецкий", "французский": "французкий", "испанский": "испанкий",
             "польский": "польский", "японский": "японский", "турецкий": "турецкий", "итальянский": "итальянский"}
TR_TEXT = [
    "я немного опоздаю", "где ближайшая аптека", "я опоздаю на десять минут", "сколько это стоит", "у меня аллергия на орехи",
    "приятного аппетита", "я ищу стажировку в области искусственного интеллекта", "мы забронировали столик на двоих",
    "неудобно", "вдохновение", "можно счёт", "я заблудился", "неловко", "мы разрабатываем систему машинного обучения",
]
SYSTEM = [
    "выруби комп", "усыпи компьютер", "отправь ноутбук в сон", "сделай скрин экрана", "заблокируй ноут",
    "перезапусти компьютер", "выключи ноутбук через полчаса", "сверни все окна на экране", "очисти корзину на компе",
    "создай папку для скриншотов", "удали файл черновик", "сделай звук на компе тише", "заверши работу системы",
]
LLM_QUESTIONS = [
    "почему небо голубое", "чем отличается python от javascript", "сколько будет двести на тридцать",
    "придумай имя для кота", "что лучше айфон или пиксель", "объясни что такое рекурсия",
    "напиши поздравление маме", "как думаешь стоит ли учить rust", "расскажи анекдот про программистов",
]
LLM_HOME = [
    "включи чайник на кухне", "выключи свет в прихожей", "поставь будильник на семь", "включи увлажнитель воздуха",
    "закрой шторы в спальне", "сделай теплее в комнате", "напомни купить хлеб вечером", "включи робот пылесос в зале",
]
VAGUE = [
    "мне нужно посмотреть один репозиторий", "хочу глянуть одну штуку", "надо кое что проверить",
    "мне нужно найти одну статью", "хочу посмотреть кое что", "мне надо проверить одно видео",
]


def _cap(rng, s):
    """Регистр как при наборе: иногда с заглавной первая буква фразы или названия."""
    return s[:1].upper() + s[1:] if s and rng.random() < 0.35 else s


def _brand(rng, name):
    if re.search("[a-z]", name) and rng.random() < 0.5:
        return " ".join(w[:1].upper() + w[1:] for w in name.split())
    return name


def _with_prefix(rng, phrase):
    p = rng.choice(PREFIX)
    return f"{p} {phrase}".strip()


# Каждая рамка: (класс, функция rng -> (phrase, intent, site, query, text, lang))
def f_site_only(rng):
    nom, prep, acc = rng.choice(HELD_SITES)
    s = _brand(rng, nom)
    v = rng.choice(OPEN)
    form = rng.randrange(5)
    if form == 0:
        return f"{v} {s}", "browser", s, "", "", "none"
    if form == 1:
        return f"{s} {rng.choice(['открой', 'запусти', 'открой мне'])}", "browser", s, "", "", "none"
    if form == 2:
        return f"мне нужен сайт {s}, открой его", "browser", s, "", "", "none"
    if form == 3:
        return f"{v} сайт {s}", "browser", s, "", "", "none"
    return f"{rng.choice(['перейди на', 'зайди на', 'открой'])} официальный сайт {s}", "browser", s, "", "", "none"


def f_site_query(rng):
    nom, prep, acc = rng.choice(HELD_SITES)
    s, sp = _brand(rng, nom), _brand(rng, prep)
    q = rng.choice(HELD_PRODUCTS + HELD_TOPICS + HELD_TITLES)
    v, f = rng.choice(OPEN), rng.choice(FIND)
    prep_word = rng.choice(["на", "в"])
    form = rng.randrange(6)
    if form == 0:
        return f"{v} {s} и {f} {q}", "browser", s, q, "", "none"
    if form == 1:
        return f"{f} {q} {prep_word} {sp}", "browser", sp, q, "", "none"
    if form == 2:
        return f"{f} {prep_word} {sp} {q}", "browser", sp, q, "", "none"
    if form == 3:
        return f"{prep_word} {sp} {f} {q}", "browser", sp, q, "", "none"
    if form == 4:
        return f"{v} {s} там {f} {q}", "browser", s, q, "", "none"
    return f"зайди на {s} и посмотри {q}", "browser", s, q, "", "none"


def f_wrapper(rng):
    """Проверка рамок с обёртками, кавычками и предлогами."""
    nom, prep, acc = rng.choice(HELD_SITES)
    sp = _brand(rng, prep)
    f = rng.choice(FIND)
    form = rng.randrange(5)
    if form == 0:
        t = rng.choice(HELD_TITLES)
        q_str = f"«{t}»" if rng.random() < 0.5 else t
        return f"{f} фильм {q_str} на {sp}", "browser", sp, t, "", "none"
    if form == 1:
        r = rng.choice(HELD_REPOS)
        return f"{f} на {sp} проект с названием {r}", "browser", sp, r, "", "none"
    if form == 2:
        p = rng.choice(HELD_PEOPLE)
        return f"{f} на {sp} информацию о {p}", "browser", sp, p, "", "none"
    if form == 3:
        t = rng.choice(HELD_TITLES)
        return f"{f} на {sp} сериал под названием {t}", "browser", sp, t, "", "none"
    a = rng.choice(HELD_ARTISTS)
    return f"{f} на {sp} музыку {a}", "browser", sp, a, "", "none"


def f_google(rng):
    q = rng.choice(HELD_TOPICS + HELD_PRODUCTS)
    g = rng.choice(GOOGLE)
    if rng.random() < 0.25 and g in ("погугли", "загугли", "найди в гугле"):
        return f"{q} {g}", "browser", "", q, "", "none"
    return f"{g} {q}", "browser", "", q, "", "none"


def f_play(rng):
    a = rng.choice(HELD_ARTISTS + HELD_TITLES)
    return f"{rng.choice(PLAY)} {a}", "browser", "", a, "", "none"


def f_app(rng):
    form = rng.randrange(6)
    if form == 0:
        a = _brand(rng, rng.choice(HELD_APPS)[0])
        cmd = rng.choice(["открой приложение", "запусти программу", "открой программу", "запусти приложение"])
        return f"{cmd} {a}", "app", a, "", "", "none"
    if form == 1:
        a, ap = rng.choice([x for x in KNOWN_HYBRID if x[0] in ("steam", "стим")])
        a = _brand(rng, a)
        g = rng.choice(HELD_GAMES)
        return f"{rng.choice(['открой', 'запусти'])} {a} и {rng.choice(FIND)} {g}", "app", a, g, "", "none"
    if form == 2:
        a, ap = rng.choice([x for x in KNOWN_HYBRID if x[0] in ("spotify", "спотифай")])
        a, ap = _brand(rng, a), _brand(rng, ap)
        ar = rng.choice(HELD_ARTISTS)
        if rng.random() < 0.5:
            w = rng.choice(["", "музыку "])
            return f"открой {a} и {rng.choice(['включи', 'найди'])} {w}{ar}", "app", a, ar, "", "none"
        return f"в {ap} {rng.choice(['включи', 'найди', 'поставь'])} {ar}", "app", ap, ar, "", "none"
    if form == 3:
        a, ap = rng.choice(KNOWN_HYBRID)
        a = _brand(rng, a)
        return f"{rng.choice(['открой', 'запусти'])} {a} {rng.choice(['в браузере', 'в вебе', 'на сайте'])}", "browser", a, "", "", "none"
    if form == 4:
        a = _brand(rng, rng.choice(HELD_APPS)[0])
        return f"{rng.choice(['открой', 'запусти'])} {a} на компьютере", "app", a, "", "", "none"
    a, ap = rng.choice([x for x in KNOWN_HYBRID if x[0] in ("steam", "стим")])
    g = rng.choice(HELD_GAMES)
    ap = _brand(rng, ap)
    return f"в {ap} {rng.choice(FIND)} игру {g}", "app", ap, g, "", "none"


def f_translate(rng):
    lang, code, adv = rng.choice(LANGS)
    t = rng.choice(TR_TEXT)
    form = rng.randrange(7)
    if form == 0:
        return f"переведи {t} на {lang}", "translate", "", "", t, code
    if form == 1:
        return f"переведи как будет {t} на {LANG_TYPO[lang]}", "translate", "", "", t, code
    if form == 2:
        return f"как будет {t} {adv}", "translate", "", "", t, code
    if form == 3:
        return f"переведи {t}", "translate", "", "", t, "none"
    if form == 4:
        return f"переведи: {t}", "translate", "", "", t, "none"
    if form == 5:
        return f"{t} переведи на {lang}", "translate", "", "", t, code
    return f"переведи на {lang} {t}", "translate", "", "", t, code


def f_system(rng):
    s = rng.choice(SYSTEM)
    return s, "system", "", s, "", "none"


def f_llm(rng):
    form = rng.randrange(5)
    if form == 0:
        return rng.choice(LLM_QUESTIONS), "llm", "", "", "", "none"
    if form == 1:
        return rng.choice(LLM_HOME), "llm", "", "", "", "none"
    if form == 2:
        nom, prep, acc = rng.choice(HELD_SITES)
        return f"давай зайдём на {_brand(rng, nom)}, {rng.choice(VAGUE)}", "llm", "", "", "", "none"
    if form == 3:
        return f"найди {rng.choice(['формулу площади круга', 'смысл жизни', 'хорошее имя для собаки', 'идею для подарка'])}", "llm", "", "", "", "none"
    nom, prep, acc = rng.choice(HELD_SITES)
    return f"{rng.choice(['как скачать видео с', 'почему не работает', 'что такое', 'как зарегистрироваться на'])} {nom}", "llm", "", "", "", "none"


FRAMES = [
    ("site_only", f_site_only, 14),
    ("site_query", f_site_query, 16),
    ("wrapper", f_wrapper, 12),
    ("google", f_google, 10),
    ("play", f_play, 6),
    ("app", f_app, 16),
    ("translate", f_translate, 14),
    ("system", f_system, 6),
    ("llm", f_llm, 14),
]


def _real_keys():
    real_path = Path(__file__).with_name("ml_real.tsv")
    return {norm(r[0]) for r in read_tsv(real_path)}


def generate(n=1200, seed=2026, include_hard=True):
    rng = random.Random(seed)
    real_seen = _real_keys()
    out, seen = [], set()
    if include_hard:
        for row in USER_HARD_CASES:
            key = norm(row[1])
            if key not in seen and key not in real_seen:
                seen.add(key)
                out.append(row)
    guard = 0
    while len(out) < n and guard < n * 50:
        guard += 1
        cls, fn, _ = rng.choices(FRAMES, weights=[f[2] for f in FRAMES])[0]
        phrase, intent, site, query, text, lang = fn(rng)
        phrase = _cap(rng, _with_prefix(rng, phrase))
        key = norm(phrase)
        if key in seen or key in real_seen:
            continue
        seen.add(key)
        out.append((cls, phrase, intent, site, query, text, lang))
    return out


def held_words():
    """Все отложенные слова: генератор (ml_vocab.py) и ml_real.tsv не должны их содержать как отдельные элементы."""
    items = (
        [x[0] for x in HELD_SITES]
        + [x[0] for x in HELD_APPS]
        + HELD_GAMES
        + HELD_ARTISTS
        + HELD_TITLES
        + HELD_REPOS
    )
    return sorted({norm(x) for x in items})


def _oracle_check_row(label: str, words: list, tags: list, intent: str, lang: str, site: str, query: str, text: str):
    phrase = " ".join(words)
    ip = [0.002] * len(INTENTS)
    ip[INTENTS.index(intent)] = 0.992
    lp = [0.001] * len(SPEC_LANGS)
    lp[SPEC_LANGS.index(lang if intent == "translate" else "none")] = 0.99
    pred = {
        "intent_p": ip,
        "lang_p": lp,
        "tags": list(tags),
        "tag_p": [0.99] * len(words),
        "truncated": False,
    }
    res = decide(phrase, words, pred)
    if res["route"] != intent:
        return f"{label}: route gold={intent} != decide={res['route']} ({res['reason']}) | {phrase!r}"
    if norm(res["site"]) != norm(site):
        return f"{label}: site gold={site!r} != decide={res['site']!r} | {phrase!r}"
    if not query_matches(res["query"], query):
        return f"{label}: query gold={query!r} != decide={res['query']!r} | {phrase!r}"
    if not text_matches(res["text"], text, phrase):
        return f"{label}: text gold={text!r} != decide={res['text']!r} | {phrase!r}"
    if intent == "translate" and res["lang"] != lang:
        return f"{label}: lang gold={lang!r} != decide={res['lang']!r} | {phrase!r}"
    return None


def audit_all(n_gen: int = 8000) -> dict:
    """Полный структурный аудит словарей, датасетов (ml_real.tsv, ml_mine.tsv, ml_exam.py),
    генератора (ml_gen.py) и правил decide() (ml_spec.py)."""
    import ml_exam as E
    import ml_gen as G
    import ml_vocab as V

    issues = []

    # 1. Проверка утечки отложенной лексики HELD_* в ml_vocab.py и ml_real.tsv
    vocab_strings = set()
    for val in vars(V).values():
        if isinstance(val, (list, tuple, set)):
            for item in val:
                if isinstance(item, str):
                    vocab_strings.add(norm(item))
                elif isinstance(item, tuple):
                    for sub in item:
                        if isinstance(sub, str):
                            vocab_strings.add(norm(sub))
    leaked_vocab = [w for w in held_words() if w in vocab_strings]
    if leaked_vocab:
        issues.append(f"HELD_* leaked into ml_vocab.py: {leaked_vocab}")

    # 2. Проверка утечки приложений в real_sites() и all_sites()
    apps = G.app_norms()
    leaked_sites = [s for s in G.all_sites() if norm(s) in apps]
    if leaked_sites:
        issues.append(f"Apps leaked into all_sites(): {leaked_sites}")

    # 3. Проверка пересечений между ml_real.tsv и проверочными наборами
    real_rows = G.real_rows()
    exam_keys = G._exam_keys()
    overlap_real = [" ".join(r["words"]) for r in real_rows if norm(" ".join(r["words"])) in exam_keys]
    if overlap_real:
        issues.append(f"ml_real.tsv overlaps with eval sets ({len(overlap_real)} rows): {overlap_real[:5]}")

    # 4. Oracle Consistency Check (ml_real.tsv, ml_mine.tsv, ml_exam.py, ml_probe.py, ml_gen.py)
    oracle_errors = []
    for idx, r in enumerate(real_rows, 1):
        err = _oracle_check_row(f"real:{idx}", r["words"], r["tags"], r["intent"], r["lang"], r["site"], r["query"], r["text"])
        if err:
            oracle_errors.append(err)

    for idx, (phrase, intent, site, query, text, lang) in enumerate(read_tsv(G.MINE_PATH), 1):
        words, tags = tag_phrase(phrase, site, query, text)
        if intent == "llm":
            tags = ["O"] * len(words)
        err = _oracle_check_row(f"mine:{idx}", words, tags, intent, lang, site, query, text)
        if err:
            oracle_errors.append(err)

    for idx, row in enumerate(E.EXAM, 1):
        phrase, intent = row[0], row[1]
        site = row[2] if len(row) > 2 else ""
        query = row[3] if len(row) > 3 else ""
        text = row[4] if len(row) > 4 else ""
        lang = row[5] if len(row) > 5 else "none"
        words, tags = tag_phrase(phrase, site, query, text)
        if intent == "llm":
            tags = ["O"] * len(words)
        err = _oracle_check_row(f"exam:{idx}", words, tags, intent, lang, site, query, text)
        if err:
            oracle_errors.append(err)

    probe_rows = generate(1200)
    for idx, (cat, phrase, intent, site, query, text, lang) in enumerate(probe_rows, 1):
        words, tags = tag_phrase(phrase, site, query, text)
        if intent == "llm":
            tags = ["O"] * len(words)
        err = _oracle_check_row(f"probe:{cat}:{idx}", words, tags, intent, lang, site, query, text)
        if err:
            oracle_errors.append(err)

    gen_samples = G.generate(n_gen, seed=42, which="all")
    rng = random.Random(42)
    pools = G.Pools("all")
    for r in real_rows:
        a = G.augment(r, rng, pools)
        if a:
            gen_samples.append(a)

    browser_app_leaks = []
    for idx, s in enumerate(gen_samples, 1):
        err = _oracle_check_row(f"gen:{s['tpl']}:{idx}", s["words"], s["tags"], s["intent"], s["lang"], s["site"], s["query"], s["text"])
        if err:
            oracle_errors.append(err)
        if s["intent"] == "browser" and s["site"] and norm(s["site"]) in apps and s["tpl"] != "browser.open_site_suffix":
            p_low = norm(" ".join(s["words"]))
            if not any(m in p_low for m in G._WEB_EXPLICIT_MARKERS):
                browser_app_leaks.append(f"{s['tpl']}: {' '.join(s['words'])} (site={s['site']!r})")

    if oracle_errors:
        issues.append(f"Oracle decide() mismatches ({len(oracle_errors)}): {oracle_errors[:5]}")
    if browser_app_leaks:
        issues.append(f"Hybrid/desktop apps in browser route without web marker ({len(browser_app_leaks)}): {browser_app_leaks[:5]}")

    return {
        "ok": len(issues) == 0,
        "real_count": len(real_rows),
        "mine_count": len(read_tsv(G.MINE_PATH)),
        "exam_count": len(E.EXAM),
        "probe_count": len(probe_rows),
        "gen_checked": len(gen_samples),
        "issues": issues,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", type=int, default=30)
    ap.add_argument("--hard", action="store_true", help="Показать только USER_HARD_CASES")
    ap.add_argument("--audit", action="store_true", help="Полный аудит датасетов, словарей, генератора и decide()")
    a = ap.parse_args()
    if a.audit:
        rep = audit_all()
        print("=== АУДИТ ДАННЫХ, ГЕНЕРАТОРА И ПРАВИЛ (ml_probe.py --audit) ===")
        print(f"  ml_real.tsv : {rep['real_count']} фраз")
        print(f"  ml_mine.tsv : {rep['mine_count']} фраз")
        print(f"  ml_exam.py  : {rep['exam_count']} фраз")
        print(f"  ml_probe.py : {rep['probe_count']} фраз")
        print(f"  ml_gen+aug  : {rep['gen_checked']} примеров проверено через decide()")
        if rep["ok"]:
            print("  СТАТУС      : OK (0 конфликтов, 0 утечек, 100% согласованность с decide())")
        else:
            print(f"  ОШИБКИ ({len(rep['issues'])}):")
            for issue in rep["issues"]:
                print("   -", issue)
            raise SystemExit(1)
    else:
        rows = USER_HARD_CASES if a.hard else generate(max(a.show, 200))[:a.show]
        for row in rows:
            print(row)
