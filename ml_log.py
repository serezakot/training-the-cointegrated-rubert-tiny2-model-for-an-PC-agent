#!/usr/bin/env python3
"""Parse log: the best source of new training phrases is your real commands.

Enabling (the log is OFF by default, nothing is written):
    export ML_PARSE_LOG=1                       # write to parse_log.jsonl next to the scripts
    export ML_PARSE_LOG=/path/to/file.jsonl     # or to a file of your choice
ml_infer.parse() appends one line per phrase: what was said and what the model decided.

Reviewing the log:
    python3 ml_log.py                            # doubtful phrases first (sent to the LLM or low confidence)
    python3 ml_log.py --all > ml_review.tsv      # all phrases, in ml_real.tsv format
Open ml_review.tsv, FIX the wrong rows (TAB-separated fields: phrase, intent, site, query, text, language)
and paste them into ml_real.tsv. Only the mistakes need fixing; correct rows may stay too, they do no harm.

Privacy: the log contains what you said. Keep it out of git (it is listed in .gitignore).
"""
import json
import os
import sys
import time
from pathlib import Path

DEFAULT_PATH = Path(__file__).with_name("parse_log.jsonl")
DOUBT = 0.90        # confidence below this value marks a phrase as "doubtful"


def log_path():
    v = os.environ.get("ML_PARSE_LOG", "").strip()
    if not v or v.lower() in {"0", "off", "false", "no"}:
        return None
    return DEFAULT_PATH if v.lower() in {"1", "on", "true", "yes"} else Path(v)


def log_result(result: dict, path=None) -> bool:
    """Appends a line to the log. Never crashes the program: a write error is simply ignored."""
    path = path or log_path()
    if path is None:
        return False
    row = {"t": time.strftime("%Y-%m-%dT%H:%M:%S")}
    row.update({k: result.get(k, "") for k in ("phrase", "route", "site", "query", "text", "lang", "confidence",
                                               "fallback", "reason")})
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return True
    except OSError:
        return False


def read_log(path) -> list:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def _key(phrase: str) -> str:
    return " ".join(phrase.casefold().replace("ё", "е").split())


def to_tsv(rows, only_doubtful=True) -> list:
    """Log -> rows in ml_real.tsv format (one per phrase; doubtful ones first, marked with #)."""
    seen, doubtful, sure = set(), [], []
    for r in rows:
        phrase = (r.get("phrase") or "").strip()
        k = _key(phrase)
        if not phrase or k in seen or "\t" in phrase:
            continue
        seen.add(k)
        route = r.get("route", "llm")
        lang = r.get("lang", "none") if route == "translate" else "none"
        site = r.get("site", "") if route in ("browser", "app") else ""
        query = r.get("query", "") if route in ("browser", "app", "system") else ""
        text = r.get("text", "") if route == "translate" else ""
        line = "\t".join([phrase, route, site, query, text, lang or "none"])
        unsure = bool(r.get("fallback")) or float(r.get("confidence") or 0) < DOUBT
        if unsure:
            why = r.get("reason") or f"confidence {float(r.get('confidence') or 0):.2f}"
            doubtful.append(f"# review: {why}\n{line}")
        elif not only_doubtful:
            sure.append(line)
    return doubtful + sure


def main(argv) -> int:
    path = Path(argv[argv.index("--file") + 1]) if "--file" in argv else (log_path() or DEFAULT_PATH)
    rows = read_log(path)
    if not rows:
        print(f"log is empty or not found: {path}\nEnable it first: export ML_PARSE_LOG=1", file=sys.stderr)
        return 1
    for line in to_tsv(rows, only_doubtful="--all" not in argv):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
