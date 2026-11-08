#!/usr/bin/env python3
"""First layer in production: phrase -> where to route it and with what data. ~10-30 ms on CPU.

    python3 ml_infer.py                    # interactive prompt
    python3 ml_infer.py "открой ютуб и найди как приготовить пирог"

In code:
    result = await ml_infer.parse("переведи мне на английский как дела")
    result["route"]   # "browser" | "app" | "system" | "translate" | "llm"
    # browser:   result["site"], result["query"]   (no site -> Google/web search, no query -> just open the site)
    # app:       result["site"], result["query"]   (desktop/hybrid app or local folder/file + optional inner query)
    # system:    result["query"]                   (PC/OS command span)
    # translate: result["text"], result["lang"]    ("none" - no target language was named)
    # llm:       the whole phrase goes to the big LLM (result["fallback"] == True if the model was unsure)

Command log (optional): export ML_PARSE_LOG=1 - every phrase and the model's decision is appended to parse_log.jsonl,
then `python3 ml_log.py` shows the doubtful ones: fix them and add them to ml_real.tsv.

Safety net: if the model is unsure (probability below a threshold) or the tagging looks odd, the phrase is not
executed "blindly" but handed to the big LLM. A mistake turns into a slower but correct answer.
"""
import asyncio
import sys
import time

import torch

import ml_log
import ml_model
from ml_spec import MODEL_DIR, decide

_engine = None


class Engine:
    def __init__(self, model_dir: str = MODEL_DIR, threads: int = 4):
        torch.set_num_threads(threads)
        self.model, self.tok = ml_model.load(model_dir)
        self.predict_one("открой ютуб")                      # warm-up

    def predict_one(self, text: str) -> dict:
        words = text.split()
        if not words:
            return {"route": "llm", "site": "", "query": "", "text": "", "lang": "none", "confidence": 0.0,
                    "fallback": True, "reason": "empty phrase", "phrase": text}
        pred = ml_model.predict(self.model, self.tok, [words])[0]
        return decide(text, words, pred)


def get_engine(model_dir: str = MODEL_DIR) -> Engine:
    global _engine
    if _engine is None:
        _engine = Engine(model_dir)
    return _engine


async def parse(text: str, model_dir: str = MODEL_DIR) -> dict:
    engine = await asyncio.to_thread(get_engine, model_dir)
    result = await asyncio.to_thread(engine.predict_one, text)
    ml_log.log_result(result)                # only when enabled: export ML_PARSE_LOG=1 (see ml_log.py)
    return result


def show(result: dict, seconds: float) -> None:
    print(f"  route : {result['route']}" + (f"   (-> LLM: {result['reason']})" if result["fallback"] else ""))
    if result["route"] in ("browser", "app"):
        print(f"  site  : {result['site']!r}\n  query : {result['query']!r}")
    elif result["route"] == "system":
        print(f"  query : {result['query']!r}")
    elif result["route"] == "translate":
        print(f"  text  : {result['text']!r}\n  lang  : {result['lang']}")
    print(f"  confidence {result['confidence']:.2f}, {seconds * 1000:.0f} ms")


def main(argv) -> int:
    try:
        sys.stdin.reconfigure(errors="replace")
    except (AttributeError, ValueError):
        pass
    try:
        import readline  # noqa: F401
    except ImportError:
        pass
    engine = Engine()
    if argv:
        t = time.perf_counter()
        show(engine.predict_one(" ".join(argv)), time.perf_counter() - t)
        return 0
    print('Type a phrase; "exit" to quit.')
    while True:
        try:
            line = input("ml> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if line.lower() in {"выход", "exit", "quit", "q"}:
            return 0
        if line:
            t = time.perf_counter()
            show(engine.predict_one(line), time.perf_counter() - t)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
