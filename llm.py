import os
import time
from dotenv import load_dotenv
from groq import Groq
from schemas import Usage

load_dotenv()

try:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
except Exception:
    client = None

MODEL = os.getenv("MODEL", "qwen/qwen3.8-27b")

# Free tiers meter OUTPUT tokens per minute (OTPM) and reserve `max_tokens` up front.
# MAX_TOKENS=0 keeps the provider default; set it (e.g. 400) when running on a small
# quota, or when the agent's answer is known to be short (a router returns one line).
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "0"))

# Week 2 habit: retry the transient HTTP failures instead of crashing the lab run.
RETRIES = int(os.getenv("LLM_RETRIES", "4"))
RETRY_STATUS = {429, 500, 502, 503, 504}
TIMEOUT = float(os.getenv("LLM_TIMEOUT", "30.0"))

USAGE: Usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "max_prompt": 0}

def reset_usage():
    for k in USAGE:
        USAGE[k] = 0

def call_llm(messages, tools=None, max_tokens=None):
    global client
    if client is None:
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    kwargs = {"model": MODEL, "messages": messages, "temperature": 0, "timeout": TIMEOUT}
    mt = max_tokens if max_tokens is not None else MAX_TOKENS
    if mt:
        kwargs["max_tokens"] = mt
    if tools:
        kwargs["tools"] = tools
    for attempt in range(RETRIES):
        try:
            resp = client.chat.completions.create(**kwargs)
            break
        except Exception as exc:                 # RateLimitError / APIStatusError / network
            status = getattr(exc, "status_code", None)
            if attempt == RETRIES - 1 or (status is not None and status not in RETRY_STATUS):
                raise                            # a real error: surface it, do not hide it
            wait = 5 * (attempt + 1)
            print(f"[llm] HTTP {status or type(exc).__name__} - "
                  f"retry {attempt + 1}/{RETRIES - 1} in {wait}s")
            time.sleep(wait)
    USAGE["calls"] += 1
    USAGE["prompt_tokens"] += resp.usage.prompt_tokens
    USAGE["completion_tokens"] += resp.usage.completion_tokens
    USAGE["max_prompt"] = max(USAGE["max_prompt"], resp.usage.prompt_tokens)
    return resp.choices[0].message
