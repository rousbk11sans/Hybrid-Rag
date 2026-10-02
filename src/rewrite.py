import json
import os

from dotenv import load_dotenv

load_dotenv()

CACHE = "cache/rewrites.json"
os.makedirs("cache", exist_ok=True)
_cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
_client = None


def _get_client():
    global _client
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic()
    return _client


def rewrite(query):
    if query in _cache:
        return _cache[query]
    r = _get_client().messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=100,
        messages=[{
            "role": "user",
            "content": (
                "Rewrite this scientific claim as a concise search query using key "
                "technical terms and synonyms. Output only the query.\n\n" + query
            ),
        }],
    )
    out = r.content[0].text.strip()
    _cache[query] = out
    with open(CACHE, "w") as f:
        json.dump(_cache, f)
    return out
