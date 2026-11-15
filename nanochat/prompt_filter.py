"""Safety filter for distillation prompts: a keyword blocklist plus a SAFE/UNSAFE judgment by a
vLLM-served model (the distillation teacher). Used by scripts/build_opd_prompts.py and
scripts/build_easy_mix.py."""

import re

import requests

BLOCK = re.compile(r"\b(" + "|".join([
    r"sex\w*", r"erotic\w*", r"porn\w*", r"nsfw", r"nude\w*", r"naked", r"fetish\w*", r"kink\w*",
    r"bdsm", r"dominan\w*", r"submissive\w*", r"\bdom\b", r"\bsub\b", r"slave\w*", r"thrall\w*",
    r"master\b", r"mistress", r"seduc\w*", r"lust\w*", r"aroused?", r"orgasm\w*", r"genital\w*",
    r"breasts?", r"boobs?", r"nipples?", r"penis", r"vagina\w*", r"lingerie", r"strip(per|ping|tease)",
    r"horny", r"hentai", r"waifu", r"incest\w*", r"rape\w*", r"molest\w*", r"gore", r"torture\w*",
    r"role-?play\w*", r"roleplay\w*", r"\brp\b", r"lewd\w*", r"x-rated", r"onlyfans", r"escort\w*",
]) + r")", re.IGNORECASE)

JUDGE_PROMPT = (
    "You are screening training prompts for a family-friendly cooking assistant. "
    "Reply with exactly one word, SAFE or UNSAFE.\n"
    "UNSAFE: sexual, erotic, romantic or fetish content; role-play involving domination, "
    "submission or intimacy; graphic violence, gore or abuse; hateful content; instructions "
    "for weapons, drugs or crime.\n"
    "SAFE: everything else, including ordinary writing, advice, coding, math and trivia.\n\n"
    "Request:\n<<<\n{prompt}\n>>>"
)


def judge(prompt, url="http://localhost:8011", model="Qwen/Qwen3.5-4B"):
    for attempt in range(4):
        try:
            r = requests.post(f"{url}/v1/chat/completions", timeout=120, json={
                "model": model, "max_tokens": 4, "temperature": 0.0,
                "messages": [{"role": "user", "content": JUDGE_PROMPT.format(prompt=prompt[:6000])}],
                "chat_template_kwargs": {"enable_thinking": False}})
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip().upper().startswith("SAFE")
        except (requests.RequestException, KeyError, ValueError):
            if attempt == 3:
                return False  # unjudged counts as unsafe
    return False
