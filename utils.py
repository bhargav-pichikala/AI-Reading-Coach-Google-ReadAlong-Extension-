"""
utils.py
────────
• align_words()      – difflib passage ↔ transcript alignment
• render_words()     – HTML coloured word chips
• analyze_fluency()  – Groq LLM fluency JSON
• explain_text()     – Groq LLM meaning JSON
• answer_question()  – Groq LLM passage-grounded Q&A
• infer_difficulty() – heuristic difficulty label
"""

from __future__ import annotations

import io
import re
import json
import difflib
import logging
from functools import lru_cache

from groq import Groq

logger = logging.getLogger(__name__)

LLM_MODEL = "llama-3.3-70b-versatile"

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clean(w: str) -> str:
    return re.sub(r"[^a-z0-9']", "", w.lower())


def _strip_fence(raw: str) -> str:
    return re.sub(r"^```json\s*|^```\s*|```$", "", raw, flags=re.MULTILINE).strip()


def _safe_json(raw: str) -> dict:
    try:
        return json.loads(_strip_fence(raw))
    except json.JSONDecodeError:
        # Try to extract first {...} block
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Word alignment
# ─────────────────────────────────────────────────────────────────────────────

def align_words(passage: str, transcript: str) -> list[dict]:
    """
    Compare passage words against full accumulated transcript.
    Returns list[dict] with keys: word, status, said(optional)
    status ∈ {correct, wrong, skipped, extra, pending}
    """
    orig_words   = passage.split()
    said_words   = transcript.split() if transcript.strip() else []
    orig_clean   = [_clean(w) for w in orig_words]
    said_clean   = [_clean(w) for w in said_words]

    matcher  = difflib.SequenceMatcher(None, orig_clean, said_clean, autojunk=False)
    opcodes  = matcher.get_opcodes()

    # Map orig_index → result dict
    per_orig : dict[int, dict] = {}
    extra    : list[dict]      = []

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            for k in range(i2 - i1):
                per_orig[i1 + k] = {"word": orig_words[i1 + k], "status": "correct"}

        elif tag == "replace":
            oc = orig_words[i1:i2]
            sc = said_words[j1:j2]
            for k, ow in enumerate(oc):
                if k < len(sc):
                    per_orig[i1 + k] = {"word": ow, "status": "wrong", "said": sc[k]}
                else:
                    per_orig[i1 + k] = {"word": ow, "status": "skipped"}
            for k in range(len(oc), len(sc)):
                extra.append({"word": sc[k], "status": "extra"})

        elif tag == "delete":
            for k, ow in enumerate(orig_words[i1:i2]):
                per_orig[i1 + k] = {"word": ow, "status": "skipped"}

        elif tag == "insert":
            for tw in said_words[j1:j2]:
                extra.append({"word": tw, "status": "extra"})

    # Build final list preserving original word order; unmatched = pending
    result = []
    for idx, ow in enumerate(orig_words):
        result.append(per_orig.get(idx, {"word": ow, "status": "pending"}))
    result.extend(extra)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# HTML rendering
# ─────────────────────────────────────────────────────────────────────────────

_STATUS_CSS = {
    "correct": "wc-correct",
    "wrong":   "wc-wrong",
    "skipped": "wc-skipped",
    "extra":   "wc-extra",
    "pending": "wc-pending",
}
_STATUS_ICON = {
    "correct": "✅",
    "wrong":   "❌",
    "skipped": "⏭",
    "extra":   "➕",
    "pending": "",
}

def render_words(aligned: list[dict]) -> str:
    parts = []
    for item in aligned:
        css  = _STATUS_CSS.get(item["status"], "wc-pending")
        icon = _STATUS_ICON.get(item["status"], "")
        tip  = ""
        if item["status"] == "wrong" and "said" in item:
            s = item["said"]
            tip = f' title="You said: {s}"'
        parts.append(f'<span class="wc {css}"{tip}>{icon} {item["word"]}</span>')
    return '<div class="passage-wrap">' + " ".join(parts) + "</div>"


def passage_as_pending(passage: str) -> str:
    words = passage.split()
    spans = [f'<span class="wc wc-pending">{w}</span>' for w in words]
    return '<div class="passage-wrap">' + " ".join(spans) + "</div>"


# ─────────────────────────────────────────────────────────────────────────────
# Stats helpers
# ─────────────────────────────────────────────────────────────────────────────

def word_counts(aligned: list[dict]) -> dict:
    c = {"correct": 0, "wrong": 0, "skipped": 0, "extra": 0, "pending": 0}
    for item in aligned:
        c[item["status"]] = c.get(item["status"], 0) + 1
    return c


def progress_pct(aligned: list[dict]) -> int:
    orig   = [i for i in aligned if i["status"] != "extra"]
    done   = [i for i in orig   if i["status"] in ("correct","wrong","skipped")]
    return int(len(done) / max(len(orig), 1) * 100)


def accuracy_pct(aligned: list[dict], total_orig: int) -> int:
    correct = sum(1 for i in aligned if i["status"] == "correct")
    return int(correct / max(total_orig, 1) * 100)


# ─────────────────────────────────────────────────────────────────────────────
# LLM — Fluency analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_fluency(passage: str, transcript: str, level: str, client: Groq) -> dict:
    if not transcript.strip():
        return {"fluency_score": 0, "feedback": "No reading detected yet.",
                "encouragement": "Press START and read aloud! 🎤",
                "suggestions": ["Make sure your microphone is working."]}

    prompt = f"""You are an encouraging reading coach for children.

Original passage:
\"\"\"{passage}\"\"\"

What the child has read so far:
\"\"\"{transcript}\"\"\"

Reading difficulty: {level}

Return ONLY valid JSON (no markdown fences, no extra text):
{{
  "fluency_score": <integer 0-100>,
  "feedback": "<2 warm, constructive sentences>",
  "encouragement": "<1 uplifting sentence with an emoji>",
  "suggestions": ["<short tip 1>", "<short tip 2>", "<short tip 3>"]
}}

Scoring: 90-100 near-perfect, 75-89 good, 60-74 decent, below 60 needs practice.
Partial reading is okay — score proportionally.
Always be positive and age-appropriate."""

    try:
        resp = client.chat.completions.create(
            model      = LLM_MODEL,
            messages   = [{"role": "user", "content": prompt}],
            temperature= 0.35,
            max_tokens = 350,
        )
        return _safe_json(resp.choices[0].message.content) or {
            "fluency_score": 50, "feedback": "Keep going!",
            "encouragement": "You're doing great! 🌟", "suggestions": []
        }
    except Exception as exc:
        logger.warning("analyze_fluency error: %s", exc)
        return {"fluency_score": 0, "feedback": f"API error: {exc}",
                "encouragement": "Please try again.", "suggestions": []}


# ─────────────────────────────────────────────────────────────────────────────
# LLM — Meaning explanation
# ─────────────────────────────────────────────────────────────────────────────

def explain_text(passage: str, level: str, client: Groq) -> dict:
    age = {"easy": "6-7", "medium": "8-9", "hard": "10-12"}.get(level, "8-9")
    prompt = f"""You are a kind teacher explaining a reading passage to a {age} year old child.

Passage:
\"\"\"{passage}\"\"\"

Return ONLY valid JSON (no markdown, no extra text):
{{
  "simple_meaning": "<1-2 very simple sentences about what the passage is about>",
  "hard_words": [
    {{"word": "<word>", "meaning": "<child-friendly meaning>"}},
    ...
  ],
  "sentence_explanation": "<2-3 fun, engaging, child-friendly sentences that explain the main idea>"
}}

Rules:
- simple_meaning: extremely simple language, max 2 sentences
- hard_words: pick 3-6 words a child might not know; empty array if none
- sentence_explanation: fun storytelling tone, age-appropriate for {age} year old"""

    try:
        resp = client.chat.completions.create(
            model      = LLM_MODEL,
            messages   = [{"role": "user", "content": prompt}],
            temperature= 0.3,
            max_tokens = 500,
        )
        return _safe_json(resp.choices[0].message.content) or {}
    except Exception as exc:
        logger.warning("explain_text error: %s", exc)
        return {"simple_meaning": f"Error: {exc}", "hard_words": [], "sentence_explanation": ""}


# ─────────────────────────────────────────────────────────────────────────────
# LLM — Q&A
# ─────────────────────────────────────────────────────────────────────────────

def answer_question(question: str, passage: str, client: Groq) -> str:
    if not question.strip():
        return ""
    prompt = f"""You are a warm, friendly reading tutor helping a child understand a passage.

Passage:
\"\"\"{passage}\"\"\"

Child's question: {question}

Rules:
- Answer ONLY using information from the passage above
- Keep your answer SHORT (2-4 sentences maximum)
- Use very simple, friendly language a child can easily understand
- If the answer is not in the passage, say: "That's a wonderful question! The passage doesn't tell us that — maybe we can find out together! 😊"
- Always end with a little encouragement"""

    try:
        resp = client.chat.completions.create(
            model      = LLM_MODEL,
            messages   = [{"role": "user", "content": prompt}],
            temperature= 0.4,
            max_tokens = 200,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        logger.warning("answer_question error: %s", exc)
        return f"Sorry, I couldn't get an answer right now. ({exc})"


# ─────────────────────────────────────────────────────────────────────────────
# Difficulty inference
# ─────────────────────────────────────────────────────────────────────────────

def infer_difficulty(passage: str) -> str:
    words    = passage.split()
    chars    = [re.sub(r"[^a-zA-Z]", "", w) for w in words]
    avg_len  = sum(len(c) for c in chars) / max(len(chars), 1)
    long_pct = sum(1 for c in chars if len(c) > 7) / max(len(chars), 1)
    if avg_len <= 4.5 and long_pct < 0.15:
        return "easy"
    if avg_len <= 5.8 and long_pct < 0.30:
        return "medium"
    return "hard"
