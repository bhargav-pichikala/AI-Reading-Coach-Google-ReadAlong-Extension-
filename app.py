"""
app.py  ·  AI Reading Coach — Real-Time Streaming
──────────────────────────────────────────────────
Architecture
  • streamlit-webrtc  → continuous mic capture
  • AudioTranscriptionProcessor → chunks PCM → Groq Whisper (background threads)
  • Streamlit auto-refresh (st.empty + time.sleep loop) → polls new transcript
  • utils.py → alignment, highlighting, LLM calls
"""

from __future__ import annotations

import os
import time
import logging
import threading

import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

from groq import Groq

from audio_processor import AudioTranscriptionProcessor
from utils import (
    align_words, render_words, passage_as_pending,
    word_counts, progress_pct, accuracy_pct,
    analyze_fluency, explain_text, answer_question,
    infer_difficulty,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Reading Coach · Live",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&family=Space+Grotesk:wght@500;600;700&display=swap');

html,body,[class*="css"]{font-family:'Nunito',sans-serif;color:#1e293b;}
.main{background:linear-gradient(140deg,#0f172a 0%,#1e1b4b 45%,#0f172a 100%) !important;min-height:100vh;}
.block-container{max-width:1080px;padding:1.4rem 1.2rem 3rem;}
h1,h2{font-family:'Space Grotesk',sans-serif;}

/* ── Glass panels ── */
.glass{
  background:rgba(255,255,255,.06);
  border:1px solid rgba(255,255,255,.11);
  border-radius:22px;padding:1.3rem 1.5rem;
  margin-bottom:1.1rem;backdrop-filter:blur(18px);
}
.glass-hi{border-color:rgba(129,140,248,.45);background:rgba(99,102,241,.08);}

/* ── Section heading ── */
.sh{
  font-family:'Space Grotesk',sans-serif;
  font-size:1rem;font-weight:700;letter-spacing:.04em;
  color:#e2e8f0;margin-bottom:.65rem;
  display:flex;align-items:center;gap:.4rem;
}

/* ── Passage word chips ── */
.passage-wrap{font-size:1.28rem;line-height:2.25;font-weight:700;letter-spacing:.01em;}
.wc{border-radius:7px;padding:2px 5px;margin:0 1px;display:inline-block;}
.wc-correct {background:#dcfce7;color:#14532d;border-bottom:3px solid #22c55e;}
.wc-wrong   {background:#fee2e2;color:#7f1d1d;border-bottom:3px solid #ef4444;}
.wc-skipped {background:#fef3c7;color:#78350f;border-bottom:3px dashed #f59e0b;}
.wc-extra   {background:#ede9fe;color:#4c1d95;border-bottom:3px dotted #8b5cf6;}
.wc-pending {color:rgba(255,255,255,.28);}

/* ── Stat cards ── */
.stats-row{display:flex;gap:.6rem;flex-wrap:wrap;margin-top:.5rem;}
.sc{flex:1;min-width:78px;border-radius:15px;padding:.6rem .35rem;text-align:center;font-weight:800;}
.sc .n{font-size:1.65rem;font-family:'Space Grotesk',sans-serif;}
.sc .l{font-size:.62rem;text-transform:uppercase;letter-spacing:.07em;}
.sg{background:#dcfce7;color:#14532d;}
.sr{background:#fee2e2;color:#7f1d1d;}
.sa{background:#fef3c7;color:#78350f;}
.sb{background:#dbeafe;color:#1e3a8a;}
.sp{background:#ede9fe;color:#4c1d95;}

/* ── Progress bar ── */
.pb-wrap{background:rgba(255,255,255,.07);border-radius:99px;height:11px;overflow:hidden;margin:.35rem 0;}
.pb-bar{height:100%;border-radius:99px;background:linear-gradient(90deg,#6366f1,#a855f7);transition:width .35s ease;}

/* ── Fluency ring ── */
.ring-wrap{text-align:center;padding:.3rem 0 .5rem;}
.ring{
  display:inline-flex;align-items:center;justify-content:center;
  width:96px;height:96px;border-radius:50%;
  font-family:'Space Grotesk',sans-serif;font-size:2.1rem;font-weight:700;
  color:white;margin-bottom:.3rem;
}

/* ── Hard-word chips ── */
.hw-grid{display:flex;flex-wrap:wrap;gap:.45rem;margin-top:.55rem;}
.hw{background:rgba(255,255,255,.06);border:1px solid rgba(167,139,250,.4);
    border-radius:10px;padding:.28rem .75rem;font-size:.82rem;color:#e2e8f0;}
.hw b{color:#a78bfa;}

/* ── Answer bubble ── */
.ans{background:rgba(255,255,255,.05);border-left:4px solid #818cf8;
     border-radius:0 14px 14px 0;padding:.85rem 1rem;
     font-size:.97rem;line-height:1.65;color:#e2e8f0;margin-top:.55rem;}

/* ── Suggestion item ── */
.sug{background:rgba(255,255,255,.04);border-left:3px solid #6366f1;
     border-radius:0 9px 9px 0;padding:.38rem .85rem;
     font-size:.86rem;color:#cbd5e1;margin-bottom:.3rem;}

/* ── Live badge ── */
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
.live-dot{display:inline-block;width:9px;height:9px;border-radius:50%;
          background:#22c55e;animation:pulse 1.1s infinite;margin-right:5px;vertical-align:middle;}
.live-badge{
  display:inline-flex;align-items:center;
  background:rgba(34,197,94,.15);border:1px solid rgba(34,197,94,.4);
  color:#86efac;border-radius:20px;padding:.18rem .75rem;
  font-size:.78rem;font-weight:800;letter-spacing:.05em;
}

/* ── Transcript scroll ── */
.tx-box{
  background:rgba(0,0,0,.25);border-radius:12px;
  padding:.7rem 1rem;max-height:110px;overflow-y:auto;
  font-size:.88rem;color:#94a3b8;line-height:1.6;
  border:1px solid rgba(255,255,255,.07);
}

/* ── Passage card accent ── */
.passage-card{
  background:rgba(255,255,255,.05);border:1.5px solid rgba(129,140,248,.3);
  border-radius:22px;padding:1.4rem 1.6rem;margin-bottom:1rem;
}

/* ── Streamlit overrides ── */
.stButton>button{
  background:linear-gradient(135deg,#6366f1,#8b5cf6) !important;
  color:white !important;border:none !important;border-radius:12px !important;
  font-weight:700 !important;padding:.5rem 1.1rem !important;
}
.stButton>button:hover{opacity:.82 !important;}
.stSelectbox label,.stTextInput label{color:#cbd5e1 !important;font-weight:700 !important;}
.stTextInput>div>div>input{
  background:rgba(255,255,255,.06) !important;
  border:1px solid rgba(255,255,255,.14) !important;
  color:#f1f5f9 !important;border-radius:10px !important;
}
[data-testid="stMarkdownContainer"] p{color:#cbd5e1;}
.stToggle label{color:#cbd5e1 !important;font-weight:600 !important;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Passages
# ─────────────────────────────────────────────────────────────────────────────
PASSAGES: dict[str, tuple[str, str]] = {
    "🌱 The Little Seed (Easy)": (
        "A little seed fell into the soft brown soil. "
        "It soaked up water and sunlight every day. "
        "Slowly a tiny green shoot pushed up through the earth. "
        "The shoot grew taller and stronger each week. "
        "One morning a beautiful yellow flower bloomed in the garden.",
        "easy",
    ),
    "🐬 Dolphins (Medium)": (
        "Dolphins are intelligent marine mammals that live in oceans around the world. "
        "They communicate using clicks and whistles, forming complex social bonds within their pods. "
        "Dolphins are known for their playful behavior, often leaping out of the water. "
        "They use echolocation to navigate and hunt for fish in murky waters.",
        "medium",
    ),
    "🚀 Space Exploration (Hard)": (
        "Space exploration has dramatically transformed our understanding of the universe. "
        "Scientists utilize sophisticated telescopes and interplanetary spacecraft to investigate celestial phenomena. "
        "Recent discoveries of exoplanets in habitable zones have intensified speculation about extraterrestrial life. "
        "Advanced propulsion technologies are being developed to facilitate crewed missions to Mars.",
        "hard",
    ),
}

# WebRTC STUN/TURN — public STUN is enough for same-LAN testing;
# for production add TURN credentials here.
RTC_CONFIG = RTCConfiguration(
    iceServers=[
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
    ]
)

# ─────────────────────────────────────────────────────────────────────────────
# Session state bootstrap
# ─────────────────────────────────────────────────────────────────────────────
_DEFAULTS = dict(
    full_transcript  = "",
    aligned          = [],
    fluency          = None,
    explanation      = None,
    last_answer      = "",
    last_question    = "",
    passage_key      = "",
    processor        = None,   # AudioTranscriptionProcessor instance
    fluency_ticker   = 0,      # count transcript updates between fluency calls
    FLUENCY_EVERY    = 5,      # call fluency every N new chunks
)
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def _reset_session():
    for k, v in _DEFAULTS.items():
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# Groq client (cached per session via session_state)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def _get_groq_client(key: str) -> Groq:
    return Groq(api_key=key)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ── API key guard ──────────────────────────────────────────────────────
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        st.error("🔑 **GROQ_API_KEY** environment variable not set.")
        st.code("$env:GROQ_API_KEY=\"gsk_your_key_here\"  # PowerShell\n"
                "export GROQ_API_KEY=gsk_your_key_here    # Linux/Mac",
                language="bash")
        st.stop()

    groq_client = _get_groq_client(api_key)

    # ── Header ────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center;padding:.8rem 0 .2rem;">
      <h1 style="font-size:2.4rem;margin:0;
        background:linear-gradient(135deg,#818cf8,#c084fc,#f472b6);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
        🎙️ AI Reading Coach
      </h1>
      <p style="color:#94a3b8;font-size:.95rem;font-weight:600;margin:.25rem 0 0;">
        Real-Time Streaming · Read aloud · Get live feedback
      </p>
    </div>""", unsafe_allow_html=True)

    # ── Passage selector ──────────────────────────────────────────────────
    col_sel, col_rst = st.columns([5, 1])
    with col_sel:
        choice = st.selectbox("", list(PASSAGES.keys()), label_visibility="collapsed")
    with col_rst:
        st.markdown("<div style='padding-top:.32rem;'>", unsafe_allow_html=True)
        if st.button("🔄 Reset"):
            _reset_session()
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    passage_text, passage_level = PASSAGES[choice]

    # Passage switch → reset
    if st.session_state["passage_key"] != choice:
        _reset_session()
        st.session_state["passage_key"] = choice

    # Custom passage
    if st.toggle("✏️ Use my own passage"):
        custom = st.text_area("Paste your passage:", height=100,
                              label_visibility="collapsed",
                              placeholder="Type or paste the text you want to practise…")
        if custom.strip():
            passage_text  = custom.strip()
            passage_level = infer_difficulty(passage_text)

    diff_label = {"easy":"🟢 Easy","medium":"🟡 Medium","hard":"🔴 Hard"}.get(passage_level, passage_level)
    st.markdown(f'<span style="font-size:.8rem;font-weight:800;color:#94a3b8;">{diff_label}</span>',
                unsafe_allow_html=True)

    total_words = len(passage_text.split())

    # ── Two-column layout ─────────────────────────────────────────────────
    left, right = st.columns([3, 2], gap="large")

    # ════════ LEFT ════════════════════════════════════════════════════════
    with left:
        # ── 📖 Passage ────────────────────────────────────────────────
        st.markdown('<div class="passage-card">', unsafe_allow_html=True)
        st.markdown('<div class="sh">📖 Passage</div>', unsafe_allow_html=True)

        prog = progress_pct(st.session_state["aligned"]) if st.session_state["aligned"] else 0
        acc  = accuracy_pct(st.session_state["aligned"], total_words) if st.session_state["aligned"] else 0
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:.6rem;margin-bottom:.45rem;">
          <div class="pb-wrap" style="flex:1;">
            <div class="pb-bar" style="width:{prog}%;"></div>
          </div>
          <span style="color:#a5b4fc;font-size:.8rem;font-weight:700;white-space:nowrap;">{prog}% read</span>
        </div>""", unsafe_allow_html=True)

        # Passage placeholder — updated by polling loop
        passage_ph = st.empty()
        if st.session_state["aligned"]:
            passage_ph.markdown(render_words(st.session_state["aligned"]), unsafe_allow_html=True)
        else:
            passage_ph.markdown(passage_as_pending(passage_text), unsafe_allow_html=True)

        st.markdown("""
        <div style="display:flex;gap:.8rem;flex-wrap:wrap;margin-top:.6rem;">
          <span style="font-size:.73rem;color:#86efac;font-weight:700;">✅ Correct</span>
          <span style="font-size:.73rem;color:#fca5a5;font-weight:700;">❌ Wrong</span>
          <span style="font-size:.73rem;color:#fde68a;font-weight:700;">⏭ Skipped</span>
          <span style="font-size:.73rem;color:rgba(255,255,255,.28);font-weight:700;">● Pending</span>
        </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── 🎤 WebRTC streamer ────────────────────────────────────────
        st.markdown('<div class="glass glass-hi">', unsafe_allow_html=True)
        st.markdown('<div class="sh">🎤 Live Microphone</div>', unsafe_allow_html=True)
        st.markdown('<p style="color:#94a3b8;font-size:.86rem;font-weight:600;margin:0 0 .5rem;">'
                    'Press <b style="color:#a5b4fc;">START</b> · Read the passage aloud · Press <b style="color:#a5b4fc;">STOP</b> when done</p>',
                    unsafe_allow_html=True)

        # Build processor lazily (one per session)
        if st.session_state["processor"] is None:
            st.session_state["processor"] = AudioTranscriptionProcessor(api_key)

        processor: AudioTranscriptionProcessor = st.session_state["processor"]

        ctx = webrtc_streamer(
            key                    = "reading-coach",
            mode                   = WebRtcMode.SENDONLY,
            rtc_configuration      = RTC_CONFIG,
            audio_frame_callback   = processor.recv_frame,
            media_stream_constraints = {"audio": True, "video": False},
            async_processing       = True,
            sendback_audio         = False,
        )

        # Status indicators
        status_ph = st.empty()
        if ctx.state.playing:
            status_ph.markdown(
                '<span class="live-badge"><span class="live-dot"></span>LIVE — listening…</span>',
                unsafe_allow_html=True)
        else:
            status_ph.markdown(
                '<span style="color:#64748b;font-size:.82rem;font-weight:700;">⏸ Not recording</span>',
                unsafe_allow_html=True)

        # Live transcript box
        st.markdown('<div class="sh" style="margin-top:.7rem;">🗒️ Captured text</div>', unsafe_allow_html=True)
        tx_ph = st.empty()
        tx_ph.markdown(
            f'<div class="tx-box">{st.session_state["full_transcript"] or "<i>Nothing yet…</i>"}</div>',
            unsafe_allow_html=True)

        # Chunk counter
        chunk_ph = st.empty()
        chunk_ph.markdown(
            f'<p style="color:#64748b;font-size:.74rem;margin:.2rem 0 0;">'
            f'Chunks processed: <b style="color:#a5b4fc;">{processor.total_chunks}</b> · '
            f'In-flight: <b style="color:#f472b6;">{processor.in_flight}</b></p>',
            unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # ── 📊 Live Stats ─────────────────────────────────────────────
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown('<div class="sh">📊 Live Stats</div>', unsafe_allow_html=True)
        wc = word_counts(st.session_state["aligned"]) if st.session_state["aligned"] \
             else {"correct":0,"wrong":0,"skipped":0,"extra":0,"pending":total_words}
        st.markdown(f"""
        <div class="stats-row">
          <div class="sc sg"><div class="n">{wc['correct']}</div><div class="l">✅ Correct</div></div>
          <div class="sc sr"><div class="n">{wc['wrong']+wc['skipped']}</div><div class="l">❌ Errors</div></div>
          <div class="sc sa"><div class="n">{wc['pending']}</div><div class="l">⏳ Left</div></div>
          <div class="sc sb"><div class="n">{acc}%</div><div class="l">🎯 Accuracy</div></div>
        </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ════════ RIGHT ═══════════════════════════════════════════════════════
    with right:
        # ── ⭐ Fluency score ──────────────────────────────────────────
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown('<div class="sh">⭐ Fluency Score</div>', unsafe_allow_html=True)
        fluency_ph = st.empty()

        def _render_fluency(fl: dict):
            sc  = fl.get("fluency_score", 0)
            col = "#22c55e" if sc>=80 else "#f59e0b" if sc>=60 else "#ef4444"
            em  = "🌟" if sc>=90 else "👍" if sc>=75 else "💪" if sc>=55 else "🌱"
            html = f"""
            <div class="ring-wrap">
              <div class="ring" style="background:linear-gradient(135deg,{col},{col}88);">{sc}</div>
              <div style="font-size:1.15rem;font-weight:800;color:{col};">{em}</div>
            </div>
            <p style="color:#c7d2fe;font-size:.88rem;text-align:center;margin:.1rem 0 .5rem;">{fl.get('encouragement','')}</p>
            <p style="color:#cbd5e1;font-size:.86rem;margin:0 0 .5rem;">{fl.get('feedback','')}</p>
            """
            for tip in fl.get("suggestions",[]):
                html += f'<div class="sug">→ {tip}</div>'
            fluency_ph.markdown(html, unsafe_allow_html=True)

        if st.session_state["fluency"]:
            _render_fluency(st.session_state["fluency"])
        else:
            fluency_ph.markdown(
                '<p style="color:#475569;font-size:.88rem;">Start reading to get your score!</p>',
                unsafe_allow_html=True)

        if st.button("🔄 Refresh Score", use_container_width=True):
            if st.session_state["full_transcript"].strip():
                with st.spinner("Analysing…"):
                    fl = analyze_fluency(passage_text, st.session_state["full_transcript"],
                                         passage_level, groq_client)
                    st.session_state["fluency"] = fl
                    _render_fluency(fl)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── 🧠 Meaning explanation ───────────────────────────────────
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown('<div class="sh">🧠 Passage Meaning</div>', unsafe_allow_html=True)
        if st.button("✨ Explain This Passage", use_container_width=True):
            with st.spinner("Thinking of a simple explanation…"):
                try:
                    ex = explain_text(passage_text, passage_level, groq_client)
                    st.session_state["explanation"] = ex
                    st.session_state["explained_for"] = passage_text
                except Exception as e:
                    st.error(str(e))

        if (st.session_state.get("explanation")
                and st.session_state.get("explained_for") == passage_text):
            ex = st.session_state["explanation"]
            if ex.get("simple_meaning"):
                st.markdown(
                    f'<div style="background:rgba(255,255,255,.05);border-left:4px solid #f59e0b;'
                    f'border-radius:0 12px 12px 0;padding:.7rem 1rem;color:#e2e8f0;font-size:.92rem;margin-bottom:.5rem;">'
                    f'📌 {ex["simple_meaning"]}</div>',
                    unsafe_allow_html=True)
            if ex.get("hard_words"):
                chips = "".join(
                    f'<div class="hw"><b>{hw["word"]}</b> — {hw["meaning"]}</div>'
                    for hw in ex["hard_words"]
                )
                st.markdown(f'<p style="color:#94a3b8;font-size:.78rem;font-weight:700;margin:.4rem 0 .1rem;">🔤 HARD WORDS</p>'
                            f'<div class="hw-grid">{chips}</div>', unsafe_allow_html=True)
            if ex.get("sentence_explanation"):
                st.markdown(
                    f'<p style="color:#a5b4fc;font-size:.88rem;font-style:italic;margin:.6rem 0 0;">'
                    f'✨ {ex["sentence_explanation"]}</p>',
                    unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── 💬 Q&A ────────────────────────────────────────────────────
        st.markdown('<div class="glass">', unsafe_allow_html=True)
        st.markdown('<div class="sh">💬 Ask a Question</div>', unsafe_allow_html=True)
        st.markdown('<p style="color:#64748b;font-size:.83rem;font-weight:600;margin:0 0 .4rem;">Curious about something in the passage?</p>',
                    unsafe_allow_html=True)
        q = st.text_input("Question:", placeholder="e.g. What did the seed need?",
                          label_visibility="collapsed")
        if st.button("🙋 Ask!", use_container_width=True) and q.strip():
            with st.spinner("Finding a child-friendly answer…"):
                try:
                    ans = answer_question(q.strip(), passage_text, groq_client)
                    st.session_state["last_answer"]   = ans
                    st.session_state["last_question"] = q.strip()
                except Exception as e:
                    st.error(str(e))

        if st.session_state["last_answer"]:
            st.markdown(
                f'<p style="color:#a5b4fc;font-size:.82rem;font-weight:700;margin:.4rem 0 .1rem;">'
                f'❓ {st.session_state["last_question"]}</p>'
                f'<div class="ans">🤖 {st.session_state["last_answer"]}</div>',
                unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────
    # Polling loop — runs while WebRTC is active, updates UI in real-time
    # ─────────────────────────────────────────────────────────────────────
    if ctx.state.playing:
        POLL_INTERVAL = 0.6   # seconds between polls
        MAX_POLLS     = 300   # safety cap (~3 min)

        for _ in range(MAX_POLLS):
            new_chunks = processor.get_new_text()

            if new_chunks:
                for chunk in new_chunks:
                    st.session_state["full_transcript"] += (" " + chunk).strip()

                # Re-align
                st.session_state["aligned"] = align_words(
                    passage_text,
                    st.session_state["full_transcript"],
                )

                # Update passage display
                passage_ph.markdown(
                    render_words(st.session_state["aligned"]),
                    unsafe_allow_html=True)

                # Update transcript box
                tx_ph.markdown(
                    f'<div class="tx-box">{st.session_state["full_transcript"]}</div>',
                    unsafe_allow_html=True)

                # Update chunk counter
                chunk_ph.markdown(
                    f'<p style="color:#64748b;font-size:.74rem;margin:.2rem 0 0;">'
                    f'Chunks processed: <b style="color:#a5b4fc;">{processor.total_chunks}</b> · '
                    f'In-flight: <b style="color:#f472b6;">{processor.in_flight}</b></p>',
                    unsafe_allow_html=True)

                # Update stats
                wc  = word_counts(st.session_state["aligned"])
                acc = accuracy_pct(st.session_state["aligned"], total_words)
                # (stat boxes re-render on next rerun; live accuracy shown in passage)

                # Periodic fluency update (every N new chunks to save API calls)
                st.session_state["fluency_ticker"] += len(new_chunks)
                if st.session_state["fluency_ticker"] >= st.session_state["FLUENCY_EVERY"]:
                    st.session_state["fluency_ticker"] = 0
                    try:
                        fl = analyze_fluency(
                            passage_text,
                            st.session_state["full_transcript"],
                            passage_level,
                            groq_client,
                        )
                        st.session_state["fluency"] = fl
                        _render_fluency(fl)
                    except Exception:
                        pass  # non-fatal

            time.sleep(POLL_INTERVAL)

            # Break out if WebRTC was stopped externally
            if not ctx.state.playing:
                processor.flush()
                break
        else:
            # Loop exhausted — flush remaining buffer
            processor.flush()


if __name__ == "__main__":
    main()
