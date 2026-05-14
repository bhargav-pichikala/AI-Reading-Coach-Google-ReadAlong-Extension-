<div align="center">

# 🎙️ AI Reading Coach
### Real-Time Streaming Read-Along powered by Groq AI

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Groq](https://img.shields.io/badge/Groq-LPU%20Inference-F55036?style=for-the-badge&logo=groq&logoColor=white)](https://groq.com)
[![WebRTC](https://img.shields.io/badge/WebRTC-Real--Time-333333?style=for-the-badge&logo=webrtc&logoColor=white)](https://webrtc.org)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

<br/>

**A production-grade, Google Read Along clone** that listens to a child read aloud in real-time,
highlights words as they're spoken, scores fluency live, explains difficult words,
and answers questions — all powered by Groq's ultra-fast AI inference.

<br/>

![Demo Banner](https://img.shields.io/badge/STATUS-LIVE%20%26%20WORKING-22c55e?style=for-the-badge)

</div>

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎤 **Real-Time Mic Streaming** | WebRTC continuous microphone capture — no manual uploads |
| 🗣️ **Live Speech-to-Text** | Groq Whisper `whisper-large-v3-turbo` transcribes every 2 seconds |
| 🎨 **Live Word Highlighting** | Green ✅ correct · Red ❌ wrong · Gray ⏳ pending — updates instantly |
| 📊 **Live Stats** | Word-by-word accuracy, errors, and reading progress bar |
| ⭐ **Fluency Scoring** | AI scores your reading 0–100 with warm, encouraging feedback |
| 🧠 **Meaning Explanation** | LLM explains the passage in child-friendly language with hard word definitions |
| 💬 **Interactive Q&A** | Ask any question about the passage — answered from the text only |
| 📖 **3 Built-in Passages** | Easy / Medium / Hard — or paste your own custom text |
| 🌙 **Beautiful Dark UI** | Glass-morphism design built for kids and adults alike |

---

## 🏗️ Architecture

```
Browser Microphone
       │
       │  WebRTC (SENDONLY)
       ▼
AudioTranscriptionProcessor
  ├─ Accumulates PCM frames (2-second chunks)
  ├─ Resamples to 16kHz (Whisper optimal)
  ├─ Encodes to WAV in-memory (zero disk I/O)
  └─ ThreadPoolExecutor (3 workers)
            │
            │  Groq Whisper API (async, parallel)
            ▼
       deque[str]  ← thread-safe text queue
            │
            │  Polled every 0.6s by Streamlit loop
            ▼
     align_words()  ←  difflib SequenceMatcher
            │
     render_words()  →  st.empty().markdown()
            │           (no page refresh)
            ▼
  analyze_fluency()  ←  every 5 chunks (LLM batching)
```

---

## 🤖 Tech Stack & Models

### Frameworks
| Tool | Purpose |
|---|---|
| **Streamlit** | Web UI framework |
| **streamlit-webrtc** | Real-time microphone capture via WebRTC |
| **Groq SDK** | API client for Whisper + LLaMA |
| **PyAV** | Audio frame decoding from WebRTC stream |
| **NumPy** | PCM audio buffering and resampling |
| **difflib** | Word alignment (SequenceMatcher) |
| **ThreadPoolExecutor** | Non-blocking async Whisper calls |

### AI Models (all via Groq)
| Task | Model |
|---|---|
| 🎤 Speech-to-Text | `whisper-large-v3-turbo` |
| ⭐ Fluency Analysis | `llama-3.3-70b-versatile` |
| 🧠 Meaning Explanation | `llama-3.3-70b-versatile` |
| 💬 Q&A | `llama-3.3-70b-versatile` |

> **Why Groq?** Groq's LPU hardware delivers ~10× faster inference than traditional GPUs.
> Whisper-turbo transcribes a 2-second audio chunk in **under 500ms** — enabling the real-time feel.

---

## 📁 Project Structure

```
ai-reading-coach/
│
├── app.py                 # Main Streamlit UI + WebRTC integration + polling loop
├── audio_processor.py     # AudioTranscriptionProcessor — PCM buffering + Whisper calls
├── utils.py               # Word alignment, HTML rendering, LLM functions
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/bhargav-pichikala/ai-reading-coach.git
cd ai-reading-coach
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Get a free Groq API key
Go to **[console.groq.com](https://console.groq.com)** → Sign up → API Keys → Create Key

### 4. Set your API key

**Windows (PowerShell):**
```powershell
$env:GROQ_API_KEY="gsk_your_key_here"
```

**Linux / macOS:**
```bash
export GROQ_API_KEY=gsk_your_key_here
```

**Make it permanent (Windows):**
```powershell
[System.Environment]::SetEnvironmentVariable("GROQ_API_KEY", "gsk_your_key_here", "User")
```

### 5. Run the app
```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501** 🎉

---

## 🎮 How to Use

1. **Choose a passage** from the dropdown (Easy / Medium / Hard) or paste your own
2. Click **START** on the microphone widget and **allow mic access**
3. **Read the passage aloud** — words highlight green as you say them correctly
4. Click **STOP** when done
5. View your **Fluency Score** on the right panel
6. Click **Explain This Passage** to understand difficult words
7. Ask a question like *"What did the seed need?"* in the Q&A box
8. Click **Reset** to try again!

---

## ⚡ Performance Optimizations

| Optimization | Impact |
|---|---|
| `MIN_RMS=150` silence gate | Skips silent chunks — saves ~40% API calls |
| Fluency LLM every 5 chunks | Prevents rate limiting, keeps score fresh |
| `SENDONLY` WebRTC mode | No audio loopback overhead |
| `st.empty()` placeholders | Targeted DOM updates — no full page rerun |
| In-memory WAV (BytesIO) | Zero disk I/O in the hot path |
| ThreadPoolExecutor (3 workers) | Parallel Whisper calls — never blocks UI |

---

## 🖥️ Browser Compatibility

| Browser | Support |
|---|---|
| ✅ Chrome / Chromium | Full support |
| ✅ Microsoft Edge | Full support |
| ✅ Firefox | Full support |
| ⚠️ Safari | Requires mic permission in Settings |

> WebRTC mic access requires **HTTPS in production**. For local development, `localhost` works fine.

---

## 🔐 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | ✅ Yes | Your Groq API key from console.groq.com |

---

## 🛠️ Error Handling

The app gracefully handles:
- ❌ Missing API key → clear setup instructions shown
- 🔇 Silent audio → RMS gate skips the chunk automatically
- 📡 API failure → non-fatal, previous state preserved
- 🔧 JSON parse errors → safe fallback with regex extraction
- 🎙️ Mic permission denied → WebRTC shows browser-native error prompt

---

## 🗺️ Roadmap

- [ ] Streamlit Cloud deployment guide
- [ ] Support for multiple languages
- [ ] Parent/teacher dashboard with session history
- [ ] PDF passage upload
- [ ] Text-to-speech for word pronunciation help
- [ ] Progress tracking across sessions

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [Groq](https://groq.com) — ultra-fast LPU inference
- [OpenAI Whisper](https://github.com/openai/whisper) — speech recognition model
- [streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc) — WebRTC integration
- [Meta LLaMA](https://llama.meta.com) — open-source LLM

---

<div align="center">

Built with ❤️ by [Bhargav Pichikala](https://github.com/bhargav-pichikala)

⭐ **Star this repo if it helped you!** ⭐

</div>
