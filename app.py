"""
voice_to_text_app.py
Streamlit app: Transcribe audio (file or live mic) ➜ GPT-4-o analysis
"""

import streamlit as st
import openai
import tempfile, os, json, time
from typing import Dict

# ------------- CONFIG --------------------------------------------------------
st.set_page_config(page_title="Voice Intelligence", page_icon="🎙️", layout="wide")
MODEL_TRANSCRIBE = "whisper-1"
# OpenAI recently re-branded GPT-4-o (May-2024).  If "gpt-4-1" is GA in your tenancy
# swap MODEL_ANALYZE below.  Fallback to gpt-4o.
MODEL_ANALYZE = "gpt-4.1-2025-04-14"       # try "gpt-4o" or "gpt-4o-2024-05-13"

RECORD_LIB = "audio_recorder_streamlit"  # pip install audio_recorder_streamlit

# Initialize OpenAI client with API key from secrets
client = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ------------- SIDEBAR -------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    temp = st.slider("LLM temperature", 0.0, 1.0, 0.3, 0.05)
    st.markdown("---")
    input_mode = st.radio(
        "Choose input source",
        ("Upload audio file", "Record from mic"),
        index=0
    )

# ------------- HELPER FUNCTIONS ---------------------------------------------
def transcribe_audio(file_path: str) -> str:
    """Send file to Whisper and return transcript text."""
    with open(file_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model=MODEL_TRANSCRIBE,
            file=f
        )
    return transcript.text

def analyze_text(text: str, temperature: float = 0.3) -> Dict:
    """Ask GPT-4-o for sentiment & threat assessment, return JSON dict."""
    system = (
        "You are an NLP analyst. Produce a JSON object with:\n"
        "overall_sentiment  (positive | neutral | negative),\n"
        "sentiment_score    (-1.0…1.0),\n"
        "summary            (concise 1-2 sentences),\n"
        "threat_level       (none | potential | high) – if violence, hate, threats.\n"
        "Only output valid JSON."
    )
    user = f"Analyze this transcript:\n```\n{text}\n```"
    completion = client.chat.completions.create(
        model=MODEL_ANALYZE,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=temperature,
        max_tokens=300
    )
    # Robust JSON parse
    first = completion.choices[0].message.content.strip()
    try:
        result = json.loads(first)
    except json.JSONDecodeError:
        # fallback if model wrapped JSON in markdown
        result = json.loads(first[first.find("{"): first.rfind("}")+1])
    return result

def save_uploaded_file(uploaded_file) -> str:
    """Write uploaded file to a temp file & return path."""
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        return tmp.name

# ------------- MAIN BODY -----------------------------------------------------
st.title("🎙️ Voice ➜ Intelligence")

audio_bytes = None
audio_filename = None

# -------- Option 1: Upload ---------------------------------------------------
if input_mode == "Upload audio file":
    file = st.file_uploader(
        "Upload audio",
        type=["mp3", "wav", "m4a", "webm", "flac", "mp4"],
        help="Common formats supported by Whisper"
    )
    if file:
        audio_bytes = file.getvalue()
        audio_filename = file.name
        st.audio(audio_bytes)

# -------- Option 2: Record ---------------------------------------------------
else:
    try:
        from audio_recorder_streamlit import audio_recorder
    except ModuleNotFoundError:
        st.error(
            f"`{RECORD_LIB}` not installed.  In terminal run:\n\n"
            f"    pip install {RECORD_LIB}\n"
        )
        st.stop()

    st.info("Click **Start Recording** then **Stop Recording**.")
    audio_bytes = audio_recorder(
        pause_threshold=3.0,
        sample_rate=44_100,
        text="Click to record"
    )
    if audio_bytes:
        audio_filename = f"mic_{int(time.time())}.wav"
        st.audio(audio_bytes)

# --------- When we have audio, process --------------------------------------
if audio_bytes:
    if st.button("🔄 Transcribe & Analyze", type="primary"):
        with st.spinner("Transcribing…"):
            path = save_uploaded_file(
                uploaded_file=file) if input_mode == "Upload audio file" else None
            if path is None:  # live recording
                # write recording bytes to temp WAV
                fd, path = tempfile.mkstemp(suffix=".wav")
                os.write(fd, audio_bytes)
                os.close(fd)

            try:
                transcript = transcribe_audio(path)
            finally:
                os.unlink(path)

        st.subheader("📝 Transcript")
        st.text_area("", transcript, height=200)

        with st.spinner("Running GPT-4-o analysis…"):
            analysis = analyze_text(transcript, temp)

        # ---------- Results ---------------------------------------------------
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sentiment")
            st.metric("Overall", analysis["overall_sentiment"].title())
            st.metric("Score", f'{analysis["sentiment_score"]:.2f}')
        with col2:
            st.subheader("Summary")
            st.write(analysis["summary"])

        threat = analysis.get("threat_level", "none")
        if threat.lower() in ("high", "potential"):
            st.error(f"⚠️ Threat detected: **{threat.upper()}**")

        # -------- Download results -------------------------------------------
        st.download_button(
            "📥 Download JSON report",
            data=json.dumps({"transcript": transcript, "analysis": analysis}, indent=2),
            file_name="voice_analysis.json",
            mime="application/json"
        )
else:
    st.info("Upload a file or record from mic to begin.")
