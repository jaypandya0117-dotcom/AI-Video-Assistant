import streamlit as st
from dotenv import load_dotenv
load_dotenv()
import os

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

os.environ["PATH"] += os.pathsep + r"D:\MTech\Minor_project\ffmpeg-9.0.1-essentials_build\ffmpeg-9.0.1-essentials_build\bin"


# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------
def run_pipeline(source: str, language: str = "english") -> dict:
    chunks = process_input(source)
    transcript = transcribe_all(chunks, language)
    title = generate_title(transcript)
    summary = summarize(transcript)
    action_item = extract_action_items(transcript)
    decisions = extract_key_decisions(transcript)
    questions = extract_questions(transcript)
    rag_chain = build_rag_chain(transcript)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_item,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }


# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background: radial-gradient(circle at 20% 0%, #1a1c2e 0%, #0e0f1a 45%, #0a0b12 100%);
        }

        section[data-testid="stSidebar"] {
            background: #12131f;
            border-right: 1px solid rgba(255,255,255,0.06);
        }

        /* Hero header */
        .hero {
            padding: 1.6rem 2rem;
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(124,58,237,0.18), rgba(59,130,246,0.12));
            border: 1px solid rgba(255,255,255,0.08);
            margin-bottom: 1.5rem;
        }
        .hero h1 {
            font-size: 1.9rem;
            font-weight: 700;
            margin: 0;
            background: linear-gradient(90deg, #a78bfa, #60a5fa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .hero p {
            color: #9ca3af;
            margin-top: 0.35rem;
            margin-bottom: 0;
            font-size: 0.95rem;
        }

        /* Card container */
        .card {
            background: rgba(255,255,255,0.03);
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 16px;
            padding: 1.4rem 1.6rem;
            margin-bottom: 1rem;
        }
        .card h4 {
            margin-top: 0;
            font-weight: 600;
            color: #e5e7eb;
        }
        .card-title-row {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            margin-bottom: 0.6rem;
        }

        /* Title pill */
        .video-title {
            display: inline-block;
            padding: 0.45rem 1rem;
            border-radius: 999px;
            background: rgba(167,139,250,0.12);
            border: 1px solid rgba(167,139,250,0.35);
            color: #c4b5fd;
            font-weight: 600;
            font-size: 0.95rem;
        }

        /* Buttons */
        .stButton > button {
            border-radius: 10px;
            border: 1px solid rgba(255,255,255,0.12);
            background: linear-gradient(135deg, #7c3aed, #4f46e5);
            color: white;
            font-weight: 600;
            padding: 0.55rem 1.2rem;
            transition: all 0.15s ease-in-out;
        }
        .stButton > button:hover {
            border-color: rgba(255,255,255,0.3);
            transform: translateY(-1px);
            box-shadow: 0 6px 18px rgba(124,58,237,0.35);
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            background: rgba(255,255,255,0.03);
            padding: 6px;
            border-radius: 12px;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            padding: 8px 16px;
            color: #9ca3af;
        }
        .stTabs [aria-selected="true"] {
            background: rgba(124,58,237,0.25) !important;
            color: #f5f3ff !important;
        }

        /* Chat bubbles */
        .stChatMessage {
            border-radius: 14px;
        }

        /* Misc text */
        .muted { color: #9ca3af; font-size: 0.88rem; }
        hr { border-color: rgba(255,255,255,0.08); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Session state
# ----------------------------------------------------------------------------
if "result" not in st.session_state:
    st.session_state.result = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processing" not in st.session_state:
    st.session_state.processing = False

# ----------------------------------------------------------------------------
# Sidebar — inputs
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎬 New Session")
    st.markdown('<p class="muted">Paste a YouTube URL or a local file path.</p>', unsafe_allow_html=True)

    source = st.text_input("Video source", placeholder="https://youtu.be/... or /path/to/file.mp4")
    language = st.selectbox("Language", options=["english", "hinglish"], index=0)

    run_clicked = st.button("🚀 Process Video", use_container_width=True)

    st.markdown("---")
    if st.session_state.result:
        if st.button("🗑️ Clear session", use_container_width=True):
            st.session_state.result = None
            st.session_state.messages = []
            st.rerun()

    st.markdown("---")
    st.markdown(
        '<p class="muted">Pipeline: audio extraction → ASR transcription → '
        'summarization → extraction → RAG indexing.</p>',
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🎬 AI Video Assistant</h1>
        <p>Summarize, extract, and chat with any video — powered by RAG.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Run pipeline
# ----------------------------------------------------------------------------
if run_clicked:
    if not source.strip():
        st.error("Please enter a YouTube URL or file path first.")
    else:
        st.session_state.messages = []
        with st.status("Processing video…", expanded=True) as status:
            st.write("🔊 Extracting and chunking audio…")
            try:
                st.write("📝 Transcribing…")
                st.write("🧠 Summarizing & extracting insights…")
                st.write("📚 Building RAG index…")
                result = run_pipeline(source.strip().strip('"').strip("'"), language)
                st.session_state.result = result
                status.update(label="✅ Done!", state="complete", expanded=False)
            except Exception as e:
                status.update(label="❌ Failed", state="error")
                st.error(f"Something went wrong: {e}")
                st.session_state.result = None

# ----------------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------------
result = st.session_state.result

if result:
    st.markdown(f'<span class="video-title">📌 {result["title"]}</span>', unsafe_allow_html=True)
    st.write("")

    tab_summary, tab_actions, tab_decisions, tab_questions, tab_transcript, tab_chat = st.tabs(
        ["📋 Summary", "✅ Action Items", "🔑 Decisions", "❓ Questions", "📄 Transcript", "💬 Chat"]
    )

    with tab_summary:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Summary")
        st.write(result["summary"])
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_actions:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Action Items")
        st.write(result["action_items"])
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_decisions:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Key Decisions")
        st.write(result["key_decisions"])
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_questions:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Open Questions")
        st.write(result["open_questions"])
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_transcript:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Full Transcript")
        st.text_area("Transcript", value=result["transcript"], height=400, label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)

    with tab_chat:
        st.markdown('<p class="muted">Ask anything about this video — answers are grounded in the transcript.</p>', unsafe_allow_html=True)
        st.write("")

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        question = st.chat_input("Ask a question about the video…")
        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    answer = ask_question(result["rag_chain"], question)
                st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.markdown(
        """
        <div class="card" style="text-align:center; padding: 3rem 1.5rem;">
            <h4>No video processed yet</h4>
            <p class="muted">Enter a YouTube URL or local file path in the sidebar and click
            <b>Process Video</b> to get started.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )