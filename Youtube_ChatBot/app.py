import streamlit as st
from chatbot import answer_question

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="YouTube RAG Bot",
    page_icon="🎥",
    layout="centered"
)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("ℹ️ How it works")
    st.markdown(
        """
        1. Paste a **YouTube video URL**  
        2. Ask a question related to the video  
        3. The bot answers **only from the transcript**

        ⚠️ If the transcript doesn’t contain the answer,  
        the bot will say **“I don’t know”**.
        """
    )
    st.markdown("---")
    st.caption("Built with LangChain + Gemini")

# ---------------- MAIN UI ----------------
st.title("🎥 YouTube Q&A Bot")
st.write("Ask questions from a YouTube video transcript")

# Input containers
with st.container():
    youtube_url = st.text_input(
        "🔗 YouTube URL",
        placeholder="https://www.youtube.com/watch?v=..."
    )

    question = st.text_input(
        "❓ Your question",
        placeholder="What is the main topic of this video?"
    )

# Buttons row
col1, col2 = st.columns([1, 1])

with col1:
    ask_clicked = st.button("🚀 Ask", use_container_width=True)

with col2:
    clear_clicked = st.button("🧹 Clear", use_container_width=True)

# Clear inputs
if clear_clicked:
    st.experimental_rerun()

# ---------------- ACTION ----------------
if ask_clicked:
    if not youtube_url or not question:
        st.warning("Please enter both the YouTube URL and a question.")
    else:
        with st.spinner("🤔 Thinking..."):
            answer = answer_question(youtube_url, question)

        st.success("📌 Answer")
        st.markdown(
            f"""
            <div style="
                background-color:#f0f2f6;
                padding:15px;
                border-radius:8px;
                font-size:16px;">
                {answer}
            </div>
            """,
            unsafe_allow_html=True
        )

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("© YouTube RAG Bot • Transcript-grounded answers only")

