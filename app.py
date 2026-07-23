import streamlit as st
import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq
from datetime import datetime
import uuid
import pytesseract
from PIL import Image
import io

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

st.set_page_config(page_title="PDF Chat", page_icon="💬", layout="wide")

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500&display=swap" rel="stylesheet">
<style>
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif; font-weight: 700; letter-spacing: -0.5px; }
    section[data-testid="stSidebar"] { background-color: #F8F7FC; border-right: 1px solid #E8E6F5; }
    .stButton button { background-color: #6C5CE7; color: white; border-radius: 10px; border: none; font-weight: 500; }
    .stButton button:hover { background-color: #5B4BD6; }
    section[data-testid="stFileUploaderDropzone"] {
        border: 2px dashed #6C5CE7 !important; border-radius: 16px !important; background-color: #FAFAFF !important;
    }
    .stChatMessage { border-radius: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
    .stExpander { border: 1px solid #E8E6F5; border-radius: 12px; }
    .welcome-text { font-size: 1.3rem; color: #6B6B76; text-align: center; margin-bottom: 0.5rem; }
    .chat-item { padding: 6px 10px; border-radius: 8px; font-size: 0.9rem; }
    .chat-item:hover { background-color: #EFEDFA; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def extract_text_from_pdf(uploaded_file):
    try:
        pdf_bytes = uploaded_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        if doc.page_count == 0:
            return None, "The PDF appears to be empty (0 pages)."

        text = "".join(page.get_text() for page in doc)

        # Fallback: if normal extraction found nothing, try OCR on rendered pages
        if not text.strip():
            ocr_text = ""
            for page in doc:
                pix = page.get_pixmap(dpi=200)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                ocr_text += pytesseract.image_to_string(img)
            text = ocr_text

        doc.close()

        if not text.strip():
            return None, "No extractable text found, even after OCR. This PDF might be a low-quality scan."

        return text, None

    except Exception as e:
        return None, f"Failed to read PDF: {e}"


def chunk_text(text):
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
        chunks = splitter.split_text(text)
        if not chunks:
            return None, "Text splitting produced no chunks."
        return chunks, None
    except Exception as e:
        return None, f"Failed to split text into chunks: {e}"


def build_vector_store(chunks, model):
    try:
        embeddings = model.encode(chunks, show_progress_bar=False)
        embeddings = np.array(embeddings).astype("float32")
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        return index, None
    except Exception as e:
        return None, f"Failed to build vector store: {e}"


def retrieve_chunks(query, index, chunks, model, k):
    try:
        query_embedding = model.encode([query]).astype("float32")
        distances, indices = index.search(query_embedding, k)
        retrieved = [chunks[i] for i in indices[0] if i < len(chunks)]
        if not retrieved:
            return None, "No relevant content found in the document for this question."
        return retrieved, None
    except Exception as e:
        return None, f"Retrieval failed: {e}"


def generate_answer(client, query, context_chunks, chat_history):
    try:
        context = "\n\n".join(context_chunks)
        recent_exchange = "\n".join(
            f"{m['role']}: {m['content']}" for m in chat_history[-4:]
        ) if chat_history else "No prior conversation."

        summary_response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": (
                    "Summarize the following document context concisely, focusing on information relevant "
                    "to the user's current question. Use the recent conversation only to understand what "
                    "the user is really asking (e.g. follow-ups like 'tell me more')."
                )},
                {"role": "user", "content": (
                    f"Recent conversation:\n{recent_exchange}\n\n"
                    f"Current question: {query}\n\nDocument context:\n{context}"
                )},
            ],
            temperature=0.3,
        )
        context_summary = summary_response.choices[0].message.content

        history_messages = [{"role": m["role"], "content": m["content"]} for m in chat_history[-6:]]

        answer_response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": (
                    "You are a friendly, conversational assistant helping someone explore a PDF document. "
                    "Answer naturally using the summary below as your source of truth. If it doesn't contain "
                    "the answer, say so honestly. When it feels natural, briefly invite the user to dive deeper "
                    "or ask something else — but vary your phrasing and don't force it every time."
                )},
                *history_messages,
                {"role": "user", "content": f"Question: {query}\n\nRelevant summary:\n{context_summary}"},
            ],
            temperature=0.4,
        )
        return answer_response.choices[0].message.content, context_summary, None

    except Exception as e:
        error_msg = str(e).lower()
        if "401" in error_msg or "invalid api key" in error_msg or "unauthorized" in error_msg:
            return None, None, "Invalid Groq API key. Please check the key and try again."
        elif "429" in error_msg or "rate limit" in error_msg:
            return None, None, "Groq API rate limit reached. Please wait a moment and try again."
        elif "timeout" in error_msg:
            return None, None, "Request timed out. Please check your internet connection and try again."
        else:
            return None, None, f"Answer generation failed: {e}"


def validate_groq_key(key):
    try:
        client = Groq(api_key=key)
        client.models.list()
        return True, None
    except Exception as e:
        error_msg = str(e).lower()
        if "401" in error_msg or "invalid" in error_msg or "unauthorized" in error_msg:
            return False, "That key doesn't look valid. Please double-check it."
        elif "timeout" in error_msg:
            return False, "Connection timed out. Check your internet and try again."
        else:
            return False, f"Couldn't verify the key: {e}"

embed_model = load_embedding_model()
# ---------------- API KEY GATE ----------------
if "groq_api_key" not in st.session_state:
    st.session_state["groq_api_key"] = None


@st.dialog("Welcome — enter your Groq API key")
def api_key_dialog():
    st.caption("You'll need this once per session to power the chatbot.")
    key_input = st.text_input("Groq API Key", type="password")
    if st.button("Continue", use_container_width=True):
        if not key_input or not key_input.strip():
            st.error("Please enter a key.")
        else:
            placeholder = st.empty()
            with placeholder.container():
                st.spinner("Verifying...")
                valid, err = validate_groq_key(key_input.strip())
            placeholder.empty()

            if valid:
                st.session_state["groq_api_key"] = key_input.strip()
                st.rerun()
            else:
                st.error(err)


if not st.session_state["groq_api_key"]:
    api_key_dialog()
    st.stop()

groq_api_key = st.session_state["groq_api_key"]

# ---------------- MULTI-CHAT STATE ----------------
if "chats" not in st.session_state:
    st.session_state["chats"] = {}  # chat_id -> {title, messages, chunks, vector_index}
if "active_chat_id" not in st.session_state:
    st.session_state["active_chat_id"] = None


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("### 💬 PDF Chat")

    if st.button("➕ New chat", use_container_width=True):
        st.session_state["active_chat_id"] = None
        st.rerun()

    top_k = st.slider("Chunks to retrieve", min_value=1, max_value=5, value=3)

    st.divider()
    search_term = st.text_input("🔍 Search chats", placeholder="Search by PDF name...")

    st.markdown("**Your chats**")
    chats = st.session_state["chats"]

    if not chats:
        st.caption("No chats yet — upload a PDF to start one.")
    else:
        # Most recent first
        sorted_ids = sorted(chats.keys(), key=lambda cid: chats[cid]["created_at"], reverse=True)
        filtered_ids = [
            cid for cid in sorted_ids
            if not search_term or search_term.lower() in chats[cid]["title"].lower()
        ]

        if not filtered_ids:
            st.caption("No chats match your search.")

        for cid in filtered_ids:
            chat = chats[cid]
            is_active = cid == st.session_state["active_chat_id"]
            button_type = "primary" if is_active else "secondary"
            if st.button(chat["title"], key=f"chat_btn_{cid}", use_container_width=True, type=button_type):
                st.session_state["active_chat_id"] = cid
                st.rerun()

# ---------------- MAIN AREA ----------------
active_id = st.session_state["active_chat_id"]

if active_id is None:
    hour = datetime.now().hour
    greeting = "Good morning" if hour < 12 else "Good afternoon" if hour < 18 else "Good evening"

    st.markdown(f"<h2 style='text-align:center;'>{greeting} 👋</h2>", unsafe_allow_html=True)
    st.markdown("<p class='welcome-text'>Upload a PDF to start a new chat.</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        uploaded_file = st.file_uploader("Drop your PDF here", type=["pdf"], label_visibility="collapsed")

    if uploaded_file is not None:
        if uploaded_file.size > 20 * 1024 * 1024:
            st.error("File too large. Please upload a PDF under 20MB.")
        else:
            with st.spinner("Reading your PDF..."):
                extracted_text, error = extract_text_from_pdf(uploaded_file)

            if error:
                st.error(error)
            else:
                with st.spinner("Indexing content..."):
                    chunks, chunk_error = chunk_text(extracted_text)

                if chunk_error:
                    st.error(chunk_error)
                else:
                    with st.spinner("Preparing semantic search..."):
                        vector_index, index_error = build_vector_store(chunks, embed_model)

                    if index_error:
                        st.error(index_error)
                    else:
                        new_id = str(uuid.uuid4())
                        st.session_state["chats"][new_id] = {
                            "title": uploaded_file.name,
                            "messages": [],
                            "chunks": chunks,
                            "vector_index": vector_index,
                            "created_at": datetime.now().timestamp(),
                        }
                        st.session_state["active_chat_id"] = new_id
                        st.rerun()

else:
    chat = st.session_state["chats"][active_id]
    chunks = chat["chunks"]
    vector_index = chat["vector_index"]

    st.success(f"Chatting about **{chat['title']}** ({len(chunks)} sections indexed)")

    for msg in chat["messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_question = st.chat_input("Ask something about this document...")

    if user_question and user_question.strip():
        chat["messages"].append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.write(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Extracting relevant sections..."):
                retrieved, retrieve_error = retrieve_chunks(user_question, vector_index, chunks, embed_model, top_k)

            if retrieve_error:
                st.error(retrieve_error)
            else:
                with st.spinner("Thinking..."):
                    client = Groq(api_key=groq_api_key)
                    answer, summary, gen_error = generate_answer(
                        client, user_question, retrieved, chat["messages"]
                    )

                if gen_error:
                    st.error(gen_error)
                else:
                    st.write(answer)
                    chat["messages"].append({"role": "assistant", "content": answer})

                    with st.expander("View context summary"):
                        st.write(summary)
                    with st.expander("View retrieved chunks"):
                        for i, c in enumerate(retrieved):
                            st.markdown(f"**Chunk {i+1}:**")
                            st.text(c)
                            st.divider()