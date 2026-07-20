import streamlit as st
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq

st.set_page_config(page_title="PDF ChatBot with RAG", page_icon="📄", layout="wide")

st.title("📄 PDF ChatBot with RAG & Dual LLM Architecture")
st.write("Upload a PDF and ask questions about its content.")


@st.cache_resource
def load_embedding_model():
    """Load the sentence transformer model once and cache it."""
    return SentenceTransformer("all-MiniLM-L6-v2")


def extract_text_from_pdf(uploaded_file):
    try:
        pdf_bytes = uploaded_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        if doc.page_count == 0:
            return None, "The PDF appears to be empty (0 pages)."

        text = "".join(page.get_text() for page in doc)
        doc.close()

        if not text.strip():
            return None, "No extractable text found. This PDF might be scanned images without OCR."

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
    """Embed chunks and build a FAISS index for similarity search."""
    try:
        embeddings = model.encode(chunks, show_progress_bar=False)
        embeddings = np.array(embeddings).astype("float32")

        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)

        return index, None

    except Exception as e:
        return None, f"Failed to build vector store: {e}"


def retrieve_chunks(query, index, chunks, model, k):
    """Find the top-k most relevant chunks for a query."""
    try:
        query_embedding = model.encode([query]).astype("float32")
        distances, indices = index.search(query_embedding, k)
        retrieved = [chunks[i] for i in indices[0] if i < len(chunks)]
        return retrieved, None
    except Exception as e:
        return None, f"Retrieval failed: {e}"


def generate_answer(client, query, context_chunks):
    """Run the dual-LLM pipeline: summarize context, then answer."""
    try:
        context = "\n\n".join(context_chunks)

        # LLM 1: Context Summarizer
        summary_response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "system", "content": "Summarize the following context concisely, focusing only on information relevant to the user's question."},
                {"role": "user", "content": f"Question: {query}\n\nContext:\n{context}"},
            ],
            temperature=0.3,
        )
        context_summary = summary_response.choices[0].message.content

        # LLM 2: Answer Generator
        answer_response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": "Answer the user's question accurately based on the provided summary. If the summary doesn't contain the answer, say so."},
                {"role": "user", "content": f"Question: {query}\n\nSummary:\n{context_summary}"},
            ],
            temperature=0.3,
        )
        final_answer = answer_response.choices[0].message.content

        return final_answer, context_summary, None

    except Exception as e:
        return None, None, f"Answer generation failed: {e}"


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("Configuration")
    groq_api_key = st.text_input("Enter your Groq API Key", type="password")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    top_k = st.slider("Number of chunks to retrieve", min_value=1, max_value=5, value=3)

# ---------------- MAIN AREA ----------------
if not groq_api_key:
    st.warning("Please enter your Groq API key in the sidebar to continue.")
elif not uploaded_file:
    st.info("Please upload a PDF file to start chatting.")
else:
    with st.spinner("Extracting text from PDF..."):
        extracted_text, error = extract_text_from_pdf(uploaded_file)

    if error:
        st.error(error)
    else:
        with st.spinner("Splitting text into chunks..."):
            chunks, chunk_error = chunk_text(extracted_text)

        if chunk_error:
            st.error(chunk_error)
        else:
            with st.spinner("Generating embeddings and building vector index..."):
                embed_model = load_embedding_model()
                vector_index, index_error = build_vector_store(chunks, embed_model)

            if index_error:
                st.error(index_error)
            else:
                st.success(f"PDF processed: {uploaded_file.name} ({len(extracted_text)} characters, {len(chunks)} chunks indexed)")

                st.session_state["chunks"] = chunks
                st.session_state["vector_index"] = vector_index
                st.session_state["embed_model"] = embed_model

                with st.expander("Preview extracted text"):
                    st.text(extracted_text[:5000])

                with st.expander(f"Preview chunks ({len(chunks)} total)"):
                    for i, chunk in enumerate(chunks[:3]):
                        st.markdown(f"**Chunk {i+1}:**")
                        st.text(chunk)
                        st.divider()

                st.divider()
                st.subheader("💬 Ask a question about your PDF")

                if "messages" not in st.session_state:
                    st.session_state["messages"] = []

                for msg in st.session_state["messages"]:
                    with st.chat_message(msg["role"]):
                        st.write(msg["content"])

                user_question = st.chat_input("Type your question here...")

                if user_question:
                    st.session_state["messages"].append({"role": "user", "content": user_question})
                    with st.chat_message("user"):
                        st.write(user_question)

                    with st.chat_message("assistant"):
                        with st.spinner("Searching document..."):
                            retrieved, retrieve_error = retrieve_chunks(
                                user_question, vector_index, chunks, embed_model, top_k
                            )

                        if retrieve_error:
                            st.error(retrieve_error)
                        else:
                            with st.spinner("Generating answer..."):
                                client = Groq(api_key=groq_api_key)
                                answer, summary, gen_error = generate_answer(client, user_question, retrieved)

                            if gen_error:
                                st.error(gen_error)
                            else:
                                st.write(answer)
                                st.session_state["messages"].append({"role": "assistant", "content": answer})

                                with st.expander("View context summary"):
                                    st.write(summary)
                                with st.expander("View retrieved chunks"):
                                    for i, c in enumerate(retrieved):
                                        st.markdown(f"**Chunk {i+1}:**")
                                        st.text(c)
                                        st.divider()