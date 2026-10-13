# ============================================================
# AskMyBooks — Chat With Your Books
# Built by Shahriar Shoaib
# ============================================================
# Flow: upload PDFs -> split into chunks -> embed -> store in Chroma
#       -> student asks a question -> find best chunks -> LLM answers
# "New Chat" deletes the vector database, so every chat starts clean.

import os
import uuid
import tempfile
import streamlit as st
from dotenv import load_dotenv
load_dotenv()  # reads GOOGLE_API_KEY and GROQ_API_KEY from your .env file

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# ---- Setting you can change: biggest PDF allowed, in MB ----
MAX_MB = 20

# ---- Page settings (must be the first Streamlit command) ----
st.set_page_config(page_title="AskMyBooks", page_icon="📚", layout="centered")

# ---- A little CSS to make the header pretty ----
st.markdown("""
<style>
.hero-title {
    font-size: 2.8rem; font-weight: 800; margin-bottom: 0;
    background: linear-gradient(90deg, #7F5AF0, #2CB67D);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-sub { font-size: 1.05rem; opacity: 0.8; margin-top: 0.2rem; }
.hero-by  { font-size: 0.9rem; opacity: 0.6; margin-bottom: 1.2rem; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Models (created once, then reused — no reloading every message)
# ============================================================
@st.cache_resource
def get_embeddings():
    return GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")

@st.cache_resource
def get_llm():
    return ChatGroq(model="openai/gpt-oss-120b")

# The prompt: tells the AI to answer ONLY from the book text we give it
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful study assistant. "
     "Use ONLY the provided context to answer the question. "
     "If the answer is not in the context, say: "
     "'I could not find the answer in the uploaded books.'"),
    ("human", "Context:\n{context}\n\nQuestion: {question}")
])


# ============================================================
# Session memory (each student's browser tab gets its own copy)
# ============================================================
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None   # the vector database (empty until books are processed)
    st.session_state.chat = []            # list of messages shown on screen
    st.session_state.books = []           # names of uploaded books
    st.session_state.uploader_key = 0     # changing this number clears the upload box
    # Unique name per student, so two students never mix their books
    st.session_state.collection = f"books_{uuid.uuid4().hex[:10]}"


# ============================================================
# Functions
# ============================================================
def build_knowledge_base(files):
    """PDFs -> pages -> chunks -> embeddings -> Chroma database (in memory)."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_chunks = []

    for f in files:
        # PyPDFLoader needs a real file on disk, so save the upload temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(f.getbuffer())
            tmp_path = tmp.name
        try:
            pages = PyPDFLoader(tmp_path).load()
        finally:
            os.remove(tmp_path)  # delete the temp file right away

        for page in pages:
            page.metadata["book"] = f.name   # remember the real book name for sources
        all_chunks.extend(splitter.split_documents(pages))

    if not all_chunks:
        return None  # PDF had no readable text (maybe a scanned book)

    # No persist_directory = database lives in memory only, nothing saved on disk
    return Chroma.from_documents(
        documents=all_chunks,
        embedding=get_embeddings(),
        collection_name=st.session_state.collection,
    )


def new_chat():
    """Delete the vector database and reset everything."""
    if st.session_state.vectorstore is not None:
        st.session_state.vectorstore.delete_collection()   # wipes the database
    st.session_state.vectorstore = None
    st.session_state.chat = []
    st.session_state.books = []
    st.session_state.collection = f"books_{uuid.uuid4().hex[:10]}"
    st.session_state.uploader_key += 1   # clears the upload box


# ============================================================
# Sidebar: upload books + New Chat
# ============================================================
with st.sidebar:
    st.header("📚 Your Books")

    files = st.file_uploader(
        "Upload PDF books",
        type="pdf",
        help=f"Max {MAX_MB} MB per PDF",
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
    )

    already_processed = st.session_state.vectorstore is not None

    if st.button("Process Books", type="primary",
                 disabled=(not files) or already_processed,
                 use_container_width=True):
        # Size check: find any PDF bigger than the limit (f.size is in bytes)
        too_big = [f.name for f in files if f.size > MAX_MB * 1024 * 1024]

        if too_big:
            st.error(f"Too big (max {MAX_MB} MB per PDF): " + ", ".join(too_big))
        else:
            with st.spinner("Reading your books... big books take a minute."):
                try:
                    vs = build_knowledge_base(files)
                    if vs is None:
                        st.error("Could not read text from these PDFs.")
                    else:
                        st.session_state.vectorstore = vs
                        st.session_state.books = [f.name for f in files]
                        st.rerun()
                except Exception as e:
                    st.error(f"Something went wrong: {e}")

    if st.session_state.books:
        st.success("Ready! Loaded:")
        for name in st.session_state.books:
            st.write(f"📖 {name}")

    st.divider()
    st.button("🔄 New Chat (clears books)", on_click=new_chat, use_container_width=True)
    st.caption("New Chat deletes your books and chat history.")
    st.caption("Built by **Shahriar Shoaib**")


# ============================================================
# Main page
# ============================================================
st.markdown('<p class="hero-title">AskMyBooks</p>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Upload your books. Ask anything. Get answers from the pages.</p>',
            unsafe_allow_html=True)
st.markdown('<p class="hero-by">Built by Shahriar Shoaib</p>', unsafe_allow_html=True)

if st.session_state.vectorstore is None:
    # No books yet -> show simple instructions
    st.info("**How it works**\n\n"
            "1. Upload one or more PDF books in the left sidebar\n"
            "2. Click **Process Books**\n"
            "3. Ask your questions here")
else:
    # Show old messages
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources"):
                    for s in msg["sources"]:
                        st.write(s)

    question = st.chat_input("Ask a question about your books...")

    if question:
        st.session_state.chat.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching your books..."):
                # Retriever = finds the best chunks for the question
                retriever = st.session_state.vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
                )
                docs = retriever.invoke(question)
                context = "\n\n".join(d.page_content for d in docs)

                final_prompt = prompt.invoke({"context": context, "question": question})
                answer = get_llm().invoke(final_prompt).content

            st.markdown(answer)

            # Sources: book name + page number (page numbers start at 0, so +1)
            sources = sorted({
                f"{d.metadata.get('book', 'Book')} — page {d.metadata.get('page', 0) + 1}"
                for d in docs
            })
            with st.expander("Sources"):
                for s in sources:
                    st.write(s)

        st.session_state.chat.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
