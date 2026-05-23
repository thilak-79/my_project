import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma


# =========================
# Load API Key
# =========================
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found. Please add it to your .env file.")
    st.stop()


# =========================
# Gemini Model
# =========================
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3

)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"

)


# =========================
# Read PDF
# =========================
def read_pdf(pdf_file):
    text = ""

    reader = PdfReader(pdf_file)

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# =========================
# Split Text into Chunks
# =========================
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300
    )

    chunks = splitter.split_text(text)

    return chunks


# =========================
# Create Vector Store
# =========================
def create_vector_store(chunks):
    vector_store = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory="db"
    )

    return vector_store


# =========================
# Build Prompt
# =========================
def build_prompt(context, question):
    prompt = f"""
You are a helpful AI study assistant.

Answer the user's question using the uploaded PDF content.

Rules:
- Use the provided context as the main source.
- If the context is enough, explain clearly.
- If the context is partial, give the best possible answer based on it.
- Do not mention unrelated topics.
- If the answer is not available in the PDF, say:
  "I could not find this information in the uploaded PDF."

Explain in:
- simple language
- clear steps
- beginner-friendly style
- examples if useful

Context:
{context}

Question:
{question}

Answer:
"""
    return prompt

# =========================
# Streamlit UI
# =========================
st.set_page_config(
    page_title="AI PDF Assistant",
    page_icon="📄",
    layout="centered"
)

st.title("📄 AI PDF Assistant")
st.write("Upload a PDF and ask questions about it.")

uploaded_file = st.file_uploader(
    "Upload your PDF file",
    type=["pdf"]
)

question = st.text_input(
    "Ask a question from the PDF"
)


if uploaded_file is not None:
    with st.spinner("Reading PDF..."):
        pdf_text = read_pdf(uploaded_file)

    if not pdf_text.strip():
        st.error("No text found in this PDF. It may be a scanned/image-based PDF.")
        st.stop()

    with st.spinner("Splitting text into chunks..."):
        chunks = split_text(pdf_text)

    with st.spinner("Creating vector database..."):
        vector_store = create_vector_store(chunks)

    st.success("PDF processed successfully!")

    if question:
        with st.spinner("Searching PDF and generating answer..."):
            docs = vector_store.similarity_search(
                question,
                k=3
            )

            context = "\n\n".join(
                [doc.page_content for doc in docs]
            )

            prompt = build_prompt(context, question)

            response = llm.invoke(prompt)

        st.subheader("Answer")
        st.write(response.content)

        with st.expander("Retrieved Context"):
            st.write(context)