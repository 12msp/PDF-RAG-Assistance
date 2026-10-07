import os
import shutil
import streamlit as st

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📄"
)

st.title("📄 PDF RAG Assistant")
st.write("Upload a PDF and ask questions about its content.")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0
)

prompt = ChatPromptTemplate.from_template("""
You are a helpful PDF document assistant.

Answer the user's question using ONLY the information
provided in the context.

If the answer is not available in the context, say:

"I could not find this information in the uploaded document."

Do not use your general knowledge.
Do not make up information.

Context:
{context}

Question:
{question}

Answer clearly and concisely.
""")

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)

if uploaded_file:

    st.write(f"Selected file: {uploaded_file.name}")

    process_button = st.button("🔄 Process PDF")

    if process_button:

        with st.spinner("Processing PDF..."):

            os.makedirs("data", exist_ok=True)

            pdf_path = os.path.join(
                "data",
                uploaded_file.name
            )

            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            loader = PyPDFLoader(pdf_path)
            documents = loader.load()

            st.write(
                f"Number of pages: {len(documents)}"
            )

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200,
                chunk_overlap=100
            )

            chunks = text_splitter.split_documents(
                documents
            )

            st.write(
                f"Number of chunks: {len(chunks)}"
            )

            with st.spinner("Creating embeddings..."):

                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )

            chroma_path = "./chroma_db"

            if os.path.exists(chroma_path):
                shutil.rmtree(chroma_path)

            with st.spinner("Creating vector database..."):

                vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory=chroma_path
                )

            st.session_state.vectorstore = vectorstore
            st.session_state.document_name = uploaded_file.name

            st.success("PDF processed successfully!")

if st.session_state.vectorstore is not None:

    st.divider()

    st.subheader("💬 Ask a question")

    query = st.text_input(
        "Enter your question:",
        placeholder="Example: What is the leave policy?"
    )

    if query:

        retriever = st.session_state.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 2,
                "fetch_k": 10,
                "lambda_mult": 0.5
            }
        )

        with st.spinner("Searching the document..."):

            results = retriever.invoke(query)

        st.subheader("🔍 Retrieved Chunks")

        for i, doc in enumerate(results):

            page = doc.metadata.get(
                "page",
                "Unknown"
            )

            if page != "Unknown":
                page = page + 1

            st.write(
                f"Result {i + 1} — Page {page}"
            )

            st.write(doc.page_content)

        context = "\n\n".join(
            doc.page_content
            for doc in results
        )

        messages = prompt.invoke({
            "context": context,
            "question": query
        })

        with st.spinner("Generating answer..."):

            response = llm.invoke(messages)

        st.subheader("🤖 Answer")
        st.write(response.content)

        st.subheader("📚 Sources")

        for doc in results:

            page = doc.metadata.get(
                "page",
                "Unknown"
            )

            if page != "Unknown":
                page = page + 1

            st.write(f"📄 Page {page}")
