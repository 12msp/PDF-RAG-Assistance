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


# 1. Load environment variables

load_dotenv()


# 2. Streamlit configuration

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📄",
    layout="centered"
)

st.title("📄 PDF RAG Assistant")

st.write(
    "Upload a PDF and ask questions about its content."
)


# 3. Session state

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None


# 4. Load Gemini

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0
)


# 5. Prompt

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


# 6. Upload PDF

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


# 7. Process PDF

if uploaded_file:

    st.write(
        f"**Selected file:** {uploaded_file.name}"
    )

    process_button = st.button(
        "🔄 Process PDF"
    )

    if process_button:

        with st.spinner("Processing PDF..."):

            # Save uploaded PDF

            os.makedirs("data", exist_ok=True)

            pdf_path = os.path.join(
                "data",
                uploaded_file.name
            )

            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())


            # Load PDF

            loader = PyPDFLoader(pdf_path)

            documents = loader.load()

            st.write(
                f"📄 Number of pages: {len(documents)}"
            )


            # Split PDF into chunks

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1200,
                chunk_overlap=100
            )

            chunks = text_splitter.split_documents(
                documents
            )

            st.write(
                f"🧩 Number of chunks: {len(chunks)}"
            )


            # Create embedding model

            with st.spinner(
                "Creating embeddings..."
            ):

                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )


            # Remove previous ChromaDB

            chroma_path = "./chroma_db"

            if os.path.exists(chroma_path):

                shutil.rmtree(
                    chroma_path
                )


            # Create ChromaDB

            with st.spinner(
                "Creating vector database..."
            ):

                vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory=chroma_path
                )


            # Save vectorstore in session

            st.session_state.vectorstore = vectorstore

            st.session_state.document_name = (
                uploaded_file.name
            )


            st.success(
                "✅ PDF processed successfully!"
            )


# 8. Question answering section

if st.session_state.vectorstore is not None:

    st.divider()

    st.subheader("💬 Ask a question")

    query = st.text_input(
        "Enter your question:",
        placeholder="Example: What is the leave policy?"
    )


    # 9. Retrieve documents

    if query:

        with st.spinner(
            "Searching the document..."
        ):

            # Create retriever

            retriever = (
                st.session_state.vectorstore
                .as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": 2,
                        "fetch_k": 10,
                        "lambda_mult": 0.5
                    }
                )
            )


            # Retrieve relevant chunks

            results = retriever.invoke(
                query
            )


        # 10. DEBUG: Show retrieved chunks

        st.subheader(
            "🔍 Retrieved Chunks"
        )

        for i, doc in enumerate(results):

            page = doc.metadata.get(
                "page",
                "Unknown"
            )

            if page != "Unknown":
                page = page + 1

            st.write(
                f"**Result {i + 1} — Page {page}**"
            )

            st.write(
                doc.page_content
            )

            st.write("---")


        # 11. Create context

        context = "\n\n".join(
            doc.page_content
            for doc in results
        )


        # 12. Create prompt

        messages = prompt.invoke({
            "context": context,
            "question": query
        })


        # 13. Generate answer with Gemini

        with st.spinner(
            "Generating answer..."
        ):

            response = llm.invoke(
                messages
            )


        # 14. Display answer

        st.subheader(
            "🤖 Answer"
        )

        st.write(
            response.content
        )


        # 15. Display sources

        st.subheader(
            "📚 Sources"
        )

        for doc in results:

            page = doc.metadata.get(
                "page",
                "Unknown"
            )

            if page != "Unknown":
                page = page + 1

            st.write(
                f"📄 Page {page}"
            )
