PDF RAG Assistant

A Streamlit-based RAG application that allows users
to upload PDF documents and ask questions about them.

Tech Stack:
- Python
- Streamlit
- LangChain
- Gemini
- HuggingFace Embeddings
- ChromaDB
- PyPDF

Architecture:

PDF
 ↓
PyPDFLoader
 ↓
Text Chunking
 ↓
HuggingFace Embeddings
 ↓
ChromaDB
 ↓
MMR Retriever
 ↓
Gemini
 ↓
Answer + Sources