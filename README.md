

# 🔍 RAG AI Assistant

This project is a question-answering assistant based on PDF documents. It uses a **Retrieval-Augmented Generation (RAG)** approach to read documents, index them, and generate answers using a local LLM.

---

## 🗂️ Project Structure

```bash
rag-assistant/
│
├── app.py                  # GUI interface using customtkinter
├── main.py                 # CLI interface for manual testing
├── rag_pipeline.py         # RAG pipeline (PDF reading, embedding, indexing, translation, search)
│
├── pdfs/                   # Folder to drop PDF files
├── faiss_index.idx         # Auto-generated FAISS index
│
├── requirements.txt        # Python dependencies
├── README.md               # Project guide
└── MODEL_INFO.md           # Details about models used and their purpose
```

- `rag_pipeline.py`: Core RAG logic + bilingual support (auto-translation of query/response).
- `main.py`: Command-line interface.
- `app.py`: Graphical interface using customtkinter.
- `pdfs/`: Folder to drop your PDF files.
- `faiss_index.idx`: Auto-generated file containing the FAISS index.

## ⚙️ Installation

1. Clone the project:

```bash
git clone https://github.com/Salsabilguizanicegid/rag_pipeline.git
cd rag_pipeline
```

2. Install the dependencies : 

```bash
pip install -r requirements.txt
```

3. Place your PDF files in the pdfs/.

## ▶️ Usage

1. Command-line mode (quick test):

```bash
python main.py
```

2. GUI Mode (Tkinter-based):
```bash
streamlit run app.py
```
💬 You can now ask questions in either French or English.
The assistant will automatically translate the query and answer if the documents are in a different language.





