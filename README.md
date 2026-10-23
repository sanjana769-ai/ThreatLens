# ThreatLens 🛡️

**AI-Powered Cybersecurity Threat Intelligence Platform**

ThreatLens is a cybersecurity intelligence platform that combines hybrid information retrieval with evidence-grounded AI generation to help users explore threat intelligence, analyze cybersecurity documents, and retrieve relevant security knowledge.

## 🚀 Features

- **Threat Intelligence Search:** Retrieve relevant cybersecurity information using hybrid search.
- **Hybrid Retrieval:** Combine semantic similarity and BM25 keyword retrieval to improve search relevance.
- **AI-Powered Answers:** Generate concise answers grounded in retrieved evidence using Groq-hosted language models.
- **Document Analysis:** Upload cybersecurity PDF documents and ask questions about their contents.
- **Evidence-Based Citations:** Support threat intelligence answers with MITRE ATT&CK technique references and document answers with page citations.
- **Evaluation Framework:** Evaluate retrieval and answer quality using project-specific evaluation scripts.
- **Security-Focused Responses:** Return an insufficient-evidence response when the available context does not support an answer.

## 🧰 Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| User Interface | Streamlit |
| Language Model API | Groq |
| Embeddings | Sentence Transformers |
| Vector Database | ChromaDB |
| Keyword Retrieval | BM25 |
| Document Processing | PyPDF |
| Environment Configuration | python-dotenv |
| Threat Intelligence | MITRE ATT&CK data |

## 🏗️ Architecture

1. **Data ingestion:** Threat intelligence data and cybersecurity documents are processed for retrieval.
2. **Document processing:** Documents are split into searchable text chunks.
3. **Hybrid retrieval:** Semantic search and BM25 retrieve relevant evidence.
4. **Context preparation:** Retrieved passages are prepared for the language model.
5. **AI generation:** The language model generates an answer using the supplied evidence.
6. **Citation handling:** The application formats and checks citations against retrieved sources.
7. **Evaluation:** Evaluation scripts assess retrieval and question-answering behavior.

## 📂 Project Structure

```text
ThreatLens/
├── app.py
├── document_retrieval.py
├── requirements.txt
├── data/
│   ├── enterprise-attack.json
│   ├── techniques.json
│   └── techniques.jsonl
├── evaluation/
│   ├── evaluate_rag.py
│   └── evaluate_retrieval.py
├── ingestion/
│   ├── download_attack.py
│   └── parse_attack.py
├── llm/
│   ├── __init__.py
│   └── groq_client.py
├── retrieval/
│   ├── __init__.py
│   └── hybrid_search.py
├── test_citation_formatting.py
├── test_citation_validation.py
├── test_evaluation.py
├── test_groq.py
├── test_llm.py
├── test_rag.py
├── test_retrieval.py
└── .gitignore
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/sanjana769-ai/ThreatLens.git
cd ThreatLens
```

### 2. Create and activate a virtual environment

**Windows PowerShell**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-20b
```

Replace `your_groq_api_key` with your own Groq API key. Keep `.env` private and never commit it to GitHub.

### 5. Run the application

```bash
streamlit run app.py
```

Streamlit will display a local URL, usually `http://localhost:8501`.

## 🧪 Testing and Evaluation

Run the project's test scripts from the repository root:

```bash
python test_citation_formatting.py
python test_citation_validation.py
python test_retrieval.py
python test_rag.py
python test_llm.py
python test_groq.py
python test_evaluation.py
```

Run the evaluation scripts:

```bash
python -m evaluation.evaluate_retrieval
python -m evaluation.evaluate_rag
```

Some tests may require configured API credentials, initialized data, or additional setup.

## 🔐 Security Considerations

- Store API credentials in environment variables.
- Do not commit `.env` files, credentials, or sensitive documents.
- Treat retrieved documents as untrusted input.
- Ground AI-generated answers in retrieved evidence.
- Verify citations and important findings before using them in security decisions.

## 🎯 Project Goals

ThreatLens explores how hybrid retrieval and evidence-grounded language models can make cybersecurity intelligence more accessible, searchable, and explainable.

## 👩‍💻 Author

**Sanjana Badugu**

GitHub: [@sanjana769-ai](https://github.com/sanjana769-ai)

---

*ThreatLens — Turning cybersecurity information into evidence-grounded intelligence.*
