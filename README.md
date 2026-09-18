# 🎓 UOBS AI Assistant — University RAG Chatbot

An AI-powered **Retrieval-Augmented Generation (RAG) chatbot** for the **University of Baltistan, Skardu (UOBS)** website.

The chatbot retrieves relevant information from a pre-built university knowledge base using **FAISS** and **Sentence Transformers**, then uses **Groq's `openai/gpt-oss-120b`** model to generate concise, context-grounded answers.

The system is designed to minimize hallucinations by instructing the LLM to answer university-specific questions **only from the retrieved knowledge base**.

---

## 🚀 Features

* 🎓 University of Baltistan, Skardu knowledge assistant
* 🔎 Semantic search using **Sentence Transformers**
* ⚡ Fast vector retrieval using **FAISS**
* 🤖 Answer generation using **Groq API**
* 🧠 Uses `openai/gpt-oss-120b`
* 📚 Displays relevant document titles and original source URLs
* 🚫 No document upload required from users
* 🛡️ Context-grounded responses to reduce hallucinations
* 💬 Natural handling of greetings and unrelated questions
* ⚙️ Streamlit caching for faster application performance
* 📱 Responsive and user-friendly Streamlit interface
* ☁️ Optimized for **Streamlit Community Cloud**
* ❌ No LangChain
* ❌ No LlamaIndex

---

## 🏗️ System Architecture

```text
                    ┌──────────────────────┐
                    │      User Query      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Sentence Transformer │
                    │   Query Embedding    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │        FAISS         │
                    │ Semantic Retrieval   │
                    └──────────┬───────────┘
                               │
                         Top-K Chunks
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Context Builder    │
                    │ Metadata + Content   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      Groq API        │
                    │ openai/gpt-oss-120b  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Grounded AI Answer   │
                    │      + Sources       │
                    └──────────────────────┘
```

---

## 🧰 Technology Stack

| Technology            | Purpose                      |
| --------------------- | ---------------------------- |
| Python                | Backend/application logic    |
| Streamlit             | Web interface                |
| Sentence Transformers | Query embeddings             |
| FAISS                 | Vector similarity search     |
| Groq                  | LLM inference API            |
| `openai/gpt-oss-120b` | Answer generation            |
| NumPy                 | Vector processing            |
| PyTorch               | Sentence Transformer backend |

The project intentionally does **not** use LangChain or LlamaIndex.

---

## 📁 Project Structure

The pre-built FAISS index and metadata files are stored directly alongside the application.

```text
UOBS-RAG-Chatbot/
│
├── app.py
├── requirements.txt
├── index.faiss
├── metadata.json
└── README.md
```

### File Description

#### `app.py`

Main Streamlit application containing:

* FAISS loading
* Metadata loading
* Sentence Transformer initialization
* Query embedding
* Vector retrieval
* Context construction
* Groq API integration
* Answer generation
* Source display
* Error handling
* Streamlit caching
* Chat interface

#### `index.faiss`

Pre-built FAISS vector index generated during the preprocessing stage.

It contains vector representations of chunks extracted from the University of Baltistan website dataset.

#### `metadata.json`

Contains metadata associated with the indexed chunks, including information such as:

* Document number
* Document title
* Source URL
* Document type
* Crawled timestamp
* Chunk text
* Chunk information
* Embedding configuration

---

## 🔄 RAG Pipeline

The chatbot follows a straightforward RAG pipeline.

### 1. User asks a question

Example:

```text
What undergraduate programs are offered at the University of Baltistan?
```

### 2. Query embedding

The question is converted into a numerical vector using the same **Sentence Transformers model** used during preprocessing.

### 3. Vector normalization

The query embedding is normalized before FAISS retrieval.

### 4. FAISS retrieval

FAISS searches the pre-built vector index and retrieves the most semantically relevant chunks.

### 5. Metadata lookup

The retrieved vector IDs are mapped to their corresponding records in `metadata.json`.

### 6. Context construction

The relevant chunks, document titles, and source URLs are provided to the LLM as context.

### 7. Grounded generation

Groq runs:

```text
openai/gpt-oss-120b
```

The model is instructed to answer using only the retrieved university information.

### 8. Source display

Relevant document titles and original URLs are displayed below the answer.

---

## 🛡️ Hallucination Control

The chatbot uses a strict grounding strategy.

The LLM is instructed to:

* Use only retrieved university information.
* Never invent university facts.
* Never fabricate dates, fees, programs, policies, names, or contact information.
* Avoid filling missing information with general knowledge.
* Clearly state when information is not available in the knowledge base.

For example, when the retrieved context does not contain an answer, the chatbot responds:

> I couldn't find this information in the University of Baltistan knowledge base.

This approach does not mathematically eliminate hallucinations, but it significantly constrains the model to the retrieved evidence.

---

## 🧠 Embedding Model Requirement

The embedding model used by the chatbot **must be exactly the same model used when generating `index.faiss`**.

For example:

```python
sentence-transformers/all-MiniLM-L6-v2
```

This model produces:

```text
Embedding dimension: 384
```

The application verifies that the embedding dimension matches the FAISS index.

### Why this matters

If the FAISS index was created using one embedding model and the application uses another model, the query vectors will not be compatible with the indexed vector space.

Therefore:

```text
Preprocessing Model
        =
Query Embedding Model
```

Both must match.

---

## 🔐 API Key Configuration

The application requires a Groq API key.

The application looks for:

```text
GROQ_API_KEY
```

### Streamlit Community Cloud

In your Streamlit application:

```text
Settings
   ↓
Secrets
```

Add:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

Do **not** commit your API key to GitHub.

Never put this directly inside `app.py`:

```python
GROQ_API_KEY = "gsk_..."
```

---

## 💻 Local Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the API key

Windows PowerShell:

```powershell
$env:GROQ_API_KEY="your_groq_api_key"
```

Linux/macOS:

```bash
export GROQ_API_KEY="your_groq_api_key"
```

### 5. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

---

## ☁️ Deploy on Streamlit Community Cloud

### Step 1 — Push the project to GitHub

Make sure the repository contains:

```text
app.py
requirements.txt
index.faiss
metadata.json
README.md
```

### Step 2 — Create a Streamlit deployment

Select your GitHub repository and choose:

```text
app.py
```

as the main application file.

### Step 3 — Configure the secret

Add:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

### Step 4 — Deploy

Streamlit Community Cloud will install the dependencies from:

```text
requirements.txt
```

and launch:

```text
app.py
```

---

## 📦 Requirements

The project uses the following core dependencies:

```text
streamlit
sentence-transformers
faiss-cpu
groq
numpy
torch
transformers
```

See [`requirements.txt`](requirements.txt) for the deployment configuration.

---

## 💬 Example Questions

You can ask questions such as:

```text
What undergraduate programs are offered?

Tell me about the Computer Science department.

What information is available about admissions?

Where is the University of Baltistan located?

What scholarships are mentioned on the university website?

What departments are available at the university?

What information is available about the university's faculty?

What contact information is provided on the website?
```

The answer depends on whether the requested information exists in the indexed knowledge base.

---

## 📚 Knowledge Base

The knowledge base is generated from the **University of Baltistan, Skardu website dataset**.

The preprocessing pipeline extracts university website content, cleans the data, preserves document metadata, divides documents into overlapping chunks, generates embeddings, and stores the resulting vectors in FAISS.

The resulting artifacts are:

```text
index.faiss
metadata.json
```

The Streamlit application consumes these pre-built artifacts directly.

Users do not need to upload documents.

---

## ⚡ Performance

The application uses Streamlit caching for expensive resources such as:

* FAISS index
* Sentence Transformer model
* Groq client

This prevents the application from repeatedly loading the same resources during normal Streamlit reruns.

The general flow is:

```text
Application Start
       ↓
Load FAISS index
       ↓
Load metadata
       ↓
Load embedding model
       ↓
Cache resources
       ↓
Wait for user query
```

---

## 🔎 Source Transparency

For retrieved university information, the application displays:

```text
Document Title
Original Source URL
Retrieval Similarity
```

This allows users to inspect the original university webpage associated with the retrieved information.

---

## 🚫 What This Project Does Not Use

This project intentionally avoids:

```text
❌ LangChain
❌ LlamaIndex
❌ User document uploads
❌ Runtime website crawling
❌ Runtime vector database creation
❌ Hard-coded university answers
```

Instead, the application uses a pre-built vector store:

```text
FAISS
+
metadata.json
+
Sentence Transformers
+
Groq
+
Streamlit
```

---

## ⚠️ Limitations

The chatbot's accuracy depends heavily on the quality of the underlying knowledge base.

Potential limitations include:

* Information missing from the crawled dataset cannot be retrieved.
* Outdated indexed content may produce outdated answers.
* Poor chunking can reduce retrieval quality.
* Incorrect metadata can result in incorrect source attribution.
* Changing the embedding model after index creation can break retrieval compatibility.
* LLM-generated responses can still contain errors despite grounding instructions.

For this reason, the FAISS index and metadata should be regenerated whenever the underlying university dataset changes significantly.

---

## 🔮 Future Improvements

Potential improvements include:

* Hybrid keyword + semantic retrieval
* Re-ranking retrieved chunks
* Query expansion
* Better duplicate detection
* Metadata-based filtering
* Conversation-aware retrieval
* Retrieval confidence thresholds
* Automatic knowledge-base updates
* Administrative dashboard
* Analytics for frequently asked questions
* Multilingual support
* Urdu support
* Streaming LLM responses
* Improved citation formatting

---

## 👨‍💻 Author

**Basit Ali**

BS Computer Science
University of Baltistan, Skardu

Interests:

* Artificial Intelligence
* Generative AI
* RAG Systems
* AI Agents
* Full-Stack Development
* Machine Learning
* Software Engineering

---

## 📄 License

This project is intended for educational and research purposes.

University website content remains subject to the University of Baltistan's applicable ownership and usage policies.

---

## ⭐ Acknowledgment

This project combines:

* University of Baltistan website data
* Sentence Transformers
* FAISS
* Groq API
* `openai/gpt-oss-120b`
* Streamlit

to demonstrate a practical **Retrieval-Augmented Generation system for university information retrieval**.
