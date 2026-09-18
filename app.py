import os
import json
import html
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

APP_TITLE = "UOBS AI Assistant"
APP_SUBTITLE = "University of Baltistan, Skardu Chatbot"

# ------------------------------------------------------------
# IMPORTANT:
# These must match the preprocessing pipeline exactly.
#
# If metadata.json contains the embedding model, the app will
# attempt to detect it automatically.
# ------------------------------------------------------------

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_DIMENSION = 384

GROQ_MODEL = "openai/gpt-oss-120b"

TOP_K = 6
MAX_CONTEXT_CHARS = 24000
MAX_SOURCE_TEXT_CHARS = 3000
BASE_DIR = Path(__file__).resolve().parent

INDEX_PATH = BASE_DIR / "index.faiss"
METADATA_PATH = BASE_DIR / "metadata.json"


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
        .stApp {
            background: #f8fafc;
        }

        .main-header {
            padding: 1.2rem 1.4rem;
            border-radius: 18px;
            background: linear-gradient(
                135deg,
                #0f172a 0%,
                #1e293b 55%,
                #064e3b 100%
            );
            color: white;
            margin-bottom: 1.2rem;
            box-shadow: 0 8px 30px rgba(15, 23, 42, 0.12);
        }

        .main-header h1 {
            margin: 0;
            font-size: 2rem;
            font-weight: 750;
        }

        .main-header p {
            margin: 0.45rem 0 0;
            color: #d1fae5;
            font-size: 1rem;
        }

        .status-card {
            padding: 0.9rem 1rem;
            border-radius: 12px;
            background: white;
            border: 1px solid #e2e8f0;
            margin-bottom: 0.7rem;
        }

        .source-card {
            padding: 0.85rem 1rem;
            margin: 0.5rem 0;
            border-radius: 12px;
            background: white;
            border: 1px solid #e2e8f0;
        }

        .source-title {
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 0.25rem;
        }

        .source-url {
            font-size: 0.82rem;
            color: #047857;
            word-break: break-word;
        }

        .source-score {
            font-size: 0.75rem;
            color: #64748b;
            margin-top: 0.35rem;
        }

        .footer {
            text-align: center;
            color: #64748b;
            font-size: 0.8rem;
            padding: 2rem 0 1rem;
        }

        div[data-testid="stChatMessage"] {
            border-radius: 14px;
        }

        .info-box {
            padding: 0.85rem 1rem;
            border-radius: 12px;
            background: #ecfdf5;
            border: 1px solid #a7f3d0;
            color: #065f46;
            margin-bottom: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_secret_api_key() -> Optional[str]:
    """
    Get GROQ_API_KEY from Streamlit Secrets first,
    then environment variables.
    """
    try:
        key = st.secrets.get("GROQ_API_KEY")
        if key:
            return str(key)
    except Exception:
        pass

    key = os.getenv("GROQ_API_KEY")

    if key:
        return key

    return None


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """
    L2-normalize embeddings.

    This must match the normalization used during preprocessing.
    """
    vector = np.asarray(vector, dtype=np.float32)

    if vector.ndim == 1:
        vector = vector.reshape(1, -1)

    faiss.normalize_L2(vector)

    return vector


def load_metadata_file(path: Path) -> Any:
    """
    Load metadata.json safely.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {path}"
        )

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def extract_metadata_records(metadata: Any) -> List[Dict[str, Any]]:
    """
    Convert common metadata.json structures into a list of records.

    Supported structures include:

    1. [
         {...},
         {...}
       ]

    2. {
         "records": [...]
       }

    3. {
         "chunks": [...]
       }

    4. {
         "metadata": [...]
       }
    """

    if isinstance(metadata, list):
        return [
            item for item in metadata
            if isinstance(item, dict)
        ]

    if isinstance(metadata, dict):

        for key in ("records", "chunks", "metadata", "documents"):
            value = metadata.get(key)

            if isinstance(value, list):
                return [
                    item for item in value
                    if isinstance(item, dict)
                ]

    raise ValueError(
        "Unsupported metadata.json format. "
        "Expected a list of metadata records or an object "
        "containing records/chunks/metadata/documents."
    )


def detect_embedding_config(
    metadata: Any,
) -> tuple[str, Optional[int]]:
    """
    Attempt to detect the embedding model and dimension from metadata.

    If the preprocessing pipeline stored these values, they will
    automatically be used.

    Otherwise fallback constants are used.
    """

    model = DEFAULT_EMBEDDING_MODEL
    dimension = DEFAULT_EMBEDDING_DIMENSION

    candidates = []

    if isinstance(metadata, dict):
        candidates.append(metadata)

        for key in ("config", "embedding", "embedding_config"):
            value = metadata.get(key)

            if isinstance(value, dict):
                candidates.append(value)

    records = extract_metadata_records(metadata)

    if records:
        candidates.extend(records[:5])

    model_keys = (
        "embedding_model",
        "embedding_model_name",
        "model_name",
        "model",
    )

    dimension_keys = (
        "embedding_dimension",
        "dimension",
        "embedding_dim",
        "vector_dimension",
    )

    for item in candidates:

        if not isinstance(item, dict):
            continue

        for key in model_keys:
            value = item.get(key)

            if isinstance(value, str) and value.strip():
                model = value.strip()
                break

        for key in dimension_keys:
            value = item.get(key)

            try:
                if value is not None:
                    dimension = int(value)
                    break
            except (TypeError, ValueError):
                pass

    return model, dimension


# ============================================================
# LOAD FAISS + METADATA
# ============================================================

@st.cache_resource(show_spinner="Loading university knowledge base...")
def load_knowledge_base():
    """
    Load FAISS index, metadata and Sentence Transformer model.

    Cached using Streamlit's resource cache so the model/index
    are not reloaded for every user interaction.
    """

    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index not found at:\n{INDEX_PATH}"
        )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file not found at:\n{METADATA_PATH}"
        )

    # Load FAISS index
    index = faiss.read_index(str(INDEX_PATH))

    # Load metadata
    metadata_raw = load_metadata_file(METADATA_PATH)

    metadata_records = extract_metadata_records(metadata_raw)

    if not metadata_records:
        raise ValueError(
            "metadata.json does not contain any metadata records."
        )

    # Detect embedding configuration
    embedding_model_name, metadata_dimension = detect_embedding_config(
        metadata_raw
    )

    # Make sure FAISS dimension agrees with expected embedding dimension
    index_dimension = index.d

    if metadata_dimension is not None:
        if index_dimension != metadata_dimension:
            raise ValueError(
                "Embedding dimension mismatch.\n\n"
                f"FAISS dimension: {index_dimension}\n"
                f"Metadata dimension: {metadata_dimension}\n\n"
                "The preprocessing and query embedding dimensions "
                "must be identical."
            )

    # Load embedding model
    embedding_model = SentenceTransformer(
        embedding_model_name
    )

    actual_model_dimension = embedding_model.get_sentence_embedding_dimension()

    if actual_model_dimension != index_dimension:
        raise ValueError(
            "Embedding model dimension does not match FAISS index.\n\n"
            f"Model: {embedding_model_name}\n"
            f"Model dimension: {actual_model_dimension}\n"
            f"FAISS dimension: {index_dimension}\n\n"
            "Use the exact same Sentence Transformers model "
            "used during preprocessing."
        )

    return (
        index,
        metadata_records,
        embedding_model,
        embedding_model_name,
        index_dimension,
    )


# ============================================================
# RETRIEVAL
# ============================================================

def get_record_text(record: Dict[str, Any]) -> str:
    """
    Extract chunk/document text from metadata.

    Supports several likely field names.
    """

    possible_keys = (
        "chunk_text",
        "text",
        "content",
        "chunk",
        "document_content",
        "page_content",
    )

    for key in possible_keys:
        value = record.get(key)

        if isinstance(value, str) and value.strip():
            return value.strip()

    return ""


def get_record_title(record: Dict[str, Any]) -> str:
    """
    Extract document title.
    """

    possible_keys = (
        "document_title",
        "title",
        "DOCUMENT TITLE",
        "TITLE",
        "name",
    )

    for key in possible_keys:
        value = record.get(key)

        if value is not None and str(value).strip():
            return str(value).strip()

    return "University of Baltistan document"


def get_record_url(record: Dict[str, Any]) -> str:
    """
    Extract original source URL.
    """

    possible_keys = (
        "source_url",
        "url",
        "SOURCE URL",
        "source",
        "original_url",
    )

    for key in possible_keys:
        value = record.get(key)

        if value is not None and str(value).strip():
            return str(value).strip()

    return ""


def retrieve_documents(
    query: str,
    index,
    metadata_records: List[Dict[str, Any]],
    embedding_model: SentenceTransformer,
    top_k: int = TOP_K,
) -> List[Dict[str, Any]]:
    """
    Embed the user query, normalize it, and search FAISS.

    The preprocessing pipeline must have used the same:
    - Sentence Transformer model
    - embedding dimension
    - normalization strategy
    - FAISS similarity configuration
    """

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    )

    query_embedding = normalize_vector(query_embedding)

    distances, indices = index.search(
        query_embedding,
        top_k,
    )

    results = []

    for rank, (idx, score) in enumerate(
        zip(indices[0], distances[0]),
        start=1,
    ):

        if idx < 0:
            continue

        if idx >= len(metadata_records):
            continue

        record = metadata_records[int(idx)]

        text = get_record_text(record)

        if not text:
            continue

        results.append(
            {
                "rank": rank,
                "index": int(idx),
                "score": float(score),
                "title": get_record_title(record),
                "url": get_record_url(record),
                "text": text,
                "metadata": record,
            }
        )

    return results


# ============================================================
# CONTEXT BUILDING
# ============================================================

def build_context(
    retrieved_documents: List[Dict[str, Any]],
    max_chars: int = MAX_CONTEXT_CHARS,
) -> str:
    """
    Build a clean context block for the LLM.
    """

    context_parts = []
    current_length = 0

    for item in retrieved_documents:

        text = item["text"]

        remaining = max_chars - current_length

        if remaining <= 0:
            break

        if len(text) > remaining:
            text = text[:remaining]

        block = (
            f"[SOURCE {item['rank']}]\n"
            f"Document Title: {item['title']}\n"
            f"Source URL: {item['url']}\n"
            f"Content:\n{text}\n"
        )

        context_parts.append(block)

        current_length += len(block)

    return "\n\n".join(context_parts)


# ============================================================
# GROQ
# ============================================================

@st.cache_resource
def get_groq_client(api_key: str):
    """
    Create and cache Groq client.
    """
    return Groq(api_key=api_key)


def generate_answer(
    question: str,
    context: str,
    client: Groq,
) -> str:
    """
    Generate a grounded answer using only retrieved context.
    """

    system_prompt = """
You are the official knowledge assistant for the University of
Baltistan, Skardu.

Your job is to answer questions using ONLY the university
information provided in the RETRIEVED CONTEXT.

STRICT RULES:

1. Use only information contained in the retrieved context.
2. Never invent facts, dates, names, programs, fees, policies,
   contact details, departments, eligibility requirements, or
   other university information.
3. If the retrieved context does not contain enough information
   to answer the question, say clearly:

   "I couldn't find this information in the University of Baltistan
   knowledge base."

4. Do not use your general knowledge to fill missing university
   information.
5. Do not assume that information is true merely because it sounds
   plausible.
6. When the answer is supported by the retrieved context, give a
   concise and direct answer.
7. If multiple retrieved sources provide relevant information,
   combine them accurately without adding unsupported details.
8. Do not mention FAISS, embeddings, retrieval, vector databases,
   prompts, or internal implementation unless the user explicitly
   asks about the chatbot's technical implementation.
9. For greetings such as "hello", "hi", "assalam o alaikum",
   respond naturally and briefly. Do not fabricate university facts.
10. For unrelated general questions, answer naturally only when the
    question does not require university-specific information.
    If answering would require university information not present
    in context, use the knowledge-base-not-found response.
11. Never claim that a webpage or document contains information
    unless that information is actually present in the supplied
    context.
12. If the user asks a question with several parts, answer only the
    parts supported by the context and explicitly identify any part
    that was not found.

RETRIEVED CONTEXT:
"""

    user_prompt = f"""
{context}

USER QUESTION:
{question}
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        ],
        temperature=0.1,
        max_tokens=700,
    )

    answer = response.choices[0].message.content

    if not answer:
        return (
            "I couldn't generate an answer from the "
            "University of Baltistan knowledge base."
        )

    return answer.strip()


# ============================================================
# SOURCE DISPLAY
# ============================================================

def display_sources(
    retrieved_documents: List[Dict[str, Any]]
):
    """
    Display document title, URL and retrieval score.
    """

    if not retrieved_documents:
        return

    st.markdown("### 📚 Sources")

    displayed_urls = set()

    for item in retrieved_documents:

        title = html.escape(item["title"])
        url = item["url"]
        safe_url = html.escape(url, quote=True)

        # Avoid displaying duplicate URLs repeatedly
        source_key = url.strip().lower()

        if source_key in displayed_urls and source_key:
            continue

        displayed_urls.add(source_key)

        score = item["score"]

        if url:
            url_html = (
                f'<a href="{safe_url}" target="_blank">'
                f"{safe_url}"
                f"</a>"
            )
        else:
            url_html = "Source URL not available"

        st.markdown(
            f"""
            <div class="source-card">
                <div class="source-title">📄 {title}</div>
                <div class="source-url">{url_html}</div>
                <div class="source-score">
                    Retrieval similarity: {score:.4f}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-header">
        <h1>🎓 UOBS AI Assistant</h1>
        <p>
            University of Baltistan, Skardu — AI-powered
            knowledge assistant
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🎓 UOBS AI Assistant")

    st.markdown(
        """
        Ask questions about information available in the
        University of Baltistan knowledge base.
        """
    )

    st.divider()

    st.markdown("### 💡 Example questions")

    example_questions = [
        "What undergraduate programs are offered?",
        "Tell me about the Computer Science department.",
        "What information is available about admissions?",
        "Where is the University of Baltistan located?",
        "What scholarships are mentioned on the university website?",
    ]

    for question in example_questions:
        if st.button(
            question,
            use_container_width=True,
        ):
            st.session_state.pending_question = question

    st.divider()

    st.markdown("### ⚙️ System")

    st.caption(
        "Knowledge source: University of Baltistan website dataset"
    )

    st.caption(
        f"LLM: {GROQ_MODEL}"
    )

    st.caption(
        "Retrieval: Sentence Transformers + FAISS"
    )

    if st.button(
        "🗑️ Clear conversation",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.rerun()


# ============================================================
# INITIALIZE KNOWLEDGE BASE
# ============================================================

try:

    (
        faiss_index,
        metadata_records,
        embedding_model,
        embedding_model_name,
        embedding_dimension,
    ) = load_knowledge_base()

except Exception as error:

    st.error("❌ Failed to load the university knowledge base.")

    st.code(
        str(error),
        language="text",
    )

    st.info(
        "Check that vector_store/index.faiss and "
        "vector_store/metadata.json are present and that the "
        "Sentence Transformers model matches the preprocessing model."
    )

    st.stop()


# ============================================================
# API KEY
# ============================================================

groq_api_key = get_secret_api_key()

if not groq_api_key:

    st.warning(
        "⚠️ GROQ_API_KEY is not configured."
    )

    st.info(
        "Add GROQ_API_KEY to Streamlit Community Cloud "
        "Secrets or configure it as an environment variable."
    )

    st.stop()


try:
    groq_client = get_groq_client(groq_api_key)

except Exception as error:

    st.error("❌ Failed to initialize Groq.")

    st.code(
        str(error),
        language="text",
    )

    st.stop()


# ============================================================
# KNOWLEDGE BASE STATUS
# ============================================================

st.markdown(
    f"""
    <div class="info-box">
        <strong>Knowledge base ready.</strong><br>
        {len(metadata_records):,} indexed chunks |
        FAISS dimension: {embedding_dimension} |
        Embedding model: {html.escape(embedding_model_name)}
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DISPLAY PREVIOUS CHAT
# ============================================================

for message in st.session_state.messages:

    role = message["role"]
    content = message["content"]

    with st.chat_message(role):
        st.markdown(content)

        if role == "assistant":

            sources = message.get("sources", [])

            if sources:
                display_sources(sources)


# ============================================================
# HANDLE SIDEBAR EXAMPLE QUESTION
# ============================================================

pending_question = st.session_state.pop(
    "pending_question",
    None,
)


# ============================================================
# CHAT INPUT
# ============================================================

user_question = st.chat_input(
    "Ask about the University of Baltistan..."
)

if pending_question:
    user_question = pending_question


# ============================================================
# PROCESS QUESTION
# ============================================================

if user_question:

    user_question = user_question.strip()

    if not user_question:
        st.stop()

    # Display user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question,
        }
    )

    with st.chat_message("user"):
        st.markdown(user_question)

    # Generate assistant response
    with st.chat_message("assistant"):

        try:

            with st.spinner(
                "Searching the university knowledge base..."
            ):

                retrieved_documents = retrieve_documents(
                    query=user_question,
                    index=faiss_index,
                    metadata_records=metadata_records,
                    embedding_model=embedding_model,
                    top_k=TOP_K,
                )

            if not retrieved_documents:

                answer = (
                    "I couldn't find this information in the "
                    "University of Baltistan knowledge base."
                )

                sources = []

            else:

                context = build_context(
                    retrieved_documents
                )

                if not context.strip():

                    answer = (
                        "I couldn't find this information in the "
                        "University of Baltistan knowledge base."
                    )

                    sources = []

                else:

                    with st.spinner(
                        "Generating an answer..."
                    ):

                        answer = generate_answer(
                            question=user_question,
                            context=context,
                            client=groq_client,
                        )

                    sources = retrieved_documents

            st.markdown(answer)

            if sources:
                display_sources(sources)

            # Store assistant response
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                }
            )

        except Exception as error:

            error_message = (
                "Sorry, I encountered a problem while processing "
                "your question. Please try again."
            )

            st.error(error_message)

            # Detailed error for deployment debugging
            with st.expander("Technical error details"):

                st.code(
                    str(error),
                    language="text",
                )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_message,
                    "sources": [],
                }
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        University of Baltistan, Skardu • RAG Knowledge Assistant
        <br>
        Answers are generated from the indexed university knowledge base.
    </div>
    """,
    unsafe_allow_html=True,
)
