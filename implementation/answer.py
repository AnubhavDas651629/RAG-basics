# %%
from pathlib import Path
import json
import os

from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_core.documents import Document


def message_content_to_str(content: object) -> str:
    """
    Flatten Gradio 6 MessageDict content (str | list | dict) to a plain string for RAG / LangChain.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [message_content_to_str(item) for item in content]
        return "\n".join(p for p in parts if p).strip()
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text
        if content.get("type") == "text" and "text" in content:
            return str(content["text"])
        path = content.get("path") or content.get("url")
        if path:
            return f"[attachment: {path}]"
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def normalize_message_dict(m: dict) -> dict:
    """Return a shallow copy of a chat message with string-only content."""
    out = dict(m)
    out["content"] = message_content_to_str(out.get("content"))
    return out


def normalize_chat_history(history: list[dict]) -> list[dict]:
    return [normalize_message_dict(m) if isinstance(m, dict) else m for m in history]


# %%
from dotenv import load_dotenv
load_dotenv(override=True)
try:
    # Works in .py files
    base_path = Path(__file__).parent.parent
except NameError:
    # Works in Notebooks
    base_path = Path.cwd()
MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"

DB_NAME = str(base_path / "vector_db")

# %%
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
RETRIEVAL_K = 10

# %%
SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
If relevant, use the given context to answer any question.
If you don't know the answer, say so.
Context:
{context}
"""

# %%
vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
retriever = vectorstore.as_retriever()

# We use the standard ChatOpenAI but change the base_url
llm = ChatOpenAI(
    model_name="nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0
)

# %%
def fetch_context(question: str) -> list[Document]:
    """
    Retrieve relevant context documents for a question.
    """
    return retriever.invoke(question, k=RETRIEVAL_K)

# %%
def combined_question(question: str, history: list[dict] = []) -> str:
    """
    Combine all the user's messages into a single string.
    """
    lines: list[str] = []
    for m in history:
        if m.get("role") != "user":
            continue
        text = message_content_to_str(m.get("content"))
        if text.strip():
            lines.append(text)
    q = message_content_to_str(question)
    if not lines:
        return q
    return "\n".join(lines) + "\n" + q

# %%
def answer_question(question: str, history: list[dict] = []) -> tuple[str, list[Document]]:
    """
    Answer the given question with RAG; return the answer and the context documents.
    """
    question = message_content_to_str(question)
    history = normalize_chat_history(history)
    combined = combined_question(question, history)
    docs = fetch_context(combined)
    context = "\n\n".join(doc.page_content for doc in docs)
    system_prompt = SYSTEM_PROMPT.format(context=context)
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(content=question))
    response = llm.invoke(messages)
    return response.content, docs

# %%



