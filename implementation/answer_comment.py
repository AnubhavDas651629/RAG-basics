

from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain_core.documents import Document

from dotenv import load_dotenv
load_dotenv(override=True)
try:
    # Works in .py files
    base_path = Path(__file__).parent.parent
except NameError:
    # Works in Notebooks
    base_path = Path.cwd()
MODEL = "deepseek/deepseek-r1:free"

DB_NAME = str(base_path / "vector_db")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
RETRIEVAL_K = 10
# retriever is saying that we are taking 10 documents of 500 characters each (chunk size = 500)

SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing the company Insurellm.
You are chatting with a user about Insurellm.
If relevant, use the given context to answer any question.
If you don't know the answer, say so.
Context:
{context}
"""

# This is where answer.py “connects” to the work done by ingest.py:
# It opens the persisted Chroma DB at DB_NAME that ingest.py created.
# Wraps it in a retriever for semantic search.
# Configures the chat model used for answering.

vectorstore = Chroma(persist_directory=DB_NAME, embedding_function=embeddings)
retriever = vectorstore.as_retriever()
from langchain_openai import ChatOpenAI
import os

# We use the standard ChatOpenAI but change the base_url
llm = ChatOpenAI(
    model_name="deepseek/deepseek-r1:free",
    openai_api_key=os.getenv("OPENROUTER_API_KEY"),
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0
)

# Retriveing Context
# In your code, retriever.invoke(question, k=RETRIEVAL_K) is performing a search. It takes the user's question, goes into the Chroma DB (where your data is stored as vectors), and pulls out the $k$ most relevant "chunks" of information.
def fetch_context(question: str) -> list[Document]:
    """
    Retrieve relevant context documents for a question.
    """
    return retriever.invoke(question, k=RETRIEVAL_K)

# Combining chat history: akes previous user messages from the chat history (list of {role, content} dicts) and concatenates them to form a richer retrieval query.
def combined_question(question: str, history: list[dict] = []) -> str:
    """
    Combine all the user's messages into a single string.
    """
    prior = "\n".join(m["content"] for m in history if m["role"] == "user")
    return prior + "\n" + question

# Main RAG function

def answer_question(question: str, history: list[dict] = []) -> tuple[str, list[Document]]:
    """
    Answer the given question with RAG; return the answer and the context documents.
    """
    combined = combined_question(question, history)
    docs = fetch_context(combined)
    context = "\n\n".join(doc.page_content for doc in docs)
    system_prompt = SYSTEM_PROMPT.format(context=context)
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(content=question))
    response = llm.invoke(messages)
    return response.content, docs

# Flow of answer_question:

# Combine history + new question for retrieval.
# Retrieve top‑K documents from Chroma (fetch_context).
# Build the system prompt by injecting the retrieved text into SYSTEM_PROMPT.
# Construct the chat messages:
# - System message with the context + instructions.
# - Prior chat history (converted to SystemMessage/HumanMessage/AIMessage objects).
# - The user’s current question.
# Call the LLM (llm.invoke) and return:
# - response.content (answer string).
# - docs (retrieved context docs), for display or evaluation.