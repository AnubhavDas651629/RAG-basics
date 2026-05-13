# %%
import os
import glob
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings


# %%
import os
from pathlib import Path
from dotenv import load_dotenv

# 1. Load environment variables
load_dotenv(override=True)

# 2. Define the Model
MODEL = "deepseek/deepseek-r1:free"

# 3. Dynamic Root Detection
try:
    # This works when running as a .py script
    base_path = Path(__file__).parent.parent
except NameError:
    # This works in Jupyter Notebooks/Cursor Cells
    # It assumes your notebook is in 'implementation/' and goes up one level
    base_path = Path.cwd().parent if Path.cwd().name == "implementation" else Path.cwd()

# 4. Define final paths relative to the project root
DB_NAME = str(base_path / "vector_db")
KNOWLEDGE_BASE = str(base_path / "knowledge-base")

# --- Verification ---
print(f"Project Root: {base_path}")
print(f"Vector DB Path: {DB_NAME}")
if os.path.exists(KNOWLEDGE_BASE):
    print(f"✅ Success: Knowledge base found at {KNOWLEDGE_BASE}")
else:
    print(f"❌ Error: Could not find {KNOWLEDGE_BASE}")

# %%
load_dotenv(override=True)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# %%
import os
import glob
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader

def fetch_documents():
    # Use Path to ensure cross-platform compatibility (Windows vs Mac)
    path_pattern = str(Path(KNOWLEDGE_BASE) / "*")
    folders = glob.glob(path_pattern)
    
    documents = []
    
    for folder in folders:
        # CRITICAL: Only proceed if it's actually a folder
        if os.path.isdir(folder):
            doc_type = os.path.basename(folder)
            print(f"Scanning folder: {doc_type}...") # Debugging line
            
            loader = DirectoryLoader(
                folder, 
                glob="**/*.md", 
                loader_cls=TextLoader, 
                loader_kwargs={"encoding": "utf-8"},
                recursive=True # Ensure it digs deep
            )
            
            folder_docs = loader.load()
            
            for doc in folder_docs:
                doc.metadata["doc_type"] = doc_type
                # Also ensure the 'source' metadata is a clean string
                doc.metadata["source"] = os.path.basename(doc.metadata.get("source", ""))
                documents.append(doc)
                
    print(f"Total documents loaded: {len(documents)}")
    return documents

# %%
def create_chunks(documents):
    # text_splitter = MarkdownTextSplitter()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    return chunks

# %%
def create_embeddings(chunks):
    if os.path.exists(DB_NAME):
        Chroma(persist_directory=DB_NAME, embedding_function=embeddings).delete_collection()

    vectorstore = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=DB_NAME
    )

    collection = vectorstore._collection
    count = collection.count()

    sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
    dimensions = len(sample_embedding)
    print(f"There are {count:,} vectors with {dimensions:,} dimensions in the vector store")
    return vectorstore

# %%
if __name__ == "__main__":
    documents = fetch_documents()
    chunks = create_chunks(documents)
    create_embeddings(chunks)
    print("Ingestion complete")


# %%



