from dotenv import load_dotenv
load_dotenv()

import os
import time

# -----------------------------
# Embeddings
# -----------------------------
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

embeddings = OllamaEmbeddings(
    base_url="http://127.0.0.1:11434",
    model="nomic-embed-text"
)

# -----------------------------
# LLM
# -----------------------------
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0
)

# -----------------------------
# Prompt
# -----------------------------
from langchain_core.prompts import PromptTemplate

prompt_template = """
You are an intelligent document question-answering assistant.

Instructions:
- Answer using ONLY the provided context.
- Do not use any external knowledge.
- Do not invent, assume, or guess information.
- The context may contain text, lists, tables, structured data, or multiple sections.
- Extract relevant information from all available context.
- If information is spread across multiple parts of the context, combine it into a clear answer.
- If only partial information is available, provide the available information.
- When answering questions related to tables or structured data, use the relevant values from the context.
- Keep answers clear, concise, and natural.
- Do not explain your reasoning process.
- Return "No information found in the document." only when the context does not contain relevant information to answer the question.

Context:
{context}

Question:
{question}

Answer:
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# -----------------------------
# Batch Embedding
# -----------------------------
def embed_in_batches(docs, embeddings, batch_size=50):
    all_texts = [doc.page_content for doc in docs]
    all_metadatas = [doc.metadata for doc in docs]

    vector_store = None
    total = len(all_texts)

    for i in range(0, total, batch_size):
        batch_texts = all_texts[i:i + batch_size]
        batch_meta = all_metadatas[i:i + batch_size]

        print(f"Embedding batch {i // batch_size + 1}/{(total + batch_size - 1) // batch_size} ...")

        if vector_store is None:
            vector_store = FAISS.from_texts(
                batch_texts,
                embeddings,
                metadatas=batch_meta
            )
        else:
            vector_store.add_texts(
                batch_texts,
                metadatas=batch_meta
            )

        time.sleep(0.3)  # give system breathing room

    return vector_store


# -----------------------------
# QA Chain
# -----------------------------
from langchain_classic.chains import RetrievalQA


def create_qa_chain_from_docs(docs, cache_name=None):

    # -------------------------
    # FAISS Cache
    # -------------------------
    index_path = f"faiss_cache_{cache_name}" if cache_name else None

    if index_path and os.path.exists(index_path):
        print(f"Loading FAISS index from cache: {index_path}")
        vector_store = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        print(f"Building FAISS index from {len(docs)} chunks...")
        vector_store = embed_in_batches(docs, embeddings, batch_size=50)

        if index_path:
            vector_store.save_local(index_path)
            print(f"FAISS index saved to cache: {index_path}")

    # -------------------------
    # BM25 Retriever
    # -------------------------
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 6

    # -------------------------
    # Vector Retriever
    # -------------------------
    vector_retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "fetch_k": 20,
            "lambda_mult": 0.7
        }
    )

    # -------------------------
    # Ensemble Retriever
    # -------------------------
    ensemble_retriever = EnsembleRetriever(
        retrievers=[
            bm25_retriever,
            vector_retriever
        ],
        weights=[0.5, 0.5]
    )

    # -------------------------
    # QA Chain
    # -------------------------
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=ensemble_retriever,
        chain_type="stuff",
        chain_type_kwargs={
            "prompt": PROMPT
        }
    )

    return qa_chain