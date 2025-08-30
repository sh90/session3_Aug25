# rag_helpdesk_bot.py
import os
import pandas as pd
import numpy as np
import faiss
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

LLM_MODEL = "gpt-4o-mini"  # or "gpt-4"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Load dataset
df = pd.read_csv("dataset-tickets-multi-lang3-4k.csv").head(n=800)

# Combine issue and resolution as context
# Combine issue and resolution as context
df["context"] = df["body"].fillna("") + "\nResolution: " + df["answer"].fillna("")

# Ensure all are strings
df["context"] = df["context"].astype(str)


# Load embedding model
EMBED_MODEL = "text-embedding-3-small"
print(f"[INFO] Using OpenAI Embeddings: {EMBED_MODEL}")
embedding_model = OpenAIEmbeddings(model=EMBED_MODEL, openai_api_key=OPENAI_API_KEY)

# Create embeddings for documents
print("Generating embeddings...")
embeddings = embedding_model.embed_documents(df["context"].tolist())

# Create FAISS index
embedding_dim = len(embeddings[0])   # embeddings is a list of lists, not np.array
index = faiss.IndexFlatL2(embedding_dim)
index.add(np.array(embeddings, dtype="float32"))

# Define RAG-style query function
def rag_query(user_query, top_k=3):
    query_embedding = embedding_model.embed_query(user_query)  # 👈 FIXED
    query_embedding = np.array([query_embedding], dtype="float32")

    distances, indices = index.search(query_embedding, top_k)

    retrieved_contexts = df.iloc[indices[0]]
    print("\nTop Retrieved Tickets:")
    for _, row in retrieved_contexts.iterrows():
        print(f"\nIssue: {row['body']}\nResolution: {row['answer']}")

    print("\n---\nSuggested Resolution:")
    print(retrieved_contexts.iloc[0]['answer'])


# --- CLI Loop ---
print("\nRAG Helpdesk Bot is ready. Type your IT issue below. Type 'exit' to quit.\n")
while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        break
    rag_query(query)
    print("\n==============================\n")
