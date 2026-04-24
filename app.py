import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
import ollama

CHROMA_DIR = "chroma_db"

st.set_page_config(page_title="PDF RAG", page_icon="📚", layout="wide")

@st.cache_resource
def load_resources():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        collection = client.get_collection("rag_docs")
        return embedder, collection
    except Exception:
        return embedder, None

def retrieve(query, embedder, collection, k=10):
    query_vec = embedder.encode([query]).tolist()
    results = collection.query(query_embeddings=query_vec, n_results=k)
    chunks = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]
    return chunks, sources

def ask_mistral(query, chunks):
    context = "\n\n---\n\n".join(chunks)
    prompt = f"""Tu es un assistant expert en appels d'offres tunisiens. 
Réponds en français en te basant UNIQUEMENT sur le contexte fourni ci-dessous.
Sois précis et cite les valeurs exactes (montants, dates, délais) trouvées dans le texte.
Si une information n'est pas dans le contexte, écris "Non mentionné dans le document."
Ne devine jamais — utilise uniquement ce qui est écrit.

Contexte:
{context}

Question: {query}
Réponse détaillée:"""
    response = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]
# --- UI ---
st.title("📚 Ask Your PDFs")
st.caption("Powered by Mistral + ChromaDB — running 100% locally")

embedder, collection = load_resources()

# Sidebar stats
with st.sidebar:
    st.header("📊 Index Info")
    if collection:
        count = collection.count()
        st.success(f"✅ {count} chunks indexed")
        st.info("Drop PDFs in /pdfs and run ingest.py to update")
    else:
        st.error("No index found. Run ingest.py first.")
        st.code("python ingest.py", language="bash")

# Chat
if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            with st.expander("📎 Sources"):
                for s in msg["sources"]:
                    st.write(f"- {s}")

query = st.chat_input("Ask a question about your documents...")

if query:
    if not collection:
        st.error("Please run `python ingest.py` first to index your PDFs.")
    else:
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating answer..."):
                chunks, sources = retrieve(query, embedder, collection)
                answer = ask_mistral(query, chunks)
            st.write(answer)
            unique_sources = list(set(sources))
            with st.expander("📎 Sources"):
                for s in unique_sources:
                    st.write(f"- {s}")

        st.session_state.history.append({"role": "user", "content": query})
        st.session_state.history.append({
            "role": "assistant",
            "content": answer,
            "sources": unique_sources
        })
