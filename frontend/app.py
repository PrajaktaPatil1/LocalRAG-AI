import streamlit as st
import requests
import json
import os
from typing import Generator, Dict, Any

# Page setup
st.set_page_config(
    page_title="Hybrid RAG + Ollama Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Backend URL configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Custom CSS for modern design
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stChatMessage {
        border-radius: 12px;
        padding: 12px 18px;
        margin-bottom: 12px;
    }
    .badge-rag {
        background: linear-gradient(135deg, #1e3a8a, #3b82f6);
        color: white;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 8px;
    }
    .badge-direct {
        background: linear-gradient(135deg, #065f46, #10b981);
        color: white;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 8px;
    }
    .citation-card {
        background: #1f2937;
        border-left: 4px solid #3b82f6;
        padding: 10px 14px;
        margin: 6px 0;
        border-radius: 4px;
        font-size: 0.88rem;
    }
    .stats-metric {
        font-size: 1.1rem;
        font-weight: bold;
        color: #60a5fa;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "llama3.2"

# ----------------- Helper Functions ----------------- #

def check_backend_health():
    try:
        res = requests.get(f"{BACKEND_URL}/api/health", timeout=3)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return None

def fetch_available_models():
    try:
        res = requests.get(f"{BACKEND_URL}/api/models", timeout=3)
        if res.status_code == 200:
            return res.json().get("models", [])
    except Exception:
        pass
    return []

def fetch_indexed_documents():
    try:
        res = requests.get(f"{BACKEND_URL}/api/documents", timeout=3)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return {"documents": [], "total_chunks": 0}

def upload_files(files):
    upload_list = [("files", (f.name, f.getvalue(), f.type)) for f in files]
    try:
        res = requests.post(f"{BACKEND_URL}/api/documents/upload", files=upload_list, timeout=60)
        return res.json()
    except Exception as e:
        return {"error": str(e)}

def delete_doc(doc_id):
    try:
        res = requests.delete(f"{BACKEND_URL}/api/documents/{doc_id}", timeout=10)
        return res.json()
    except Exception as e:
        return {"error": str(e)}

def stream_query_response(payload: Dict[str, Any]):
    """Generator that streams response tokens and captures metadata from SSE endpoint."""
    try:
        with requests.post(
            f"{BACKEND_URL}/api/query/stream",
            json=payload,
            stream=True,
            timeout=(10, 300)
        ) as response:
            if response.status_code != 200:
                yield {"type": "token", "content": f"⚠️ Server Error: {response.text}"}
                return

            for line in response.iter_lines():
                if not line:
                    continue
                line_str = line.decode("utf-8")
                if line_str.startswith("data: "):
                    raw_data = line_str[6:]
                    try:
                        event_data = json.loads(raw_data)
                        yield event_data
                    except json.JSONDecodeError:
                        continue
    except requests.exceptions.Timeout:
        yield {"type": "token", "content": "\n\n⚠️ Request timed out. The local LLM or reranker is still generating. Please try again."}
    except Exception as e:
        yield {"type": "token", "content": f"\n\n⚠️ Connection Error: {str(e)}"}


# ----------------- UI Layout ----------------- #

# Sidebar
with st.sidebar:
    st.title("⚙️ RAG Engine Settings")
    
    # 1. Health & Model Selection
    health_info = check_backend_health()
    if health_info:
        st.success("🟢 Backend Connected")
    else:
        st.error("🔴 Backend Offline (Run FastAPI on port 8000)")
        
    models_list = fetch_available_models()
    if not models_list:
        models_list = ["llama3.2", "llama3", "mistral", "phi3", "gemma2"]
    
    selected_model = st.selectbox(
        "🧠 LLM Model (Ollama)",
        options=models_list,
        index=0 if st.session_state.selected_model not in models_list else models_list.index(st.session_state.selected_model)
    )
    st.session_state.selected_model = selected_model
    
    st.divider()
    
    # 2. Document Upload Section
    st.subheader("📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, TXT, or MD",
        type=["pdf", "docx", "txt", "md", "csv"],
        accept_multiple_files=True
    )
    
    if uploaded_files and st.button("🚀 Process & Index Documents", use_container_width=True):
        with st.spinner("Parsing, chunking, and embedding into Chroma + BM25..."):
            result = upload_files(uploaded_files)
            if "uploaded" in result:
                st.success(f"Indexed successfully! Total chunks: {result.get('total_chunks_now', 0)}")
                st.rerun()
            else:
                st.error(f"Upload failed: {result}")
                
    st.divider()
    
    # 3. Knowledge Base Status
    st.subheader("📚 Knowledge Base")
    doc_info = fetch_indexed_documents()
    docs = doc_info.get("documents", [])
    total_chunks = doc_info.get("total_chunks", 0)
    
    col1, col2 = st.columns(2)
    col1.metric("Documents", len(docs))
    col2.metric("Total Chunks", total_chunks)
    
    if docs:
        with st.expander("View Indexed Documents", expanded=False):
            for doc in docs:
                c1, c2 = st.columns([3, 1])
                c1.write(f"📄 **{doc['source']}** ({doc['chunk_count']} chunks)")
                if c2.button("🗑️", key=f"del_{doc['doc_id']}", help="Delete document"):
                    delete_doc(doc["doc_id"])
                    st.rerun()
    else:
        st.info("No documents uploaded yet. Queries will use direct LLM knowledge.")
        
    st.divider()
    
    # 4. Retrieval & Router Tuning
    st.subheader("🎯 Pipeline Parameters")
    
    routing_mode = st.radio(
        "Smart Routing Mode",
        options=["auto", "rag_only", "direct_only"],
        format_func=lambda x: {
            "auto": "🧠 Smart Auto (Adaptive RAG + Direct)",
            "rag_only": "📄 Strict RAG (Document context only)",
            "direct_only": "🤖 Direct LLM (General Knowledge only)"
        }[x]
    )
    
    relevance_threshold = st.slider(
        "Relevance Gate Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.30,
        step=0.05,
        help="Queries scoring below this threshold route smoothly to direct LLM knowledge."
    )
    
    top_k_retrieval = st.slider("Hybrid Retrieval Top-K", min_value=2, max_value=20, value=8)
    top_k_rerank = st.slider("Reranker Top-K Chunks", min_value=1, max_value=8, value=4)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    
    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# Main Chat Interface
st.header("🧠 Hybrid RAG with Adaptive Routing & Streaming")
st.caption("Combines Dense Vector + BM25 Sparse Search, Cross-Encoder Reranker, Relevance Gating, and Ollama.")

# Display previous conversation
for msg in st.session_state.messages:
    role = msg["role"]
    with st.chat_message(role):
        if role == "assistant":
            mode = msg.get("mode", "direct_llm")
            score = msg.get("relevance_score", 0.0)
            
            if mode == "rag_grounded":
                st.markdown(f'<div class="badge-rag">📄 Grounded on Documents (Confidence: {score:.2f})</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="badge-direct">🤖 Direct LLM Knowledge</div>', unsafe_allow_html=True)
                
            st.markdown(msg["content"])
            
            # Show citations if present
            citations = msg.get("citations", [])
            if citations:
                with st.expander(f"📑 View {len(citations)} Source Citation(s)", expanded=False):
                    for c in citations:
                        st.markdown(f"""
                        <div class="citation-card">
                            <strong>[Doc #{c['citation_num']}] {c['source']} (Page {c['page']})</strong> — <em>Relevance: {c['score']:.2f}</em><br>
                            "{c['snippet']}"
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.markdown(msg["content"])


# User query input
if prompt := st.chat_input("Ask a question about your documents or general topic..."):
    # Append user prompt
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare Assistant response container
    with st.chat_message("assistant"):
        badge_placeholder = st.empty()
        response_placeholder = st.empty()
        citation_placeholder = st.empty()
        
        payload = {
            "query": prompt,
            "model": selected_model,
            "relevance_threshold": relevance_threshold,
            "top_k_retrieval": top_k_retrieval,
            "top_k_rerank": top_k_rerank,
            "temperature": temperature,
            "force_mode": routing_mode
        }
        
        stream_generator = stream_query_response(payload)
        
        full_response = ""
        meta_info = {}
        citations = []
        
        for event in stream_generator:
            event_type = event.get("type")
            
            if event_type == "meta":
                meta_info = event
                mode = meta_info.get("mode", "direct_llm")
                score = meta_info.get("relevance_score", 0.0)
                citations = meta_info.get("citations", [])
                
                if mode == "rag_grounded":
                    badge_placeholder.markdown(
                        f'<div class="badge-rag">📄 Grounded on Documents (Confidence: {score:.2f})</div>',
                        unsafe_allow_html=True
                    )
                else:
                    badge_placeholder.markdown(
                        f'<div class="badge-direct">🤖 Direct LLM Knowledge</div>',
                        unsafe_allow_html=True
                    )
                    
            elif event_type == "token":
                full_response += event.get("content", "")
                response_placeholder.markdown(full_response + "▌")
                
            elif event_type == "done":
                break

        # Final render without cursor
        response_placeholder.markdown(full_response)
        
        # Display citations
        if citations:
            with citation_placeholder.expander(f"📑 View {len(citations)} Source Citation(s)", expanded=False):
                for c in citations:
                    st.markdown(f"""
                    <div class="citation-card">
                        <strong>[Doc #{c['citation_num']}] {c['source']} (Page {c['page']})</strong> — <em>Relevance: {c['score']:.2f}</em><br>
                        "{c['snippet']}"
                    </div>
                    """, unsafe_allow_html=True)

        # Save assistant message to session history
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "mode": meta_info.get("mode", "direct_llm"),
            "relevance_score": meta_info.get("relevance_score", 0.0),
            "citations": citations
        })
