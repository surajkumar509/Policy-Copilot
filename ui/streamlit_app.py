import sys
import os
from dotenv import load_dotenv

# ✅ Add project root to Python path FIRST (before any agent imports)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
load_dotenv()

from agent.tools import rephrase_response, vary_response  
import random
import time
from datetime import datetime

import streamlit as st
from rag.loader import load_documents
from rag.chunker import chunk_text
from rag.azure_embeddings import embed_text
from rag.shared_store import vector_db
from agent.agent import agent_run
from agent.tools import TOTAL_QUERIES, CACHE_HITS, API_CALLS, get_cost_savings

# ✅ Session state to track loaded document sources
if 'loaded_sources' not in st.session_state:
    st.session_state.loaded_sources = set()
    st.session_state.last_sync_time = 0
    st.session_state.load_error = None

# ✅ Auto-sync interval in seconds
AUTO_SYNC_INTERVAL = 30

def load_new_documents():
    """Load only NEW documents from Azure Blob that haven't been loaded yet"""
    try:
        docs = load_documents()
        added_count = 0
        st.session_state.load_error = None
        
        for doc in docs:
            # ✅ Skip if already loaded
            if doc['source'] in st.session_state.loaded_sources:
                continue
                
            chunk_records = [
                {
                    'source': doc['source'],
                    'text': chunk
                }
                for chunk in chunk_text(doc["text"])
            ]
            vectors = [embed_text(chunk['text']) for chunk in chunk_records]
            vector_db.add(vectors, chunk_records)
            
            # ✅ Mark as loaded
            st.session_state.loaded_sources.add(doc['source'])
            added_count += 1
            
            # ✅ Print loaded policy
            print(f"📄 Loaded policy: {doc['source']}")
        
        # ✅ Update sync time only when we attempt a load
        st.session_state.last_sync_time = time.time()
        
        if added_count > 0:
            print(f"✅ Total policies loaded: {len(st.session_state.loaded_sources)}")
        
        return added_count
    except Exception as e:
        st.session_state.load_error = str(e)
        print(f"❌ Error loading documents: {e}")
        return 0

# ✅ Auto-sync on every page load if interval exceeded
current_time = time.time()
if current_time - st.session_state.last_sync_time > AUTO_SYNC_INTERVAL:
    load_new_documents()

st.set_page_config(page_title="Policy & SOP Copilot", layout="wide")

st.title("📘 Policy & SOP Compliance Copilot")
st.write("Agentic AI + RAG powered policy assistant")

# ✅ Sidebar with status
with st.sidebar:
    st.header("📚 Documents")
    st.write(f"✅ Total loaded: {len(st.session_state.loaded_sources)}")
    
    # ✅ Manual sync button
    if st.button("🔄 Sync Now"):
        new_count = load_new_documents()
        if new_count > 0:
            st.rerun()
    
    # ✅ Show all loaded policies
    st.subheader("Loaded Policies:")
    if st.session_state.loaded_sources:
        for idx, policy in enumerate(sorted(st.session_state.loaded_sources), 1):
            st.write(f"{idx}. {policy}")
    else:
        st.write("No policies loaded yet")
    
    # ✅ Show last sync time
    last_sync = datetime.fromtimestamp(st.session_state.last_sync_time).strftime('%H:%M:%S') if st.session_state.last_sync_time > 0 else "Never"
    st.caption(f"Last sync: {last_sync}")
    st.caption(f"Auto-syncs every {AUTO_SYNC_INTERVAL}s")

    if st.session_state.load_error:
        st.error(f"Document load error: {st.session_state.load_error}")

    st.write(f"Total loaded: {len(st.session_state.loaded_sources)}")

user_query = st.text_area("Enter your policy question or request:")


if st.button("Submit"):
    if user_query.strip():
        with st.spinner("Thinking..."):
            response, source = agent_run(user_query)

        # ✅ Apply variation ONLY for cache responses
        if source != "API_CALL":
            if random.random() < 0.2:   # 20% rephrase
                response = rephrase_response(response)
            else:
                response = vary_response(response)

        # ✅ Indicator (✅ = fresh, ☑️ = cached)
        dot = "☑️" if source != "API_CALL" else "✅"

        # ✅ Indicator on top-right
        # st.markdown(
        #     f"<div style='text-align:right; font-size:20px'>{dot}</div>",
        #     unsafe_allow_html=True
        # )

        # ✅ Header
        st.markdown(f"{dot} Response")

        # ✅ Clean formatting
        formatted = response.replace("\n", "<br>")  
        st.sidebar.markdown("### 📊 Cost Optimization")
        st.sidebar.write(f"Total Queries: {TOTAL_QUERIES}")
        st.sidebar.write(f"Cache Hits: {CACHE_HITS}")
        st.sidebar.write(f"API Calls: {API_CALLS}")
        st.sidebar.success(f"💰 Savings: {get_cost_savings()}%")
        st.markdown(
            f"""
            <div style="
                background-color:#f5f5f5;
                padding:15px;
                border-radius:10px;
                border:1px solid #ddd;
                font-size:15px;
                line-height:1.6;
            ">
                {formatted}
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        st.warning("Please enter a query.")