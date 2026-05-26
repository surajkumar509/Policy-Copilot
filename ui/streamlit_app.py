import sys
import os
from dotenv import load_dotenv

# ✅ Setup path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
load_dotenv()

import random
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import streamlit as st

# ✅ Imports
import agent.tools as tools
from rag.loader import load_documents
from rag.chunker import chunk_text
from rag.azure_embeddings import embed_text
from rag.shared_store import vector_db
from agent.agent import agent_run

# =========================
# ✅ SESSION STATE INIT
# =========================
if 'loaded_sources' not in st.session_state:
    st.session_state.loaded_sources = set()

if 'last_sync_time' not in st.session_state:
    st.session_state.last_sync_time = 0

if 'load_error' not in st.session_state:
    st.session_state.load_error = None

if 'initial_load_done' not in st.session_state:
    st.session_state.initial_load_done = False

if 'initial_load_count' not in st.session_state:
    st.session_state.initial_load_count = 0


# =========================
# ✅ PARALLEL EMBEDDING
# =========================
def embed_chunks_parallel(chunks):
        with ThreadPoolExecutor(max_workers=5) as executor:
                return list(executor.map(embed_text, chunks))


# =========================
# ✅ LOAD NEW DOCUMENTS
# =========================
def load_new_documents():
    try:
        docs = load_documents()
        added_count = 0
        st.session_state.load_error = None

        for doc in docs:

            # ✅ Skip already loaded
            if doc['source'] in st.session_state.loaded_sources:
                print(f"⏭️ Skipped: {doc['source']}")
                continue

            try:
                # ✅ STEP 1: Chunk text
                chunks = chunk_text(doc["text"])

                # ✅ STEP 3: LIMIT chunks (VERY IMPORTANT)
                MAX_CHUNKS = 80
                chunks = chunks[:MAX_CHUNKS]

                print(f"📊 {doc['source']} → {len(chunks)} chunks")

                # ✅ STEP 2: Embed (ONLY strings)
                vectors = embed_chunks_parallel(chunks)

                # ✅ Prepare records
                chunk_records = [
                    {"source": doc['source'], "text": chunk}
                    for chunk in chunks
                ]

                # ✅ Store in vector DB
                vector_db.add(vectors, chunk_records)

                # ✅ Mark as loaded
                st.session_state.loaded_sources.add(doc['source'])
                added_count += 1

                print(f"✅ Indexed: {doc['source']}")

            except Exception as e:
                print(f"❌ Error processing {doc['source']}: {e}")

        st.session_state.last_sync_time = time.time()

        return added_count

    except Exception as e:
        st.session_state.load_error = str(e)
        print(f"❌ Load error: {e}")
        return 0
        
# =========================
# ✅ PAGE CONFIG
# =========================
st.set_page_config(page_title="Policy & SOP Copilot", layout="wide")

st.title("📘 Policy & SOP Compliance Copilot")
st.write("Agentic AI + RAG powered policy assistant")

# One-time auto load: runs only once per session so reruns stay fast.
if not st.session_state.initial_load_done:
    with st.spinner("First-time setup: loading policy documents..."):
        st.session_state.initial_load_count = load_new_documents()
        st.session_state.initial_load_done = True

# =========================
# ✅ SIDEBAR
# =========================
with st.sidebar:
    st.header("📚 Documents")
    st.write(f"✅ Loaded: {len(st.session_state.loaded_sources)}")

    # ✅ Manual Sync Button
    if st.button("🔄 Sync Now"):
        with st.spinner("Syncing policies..."):
            new_count = load_new_documents()

        if new_count > 0:
            st.success(f"✅ {new_count} new policies added")
        else:
            st.info("No new policies found")

    st.subheader("Loaded Policies")

    if st.session_state.loaded_sources:
        for i, policy in enumerate(sorted(st.session_state.loaded_sources), 1):
            st.write(f"{i}. {policy}")
    else:
        st.write("No policies loaded")

    if st.session_state.load_error:
        st.error(st.session_state.load_error)

# =========================
# ✅ USER INPUT FORM
# =========================
with st.form("query_form", clear_on_submit=True):
    user_query = st.text_area("Enter your policy question:")
    submitted = st.form_submit_button("Submit")

# =========================
# ✅ QUERY HANDLING
# =========================
if submitted:
    if user_query.strip():
        # # ✅ LAZY LOAD (ONLY FIRST TIME)
        # if not st.session_state.initial_load_done:
        #     with st.spinner("🔄 Loading policies (one-time setup)..."):
        #         load_new_documents()
        #         st.session_state.initial_load_done = True

        # ✅ RUN AGENT
        with st.spinner("Thinking..."):
            response, source = agent_run(user_query)

        # ✅ VARIATION LOGIC
        if source != "API_CALL":
            if random.random() < 0.2:
                response = tools.rephrase_response(response)
            else:
                response = tools.vary_response(response)

        dot = "☑️" if source != "API_CALL" else "✅"

        # ✅ DISPLAY QUERY (TOP RIGHT)
        st.markdown(
            f"""
            <div style='text-align:right; font-size:16px; color:#555; margin-bottom:10px;'>
                <strong>Query:</strong> {user_query}
            </div>
            """,
            unsafe_allow_html=True
        )

        # ✅ FORMAT RESPONSE
        formatted = response.replace("\n", "<br>")

        st.markdown(f"{dot} Response")

        st.markdown(
            f"""
            <div style="
                background-color:#f5f5f5;
                padding:15px;
                border-radius:10px;
                border:1px solid #ddd;
            ">
                {formatted}
            </div>
            """,
            unsafe_allow_html=True
        )

        # ✅ COST METRICS
        st.sidebar.markdown("### 📊Token Cost Optimization")
        st.sidebar.write(f"Total Queries: {tools.TOTAL_QUERIES}")
        st.sidebar.write(f"Cache Hits: {tools.CACHE_HITS}")
        st.sidebar.write(f"API Calls: {tools.API_CALLS}")
        st.sidebar.success(f"💰 Savings: {tools.get_cost_savings()}%")

    else:
        st.warning("Please enter a query.")