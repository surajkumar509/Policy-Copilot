import sys
import os
from dotenv import load_dotenv

# ✅ Add project root to Python path FIRST (before any agent imports)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
load_dotenv()

from agent.tools import rephrase_response, vary_response  
import random

import streamlit as st
from rag.loader import load_documents
from rag.chunker import chunk_text
from rag.azure_embeddings import embed_text
from rag.shared_store import vector_db
from agent.agent import agent_run

@st.cache_resource
def initialize_vector_db():
    docs = load_documents()
    for doc in docs:
        chunk_records = [
            {
                'source': doc['source'],
                'text': chunk
            }
            for chunk in chunk_text(doc["text"])
        ]
        vectors = [embed_text(chunk['text']) for chunk in chunk_records]
        vector_db.add(vectors, chunk_records)
    return True

initialize_vector_db()

st.set_page_config(page_title="Policy & SOP Copilot", layout="wide")

st.title("📘 Policy & SOP Compliance Copilot")
st.write("Agentic AI + RAG powered policy assistant")

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