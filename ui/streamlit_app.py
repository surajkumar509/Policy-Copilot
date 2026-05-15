import sys
import os
from dotenv import load_dotenv

# ✅ Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
load_dotenv()

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
     
        # ✅ Decide indicator
        dot = "☑️" if source != "API_CALL" else "✅"
        print("Source", source, dot)  # Debug: check source and indicator
        # ✅ Show indicator (clean, no text)
        st.write(dot, "Response")           
        st.write(response)
    else:
        st.warning("Please enter a query.")