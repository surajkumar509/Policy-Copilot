
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
load_dotenv()

from rag.azure_embeddings import embed_text
from rag.loader import load_documents
from rag.chunker import chunk_text
from rag.vector_store import VectorStore
from agent.tools import set_vector_db
from ui.cli import start_cli

vector_db = VectorStore()
docs = load_documents()
if not docs:
    print('⚠️  No policy documents found. Please configure Azure Blob storage with policy documents.')
else:
    chunk_count = 0
    for doc in docs:
        chunk_records = [
            {
                'source': doc['source'],
                'text': chunk
            }
            for chunk in chunk_text(doc['text'])
        ]
        vectors = [embed_text(chunk['text']) for chunk in chunk_records]
        vector_db.add(vectors, chunk_records)
        chunk_count += len(chunk_records)
    print(f'✅ Policies indexed: {len(docs)} documents, {chunk_count} chunks')

# Share the vector database with the tools module
set_vector_db(vector_db)
start_cli()
    