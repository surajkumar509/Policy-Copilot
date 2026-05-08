"""Final validation of the Policy Copilot application"""
import sys
import os
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(__file__))
load_dotenv()

from rag.loader import load_documents
from rag.chunker import chunk_text
from rag.azure_embeddings import embed_text
from rag.vector_store import VectorStore
from agent.tools import set_vector_db
from agent.agent import agent_run

print("=" * 60)
print("FINAL VALIDATION: Policy Copilot Application")
print("=" * 60)

# Load and index data
vector_db = VectorStore()
try:
    docs = load_documents()
    print(f"\n✅ Loaded {len(docs)} policy documents from Azure Blob")
except Exception as e:
    print(f"\n❌ Failed to load documents: {e}")
    print("⚠️  Ensure BLOB_CONNECTION_STRING is set in .env")
    sys.exit(1)

total_chunks = 0
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
    total_chunks += len(chunk_records)
    print(f"   - {doc['source']}: {len(chunk_records)} chunks")

print(f"\n✅ Total chunks indexed: {total_chunks}")

# Share vector database
set_vector_db(vector_db)

# Test queries
print("\n" + "=" * 60)
print("RUNNING AGENT TESTS")
print("=" * 60)

test_queries = [
    ("leave policy", "Testing policy search"),
    ("create a checklist", "Testing checklist generation"),
    ("draft email", "Testing email generation"),
    ("what are holidays", "Testing general query"),
]

all_tests_passed = True
for query, description in test_queries:
    print(f"\n[{description}]")
    print(f"Query: {query}")
    try:
        response = agent_run(query)
        print(f"Response: {response[:100]}...")
        if response.startswith("[Azure OpenAI chat error]"):
            print("⚠️ Azure chat error returned")
            all_tests_passed = False
        else:
            print("✅ Success")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        all_tests_passed = False

print("\n" + "=" * 60)
if all_tests_passed:
    print("✅ ALL VALIDATION TESTS PASSED")
    print("=" * 60)
    print("\nApplication is fully functional and ready for use!")
    print("Run 'python app.py' to start the CLI interface.")
else:
    print("❌ VALIDATION DETECTED FAILURES")
    print("=" * 60)
    print("\nPlease fix the errors above before using the application in production.")
