from rag.azure_embeddings import embed_text
import rag.shared_store as shared_store
from rag.azure_chat import chat_with_context
def set_vector_db(db):
    shared_store.vector_db = db


def search_policies(query):
    db = shared_store.vector_db
    if db.index is None or db.index.ntotal == 0:
        return []

    query_vector = embed_text(query)
    search_results = db.search(query_vector)

    unique_results = []
    seen_sources = set()
    for item in search_results:
        source = item.get('source') if isinstance(item, dict) else None
        if source and source not in seen_sources:
            unique_results.append(item)
            seen_sources.add(source)
        if len(unique_results) >= 3:
            break

    return unique_results


def generate_answer(context, query):
    if not context:
        return "⚠️ No relevant policy documents found for this query."

    unique_sources = []
    source_set = set()
    context_texts = []
    doc_excerpts = []
    for item in context:
        if not isinstance(item, dict):
            continue
        context_texts.append(f"Source: {item['source']}\n{item['text']}")
        if item['source'] not in source_set:
            source_set.add(item['source'])
            unique_sources.append(item['source'])
            excerpt = item['text'].strip().replace('\n', ' ')
            if len(excerpt) > 300:
                excerpt = excerpt[:300].rsplit(' ', 1)[0] + '...'
            doc_excerpts.append(f"- {item['source']}: {excerpt}")

    response = chat_with_context("\n\n".join(context_texts), query)
    return response

def create_checklist(context):
    return "Checklist:\n- Step 1\n- Step 2"

def draft_email(context):
    return "Email drafted using policy context."