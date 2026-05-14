from rag.azure_embeddings import embed_text
import rag.shared_store as shared_store
from rag.azure_chat import chat_with_context
import numpy as np

# Exact cache (simple dictionary)
RESPONSE_CACHE = {}

# Semantic cache (list of embeddings)
SEMANTIC_CACHE = []

def normalize_query(query):
    q = query.lower().strip()

    # remove filler words
    fillers = [
        "please", "can you", "tell me", "how to",
        "about", "what is", "explain"
    ]

    for f in fillers:
        q = q.replace(f, "")

    return q.strip()

def cosine_similarity(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)

    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    query_vec = embed_text(query)

    for item in SEMANTIC_CACHE:
        sim = cosine_similarity(query_vec, item["embedding"])

        if sim > threshold:
            return item["response"]

    return None

def store_semantic_cache(query, response, intent):
    SEMANTIC_CACHE.append({
        "query": query,
        "embedding": embed_text(query),
        "response": response,
        "intent": intent
    })

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

def find_similar_query(query, intent, threshold=0.75):
    query_vec = embed_text(query)

    for item in SEMANTIC_CACHE:

        # ✅ Intent must match
        if item["intent"] != intent:
            continue

        sim = cosine_similarity(query_vec, item["embedding"])

        if sim > threshold:
            return item["response"]

    return None

def generate_answer(context, query):

    normalized = normalize_query(query)
    cache_key = f"ANSWER::{normalized}"

    print("\n🔎 Incoming Query:", query)

    # ✅ Exact cache
    if cache_key in RESPONSE_CACHE:
        print("✅ Exact cache HIT (No API call)")
        return RESPONSE_CACHE[cache_key]

    # ✅ Semantic cache
    semantic_result = find_similar_query(query, "ANSWER")
    if semantic_result:
        print("✅ Semantic cache HIT (No API call)")
        return semantic_result

    print("🚀 Cache MISS → Calling Azure OpenAI API")

    # ✅ Original logic
    if not context:
        return "⚠️ No relevant policy documents found."
    context_texts = []
    for item in context:
        if isinstance(item, dict):
            context_texts.append(f"{item['source']}\n{item['text']}")

    # 🔥 THIS IS YOUR ACTUAL API CALL
    response = chat_with_context("\n\n".join(context_texts), query)

    print("✅ Azure OpenAI API CALL COMPLETED")

    # ✅ Store in cache
    RESPONSE_CACHE[cache_key] = response
    store_semantic_cache(query, response, "ANSWER")

    return response

def create_checklist(context, query):
    if not context:
        return "⚠️ No relevant policy content found to generate checklist."

    context_texts = []
    for item in context:
        if isinstance(item, dict):
            context_texts.append(f"{item['text']}")

    prompt = """
You are an enterprise compliance assistant.

Using ONLY the provided policy context:
- Create a clear, step-by-step compliance checklist
- Include mandatory actions and approvals
- Do not add steps not supported by the policy

Policy Context:
""" + "\n\n".join(context_texts)

    return chat_with_context(prompt, query)

def draft_email(context, query):
    if not context:
        return "⚠️ No relevant policy content found to draft email."

    context_texts = []
    for item in context:
        if isinstance(item, dict):
            context_texts.append(f"{item['text']}")

    prompt = """
You are an enterprise compliance assistant.

Using ONLY the provided policy context:
- Draft a professional corporate email
- Explain required actions clearly
- Use formal and concise language

Policy Context:
""" + "\n\n".join(context_texts)

    return chat_with_context(prompt, query)