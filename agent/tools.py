from rag.azure_embeddings import embed_text
import rag.shared_store as shared_store
from rag.azure_chat import chat_with_context
import numpy as np
import time
import random

# ✅ Caches
RESPONSE_CACHE = {}
SEMANTIC_CACHE = []

# ✅ Tracking
LAST_CACHE_CLEAR = time.time()
CACHE_TTL = 20

TOTAL_QUERIES = 0
CACHE_HITS = 0
API_CALLS = 0


# ✅ Auto clear cache
def auto_clear_cache():
    global LAST_CACHE_CLEAR

    if time.time() - LAST_CACHE_CLEAR > CACHE_TTL:
        RESPONSE_CACHE.clear()
        SEMANTIC_CACHE.clear()
        LAST_CACHE_CLEAR = time.time()
        print("🧹 Cache cleared automatically")


# ✅ Normalize query
def normalize_query(query):
    q = query.lower()

    # unify
    q = q.replace("check list", "checklist")

    replacements = {
        "steps": "checklist",
        "procedure": "checklist",
        "mail": "email",
        "draft": "email"
    }

    for k, v in replacements.items():
        q = q.replace(k, v)

    # remove fillers
    for f in ["please", "can you", "tell me", "how to"]:
        q = q.replace(f, "")

    # ✅ CRITICAL FIX
    q = " ".join(q.split())   # removes extra spaces

    return q


# ✅ Cosine similarity FIXED
def cosine_similarity(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)

    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return 0

    return np.dot(v1, v2) / denom


# ✅ Semantic cache lookup
def find_similar_query(query, intent, threshold=0.60):
    query = normalize_query(query)
    query_vec = embed_text(query)

    for item in SEMANTIC_CACHE:
        if item["intent"] != intent:
            continue

        sim = cosine_similarity(query_vec, item["embedding"])

        if sim > threshold:
            return item["response"]

    return None


# ✅ Store semantic cache
def store_semantic_cache(query, response, intent):
    query = normalize_query(query)

    SEMANTIC_CACHE.append({
        "query": query,
        "embedding": embed_text(query),
        "response": response,
        "intent": intent
    })


# ✅ Search policies
def search_policies(query):
    db = shared_store.vector_db

    if db.index is None or db.index.ntotal == 0:
        return []

    query_vector = embed_text(query)
    results = db.search(query_vector)

    unique = []
    seen = set()

    for item in results:
        src = item.get('source')
        if src and src not in seen:
            unique.append(item)
            seen.add(src)

    return unique


# ✅ Cost saving FIXED
def get_cost_savings():
    if TOTAL_QUERIES == 0:
        return 0

    saved = TOTAL_QUERIES - API_CALLS
    return round((saved / TOTAL_QUERIES) * 100)


# ✅ Vary response (KEEP SINGLE VERSION)
def vary_response(response):
    styles = [
        response,
        "👉 " + response,
        "📌 " + response
    ]
    return random.choice(styles)


# ✅ Generate answer
def generate_answer(context, query):
    global TOTAL_QUERIES, CACHE_HITS, API_CALLS

    auto_clear_cache()
    TOTAL_QUERIES += 1

    normalized = normalize_query(query)
    cache_key = f"ANSWER::{normalized}"

    print("\n🔎 Query:", query)

    # ✅ Exact cache
    if cache_key in RESPONSE_CACHE:
        CACHE_HITS += 1
        print("✅ Exact cache HIT")
        return RESPONSE_CACHE[cache_key], "EXACT_CACHE"

    # ✅ Semantic cache
    semantic = find_similar_query(query, "ANSWER")
    if semantic:
        CACHE_HITS += 1
        print("✅ Semantic cache HIT")
        return semantic, "SEMANTIC_CACHE"

    print("🚀 API CALL STARTED")

    if not context:
        return "⚠️ No relevant policy documents found.", "API_CALL"

    context_text = "\n\n".join([f"{c['source']}\n{c['text']}" for c in context])

    response = chat_with_context(context_text, query)

    API_CALLS += 1
    print("✅ API DONE")

    RESPONSE_CACHE[cache_key] = response
    store_semantic_cache(query, response, "ANSWER")

    return response, "API_CALL"


# ✅ Checklist
def create_checklist(context, query):
    global TOTAL_QUERIES, CACHE_HITS, API_CALLS

    auto_clear_cache()
    TOTAL_QUERIES += 1
    normalized = normalize_query(query)
    cache_key = f"CHECKLIST::{normalized}"

    if cache_key in RESPONSE_CACHE:
        CACHE_HITS += 1 
        return RESPONSE_CACHE[cache_key], "EXACT_CACHE"
    
    normalized = normalize_query(query)
    semantic = find_similar_query(query, "CHECKLIST")
    if semantic:
        CACHE_HITS += 1 
        return semantic, "SEMANTIC_CACHE"

    if not context:
        return "⚠️ No policy content", "API_CALL"

    prompt = "Create max 5 step checklist:\n" + "\n\n".join(c["text"] for c in context)

    response = chat_with_context(prompt, query)
    API_CALLS += 1

    RESPONSE_CACHE[cache_key] = response
    store_semantic_cache(query, response, "CHECKLIST")

    return response, "API_CALL"


# ✅ Email
def draft_email(context, query):
    global TOTAL_QUERIES, CACHE_HITS, API_CALLS

    auto_clear_cache()
    TOTAL_QUERIES += 1
    normalized = normalize_query(query)
    cache_key = f"EMAIL::{normalized}"

    if cache_key in RESPONSE_CACHE:
        CACHE_HITS += 1 
        return RESPONSE_CACHE[cache_key], "EXACT_CACHE"

    semantic = find_similar_query(query, "EMAIL")
    if semantic:
        CACHE_HITS += 1
        return semantic, "SEMANTIC_CACHE"

    if not context:
        return "⚠️ No policy content", "API_CALL"

    prompt = "Write short professional email:\n" + "\n\n".join(c["text"] for c in context)

    response = chat_with_context(prompt, query)
    API_CALLS += 1

    RESPONSE_CACHE[cache_key] = response
    store_semantic_cache(query, response, "EMAIL")

    return response, "API_CALL"

def rephrase_response(text):
    prompt = f"""
Rephrase the following response using different wording.
Do NOT change the meaning.
Keep it concise and professional.

Response:
{text}
"""
    return chat_with_context("", prompt)
