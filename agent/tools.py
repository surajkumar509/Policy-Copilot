from rag.azure_embeddings import embed_text
import rag.shared_store as shared_store
from rag.azure_chat import chat_with_context
import numpy as np
import time
import random


# Exact cache (simple dictionary)
RESPONSE_CACHE = {}

# Semantic cache (list of embeddings)
SEMANTIC_CACHE = []

# ✅ Track last clear time
LAST_CACHE_CLEAR = time.time()
CACHE_TTL = 60

def auto_clear_cache():
    global LAST_CACHE_CLEAR

    current_time = time.time()

    # ✅ Check if TTL expired
    if current_time - LAST_CACHE_CLEAR > CACHE_TTL:
        RESPONSE_CACHE.clear()
        SEMANTIC_CACHE.clear()

        LAST_CACHE_CLEAR = current_time

        print("🧹 Cache cleared automatically")

def rephrase_response(text):
    prompt = f"""
Rephrase the following response using different wording.
Do NOT change the meaning.
Keep it concise and professional.

Response:
{text}
""" 
    # ✅ Use your existing Azure chat function
    return chat_with_context("", prompt)

def vary_response(response):
    styles = [
        response,
        "👉 " + response,
        "📌 " + response,
        "\n".join([f"• {line}" for line in response.split("\n") if line.strip()]),
        response.replace(". ", ".\n\n"),
    ]
    return random.choice(styles)

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

def vary_response(response):
    styles = [
        response,
        "👉 " + response,
        "📌 " + response,
        response.replace("Technical certification", "Technical certification is"),
        response.replace("refers to", "means"),
    ]
    
    return random.choice(styles)

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

    # ✅ Normalize query for exact cache
    normalized = normalize_query(query)
    cache_key = f"ANSWER::{normalized}"

    print("\n🔎 Incoming Query:", query)

    # ✅ 1. Exact cache check
    if cache_key in RESPONSE_CACHE:
        print("✅ Exact cache HIT (No API call)")
        return RESPONSE_CACHE[cache_key], "EXACT_CACHE"

    # ✅ 2. Semantic cache check
    semantic_result = find_similar_query(query, "ANSWER")
    if semantic_result:
        print("✅ Semantic cache HIT (No API call)")
        return semantic_result, "SEMANTIC_CACHE"

    print("🚀 Cache MISS → Calling Azure OpenAI API")

    # ✅ 3. Guard: no context
    if not context:
        return "⚠️ No relevant policy documents found.", "API_CALL"

    # ✅ 4. Prepare context for RAG
    context_texts = [
        f"{item['source']}\n{item['text']}"
        for item in context
        if isinstance(item, dict)
    ]

    # ✅ 5. Azure OpenAI call (only here)
    response = chat_with_context("\n\n".join(context_texts), query)

    print("✅ Azure OpenAI API CALL COMPLETED")

    # ✅ 6. Store in caches
    RESPONSE_CACHE[cache_key] = response
    store_semantic_cache(query, response, "ANSWER")

    # ✅ 7. Always return tuple
    return response, "API_CALL"

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

    normalized = normalize_query(query)
    cache_key = f"CHECKLIST::{normalized}"

    if cache_key in RESPONSE_CACHE:
        return RESPONSE_CACHE[cache_key], "EXACT_CACHE"

    semantic_result = find_similar_query(query, "CHECKLIST")
    if semantic_result:
        return semantic_result, "SEMANTIC_CACHE"

    if not context:
        return "⚠️ No relevant policy content found.", "API_CALL"

    context_texts = [item["text"] for item in context if isinstance(item, dict)]

    prompt = """
Create a clear, step-by-step compliance checklist using ONLY the policy context below.
Limit the checklist to a maximum of 5 steps.

Policy Context:
""" + "\n\n".join(context_texts)

    response = chat_with_context(prompt, query)

    RESPONSE_CACHE[cache_key] = response
    store_semantic_cache(query, response, "CHECKLIST")

    return response, "API_CALL"

def draft_email(context, query):

    normalized = normalize_query(query)
    cache_key = f"EMAIL::{normalized}"

    if cache_key in RESPONSE_CACHE:
        return RESPONSE_CACHE[cache_key], "EXACT_CACHE"

    semantic_result = find_similar_query(query, "EMAIL")
    if semantic_result:
        return semantic_result, "SEMANTIC_CACHE"

    if not context:
        return "⚠️ No relevant policy content found.", "API_CALL"

    context_texts = [item["text"] for item in context if isinstance(item, dict)]

    prompt = """
Draft a professional corporate email using ONLY the policy context below.
Keep the email concise (under 150 words).

Policy Context:
""" + "\n\n".join(context_texts)

    response = chat_with_context(prompt, query)

    RESPONSE_CACHE[cache_key] = response
    store_semantic_cache(query, response, "EMAIL")

    return response, "API_CALL"
    
def detect_intent(query):    
    q = query.lower()

    if any(word in q for word in ["checklist", "steps", "procedure", "process"]):
        return "CHECKLIST"

    if any(word in q for word in ["email", "mail", "draft"]):
        return "EMAIL"

    return "ANSWER"