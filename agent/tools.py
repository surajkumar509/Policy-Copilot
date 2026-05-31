from rag.azure_embeddings import embed_text
import rag.shared_store as shared_store
from rag.azure_chat import chat_with_context
import numpy as np
import time
import random
from datetime import datetime

# ✅ Caches
RESPONSE_CACHE = {}
SEMANTIC_CACHE = []

# ✅ Tracking
LAST_CACHE_CLEAR = time.time()
CACHE_TTL = 120

TOTAL_QUERIES = 0
CACHE_HITS = 0
API_CALLS = 0

STOP_TERMS = {
    "the", "a", "an", "and", "or", "for", "to", "of", "in", "on", "with",
    "what", "is", "are", "can", "you", "please", "tell", "me", "about"
}


def set_vector_db(db):
    """Allow application entrypoints to replace the shared in-memory vector store."""
    shared_store.vector_db = db


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

    normalized_query = normalize_query(query)
    query_terms = [t for t in normalized_query.split() if t and t not in STOP_TERMS]

    query_vector = embed_text(query)
    # Pull a wider candidate set and rerank to avoid unrelated policy bleed.
    k = min(40, db.index.ntotal)
    results = db.search(query_vector, k=k)

    leave_intent = any(t in normalized_query for t in ("leave", "holiday", "lta"))
    if leave_intent:
        # Make sure leave/holiday-named policies are considered even if vector rank is low.
        seeded = {}
        for item in db.items:
            src = (item.get("source") or "").lower()
            if any(t in src for t in ("leave", "holiday", "lta")) and src not in seeded:
                seeded[src] = item
        if seeded:
            results.extend(seeded.values())

    def source_and_text(item):
        src = (item.get("source") or "").lower()
        txt = (item.get("text") or "").lower()
        return src, txt

    def relevance_score(item):
        src, txt = source_and_text(item)
        score = 0

        for term in query_terms:
            if term in src:
                score += 3
            if term in txt:
                score += 2

        # Leave/holiday questions should strongly prefer matching leave/holiday sources.
        if leave_intent:
            if any(t in src for t in ("leave", "holiday", "lta")):
                score += 6
            if any(t in txt for t in ("leave", "holiday", "lta")):
                score += 3

        return score

    ranked = sorted(results, key=relevance_score, reverse=True)

    unique = []
    seen = set()
    max_sources = 2 if leave_intent else 3

    for item in ranked:
        src = item.get('source')
        if src and src not in seen:
            unique.append(item)
            seen.add(src)

        if len(unique) >= max_sources:
            break

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

    answer_prompt = f"""
You are a policy compliance assistant. Use only the provided policy context.

User question:
{query}

Response style:
- Write a rich, descriptive answer that is clear and practical.
- Include concrete details from policy context (eligibility, conditions, limits, timelines, exceptions, and process steps when available).
- Use short sections with headings in this order when relevant:
    1) Direct Answer
    2) Key Policy Details
    3) What To Do Next
    4) Caveats or Missing Information
- Keep tone professional and easy to understand.

Grounding and safety:
- Do not invent policy facts.
- If context is partial, explicitly state what is missing and what document detail would be needed.
- If multiple policy documents differ, mention the difference briefly.
- End with exactly one line in this format:
    Source: <comma-separated policy file names actually used>
""".strip()

    response = chat_with_context(context_text, answer_prompt)

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


def simulate_email_send(recipients, subject, body, channel="UI_SIMULATOR"):
    """Return a fake send receipt without sending a real email."""
    cleaned = [r.strip() for r in recipients if r and r.strip()]

    if not cleaned:
        return {
            "ok": False,
            "message": "Simulation failed: at least one recipient is required.",
            "details": {
                "channel": channel,
                "recipient_count": 0,
                "subject": subject,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        }

    fake_id = f"SIM-{int(time.time())}-{len(cleaned)}"
    return {
        "ok": True,
        "message": "Simulated send successful (no real email delivered).",
        "details": {
            "channel": channel,
            "message_id": fake_id,
            "recipient_count": len(cleaned),
            "recipients": cleaned,
            "subject": subject,
            "body_preview": body[:180],
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    }
