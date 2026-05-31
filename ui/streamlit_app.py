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
import io

import streamlit as st
from docx import Document
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

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
if "loaded_sources" not in st.session_state:
    st.session_state.loaded_sources = set()

if "last_sync_time" not in st.session_state:
    st.session_state.last_sync_time = 0

if "load_error" not in st.session_state:
    st.session_state.load_error = None

if "initial_load_done" not in st.session_state:
    st.session_state.initial_load_done = False

if "initial_load_count" not in st.session_state:
    st.session_state.initial_load_count = 0

if "last_user_query" not in st.session_state:
    st.session_state.last_user_query = ""

if "last_response" not in st.session_state:
    st.session_state.last_response = ""

if "last_source" not in st.session_state:
    st.session_state.last_source = ""

if "last_sim_receipt" not in st.session_state:
    st.session_state.last_sim_receipt = None

if "simulated_email_history" not in st.session_state:
    st.session_state.simulated_email_history = []

if "download_choice" not in st.session_state:
    st.session_state.download_choice = "No"

if "download_format" not in st.session_state:
    st.session_state.download_format = "DOCX"

if "download_ready" not in st.session_state:
    st.session_state.download_ready = False

if "download_payload" not in st.session_state:
    st.session_state.download_payload = b""

if "download_filename" not in st.session_state:
    st.session_state.download_filename = ""

if "download_mime" not in st.session_state:
    st.session_state.download_mime = ""

if "download_label" not in st.session_state:
    st.session_state.download_label = ""

if "prepared_download_format" not in st.session_state:
    st.session_state.prepared_download_format = ""


# =========================
# ✅ PARALLEL EMBEDDING
# =========================
def embed_chunks_parallel(chunks):
    with ThreadPoolExecutor(max_workers=5) as executor:
        return list(executor.map(embed_text, chunks))


def parse_email_subject_and_body(response_text, user_query):
    lines = [line.strip() for line in response_text.splitlines() if line.strip()]
    if not lines:
        return "Policy Update", response_text

    first = lines[0]
    if first.lower().startswith("subject:"):
        subject = first.split(":", 1)[1].strip() or "Policy Update"
        body = "\n".join(response_text.splitlines()[1:]).strip()
        return subject, (body if body else response_text)

    short_query = user_query.strip()[:60] if user_query.strip() else "Policy Update"
    return f"Policy Follow-up: {short_query}", response_text


def build_download_filename(prefix, query):
    cleaned = "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in query.lower().strip()
    )
    cleaned = "_".join([part for part in cleaned.split("_") if part])
    if not cleaned:
        cleaned = "policy"
    return f"{prefix}_{cleaned[:40]}"


def build_checklist_pdf(query, response):
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    y = height - 50
    line_height = 14

    def draw_line(text):
        nonlocal y
        if y < 50:
            pdf.showPage()
            y = height - 50
        pdf.drawString(40, y, text)
        y -= line_height

    pdf.setFont("Helvetica-Bold", 14)
    draw_line("Policy Checklist")
    y -= 4
    pdf.setFont("Helvetica-Bold", 11)
    draw_line("Query:")
    pdf.setFont("Helvetica", 10)
    for part in query.splitlines() or [query]:
        for chunk in [part[i : i + 95] for i in range(0, len(part), 95)] or [""]:
            draw_line(chunk)
    y -= 4
    pdf.setFont("Helvetica-Bold", 11)
    draw_line("Response:")
    pdf.setFont("Helvetica", 10)
    for part in response.splitlines() or [response]:
        for chunk in [part[i : i + 95] for i in range(0, len(part), 95)] or [""]:
            draw_line(chunk)

    pdf.save()
    buffer.seek(0)
    return buffer.getvalue()


def build_checklist_docx(query, response):
    doc = Document()
    doc.add_heading("Policy Checklist", level=1)
    doc.add_heading("Query", level=2)
    doc.add_paragraph(query)
    doc.add_heading("Response", level=2)
    doc.add_paragraph(response)
    buffer = io.BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer.getvalue()


def build_checklist_txt(query, response):
    content = (
        "Policy Checklist\n"
        "================\n\n"
        "Query:\n" + query + "\n\n" + "Response:\n" + response + "\n"
    )
    return content.encode("utf-8")


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
            if doc["source"] in st.session_state.loaded_sources:
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
                    {"source": doc["source"], "text": chunk} for chunk in chunks
                ]

                # ✅ Store in vector DB
                vector_db.add(vectors, chunk_records)

                # ✅ Mark as loaded
                st.session_state.loaded_sources.add(doc["source"])
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

        st.session_state.last_user_query = user_query
        st.session_state.last_response = response
        st.session_state.last_source = source
        st.session_state.download_choice = "No"
        st.session_state.download_format = "DOCX"
        st.session_state.download_ready = False
        st.session_state.download_payload = b""
        st.session_state.download_filename = ""
        st.session_state.download_mime = ""
        st.session_state.download_label = ""
        st.session_state.prepared_download_format = ""

if st.session_state.last_response:
    user_query = st.session_state.last_user_query
    response = st.session_state.last_response
    source = st.session_state.last_source
    dot = "☑️" if source != "API_CALL" else "✅"

    # ✅ DISPLAY QUERY (TOP RIGHT)
    st.markdown(
        f"""
        <div style='text-align:right; font-size:16px; color:#555; margin-bottom:10px;'>
            <strong>Query:</strong> {user_query}
        </div>
        """,
        unsafe_allow_html=True,
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
        unsafe_allow_html=True,
    )

    # ✅ COST METRICS
    st.sidebar.markdown("### 📊Token Cost Optimization")
    st.sidebar.write(f"Total Queries: {tools.TOTAL_QUERIES}")
    st.sidebar.write(f"Cache Hits: {tools.CACHE_HITS}")
    st.sidebar.write(f"API Calls: {tools.API_CALLS}")
    st.sidebar.success(f"💰 Savings: {tools.get_cost_savings()}%")

    normalized_query = tools.normalize_query(user_query)

    if "checklist" in normalized_query:
        st.markdown("### Download Checklist")
        st.radio(
            "Would you like to download this checklist?",
            options=["No", "Yes"],
            key="download_choice",
            horizontal=True,
        )

        if st.session_state.download_choice == "Yes":
            st.radio(
                "Which format do you want?",
                options=["DOCX", "PDF", "TXT"],
                key="download_format",
                horizontal=True,
            )

            if st.button("Run Download Automation", use_container_width=True):
                with st.status("Automation running...", expanded=True) as status:
                    st.write("1. Reading checklist response")
                    st.write(f"2. Preparing {st.session_state.download_format} file")

                    base_name = build_download_filename("checklist", user_query)
                    selected_format = st.session_state.download_format

                    if selected_format == "DOCX":
                        payload = build_checklist_docx(user_query, response)
                        ext = ".docx"
                        mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    elif selected_format == "PDF":
                        payload = build_checklist_pdf(user_query, response)
                        ext = ".pdf"
                        mime = "application/pdf"
                    else:
                        payload = build_checklist_txt(user_query, response)
                        ext = ".txt"
                        mime = "text/plain"

                    st.session_state.download_payload = payload
                    st.session_state.download_filename = f"{base_name}{ext}"
                    st.session_state.download_mime = mime
                    st.session_state.download_label = (
                        f"Download Checklist ({selected_format})"
                    )
                    st.session_state.download_ready = True
                    st.session_state.prepared_download_format = selected_format

                    st.write("3. File prepared successfully")
                    status.update(label="Automation completed", state="complete")

            if (
                st.session_state.download_ready
                and st.session_state.prepared_download_format
                == st.session_state.download_format
            ):
                st.success(
                    f"Automation done. Ready to download {st.session_state.download_filename}"
                )
                st.download_button(
                    label=st.session_state.download_label,
                    data=st.session_state.download_payload,
                    file_name=st.session_state.download_filename,
                    mime=st.session_state.download_mime,
                )

    if "email" in normalized_query:
        st.markdown("### Email Simulation (No Real Send)")
        suggested_subject, suggested_body = parse_email_subject_and_body(
            response, user_query
        )

        with st.form("email_simulation_form"):
            to_raw = st.text_input("To", value="manager@company.com")
            cc_raw = st.text_input("CC (optional)", value="")
            subject = st.text_input("Subject", value=suggested_subject)
            body = st.text_area("Body", value=suggested_body, height=180)
            simulate_submit = st.form_submit_button("Simulate Send")

        if simulate_submit:
            recipients = [
                x.strip() for x in (to_raw + "," + cc_raw).split(",") if x.strip()
            ]
            st.session_state.last_sim_receipt = tools.simulate_email_send(
                recipients, subject, body
            )
            if st.session_state.last_sim_receipt["ok"]:
                st.session_state.simulated_email_history.append(
                    st.session_state.last_sim_receipt["details"]
                )

        if st.session_state.last_sim_receipt:
            receipt = st.session_state.last_sim_receipt
            if receipt["ok"]:
                detail = receipt["details"]
                st.success(
                    f"Mail send simulated successfully for {detail['recipient_count']} recipient(s). "
                    f"Reference: {detail['message_id']}"
                )
            else:
                st.error(receipt["message"])

        if st.session_state.simulated_email_history:
            with st.expander("View Simulation History"):
                for idx, item in enumerate(
                    reversed(st.session_state.simulated_email_history), 1
                ):
                    st.write(
                        f"{idx}. {item['timestamp']} | {item['message_id']} | {item['recipient_count']} recipient(s)"
                    )
elif submitted:
    st.warning("Please enter a query.")
