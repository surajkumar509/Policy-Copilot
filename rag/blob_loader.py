import os
from io import BytesIO
from azure.storage.blob import BlobServiceClient
from PyPDF2 import PdfReader
from docx import Document

MAX_FILE_SIZE_MB = 15   # ✅ increase if needed


def extract_pdf(file_bytes):
    text = ""
    try:
        reader = PdfReader(BytesIO(file_bytes))
        for page in reader.pages:
            text += page.extract_text() or ""
    except Exception as e:
        print(f"❌ PDF Error: {e}")
    return text


def extract_docx(file_bytes):
    text = ""
    try:
        doc = Document(BytesIO(file_bytes))
        for para in doc.paragraphs:
            text += para.text + "\n"
    except Exception as e:
        print(f"❌ DOCX Error: {e}")
    return text


def load_documents():
    client = BlobServiceClient.from_connection_string(
        os.environ["BLOB_CONNECTION_STRING"]
    )

    container = client.get_container_client("policies")

    documents = []

    blobs = list(container.list_blobs())
    print(f"🔍 Total blobs in Azure: {len(blobs)}")

    for blob in blobs:
        print(f"➡️ Processing: {blob.name}")

        try:
            # ✅ Skip very large files
            if blob.size and blob.size > MAX_FILE_SIZE_MB * 1024 * 1024:
                print(f"⚠️ Skipping large file: {blob.name}")
                continue

            data = container.download_blob(blob.name).readall()

            text = ""

            # ✅ FILE TYPE HANDLING
            if blob.name.endswith(".pdf"):
                text = extract_pdf(data)

            elif blob.name.endswith(".docx"):
                text = extract_docx(data)

            elif blob.name.endswith(".txt"):
                text = data.decode("utf-8", errors="ignore")

            else:
                print(f"⚠️ Unsupported format: {blob.name}")
                continue

            # ✅ Validate content
            if not text or len(text.strip()) < 20:
                print(f"⚠️ Empty/invalid content: {blob.name}")
                continue

            documents.append({
                "source": blob.name,
                "text": text
            })

            print(f"✅ Loaded: {blob.name}")

        except Exception as e:
            print(f"❌ Failed: {blob.name} | {e}")

    print(f"✅ Final usable documents: {len(documents)}")
    return documents