
import os
from .blob_loader import load_documents as load_documents_from_blob

def load_documents():
    if not os.environ.get("BLOB_CONNECTION_STRING"):
        raise ValueError(
            "Azure Blob storage is required. Please set BLOB_CONNECTION_STRING in .env."
        )

    try:
        return load_documents_from_blob()
    except Exception as e:
        raise RuntimeError(f"Failed to load documents from Azure Blob: {e}")

