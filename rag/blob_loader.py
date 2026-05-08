import os
from azure.storage.blob import BlobServiceClient

def load_documents():
    client = BlobServiceClient.from_connection_string(
        os.environ["BLOB_CONNECTION_STRING"]
    )
    container = client.get_container_client("policies")

    documents = []
    for blob in container.list_blobs():
        data = container.download_blob(blob.name).readall()
        documents.append({
            "source": blob.name,
            "text": data.decode("utf-8", errors="ignore")
        })

    return documents