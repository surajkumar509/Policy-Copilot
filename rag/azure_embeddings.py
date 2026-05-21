import os
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv()

required_vars = [
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
]
missing = [name for name in required_vars if not os.environ.get(name)]
if missing:
    raise RuntimeError(
        "Missing required Azure OpenAI environment variables: "
        + ", ".join(missing)
        + ". Please set them in a .env file or export them in your shell before running Streamlit."
    )

client = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01")
)

def embed_text(text: str):
    response = client.embeddings.create(
        input=text,
        model=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]
    )
    return response.data[0].embedding