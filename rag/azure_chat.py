import os

try:
    from openai import AzureOpenAI
except ImportError:
    AzureOpenAI = None

azure_configured = all(
    os.environ.get(var)
    for var in [
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_CHAT_DEPLOYMENT",
    ]
)

client = None
if azure_configured and AzureOpenAI is not None:
    client = AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version="2024-02-01"
    )

def chat_with_context(context: str, user_query: str) -> str:
    """
    Sends policy-grounded context to Azure OpenAI Chat model
    """

    messages = [
        {
            "role": "system",
            "content": "Answer ONLY using the provided policy context."
        },
        {
            "role": "user",
            "content": f"Policy Context:\n{context}\n\nUser Query:\n{user_query}"
        }
    ]

    if client is None:
        return (
            f"[Azure OpenAI chat not configured] Query: {user_query}. "
            "Install the OpenAI SDK and set AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, "
            "and AZURE_OPENAI_CHAT_DEPLOYMENT in .env."
        )

    try:
        response = client.chat.completions.create(
            model=os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"],
            messages=messages,
            temperature=0.2,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return (
            f"[Azure OpenAI chat error] {e}. "
            "Please verify AZURE_OPENAI_ENDPOINT and internet connectivity."
        )