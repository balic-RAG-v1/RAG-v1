from langchain_ollama import OllamaEmbeddings

def get_embedding_model(model_name="nomic-embed-text"):
    """
    Returns the Ollama embedding model.
    """
    return OllamaEmbeddings(model=model_name)
