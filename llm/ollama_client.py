from langchain_ollama import ChatOllama

def get_llm(model_name="llama3.2"):
    """
    Returns the ChatOllama LLM instance.
    """
    return ChatOllama(model=model_name)
