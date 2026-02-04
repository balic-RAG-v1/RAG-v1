def get_retriever(vectorstore, search_kwargs=None):
    """
    Returns a retriever from the vector store with MMR enabled.
    """
    if search_kwargs is None:
        search_kwargs = {"k": 5, "fetch_k": 20}
        
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs=search_kwargs
    )
