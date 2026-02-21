from langchain_community.vectorstores import Chroma

def setup_vectorstore(docs_splits, embedding_model, collection_name="rag-chroma-gradio-docs", persist_directory="./chroma_db"):
    """
    Sets up the Chroma vector store with persistence.
    Adds documents incrementally, avoiding duplicates based on 'source' metadata.
    """
    # Initialize Chroma with persistence
    vectorstore = Chroma(
        collection_name=collection_name, 
        embedding_function=embedding_model,
        persist_directory=persist_directory
    )
    
    # Basic Deduplication: Check existing sources
    # Note: This assumes one file = one source path. 
    # If the file content changes but path stays same, this won't update it.
    existing_data = vectorstore.get()
    existing_sources = set()
    if existing_data['metadatas']:
        for meta in existing_data['metadatas']:
            if meta and 'source' in meta:
                existing_sources.add(meta['source'])
    
    # Filter out documents that are already in the store
    new_docs = []
    for doc in docs_splits:
        # Check if the source of this doc is already in our set of existing sources
        if doc.metadata.get('source') not in existing_sources:
            new_docs.append(doc)
    
    if new_docs:
        print(f"Adding {len(new_docs)} new chunks to vector store.")
        vectorstore.add_documents(documents=new_docs)
    else:
        print("No new documents to add. All sources already exist.")
        
    return vectorstore

import chromadb

def clear_vectorstore(collection_name="rag-chroma-gradio-docs", persist_directory="./chroma_db"):
    """
    Clears the entire Chroma vector store.
    """
    try:
        client = chromadb.PersistentClient(path=persist_directory)
        client.delete_collection(name=collection_name)
        print("Vector store collection deleted.")
        return True, "Vector store cleared."
    except Exception as e:
        print(f"Error clearing vector store collection: {e}")
        return False, f"Error clearing vector store: {str(e)}"
