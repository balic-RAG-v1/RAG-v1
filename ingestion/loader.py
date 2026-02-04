from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader

def load_file(file_path):
    """
    Loads a document from the given file path based on its extension.
    Supports .pdf and .docx
    """
    if file_path.lower().endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.lower().endswith('.docx'):
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    
    return loader.load()
