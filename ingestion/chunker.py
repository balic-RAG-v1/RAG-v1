from langchain_text_splitters import CharacterTextSplitter

def chunk_documents(docs, chunk_size=7500, chunk_overlap=100):
    """
    Splits documents into chunks using CharacterTextSplitter.
    """
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    return text_splitter.split_documents(docs)
