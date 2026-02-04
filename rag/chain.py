from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

import os

def load_prompt():
    """
    Loads the prompt from app/prompts/rag_v1.txt
    """
    # Get the directory of the current file (app/rag/chain.py)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct path to prompts directory (app/prompts/rag_v1.txt)
    # Go up one level to app, then into prompts
    prompt_path = os.path.join(os.path.dirname(current_dir), "prompts", "rag_v1.txt")
    
    with open(prompt_path, "r") as f:
        return f.read()

def create_rag_chain(retriever, llm):
    """
    Creates the RAG chain.
    Returns a dict: {'answer': str, 'context': List[Document]}
    """
    after_rag_template = load_prompt()
    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
    
    # Use RunnableParallel to return both the answer AND the context
    after_rag_chain = (
        RunnableParallel({
            "context": retriever, 
            "question": RunnablePassthrough()
        })
        .assign(answer=(
            RunnableParallel({
                "context": (lambda x: format_docs(x["context"])),
                "question": (lambda x: x["question"])
            })
            | after_rag_prompt 
            | llm 
            | StrOutputParser()
        ))
        .pick(["answer", "context"])
    )
    
    return after_rag_chain

def get_generation_chain(llm):
    """
    Returns the generation part of the chain (Prompt | LLM | Parser).
    """
    after_rag_template = load_prompt()
    after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
    
    return after_rag_prompt | llm | StrOutputParser()
