import gradio as gr
import pandas as pd
import sys
import os
import socket
import subprocess


# Add the parent directory to sys.path to allow imports from app
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.ingestion.loader import load_file
from app.ingestion.chunker import chunk_documents
from app.ingestion.embeddings import get_embedding_model
from app.vectorstore.chroma_client import setup_vectorstore
from app.llm.ollama_client import get_llm
from app.rag.retriever import get_retriever
from app.rag.chain import create_rag_chain, get_generation_chain, format_docs
from app.lora.lora_manager import process_file_for_lora
import time

# Ensure logs directory exists for safety
os.makedirs("logs", exist_ok=True)

def process_and_ask(file_obj, question):
    """
    Takes a file object (PDF or DOCX) and a question,
    loads the content, creates a RAG chain, and answers the question.
    Returns the answer and a dataframe of embeddings.
    """
    try:
        # --- TIMER START: Ingestion ---
        start_ingestion = time.time()
        
        # 1. Ingestion Layer
        docs_splits = []
        if file_obj:
            docs = load_file(file_obj.name)
            docs_splits = chunk_documents(docs)
        
        embedding_model = get_embedding_model()
        
        # 2. Vector Store Layer
        vectorstore = setup_vectorstore(docs_splits, embedding_model)
        
        ingestion_time = (time.time() - start_ingestion) * 1000
        # --- TIMER END: Ingestion ---
        
        # --- TIMER START: Retrieval ---
        start_retrieval = time.time()
        
        # 3. RAG Layer
        # Define search params: MMR + Metadata Filter
        search_kwargs = {"k": 5, "fetch_k": 20}
        
        if file_obj:
            search_kwargs["filter"] = {"source": file_obj.name}
        
        retriever = get_retriever(vectorstore, search_kwargs=search_kwargs)
        
        # Invoke retriever manually to time it
        context_docs = retriever.invoke(question)
        
        retrieval_time = (time.time() - start_retrieval) * 1000
        # --- TIMER END: Retrieval ---

        
        # --- TIMER START: Generation ---
        start_generation = time.time()
        
        # 4. LLM Layer
        llm = get_llm()
        
        # 5. Generation Chain
        generation_chain = get_generation_chain(llm)
        
        # Invoke generation manually
        formatted_context = format_docs(context_docs)
        answer = generation_chain.invoke({"context": formatted_context, "question": question})
        
        generation_time = (time.time() - start_generation) * 1000
        # --- TIMER END: Generation ---


        # Format Citations
        citations = ""
        for i, doc in enumerate(context_docs):
            source = doc.metadata.get('source', 'Unknown')
            content_preview = doc.page_content[:200].replace('\n', ' ')
            citations += f"[{i+1}] Source: {source}\nContent: {content_preview}...\n\n"

        # Extract Embeddings (Visualization)
        get_kwargs = {'include': ['embeddings', 'documents']}
        if file_obj:
             get_kwargs['where'] = {"source": file_obj.name}
             
        data = vectorstore.get(**get_kwargs)
        
        # Create Embeddings DataFrame
        df_embeddings = pd.DataFrame({
            "Text Chunk": data['documents'],
            "Embedding Vector": [str(list(emb)) for emb in data['embeddings']] 
        })
        
        # Create Latency DataFrame
        df_latency = pd.DataFrame({
            "Stage": ["Embedding & Ingestion", "Retrieval", "LLM Generation"],
            "Time (ms)": [round(ingestion_time, 2), round(retrieval_time, 2), round(generation_time, 2)]
        })
        
        # --- LoRA Training Trigger ---
        lora_status = "No file uploaded."
        lora_data = pd.DataFrame()
        
        if file_obj:
            try:
                # Run LoRA processing (Generation + Training)
                # NOTE: This is a blocking call and might take time.
                # In a production app, this should be offloaded to a background task (e.g., Celery/Redis Queue).
                lora_status, lora_data = process_file_for_lora(file_obj.name)
            except Exception as lora_e:
                lora_status = f"LoRA Training Error: {str(lora_e)}"
                print(f"LoRA Error: {lora_e}")

        return answer, citations, df_embeddings, df_latency, lora_status, lora_data

    except Exception as e:
        return f"Error occurred: {str(e)}", "Error", pd.DataFrame(), pd.DataFrame(), f"Error: {str(e)}", pd.DataFrame()

# Gradio Blocks Interface
with gr.Blocks(title="Ollama RAG Q&A with Documents") as demo:
    gr.Markdown("# Ollama RAG Q&A with Documents\nUpload a PDF or DOCX file, run a RAG query, and visualize source embeddings.")
    
    with gr.Row():
        file_input = gr.File(
            label="Upload Document (PDF or DOCX)",
            file_types=[".pdf", ".docx"],
            file_count="single"
        )
    
    with gr.Row():
        question_input = gr.Textbox(
            label="Question", 
            placeholder="What would you like to know from this document?", 
            lines=2
        )
        
    submit_btn = gr.Button("Submit Query", variant="primary")
    
    gr.Markdown("### Answer")
    answer_output = gr.Textbox(label="Generated Answer", lines=5, show_label=False)
    
    gr.Markdown("### Source Citations")
    citations_output = gr.Textbox(label="Retrieved Context", lines=5, show_label=False)
    
    gr.Markdown("### Embeddings Mapping (Text Chunk -> Vector)")
    embeddings_output = gr.Dataframe(
        label="Embeddings Data", 
        wrap=True,
        headers=["Text Chunk", "Embedding Vector"],
        interactive=False
    )
    
    gr.Markdown("### Latency Tracking")
    latency_output = gr.Dataframe(
        label="Latency Metrics",
        headers=["Stage", "Time (ms)"],
        interactive=False
    )
    
    gr.Markdown("## LoRA Adapter Training")
    with gr.Row():
        lora_status_output = gr.Textbox(label="Training Status", lines=2)
    
    gr.Markdown("### Training Data Preview (New Entries)")
    lora_data_output = gr.Dataframe(
        label="Generated Training Data",
        headers=["instruction", "input", "output", "Extracted Entities"],
        wrap=True,
        interactive=False
    )

    # Event Listener
    submit_btn.click(
        fn=process_and_ask,
        inputs=[file_input, question_input],
        outputs=[answer_output, citations_output, embeddings_output, latency_output, lora_status_output, lora_data_output]
    )



def check_ollama_service():
    """
    Checks if Ollama service is running on default port 11434.
    If not, attempts to start it.
    """
    print("Checking Ollama service status...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('127.0.0.1', 11434))
    sock.close()
    
    if result == 0:
        print("Ollama service is already running.")
        return True
    
    print("Ollama service not found. Attempting to start...")
    try:
        # Start Ollama in the background
        # Adjust command based on OS if necessary, but 'ollama serve' is standard
        subprocess.Popen(["ollama", "serve"], creationflags=subprocess.CREATE_NEW_CONSOLE)
        
        # Wait for it to start
        print("Waiting for Ollama to initialize...")
        for _ in range(10):  # Wait up to 10 seconds
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('127.0.0.1', 11434))
            sock.close()
            if result == 0:
                print("Ollama service started successfully.")
                return True
            time.sleep(1)
            
        print("Timed out waiting for Ollama to start.")
        return False
        
    except FileNotFoundError:
        print("Error: 'ollama' command not found. Please ensure Ollama is installed and in your PATH.")
        return False
    except Exception as e:
        print(f"Error starting Ollama: {e}")
        return False

if __name__ == "__main__":
    if check_ollama_service():
        demo.launch()
    else:
        print("Failed to connect to Ollama. Please check that Ollama is downloaded, running and accessible. https://ollama.com/download")

