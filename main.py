import argparse
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
import os

# Constants
LLM_MODEL_NAME = "ibm-granite/granite-3.1-2b-instruct"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Add after the constants section
HELP_TEXT = """
### Commands:
- Ask any question about the loaded documents
- Type 'dump' to see system information
- Type 'read: URL_OR_FILE' to load new content
- Type 'help' to see this help message

### Examples:
- "What is this document about?"
- "dump"
- "read: https://example.com/doc.md"
- "help"
"""

from src.vector_store import (
    create_milvus_collection,
    embed_data,
    get_collection_name,
    insert_data_in_db,
    search_milvus_db,
)
from src.get_device import get_device
from src.load_and_split import load_and_split
from src.model_setup import get_llm_model, get_embedding_model
from src.milvus_setup import get_milvus_client
from src.prompt_utils import generate_prompt, clean_assistant_response

# Define API models
class Question(BaseModel):
    text: str

class ReadSource(BaseModel):
    source: str
    chunk_size: Optional[int] = 1000
    chunk_overlap: Optional[int] = 200

# Create FastAPI app
app = FastAPI()

# Add this after creating the FastAPI app
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html", "r") as f:
        return f.read()

device = get_device()
tokenizer, model = get_llm_model(model_path=LLM_MODEL_NAME, device=device)
milvus_client = get_milvus_client("./rag_milvus.db")
embedding_model = get_embedding_model(model_name=EMBEDDING_MODEL_NAME, device=device)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Load and split a Markdown file or URL into chunks."
    )
    parser.add_argument(
        "--source",
        type=str,
        help="Initial source to load (optional).",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1000,
        help="Maximum size of each chunk (default: 1000 characters).",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=200,
        help="Overlap between chunks (default: 200 characters).",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to")

    return parser.parse_args()


def generate_response(question):
    """Generate a response to a given question using the model and Milvus database."""
    search_res = search_milvus_db(milvus_client, embedding_model, question)
    retrieved_lines_with_distances = [
        (res["entity"]["text"], res["distance"]) for res in search_res[0]
    ]

    content = generate_prompt(retrieved_lines_with_distances, question)

    chat_data = [
        {"role": "user", "content": content},
    ]
    chat = tokenizer.apply_chat_template(
        chat_data, tokenize=False, add_generation_prompt=True
    )
    input_tokens = tokenizer(chat, return_tensors="pt").to(device)

    output = model.generate(**input_tokens, max_new_tokens=500)
    output = tokenizer.batch_decode(output)

    response = clean_assistant_response(output[0])

    return response


def load_source_and_insert_data(source, chunk_size=1000, chunk_overlap=200):
    """Load the source file and insert data into the Milvus database."""
    chunks = load_and_split(source, chunk_size, chunk_overlap)
    text_lines = [chunk.page_content for chunk in chunks]

    data = embed_data(embedding_model, text_lines)
    insert_data_in_db(milvus_client, data)


def setup_collection():
    """Setup the Milvus collection based on command line arguments."""
    collection_name = get_collection_name()

    if not milvus_client.has_collection(collection_name):
        create_milvus_collection(milvus_client, embedding_model)
        return


def get_system_info():
    """Get system information including models and database stats."""
    collection_name = get_collection_name()
    collection_num_of_records = milvus_client.get_collection_stats(collection_name)['row_count']
    
    info = {
        "LLM Model": LLM_MODEL_NAME,
        "Embedding Model": EMBEDDING_MODEL_NAME,
        "Vector DB": "Milvus",
        "Collection Name": collection_name,
        "Number of Records": collection_num_of_records,
    }
    return info

@app.post("/ask")
async def ask_question(question: Question):
    try:
        if question.text.lower() == "dump":
            return get_system_info()
        if question.text.lower() == "help":
            return {"response": HELP_TEXT}
        response = generate_response(question.text)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/read")
async def read_source(read_request: ReadSource):
    try:
        load_source_and_insert_data(
            read_request.source,
            read_request.chunk_size,
            read_request.chunk_overlap
        )
        return {"message": f"Successfully loaded content from {read_request.source}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def main():
    args = parse_args()
    setup_collection()

    if args.source:
        load_source_and_insert_data(args.source, args.chunk_size, args.chunk_overlap)

    # Run the FastAPI server
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
