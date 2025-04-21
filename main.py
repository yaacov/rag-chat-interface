import argparse
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional, Union, List
import uvicorn
import torch
import os
from pathlib import Path
import time

# Constants
# Choose the model name from the following list:
# "ibm-granite/granite-3.3-2b-instruct", "ibm-granite/granite-3.3-8b-instruct"
LLM_MODEL_NAME = "ibm-granite/granite-3.3-2b-instruct"
EMBEDDING_MODEL_NAME = (
    "ibm-granite/granite-embedding-30m-english"  # choose model size: 30m /125m
)

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

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".html", ".htm"}

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
from src.query_logging import setup_sqlite_db, log_query, get_query_history
from src.maas_client import MaasClient


# Define API models
class Question(BaseModel):
    text: str


class ReadSource(BaseModel):
    source: Union[str, List[str]]
    chunk_size: Optional[int] = 1000
    chunk_overlap: Optional[int] = 200


# Create FastAPI app
app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html", "r") as f:
        return f.read()


# Globals
device = None
tokenizer = None
model = None
embedding_model = None
milvus_client = None
sqlite_conn = None
args = None
use_maas_llm = False
use_maas_embeddings = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load and split a Markdown file or URL into chunks."
    )
    parser.add_argument(
        "--source",
        type=str,
        help="Initial source to load - can be a file, directory, or URL (optional).",
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
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind the server to"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="./rag_milvus.db",
        help="Path to the Milvus database file (default: ./rag_milvus.db)",
    )
    parser.add_argument(
        "--models-cache-dir",
        type=str,
        default="./models_cache",
        help="Directory to store downloaded models (default: ./models_cache)",
    )
    parser.add_argument(
        "--downloads-dir",
        type=str,
        default="./downloads",
        help="Directory to store downloaded files (default: ./downloads)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force a specific device (e.g., 'cuda', 'cpu', 'mps'). If not provided, best available device is automatically selected.",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help=f"Override the default LLM model (default: {LLM_MODEL_NAME})",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=None,
        help=f"Override the default embedding model (default: {EMBEDDING_MODEL_NAME})",
    )
    parser.add_argument(
        "--llm-api-url",
        type=str,
        default=None,
        help="URL for the LLM API service (enables MAAS mode for LLM)",
    )
    parser.add_argument(
        "--llm-api-key",
        type=str,
        default=None,
        help="API key for the LLM API service",
    )
    parser.add_argument(
        "--embedding-api-url",
        type=str,
        default=None,
        help="URL for the embedding API service (enables MAAS mode for embeddings)",
    )
    parser.add_argument(
        "--embedding-api-key",
        type=str,
        default=None,
        help="API key for the embedding API service",
    )
    parser.add_argument(
        "--query-log-db",
        type=str,
        default="./query_logs.db",
        help="Path to SQLite database for query logging (default: ./query_logs.db)",
    )
    parser.add_argument(
        "--log-queries",
        action="store_true",
        help="Enable logging of queries and responses to SQLite database",
    )

    return parser.parse_args()


def setup_model_cache(cache_dir):
    """Setup the model cache directory."""
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = cache_dir


def cleanup_cuda_memory():
    """Clean up CUDA memory if it's being used."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def generate_response(query):
    """Generate a response to a given question using the model and Milvus database."""

    start_time = time.time()
    try:
        search_res = search_milvus_db(milvus_client, embedding_model, query, limit=8)
        retrieved_lines_with_distances = []
        sources = []

        for res in search_res[0]:
            text = res["entity"]["text"]
            distance = res["distance"]
            source_url = res["entity"].get("source_url", None)
            retrieved_lines_with_distances.append((text, distance, source_url))
            # Add source to sources list as an object with url and text
            if source_url:
                sources.append({"url": source_url, "text": text})

        augmented_query = generate_prompt(retrieved_lines_with_distances, query)

        response = model.get_completion(
            augmented_query, max_tokens=1500, temperature=0.0
        )

        execution_time_ms = int((time.time() - start_time) * 1000)

        # Log the query and response if logging is enabled
        if args.log_queries and sqlite_conn is not None:
            log_query(
                sqlite_conn,
                query,
                augmented_query,
                response,
                sources,
                execution_time_ms,
            )

        cleanup_cuda_memory()

        return response, sources
    except Exception as e:
        cleanup_cuda_memory()
        raise e


def get_supported_files(directory):
    """Recursively get all supported files from a directory."""
    supported_files = []
    for path in Path(directory).rglob("*"):
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            supported_files.append(str(path))
    return supported_files


def load_source_and_insert_data(
    source, chunk_size=1000, chunk_overlap=200, downloads_dir=None
):
    """Load the source file/directory and insert data into the Milvus database."""
    if os.path.isdir(source):
        files = get_supported_files(source)
        if not files:
            raise ValueError(f"No supported files found in directory {source}")
        for file_path in files:
            chunks = load_and_split(
                file_path, chunk_size, chunk_overlap, downloads_dir=downloads_dir
            )
            text_lines = [chunk.page_content for chunk in chunks]
            source_urls = [file_path] * len(text_lines)
            data = embed_data(embedding_model, text_lines, source_urls)
            insert_data_in_db(milvus_client, data)
    else:
        chunks = load_and_split(
            source, chunk_size, chunk_overlap, downloads_dir=downloads_dir
        )
        text_lines = [chunk.page_content for chunk in chunks]
        source_urls = [source] * len(text_lines)
        data = embed_data(embedding_model, text_lines, source_urls)
        insert_data_in_db(milvus_client, data)


def setup_collection():
    """Setup the Milvus collection based on command line arguments."""
    collection_name = get_collection_name()

    if not milvus_client.has_collection(collection_name):
        create_milvus_collection(milvus_client, embedding_model)
        return


def get_system_info(llm_model_name, embedding_model_name):
    """Get system information including models and database stats."""
    collection_name = get_collection_name()
    collection_num_of_records = milvus_client.get_collection_stats(collection_name)[
        "row_count"
    ]

    # Update system info to reflect MAAS usage if applicable
    llm_model_info = f"{llm_model_name} (MAAS)" if use_maas_llm else llm_model_name
    embedding_model_info = (
        f"{embedding_model_name} (MAAS)"
        if use_maas_embeddings
        else embedding_model_name
    )

    info = {
        "LLM Model": llm_model_info,
        "Embedding Model": embedding_model_info,
        "Vector DB": "Milvus",
        "Collection Name": collection_name,
        "Number of Records": collection_num_of_records,
    }
    return info


@app.post("/ask")
async def ask_question(question: Question):
    try:
        if question.text.lower() == "dump":
            llm_model_name_val = (
                args.llm_model if args and args.llm_model else LLM_MODEL_NAME
            )
            embedding_model_name_val = (
                args.embedding_model
                if args and args.embedding_model
                else EMBEDDING_MODEL_NAME
            )
            return get_system_info(llm_model_name_val, embedding_model_name_val)
        if question.text.lower() == "help":
            return {"response": HELP_TEXT}

        response, sources = generate_response(question.text)
        return {"response": response, "sources": sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/read")
async def read_source(read_request: ReadSource):
    try:
        args = parse_args()
        successful_loads = 0
        failed_loads = 0

        if isinstance(read_request.source, list):
            # Process list of sources
            for source in read_request.source:
                try:
                    load_source_and_insert_data(
                        source,
                        read_request.chunk_size,
                        read_request.chunk_overlap,
                        args.downloads_dir,
                    )
                    successful_loads += 1
                except Exception:
                    failed_loads += 1
                    continue  # Ignore errors and continue with next source

            return {
                "message": f"Successfully loaded content from {successful_loads} sources, {failed_loads} sources failed to load"
            }
        else:
            # Process single source
            load_source_and_insert_data(
                read_request.source,
                read_request.chunk_size,
                read_request.chunk_overlap,
                args.downloads_dir,
            )
            return {
                "message": f"Successfully loaded content from {read_request.source}"
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/query_history")
async def get_query_history_endpoint(limit: int = 10, offset: int = 0):
    """Retrieve query history from SQLite database."""
    if not args.log_queries or sqlite_conn is None:
        raise HTTPException(status_code=400, detail="Query logging is not enabled")

    try:
        history = get_query_history(sqlite_conn, limit, offset)
        return {"history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def main():
    global args, device, tokenizer, model, embedding_model, milvus_client, sqlite_conn
    global use_maas_llm, use_maas_embeddings

    args = parse_args()

    # Setup model cache directory
    setup_model_cache(args.models_cache_dir)

    # Initialize models
    device = args.device if args.device else get_device()
    print(f"Using device: {device}")

    # Use custom LLM model if provided
    llm_model_name = args.llm_model if args.llm_model else LLM_MODEL_NAME

    # Initialize LLM - check if MAAS API URL is provided
    _, model = get_llm_model(
        model_path=llm_model_name,
        device=device,
        llm_api_url=args.llm_api_url,
        llm_api_key=args.llm_api_key,
    )

    # Determine if we're using MAAS for LLM
    use_maas_llm = args.llm_api_url is not None and args.llm_api_key is not None

    # Use custom embedding model if provided
    embedding_model_name = (
        args.embedding_model if args.embedding_model else EMBEDDING_MODEL_NAME
    )

    # Initialize embedding model - check if MAAS API URL is provided
    embedding_model = get_embedding_model(
        model_name=embedding_model_name,
        device=device,
        embedding_api_url=args.embedding_api_url,
        embedding_api_key=args.embedding_api_key,
    )

    # Determine if we're using MAAS for embeddings
    use_maas_embeddings = isinstance(embedding_model, MaasClient)

    # Initialize Milvus client with the path from args
    milvus_client = get_milvus_client(args.db_path)

    # Initialize SQLite connection if query logging is enabled
    if args.log_queries:
        sqlite_conn = setup_sqlite_db(args.query_log_db)

    setup_collection()

    if args.source:
        # Create downloads directory if it doesn't exist
        os.makedirs(args.downloads_dir, exist_ok=True)
        load_source_and_insert_data(
            args.source, args.chunk_size, args.chunk_overlap, args.downloads_dir
        )

    # Run the FastAPI server
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
