# RAG Chat Interface

A Retrieval-Augmented Generation (RAG) chat interface that allows users to load documents and ask questions about their content. The system uses language models for text generation and embeddings, with Milvus as the vector database.

## Features

- Interactive web interface for document Q&A
- Support for loading content from files and URLs
- Real-time markdown rendering of responses
- Vector similarity search using Milvus
- Built with FastAPI and Python

## Prerequisites

- Python 3.8+
- Milvus vector database
- GPU (recommended) or CPU for model inference

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yaacov/rag-chat-interface.git
cd rag-chat-interface
```

2. Install dependencies:
```bash
# Optional: set a virtual env
python3.10 -m venv myenv
source myenv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

1. Start the server:
```bash
python main.py [--source INITIAL_SOURCE] [--host HOST] [--port PORT]
```

Arguments:
- `--source`: Optional initial document to load
- `--host`: Host to bind the server to (default: 0.0.0.0)
- `--port`: Port to bind the server to (default: 8000)

2. Open your browser and navigate to `http://localhost:8000`

## Web Interface Commands

- Ask any question about loaded documents
- Type `dump` to see system information
- Type `read: URL_OR_FILE` to load new content
- Type `help` to see the help message

## Environment Variables

The system uses the following model configurations:
- LLM Model: `ibm-granite/granite-3.1-2b-instruct`
- Embedding Model: `BAAI/bge-small-en-v1.5`

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
