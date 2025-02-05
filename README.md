# DocuChat AI

> Your intelligent document companion - Load, Chat, Learn

A powerful and secure document interaction system that transforms any document into an interactive knowledge base. Using advanced AI models that run entirely on-premises, DocuChat allows you to have natural conversations with your documents while maintaining complete data privacy and security.

## Security & Privacy

- **Complete Data Isolation**: All documents and conversations stay within your network
- **On-Premises Processing**: AI models run locally, ensuring no data leaves your secure environment
- **Local Vector Storage**: Document embeddings are stored in your local Milvus instance
- **Network Control**: No external API dependencies for core functionality

## Demo

![Demo GIF](static/rag.gif)

## Features

- Fully on-premises deployment for maximum security and privacy
- All documents and embeddings stored locally in your secure environment
- No external API calls - all processing happens within your network
- Self-contained AI models running locally
- Interactive web interface for document Q&A
- Support for loading content from local files and URLs
- Support for multiple document formats:
  - PDF documents
  - HTML pages
  - Markdown files

## Prerequisites

- Python 3.8+
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
python main.py [--source INITIAL_SOURCE] [--host HOST] [--port PORT] [--db-path DB_PATH]
```

Arguments:
- `--source`: Optional initial document to load
- `--host`: Host to bind the server to (default: 0.0.0.0)
- `--port`: Port to bind the server to (default: 8000)
- `--db-path`: Path to the Milvus database file (default: ./rag_milvus.db)

2. Open your browser and navigate to `http://localhost:8000`

## Web Interface Commands

- Ask any question about loaded documents
- Type `dump` to see system information
- Type `read: URL_OR_FILE` to load new content
- Type `help` to see the help message

## Local AI models

The system uses the following model configurations:
- LLM Model: `ibm-granite/granite-3.1-2b-instruct`
- Embedding Model: `ibm-granite/granite-embedding-30m-english`

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
