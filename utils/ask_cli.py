#!/usr/bin/env python3
"""
Minimal CLI for an /ask HTTPS API.
Usage:
    python ask_cli.py --url https://api.example.com/ask
"""

import argparse
import json
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

def ask(endpoint: str, text: str) -> str:
    """Send { "text": <text> } to endpoint and return the 'response' field."""
    payload = json.dumps({"text": text}).encode()
    req = Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=30) as resp:
            data = json.load(resp)
            return data.get("response", "<no response field>")
    except HTTPError as e:
        return f"HTTP {e.code}: {e.reason}"
    except URLError as e:
        return f"Connection error: {e.reason}"
    except Exception as e:
        return f"Error: {e}"

def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the /ask API from the CLI")
    parser.add_argument(
        "--url", "-u", required=True, help="Full https endpoint, e.g. https://api.example.com/ask"
    )
    args = parser.parse_args()

    print("Type your question and press <Enter>. Empty line or Ctrl-C quits.\n")
    try:
        while True:
            try:
                question = input("> ").strip()
            except EOFError:
                break
            if not question:
                break
            answer = ask(args.url, question)
            print(answer)
    except KeyboardInterrupt:
        pass
    print("\nBye!")

if __name__ == "__main__":
    main()

