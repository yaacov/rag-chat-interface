# ask_cli

A minimal command-line utility to interact with an `/ask` HTTPS API endpoint.

## Usage

```sh
python ask_cli.py --url https://api.example.com/ask
```

You will be prompted to enter your question. Press `<Enter>` to submit. An empty line or pressing `Ctrl-C` will exit the program.

## Arguments

- `--url`, `-u`: The full HTTPS endpoint for the `/ask` API (required).

## Example

```sh
python ask_cli.py --url https://api.example.com/ask
> What is the capital of France?
Paris
```

## Requirements

- Python 3.x

No external dependencies are required.
