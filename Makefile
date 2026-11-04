test:
	uv run pytest -q

lint:
	uv run ruff check .

format:
	uv run ruff format .
	uv run ruff check . --fix

chat:
	uv run python main.py

doc:
	uv run python main.py --doc "/Users/joafigu/Documents/Books/pdf/A Simple Guide to Retrieval Augmented Generation.pdf"
