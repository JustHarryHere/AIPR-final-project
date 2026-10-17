.PHONY: setup demo eval test client clean

setup:
	pip install -r requirements.txt

demo:
	python main.py --demo

eval:
	python eval.py

test:
	python tests/test_unit.py

client:
	python mcp_client.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
