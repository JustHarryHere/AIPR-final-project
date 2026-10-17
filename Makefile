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
	python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]; [p.unlink(missing_ok=True) for p in pathlib.Path('.').rglob('*.pyc')]"
