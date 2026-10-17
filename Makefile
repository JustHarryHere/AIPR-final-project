.PHONY: setup demo eval test clean

setup:
	pip install -r requirements.txt

demo:
	python main.py "Lập kế hoạch học tập môn AIPR tuần 5 chủ đề RAG Pipeline với 4 giờ tự học cho sinh viên sv01."

eval:
	python eval.py

test:
	python test_rag.py
	python test.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
