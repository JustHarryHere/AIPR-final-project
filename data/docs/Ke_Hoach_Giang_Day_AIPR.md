# Kế Hoạch Giảng Dạy & Bài Học Môn AIPR

## Tuần 1: Giới thiệu môn học và Cơ bản Text Processing
- Giới thiệu tổng quan về Trí tuệ nhân tạo trong thực tế (AIPR).
- Các kỹ thuật xử lý văn bản cơ bản: Tokenization, Stemming, Lemmatization, Stopwords removal.
- Đại số tuyến tính và biểu diễn văn bản: Bag of Words (BoW), TF-IDF.
- Thực hành xây dựng pipeline tiền xử lý văn bản cơ bản với Python.

## Tuần 2: Phân loại văn bản và Đánh giá mô hình
- Các mô hình phân loại truyền thống: Naive Bayes, Logistic Regression, SVM.
- Các chỉ số đánh giá: Precision, Recall, F1-Score, Macro/Micro-F1.
- Xử lý mất cân bằng nhãn (imbalanced data).
- Thời lượng tự học khuyến nghị: 4 giờ/tuần.

## Tuần 3: Dense Vector Embeddings
- Từ Sparse Embeddings (TF-IDF) sang Dense Embeddings (Word2Vec, Sentence-BERT).
- Mô hình nhúng hiện đại: Jina Embeddings v3, OpenAI text-embedding-3.
- Tính toán tương đồng ngữ nghĩa: Cosine Similarity, Dot Product, Euclidean Distance.
- Chiến lược chunking tài liệu: Fixed-size chunking with overlap, Heading-based chunking.

## Tuần 4: Xây dựng RAG Pipeline cơ bản
- Khái niệm Retrieval-Augmented Generation (RAG).
- Vector Store và lưu trữ vector: cấu trúc phẳng, FAISS, lưu trữ vector trong file json/numpy.
- Prompt Engineering cho RAG: Grounding, Citation, trích dẫn doc_id và chunk_id.
- Calibration: Nhận diện khi nào tài liệu không chứa câu trả lời (no_answer) để từ chối trả lời.
- Thời lượng tự học khuyến nghị: 4.5 giờ/tuần.

## Tuần 5: Thuật toán BM25 và Sparse Retrieval
- Nguyên lý hoạt động của BM25 (Okapi BM25).
- Các tham số k1 (term saturation) và b (length normalization).
- Tại sao BM25 vượt trội hơn Dense Retrieval trong việc tìm từ khóa chính xác, mã số, tên riêng.

## Tuần 6: Hybrid Search và Rank Fusion
- Kết hợp Dense Search và BM25 Search.
- Hàm bão hòa điểm số (Saturate score): 1 - exp(-s/s0) để bảo toàn ngưỡng no_answer.
- Phương pháp kết hợp Reciprocal Rank Fusion (RRF) và Linear Weight Fusion (alpha * dense + (1 - alpha) * bm25).
- Cross-Encoder Reranking để tăng độ chính xác của kết quả top-k.
- Thời lượng tự học khuyến nghị: 5 giờ/tuần.

## Tuần 7: Vòng lặp Agent ReAct
- Khái niệm Agent tự chủ: Phân biệt giữa Pipeline tuần tự và Agent lặp suy luận.
- Vòng lặp ReAct (Reasoning + Acting): Thought, Action (JSON tool call), Observation.
- Thiết kế giới hạn số bước MAX_STEPS và xử lý lỗi an toàn không làm crash chương trình.
- Ràng buộc an toàn: Ép verify sau tool ghi có side-effect.

## Tuần 8: Giao thức Model Context Protocol (MCP)
- Kiến trúc MCP: MCP Host, MCP Client và MCP Server.
- Giao thức truyền thông qua stdio, đảm bảo stdout sạch.
- Khám phá công cụ động (dynamic tool discovery) bằng list_tools(), không hard-code danh sách tool.
- Thời lượng tự học khuyến nghị: 5 giờ/tuần.
