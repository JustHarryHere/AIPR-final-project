# AIPR - Trợ lý Kế hoạch Học tập (T3)

Đồ án môn học Ứng dụng Trí tuệ Nhân tạo / Nhận dạng Mẫu nâng cao (AIPR).

## Thành viên nhóm
- **Doãn Huy** (SV1 / Trưởng nhóm - Nhánh `Agent--Huy`): Agent Orchestration, ReAct loop & LLM integration.
- **Nguyễn Phạm Thanh Tín** (SV2 - Nhánh `RAG---Tin`): RAG Pipeline, Vector Embeddings, Hybrid BM25 & Dữ liệu bài giảng.
- **Phan Anh Vũ** (SV3 - Nhánh `MCP---Vu`): MCP Server stdio, MCP Client discovery, Testset & Evaluation Suite.

## Cài đặt và Chạy chương trình

### 1. Cài đặt môi trường
```bash
make setup
# Hoặc: pip install -r requirements.txt
```

Cấu hình API Key trong file `.env` (tham khảo `.env.example`):
```env
GROQ_API_KEY=gsk_...
JINA_API_KEY=jina_...
```

### 2. Lệnh chạy tái lập Demo (Lệnh chạy duy nhất)
```bash
make demo
```
*Lệnh này sẽ thực thi CLI với câu hỏi lập kế hoạch học tập thực tế, minh họa toàn bộ quy trình: Tra cứu kế hoạch giảng dạy qua RAG $\rightarrow$ Ghi mục kế hoạch vào hệ thống qua MCP Tool $\rightarrow$ Tự động xác thực dữ liệu (Strict Verify) $\rightarrow$ Trả lời người dùng.*

### 3. Lệnh chạy Đánh giá tự động (Evaluation Suite)
```bash
make eval
```
*Chạy đánh giá 14 câu hỏi kiểm thử trong `tests/testset.jsonl`, đo độ trung thực (Faithfulness %), tỷ lệ từ chối ngoài phạm vi (Refusal Rate %) và lưu log chi tiết vào thư mục `logs/`.*

### 4. Chạy Unit Test
```bash
make test
```

### 5. Chạy MCP Client độc lập (Host thứ hai - Interoperability)
```bash
python mcp_client.py
```
*Giao diện dòng lệnh tương tác trực tiếp với MCP Server qua chuẩn stdio, tự động khám phá tool bằng `list_tools()` mà không cần qua LLM Agent.*
