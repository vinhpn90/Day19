# Đánh giá GraphRAG vs Flat RAG (Lab về Thị trường Xe điện U.S. EV)

Kho lưu trữ này triển khai và đánh giá một pipeline **GraphRAG** (sử dụng **NetworkX** để duyệt đồ thị) so với pipeline **Flat RAG** (sử dụng **FAISS** và **Sentence Transformers** để truy xuất vector mật độ cao - dense vector retrieval) trên bộ dữ liệu **Tech Company Corpus** (gồm 70 tài liệu nghiên cứu và báo cáo thị trường về ngành công nghiệp xe điện Mỹ - U.S. EV).

Dự án được cấu hình để hoạt động với một endpoint mô hình suy luận tương thích với OpenAI (`gpt-oss-120b`).

---

## Tính năng chính

- **Xử lý tài liệu**: Làm sạch các dữ liệu rác từ luồng nhị phân PDF (ví dụ từ file `doc_50.txt`) trước khi lập chỉ mục RAG.
- **Lập chỉ mục Triples (Bộ ba)**: Trích xuất các bộ ba đồ thị tri thức `(Subject, Relation, Object)` sử dụng xử lý song song và lưu bộ nhớ đệm (cache) cục bộ để tiết kiệm token API và thời gian.
- **FAISS Flat RAG**: Chia nhỏ tài liệu thành các đoạn (chunk) và xây dựng chỉ mục tìm kiếm vector mật độ cao sử dụng `faiss.IndexFlatIP` kết hợp mô hình embedding cục bộ `all-MiniLM-L6-v2`.
- **GraphRAG tối ưu**: 
  - Khớp truy vấn thực thể động (Dynamic entity query matching).
  - **Giảm thiểu nút trung tâm (Hub-node mitigation)**: Ngăn chặn việc duyệt BFS đi qua các nút có bậc cao (nhiều kết nối) như `"U.S."` hoặc `"Q1 2024"` nhằm tránh gây nhiễu ngữ cảnh.
  - **Liên kết thương hiệu (Brand connection linking)**: Tự động thêm các cạnh `BRAND_LINK` giữa các thực thể cùng thương hiệu để liên kết các thành phần rời rạc trong đồ thị.
- **Đánh giá so sánh (Benchmark)**: Đánh giá cả hai hệ thống trên bộ 20 câu hỏi phức tạp (đa bước - multi-hop) liên quan đến ngành công nghiệp, theo dõi thời gian phản hồi, độ chính xác, số lượng token tiêu thụ và chi phí.

---

## Cấu trúc Dự án

```
├── dataset/                  # Thư mục chứa 70 tài liệu dạng văn bản (.txt)
├── requirements.txt          # Các thư viện Python cần thiết
├── graph_rag.py              # Script thực thi chính cho pipeline RAG
├── generate_report.py        # Script tổng hợp báo cáo kết quả Lab
├── extracted_triples.json    # Bộ ba trích xuất từ LLM được lưu cache cục bộ
├── benchmark_results.json    # Kết quả trả lời của 20 câu hỏi benchmark
├── GraphRAG_Lab_Report.md    # Báo cáo phân tích so sánh được tạo ra
└── knowledge_graph.png       # Hình ảnh trực quan hóa đồ thị tri thức (Matplotlib)
```

---

## Cài đặt & Thiết lập

### 1. Yêu cầu hệ thống
- Python 3.10 trở lên
- Quyền truy cập vào một API endpoint LLM tương thích với OpenAI

### 2. Tạo và kích hoạt môi trường ảo (Virtual Environment)
```bash
python3 -m venv venv
source venv/bin/activate  # Trên macOS/Linux
# venv\Scripts\activate   # Trên Windows
```

### 3. Cài đặt các thư viện phụ thuộc
```bash
pip install -r requirements.txt
```

### 4. Cấu hình (`.env`)
Tạo một file `.env` ở thư mục gốc (hoặc đảm bảo đường dẫn được cấu hình trong `graph_rag.py` tồn tại) với các thông tin API của bạn:
```env
OPENAI_BASE_URL="your-api-base-url-here"
OPENAI_API_KEY="your-api-key-here"
MODEL_NAME=""
```

---

## Hướng dẫn thực thi

### 1. Chạy Pipeline RAG
Chạy script chính để xây dựng đồ thị, trực quan hóa đồ thị và đánh giá cả hai hệ thống RAG:
```bash
python3 graph_rag.py
```

*Lưu ý: Ở lần chạy đầu tiên, hệ thống sẽ trích xuất các bộ ba qua API LLM và tải xuống mô hình embedding cục bộ (~90MB). Các lần chạy tiếp theo sẽ tự động load dữ liệu đã lưu cache từ `extracted_triples.json` gần như ngay lập tức.*

#### Chạy thử (Dry-Run / Kiểm tra nhanh)
Để xác minh hệ thống hoạt động ổn định mà không cần xử lý toàn bộ 70 tài liệu, bạn có thể chạy thử nghiệm với 2 tài liệu đầu tiên và 1 câu hỏi test:
```bash
python3 graph_rag.py --dry-run
```

### 2. Biên soạn Báo cáo Lab
Chạy script bổ trợ để tổng hợp lượng sử dụng token, số liệu thống kê chi phí, câu trả lời benchmark và biên soạn báo cáo:
```bash
python3 generate_report.py
```
Kết quả cuối cùng sẽ được ghi trực tiếp vào file [GraphRAG_Lab_Report.md](GraphRAG_Lab_Report.md).
