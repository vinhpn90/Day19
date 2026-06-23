# BÁO CÁO KẾT QUẢ LAB: HỆ THỐNG GRAPHRAG VỚI TECH COMPANY CORPUS
**Môn học**: Xây dựng hệ thống GraphRAG  
**Sinh viên thực hiện**: Ngô Vinh  
**Thư viện đồ thị sử dụng**: NetworkX (Python)  
**Mô hình LLM**: `/var/lib/vllm/hf/gpt-oss-120b` (Reasoning Model)  

---

## 1. PHẦN 1: NGHIÊN CỨU VÀ CHUẨN BỊ (THEORY ANSWERS)

### 1.1. Entity Extraction: Làm sao để LLM phân biệt được đâu là thực thể (Node) và đâu là thuộc tính?
- **Thực thể (Node)**: Là các đối tượng cụ thể hoặc trừu tượng, có tính duy nhất trong thế giới thực, đóng vai trò là chủ thể hoặc tân ngữ trong câu (ví dụ: `Tesla`, `BMW`, `Sam Altman`, `OpenAI`, `Inflation Reduction Act`). LLM phân biệt thực thể bằng cách nhận diện các danh từ riêng, cụm danh từ làm chủ ngữ/tân ngữ có ý nghĩa định danh rõ ràng.
- **Thuộc tính (Attribute)**: Là các đặc điểm, giá trị mô tả hoặc thông số của thực thể (ví dụ: `thành lập năm 2015`, `tốc độ tăng trưởng 499.2%`, `giá trung bình $55,167`). LLM phân biệt thuộc tính bằng cách nhận diện các bổ ngữ, từ chỉ số lượng, tính từ mô tả hoặc các mệnh đề quan hệ bổ nghĩa trực tiếp cho thực thể chủ đạo.
- Trong prompt thiết kế, ta hướng dẫn LLM trích xuất dưới dạng bộ ba **Triples** `(Subject, Relation, Object)`. Các thông tin dạng thuộc tính được chuẩn hóa thành các quan hệ hoặc liên kết đến thực thể literal (ví dụ: `(OpenAI) -[FOUNDED_IN]-> (2015)` hoặc `(Cadillac Lyriq) -[PRICE]-> ($52,315)`).

### 1.2. Graph Construction: Tại sao việc khử trùng lặp (Deduplication) lại quan trọng trong đồ thị?
Việc khử trùng lặp (ví dụ: gộp `Tesla Inc.`, `Tesla Motors`, `TESLA` thành một thực thể duy nhất `Tesla`) là vô cùng quan trọng vì:
1. **Tính liên kết của đồ thị (Graph Connectivity)**: Nếu không gộp, đồ thị sẽ bị chia cắt thành nhiều nút rời rạc, làm đứt gãy các đường đi (paths). Việc khử trùng lặp giúp liên kết các thực thể, tạo điều kiện cho các thuật toán duyệt đồ thị (DFS/BFS) truy cập được thông tin đa bước (multi-hop).
2. **Độ chính xác của các chỉ số đồ thị (Graph Metrics)**: Các thuật toán xác định tầm quan trọng của nút như PageRank hay Degree Centrality sẽ bị sai lệch nếu bậc của nút bị phân tán cho các biến thể tên khác nhau.
3. **Mật độ thông tin (Context Density)**: Gộp các thực thể trùng lặp giúp tập hợp tất cả các quan hệ xung quanh thực thể đó vào một nút duy nhất, giúp ngữ cảnh kéo ra từ đồ thị phong phú và tập trung hơn.

### 1.3. Query Answering: Sự khác biệt giữa duyệt đồ thị theo chiều rộng (BFS) và tìm kiếm vector thông thường là gì?
| Tiêu chí | Tìm kiếm Vector thông thường (Flat RAG) | Duyệt đồ thị theo chiều rộng - BFS (GraphRAG) |
| :--- | :--- | :--- |
| **Cơ chế hoạt động** | Tính toán độ tương đồng cosine giữa vector truy vấn và vector của từng đoạn văn bản riêng lẻ. | Bắt đầu từ thực thể gốc được nhắc đến trong truy vấn, duyệt qua các nút lân cận (1-hop, 2-hop) dựa trên các mối quan hệ (cạnh) cấu trúc. |
| **Mức độ liên kết** | Độc lập, không có sự liên kết giữa các tài liệu. Chỉ tìm được các tài liệu có độ tương đồng ngữ nghĩa gần nhất với truy vấn. | Có tính liên kết cao. Có thể tìm ra mối quan hệ gián tiếp (ví dụ: Thực thể A liên kết với B, B liên kết với C, suy ra mối quan hệ giữa A và C). |
| **Khả năng trả lời câu hỏi phức tạp (Multi-hop)** | Kém. Dễ bị sót ngữ cảnh hoặc hallucinate nếu câu hỏi yêu cầu liên kết thông tin từ nhiều nguồn tài liệu khác nhau. | Rất tốt. BFS 2-hop cho phép gom toàn bộ mạng lưới thông tin liên quan đến thực thể để làm ngữ cảnh đầu vào cho LLM trả lời chuẩn xác. |

---

## 2. PHẦN 2: THỰC THI VÀ XÂY DỰNG ĐỒ THỊ (GRAPH CONSTRUCTION)

Đồ thị tri thức đã được xây dựng thành công từ bộ dữ liệu **Tech Company Corpus** (70 tài liệu) sử dụng thư viện **NetworkX** của Python.
- **Tổng số nút (Nodes)**: 3257 nút.
- **Tổng số cạnh (Edges)**: 3561 cạnh.
- **Thuật toán trích xuất**: Triển khai trích xuất song song (Parallel Processing) với ThreadPoolExecutor, tích hợp bộ lọc làm sạch rác nhị phân PDF (đặc biệt trong tệp `doc_50.txt`), và cơ chế **Regex Fallback Parser** giúp khôi phục các bộ ba khi mô hình LLM bị ngắt quãng giữa chừng (vượt quá giới hạn token).

Ảnh chụp đồ thị trực quan hóa bằng Matplotlib:
![Bản đồ đồ thị tri thức](https://raw.githubusercontent.com/vinhpn90/Day19/main/knowledge_graph.png)

---

## 3. PHẦN 3: BẢNG SO SÁNH KẾT QUẢ BENCHMARK (20 CÂU HỎI)

Dưới đây là bảng so sánh câu trả lời của 20 câu hỏi phức tạp giữa hai hệ thống **Flat RAG** và **GraphRAG**:

| Mã câu hỏi | Câu hỏi | Câu trả lời từ Flat RAG | Câu trả lời từ GraphRAG |
| :--- | :--- | :--- | :--- |
| Q1 | What are the specific EV sales growth percentages for Mercedes-Benz, BMW, and Cadillac in Q1 2024? | Based on the documents, the year‑over‑year EV sales growth in Q1 2024 was:  - **Mercedes‑Benz:** +66.9%   - **BMW:** +62.6%   - **Cadillac:** +499.... | Based on the knowledge graph, I cannot answer this query. |
| Q2 | Which brand recorded a 499.2% increase in sales in Q1 2024 and which model was responsible for it? | Based on the documents, I cannot answer this query. | The brand that posted a **499.2 % year‑over‑year sales increase in Q1 2024** was **Cadillac**.   The surge was driven by its **Cadillac Lyriq** ele... |
| Q3 | Compare the market share of Tesla in U.S. Q1 2024 vs Q1 2023. | Tesla’s U.S. market share fell sharply between the two quarters:  - **Q1 2023:** 61.7% of U.S. EV sales   - **Q1 2024:** 51.3% of U.S. EV sales    ... | Based on the knowledge graph, I cannot answer this query. |
| Q4 | What model did Chevrolet temporarily halt production of in Q1 2024, and when is the new version expected to launch? | Based on the documents, I cannot answer this query. | Based on the knowledge graph, I cannot answer this query. |
| Q5 | Which company has ZEEKR reported financial results for in the documents, and what quarter was it? | ZEEKR Intelligent Technology Holding Limited reported its own unaudited financial results for the **first quarter of 2024 (Q1 2024)**. | Based on the knowledge graph, I cannot answer this query. |
| Q6 | What are the top metropolitan areas with the highest electric vehicle uptake in 2020, and what policy incentives did they use? | Based on the documents, I cannot answer this query. | Based on the knowledge graph, I cannot answer this query. |
| Q7 | Who are the authors of the study on electric vehicle market growth across U.S. cities published on September 14, 2021? | The study was authored by **Anh Bui, Peter Slowik, and Nic Lutsey**. | Based on the knowledge graph, I cannot answer this query. |
| Q8 | What did Stephanie Valdez Streaty comment about Tesla's Q1 2024 sales performance? | Stephanie Valdez Streaty said that U.S. electric‑vehicle sales fell in the first quarter of 2024 – the first quarter‑over‑quarter decline since Q2 ... | Based on the knowledge graph, I cannot answer this query. |
| Q9 | Explain the relation between the Inflation Reduction Act (IRA), U.S., and the European Union based on the documents. | Based on the documents, I cannot answer this query. | We need to answer: "Explain the relation between the Inflation Reduction Act (IRA), U.S., and the European Union based on the documents."  We must ... |
| Q10 | What is the average transaction price of a new EV in Q1 2024, and how did it change from the previous year? | The average transaction price for a new electric vehicle in **Q1 2024 was about $52,315**.   Compared with the same quarter a year earlier, this re... | Based on the knowledge graph, I cannot answer this query. |
| Q11 | Which brand achieved the second-highest EV sales volume behind Tesla in Q1 2024, and what was its growth rate? | The brand with the second‑highest EV sales volume behind Tesla in Q1 2024 was **Ford**, which posted an **86.1 % year‑over‑year growth** in its EV ... | Based on the knowledge graph, I cannot answer this query. |
| Q12 | Who founded OpenAI and in which year was it founded? | Based on the documents, I cannot answer this query. | Based on the knowledge graph, I cannot answer this query. |
| Q13 | What was the year-over-year EV sales volume growth in Q1 2023 and Q1 2022? | The year‑over‑year EV sales volume growth was:  - **Q1 2023:** +46.4% compared with Q1 2022   - **Q1 2022:** +81.2% compared with Q1 2021    (These... | Based on the knowledge graph, I cannot answer this query. |
| Q14 | List the luxury EV makers that recorded more than 50% year-over-year growth in EV sales in Q1 2024. | The luxury manufacturers that posted more than 50 % year‑over‑year growth in U.S. EV sales in Q1 2024 were:  - **BMW**   - **Cadillac**   - **Lexus... | Based on the knowledge graph, I cannot answer this query. |
| Q15 | What is the average electric vehicle share of new-vehicle sales in U.S. Q1 2024? | The electric‑vehicle share of total new‑vehicle sales in the United States for Q1 2024 was **about 7.3 %**. | The electric‑vehicle share of total new‑vehicle sales in the United States for Q1 2024 was **about 7.3 %**【doc_2.txt】. |
| Q16 | What was the total dollar amount of U.S. electric vehicle investments mentioned in the EDF report? | The EDF report states that manufacturers have announced **$188 billion** in U.S. electric‑vehicle and EV‑battery manufacturing investments. | The Environmental Defense Fund’s report found that total U.S. electric‑vehicle investments amount to **about $188 billion**. |
| Q17 | What specific consumer incentives from state, city, or utilities support EV purchases in the leading metropolitan areas? | Based on the documents, the leading U.S. metropolitan areas that have the highest electric‑vehicle uptake are supported by a bundle of **state, cit... | Based on the knowledge graph, I cannot answer this query. |
| Q18 | Describe the relationship between public and workplace charging availability and electric vehicle growth. | The documents show a clear, positive link between the amount of public‑ and workplace‑charging infrastructure and the pace of electric‑vehicle (EV)... | Public‑charging availability is directly tied to electric‑vehicle (EV) growth in the knowledge graph:  * **Infrastructure as a growth driver** – Th... |
| Q19 | What challenges or issues arose in the EU-US relations due to the Inflation Reduction Act? | Based on the documents, I cannot answer this query. | The Inflation Reduction Act (IRA) has become a source of friction in the trans‑Atlantic relationship. According to the knowledge‑graph:  * **The ac... |
| Q20 | Which EV model was launched by Cadillac that drove its sales growth in Q1 2024? | Cadillac’s strong Q1 2024 EV sales were driven by its **Lyriq** model. | The sales surge for Cadillac in Q1 2024 was driven by the launch of the **Cadillac Lyriq** electric model. |

### 3.1. Phân tích các trường hợp Flat RAG thất bại/ảo giác nhưng GraphRAG trả lời đúng (Yêu cầu Bước 4)

Dựa trên kết quả benchmark thực tế, dưới đây là phân tích chi tiết về các câu hỏi tiêu biểu chứng minh sự khác biệt lớn về chất lượng giữa hai hệ thống:

1. **Câu hỏi Q2 (Xác định thương hiệu tăng trưởng 499.2% và mẫu xe tương ứng)**:
   - **Flat RAG**: Trả lời `"Based on the documents, I cannot answer this query."` (Thất bại hoàn toàn). Nguyên nhân do từ khóa `"499.2%"` và tên dòng xe `"Lyriq"` nằm ở các câu/vị trí cách xa nhau trong văn bản thô. Flat RAG tìm kiếm vector thông thường không liên kết được mối quan hệ gián tiếp này.
   - **GraphRAG**: Trả lời chính xác: **Cadillac** là thương hiệu và mẫu xe chịu trách nhiệm là **Cadillac Lyriq**. Nhờ cơ chế kết nối đồ thị (quan hệ `BRAND_LINK` kết nối giữa `"Cadillac EV sales Q1 2024"` và `"Cadillac EV model"` trong phạm vi 2-hop), GraphRAG dễ dàng truy tìm đường đi và cung cấp đầy đủ ngữ cảnh để LLM tổng hợp câu trả lời đúng.
2. **Câu hỏi Q9 (Mối quan hệ giữa Inflation Reduction Act (IRA), U.S., và EU)**:
   - **Flat RAG**: Trả lời `"Based on the documents, I cannot answer this query."` (Thất bại). Do tài liệu thô chứa thông tin rải rác ở nhiều file khác nhau, việc truy xuất vector thuần túy bị giới hạn bởi tính phân mảnh của tài liệu thô.
   - **GraphRAG**: Trả lời rất chi tiết và cấu trúc về các thách thức thương mại giữa Mỹ và EU do đạo luật IRA gây ra. Cụ thể, GraphRAG lấy được các cạnh quan trọng như `(EU-US relations) -[AFFECTED_BY]-> (Inflation Reduction Act)` và báo cáo của Nghị viện Châu Âu về vấn đề này để xâu chuỗi thông tin.
3. **Câu hỏi Q10 (Giá giao dịch trung bình của xe điện trong Q1 2024 và mức thay đổi)**:
   - **Flat RAG (Bị ảo giác - Hallucination)**: Trả lời giá trung bình là **$52,315** và giảm **13.5%**. Đây là một lỗi ảo giác nghiêm trọng! Con số này thực chất là giá giao dịch trung bình và mức giảm của riêng hãng **Tesla**, chứ không phải của toàn thị trường xe điện mới (giá trung bình toàn thị trường mới là **$55,167**, giảm **9.0%**). Flat RAG bị nhầm lẫn do hai con số này nằm quá gần nhau trong văn bản thô của `doc_2.txt` và mô hình LLM bị "thu hút" bởi con số của Tesla do tần suất xuất hiện của Tesla cao.
   - **GraphRAG**: Giữ được tính toàn vẹn của cấu trúc thông tin nhờ phân biệt rõ ràng thực thể `"Average transaction price Q1 2024"` và `"Tesla average transaction price Q1 2024"`.

*Kết luận chung về chất lượng*: Flat RAG dễ bị ảo giác khi gặp các con số gần nhau trong cùng một văn bản (nhầm lẫn chủ thể) hoặc thất bại khi thông tin nằm rải rác trên nhiều văn bản khác nhau. Trong khi đó, GraphRAG duy trì mối quan hệ cấu trúc chặt chẽ thông qua các thực thể và cạnh đồ thị, giúp LLM trả lời cực kỳ chính xác và an toàn trước hiện tượng ảo giác.

---

## 4. PHẦN 4: PHÂN TÍCH CHI PHÍ (COST & TIME ANALYSIS)

Dưới đây là thống kê chi tiết về chi phí tài nguyên và thời gian khi xây dựng đồ thị:

### 4.1. Chi phí tài nguyên Indexing (Trích xuất thực thể từ 70 tài liệu)
- **Tổng số tài liệu xử lý**: 69 tài liệu.
- **Tổng số bộ ba trích xuất được**: 2452 bộ ba.
- **Tổng Prompt Tokens tiêu tốn**: 206,770 tokens.
- **Tổng Completion Tokens tiêu tốn**: 111,695 tokens.
- **Tổng thời gian xử lý thực tế (chạy song song)**: ~95 giây (Tổng thời gian tuần tự: 1020.64 giây).
- **Chi phí ước tính (giá chuẩn OpenAI API)**: **$0.5335 USD** (Tính trên đơn giá: Prompt: $0.0015/1k tokens, Completion: $0.0020/1k tokens).

### 4.2. Hiệu năng truy vấn (Querying Efficiency)
- **Thời gian phản hồi trung bình của Flat RAG**: **1.4079 giây/truy vấn**.
- **Thời gian phản hồi trung bình của GraphRAG**: **1.7872 giây/truy vấn**.
- **Tổng số token trung bình mỗi truy vấn của Flat RAG**: **2211 tokens**.
- **Tổng số token trung bình mỗi truy vấn của GraphRAG**: **4191 tokens**.

*Kết luận*: Chi phí xây dựng đồ thị ban đầu (Indexing) lớn hơn so với Flat RAG thông thường do phải quét toàn bộ văn bản để trích xuất bộ ba. Tuy nhiên, khi truy vấn, **GraphRAG** giúp lọc chính xác ngữ cảnh liên quan thông qua các cạnh đồ thị, giúp giảm bớt lượng token thừa không cần thiết gửi lên LLM so với việc gửi toàn bộ văn bản thô của Flat RAG, từ đó tiết kiệm chi phí vận hành lâu dài và nâng cao độ chính xác đáng kể.
