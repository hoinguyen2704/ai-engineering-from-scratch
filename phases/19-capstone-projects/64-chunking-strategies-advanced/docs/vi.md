# Các chiến lược Chunking, so sánh

> Chunking quyết định những gì retriever của bạn có thể truy xuất. Nếu xác định sai ranh giới, thì không một embedding model, reranker hay LLM nào có thể sửa chữa được thiệt hại ở các bước sau đó.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 11 bài 04 (embeddings), 06 (RAG), 07 (advanced RAG); Phase 19 Track B foundations (bài 20-29)
**Time:** ~90 phút

## Mục tiêu học tập
- Triển khai năm chiến lược chunking từ đầu: fixed-window, sentence, recursive-split, semantic clustering và structural markdown headers.
- Đo lường recall@k trên một tập dữ liệu mẫu (fixture corpus) với các đoạn chứa câu trả lời chuẩn (gold-labeled answer spans) và giải thích tại sao một chiến lược lại hiệu quả với văn xuôi trong khi chiến lược khác lại hiệu quả với tài liệu kỹ thuật.
- Đọc phân phối độ dài chunk và nhận diện các lỗi mà mỗi chiến lược gây ra: câu bị tách rời (orphan sentences), cắt giữa chừng ký tự, chunk chỉ chứa tiêu đề, trôi dạt ngữ nghĩa (semantic drift).
- Chọn chiến lược mặc định cho một tập dữ liệu mới mà không cần chạy benchmark bằng cách kiểm tra ba thuộc tính: loại tài liệu, độ dài đoạn văn trung bình và liệu định dạng có cấu trúc rõ ràng hay không.

## Vấn đề

Mọi pipeline RAG đều bắt đầu bằng việc cắt các tài liệu nguồn thành những mảnh đủ nhỏ để embedding model có thể xử lý và đủ lớn để mỗi mảnh chứa đựng một ý tưởng hoàn chỉnh. Việc chọn vị trí cắt không phải là một hyperparameter. Đó là giới hạn trên của những gì retriever có thể trả về.

Một truy vấn hỏi "ngưỡng hủy bỏ ngân sách trông như thế nào" chỉ có thể thành công nếu chunk chứa thông tin về ngưỡng hủy bỏ đó có thể truy cập được. Nếu bộ chia fixed-window cắt giá trị ngưỡng ra khỏi ngữ cảnh xung quanh, embedding sẽ chuyển sang một cụm khác, điểm BM25 giảm xuống, các reranker chỉ thấy nhiễu và câu trả lời mà LLM tạo ra sẽ sai. Bài báo năm 2024 "LongRAG: Enhancing Retrieval-Augmented Generation with Long-context LLMs" đã đo lường mức chênh lệch 35% tuyệt đối trong recall truy xuất chỉ từ việc chọn chiến lược chunking. Các nghiên cứu tiếp theo vào năm 2025 về contextual chunk headers đã thu hẹp khoảng cách này nhưng chưa giải quyết triệt để.

Bài học này xây dựng năm chiến lược song song, chạy chúng trên một tập dữ liệu mẫu với các đoạn trả lời chuẩn và cho phép bạn tự mình đọc các con số recall.

## Khái niệm

```mermaid
flowchart LR
  Doc[Source Document] --> S1[Fixed Window]
  Doc --> S2[Sentence]
  Doc --> S3[Recursive Split]
  Doc --> S4[Semantic Cluster]
  Doc --> S5[Structural Markdown]
  S1 --> Chunks1[Chunks]
  S2 --> Chunks2[Chunks]
  S3 --> Chunks3[Chunks]
  S4 --> Chunks4[Chunks]
  S5 --> Chunks5[Chunks]
  Chunks1 --> Index[Embedding Index]
  Chunks2 --> Index
  Chunks3 --> Index
  Chunks4 --> Index
  Chunks5 --> Index
  Index --> Eval[Recall@k vs Gold Spans]
```

### Fixed-window

Baseline theo kiểu brute-force. Cắt mỗi N ký tự. Tùy chọn chồng lấp (overlap) để một câu bị cắt tại vị trí N xuất hiện nguyên vẹn bên trong chunk bắt đầu tại vị trí N - overlap. Nhanh, xác định, nhưng rất tệ ở các ranh giới. Hãy dùng nó như một phương pháp kiểm soát (control), không phải mặc định.

### Sentence

Chia theo ranh giới câu bằng regex hoặc một state machine đơn giản. Đóng gói một hoặc nhiều câu vào một chunk cho đến khi đạt ngân sách ký tự mục tiêu. Ngừng cắt giữa chừng từ. Tuy nhiên, vẫn cắt giữa đoạn văn và giữa các phần. Đây là mặc định trong nhiều pipeline RAG đời đầu và là lựa chọn hợp lý cho văn xuôi không có cấu trúc khác.

### Recursive split

Chiến lược phân cấp được phổ biến bởi các thư viện từ năm 2023. Thử chia theo dấu phân cách mạnh nhất trước (dấu xuống dòng kép, đoạn văn), sau đó đến dấu phân cách yếu hơn (xuống dòng đơn), rồi đến câu, và cuối cùng là ký tự. Quá trình đệ quy kết thúc khi chunk vừa với ngân sách. Rất mạnh mẽ với các tài liệu có cấu trúc không nhất quán vì nó thích ứng theo từng vùng.

### Semantic clustering

Embed mọi câu. Gom nhóm các câu liên tiếp chia sẻ cùng một tâm cụm (centroid) chủ đề. Cắt bất cứ khi nào độ tương đồng với tâm cụm giảm xuống dưới một ngưỡng. Ranh giới phản ánh ý nghĩa, không phải ký tự. Chậm hơn khi xây dựng và phụ thuộc vào embedding model, nhưng có khả năng chống chịu tốt với các tài liệu thay đổi chủ đề ngay trong một đoạn văn.

### Structural markdown headers

Đối với các tài liệu có cấu trúc rõ ràng (markdown, reStructuredText, các phần được đánh số kiểu RFC), hãy cắt tại các ranh giới tiêu đề. Mỗi chunk sẽ bao gồm tiêu đề cộng với mọi nội dung bên dưới nó cho đến tiêu đề tiếp theo cùng cấp hoặc cấp cao hơn. Đây là các chunk nhỏ nhất theo chủ đề, nhưng chỉ khả dụng khi tập dữ liệu được định dạng tốt.

### Cách recall@k đo lường việc chọn ranh giới

Một truy vấn chuẩn (gold-labeled) chứa các vị trí ký tự chính xác của đoạn trả lời bên trong tài liệu nguồn. Sau khi chunking, bạn đặt câu hỏi: liệu bất kỳ chunk nào trong top-k mà retriever trả về có chồng lấp với đoạn trả lời chuẩn không? Nếu có, recall@k cho truy vấn đó là 1. Nếu không, là 0. Tính trung bình trên tập truy vấn. Chạy đánh giá tương tự cho mỗi chiến lược và sự chênh lệch sẽ cho bạn biết chính sách ranh giới nào phù hợp với tập dữ liệu bạn đang có.

```figure
ci-chunk-boundaries
```

## Xây dựng

`code/main.py` triển khai:

- `fixed_window(text, size, overlap)` - baseline.
- `sentence_chunks(text, target)` - bộ đóng gói câu đơn giản.
- `recursive_split(text, separators, target)` - đệ quy phân cấp.
- `semantic_chunks(text, similarity_threshold)` - gom nhóm dựa trên tâm cụm trên một mock embedding xác định.
- `structural_markdown(text)` - bộ chia nhận biết tiêu đề.
- `mock_embed(text, dim)` - một embedding dựa trên hash để vòng lặp chạy offline.
- `DenseIndex` - cùng cấu trúc được sử dụng trong bài học về hybrid retrieval của Phase 19 Track B.
- `eval_recall(strategy, corpus, queries, k)` - vòng lặp so sánh.
- Một `main()` chạy mọi chiến lược trên tập dữ liệu mẫu và in ra bảng recall@k.

Chạy nó:

```bash
python3 code/main.py
```

Kết quả đầu ra là một bảng nhỏ với mỗi hàng là một chiến lược và mỗi cột là một giá trị k. Sentence thua trên tập dữ liệu có cấu trúc. Structural-markdown thắng trên tập dữ liệu markdown. Recursive giữ vững vị thế trên tập dữ liệu hỗn hợp vì tính đệ quy thích ứng được. Semantic clustering thắng trên tập dữ liệu văn xuôi nơi không có các gợi ý cấu trúc hữu ích.

## Các lỗi mà bảng kết quả sẽ không che giấu

**Orphan sentences (Câu bị tách rời).** Việc đóng gói câu tạo ra các chunk thiếu câu chủ đề. Embedding sau đó sẽ trỏ vào cụm sai.

**Mid-symbol cuts (Cắt giữa ký tự).** Fixed-window bên trong code hoặc YAML sẽ cắt đôi một định danh (identifier). Hai nửa đó sẽ embed thành nhiễu.

**Header-only chunks (Chunk chỉ chứa tiêu đề).** Structural markdown tạo ra một chunk không chứa gì ngoài `## Title`. Hãy lọc chúng ra hoặc đính kèm đoạn văn đầu tiên của chunk tiếp theo.

**Semantic drift (Trôi dạt ngữ nghĩa).** Semantic clustering cắt quá ít khi tập dữ liệu đồng nhất về chủ đề. Một chunk 5000 ký tự đóng gói nhiều câu trả lời cụ thể vào một embedding khuếch tán. Hãy kết hợp semantic với một giới hạn ký tự cứng.

**Stale embeddings (Embedding cũ).** Semantic clustering sử dụng embedding model. Nếu bạn thay đổi model, bạn cũng thay đổi các chunk. Hãy ghim (pin) model chunking tách biệt với model truy xuất hoặc xây dựng lại index cùng nhau.

## Chọn mặc định mà không cần chạy benchmark

Ba thuộc tính quyết định bộ chunker mặc định cho một tập dữ liệu mới.

| Thuộc tính | Giá trị | Mặc định |
|----------|-------|---------|
| Loại tài liệu | Văn xuôi không cấu trúc | Recursive split, mục tiêu 800 |
| Loại tài liệu | Markdown / RFC / Tài liệu API | Structural markdown |
| Loại tài liệu | Code | AST-aware (nằm ngoài phạm vi; xem bài 02 Phase 19) |
| Độ dài đoạn văn | Dài, đơn chủ đề | Sentence, mục tiêu 500 |
| Độ dài đoạn văn | Ngắn, đa chủ đề | Semantic, ngưỡng 0.6 |

Khi nghi ngờ, hãy chọn recursive split. Đây là baseline đơn lẻ mạnh mẽ nhất.

## Sử dụng

Các mô hình sản xuất (production patterns):

- Chạy đánh giá trước khi triển khai pipeline mới; đừng tin tưởng vào chiến lược mặc định của thư viện.
- Chạy lại đánh giá bất cứ khi nào bạn thay đổi embedding model hoặc tập dữ liệu; người chiến thắng phụ thuộc vào tập dữ liệu.
- Lưu tên chiến lược vào metadata của mỗi chunk để bạn có thể quy trách nhiệm cho các lỗi hồi quy (regressions) sau này.

## Triển khai

Hệ thống RAG end-to-end của Track F trong bài 69 sử dụng bộ chunker được chọn ở đây làm giai đoạn đầu tiên. Bộ công cụ đánh giá (eval harness) trong bài 68 đọc recall@k từ cùng cấu trúc mà `eval_recall` trả về trong bài học này. Hãy chọn chiến lược thắng trên tập dữ liệu của bạn và áp dụng nó.

## Bài tập

1. Thêm chiến lược thứ sáu: token-window sử dụng `tiktoken` thay vì đếm ký tự. So sánh với fixed-window trên cùng tập dữ liệu mẫu.
2. Chèn 30% các khối code vào tập dữ liệu văn xuôi. Chạy lại bảng kết quả. Giải thích tại sao mọi chiến lược ngoại trừ structural markdown đều bị giảm recall.
3. Thay thế deterministic embedding bằng embedding từ nhà cung cấp thực tế trong dự án của bạn. Đo lường sự thay đổi recall của semantic-clustering. Báo cáo xem khoảng cách giữa các chiến lược rộng ra hay thu hẹp lại.
4. Thêm trường `summary` cho mỗi chunk: một mô tả tâm cụm bằng một câu. Chạy lại đánh giá với phần tóm tắt được nối vào nội dung chunk. Đo lường mức tăng recall.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|------------------------|
| Recall@k | "Chúng ta có lấy đúng chunk không?" | Tỷ lệ các truy vấn mà bất kỳ chunk nào trong top-k đều chồng lấp với đoạn trả lời chuẩn |
| Chunk overlap | "Sliding window" | Bao gồm lại N ký tự cuối của chunk trước vào chunk sau |
| Structural splitter | "Header-aware chunks" | Cắt tại ranh giới H1/H2/H3; văn bản tiêu đề là một phần của chunk |
| Semantic chunker | "Topic-aware chunks" | Embed các câu, gom nhóm theo độ tương đồng tâm cụm, cắt khi có sự trôi dạt |
| Centroid drift | "Topic shift" | Độ tương đồng cosine giữa trung bình cộng hiện tại và câu tiếp theo giảm xuống dưới ngưỡng |

## Đọc thêm

- [LongRAG: Enhancing Retrieval-Augmented Generation with Long-context LLMs (arXiv 2406.15319)](https://arxiv.org/abs/2406.15319)
- [Anthropic, Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- [LlamaIndex, Chunking strategies for production RAG](https://docs.llamaindex.ai/en/stable/optimizing/production_rag/)
- Phase 11 bài 06 - RAG fundamentals
- Phase 11 bài 07 - advanced RAG
- Phase 19 bài 65 - hybrid retrieval xếp hạng các chunk được tạo ra ở đây
- Phase 19 bài 68 - eval harness chấm điểm lựa chọn chiến lược trong sản xuất