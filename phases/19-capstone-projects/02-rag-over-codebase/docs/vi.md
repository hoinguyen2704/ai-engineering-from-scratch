# Capstone 02 — RAG cho Codebase (Tìm kiếm ngữ nghĩa liên kho lưu trữ)

> Mọi tổ chức kỹ thuật nghiêm túc vào năm 2026 đều vận hành một hệ thống tìm kiếm mã nguồn nội bộ có khả năng hiểu được ý nghĩa, thay vì chỉ tìm kiếm theo chuỗi ký tự. Sourcegraph Amp, codebase answers của Cursor, enterprise graph của Augment, repomap của Aider, MCP nội bộ của Pinterest — tất cả đều có cùng hình thái. Hệ thống sẽ nạp nhiều kho lưu trữ (repo), phân tích bằng tree-sitter, nhúng (embed) các đoạn mã ở cấp độ hàm và lớp, thực hiện tìm kiếm lai (hybrid-search), tái xếp hạng (re-rank) và trả lời kèm theo trích dẫn. Capstone này yêu cầu bạn xây dựng một hệ thống xử lý 2 triệu dòng mã trên 10 repo và duy trì khả năng tái lập chỉ mục (re-indexing) tăng dần sau mỗi lần git push.

**Type:** Capstone
**Languages:** Python (nạp dữ liệu), TypeScript (API + UI)
**Prerequisites:** Phase 5 (nền tảng NLP), Phase 7 (transformers), Phase 11 (kỹ thuật LLM), Phase 13 (công cụ), Phase 17 (cơ sở hạ tầng)
**Phases exercised:** P5 · P7 · P11 · P13 · P17
**Time:** 30 giờ

## Vấn đề

Đến năm 2026, mọi tác nhân lập trình tiên tiến đều được trang bị một lớp truy xuất codebase vì context window đơn thuần không giải quyết được các câu hỏi liên kho lưu trữ. Context 1 triệu token của Claude rất hữu ích, nhưng nó không loại bỏ được nhu cầu về truy xuất có xếp hạng. Tìm kiếm cosine ngây thơ trên các đoạn mã thô sẽ làm nhiễu kết quả đối với mã được tạo tự động, các monorepo trùng lặp và các ký hiệu (symbol) hiếm khi được import. Giải pháp thực tế là tìm kiếm lai (dày đặc + BM25) trên các đoạn mã hiểu được AST (cây cú pháp trừu tượng) với một bộ tái xếp hạng, được hỗ trợ bởi đồ thị các tham chiếu ký hiệu.

Bạn sẽ học điều này bằng cách lập chỉ mục một đội ngũ thực tế — không phải một repo hướng dẫn — và đo lường MRR@10, độ trung thực của trích dẫn và tính cập nhật tăng dần. Các chế độ lỗi thường nằm ở cơ sở hạ tầng: một monorepo 100k tệp, một lần push thay đổi một nửa số tệp, một truy vấn cần đi qua bốn repo để trả lời chính xác.

## Khái niệm

Một pipeline nạp dữ liệu hiểu AST sẽ phân tích từng tệp bằng tree-sitter, trích xuất các nút hàm và lớp, đồng thời chia đoạn (chunk) tại ranh giới nút thay vì các cửa sổ token cố định. Mỗi đoạn mã nhận ba biểu diễn: một embedding dày đặc (Voyage-code-3 hoặc nomic-embed-code), các thuật ngữ BM25 thưa thớt và một bản tóm tắt ngôn ngữ tự nhiên ngắn gọn. Bản tóm tắt bổ sung thêm một phương thức truy xuất thứ ba — người dùng hỏi "X được ủy quyền như thế nào" và bản tóm tắt đề cập đến "authz", ngay cả khi mã nguồn chỉ có `check_permission`.

Truy xuất là dạng lai. Một truy vấn kích hoạt cả tìm kiếm dày đặc và BM25, hợp nhất top-k và chuyển kết quả cho bộ tái xếp hạng cross-encoder (Cohere rerank-3 hoặc bge-reranker-v2-gemma-2b). Danh sách đã tái xếp hạng được gửi đến bộ tổng hợp ngữ cảnh dài (Claude Sonnet 4.7 với prompt caching, hoặc Llama 3.3 70B tự lưu trữ) với hướng dẫn trích dẫn mọi khẳng định theo tệp và phạm vi dòng. Các câu trả lời không có trích dẫn sẽ bị bộ lọc hậu kiểm loại bỏ.

Tính cập nhật tăng dần là vấn đề về cơ sở hạ tầng. Git push kích hoạt một diff: tệp nào đã thay đổi, ký hiệu nào đã thay đổi. Chỉ các đoạn mã bị ảnh hưởng mới được nhúng lại. Các cạnh ký hiệu liên tệp bị ảnh hưởng (import, lời gọi phương thức) được tính toán lại. Chỉ mục vẫn nhất quán mà không cần xử lý lại 2 triệu dòng mã mỗi lần commit.

## Kiến trúc

```
git push --> webhook --> ingest worker (LlamaIndex Workflow)
                           |
                           v
             tree-sitter parse + AST chunk
                           |
            +--------------+----------------+
            v              v                v
          dense        BM25 index       summary (LLM)
        (Voyage / bge)  (Tantivy)        (Haiku 4.5)
            |              |                |
            +------> Qdrant / pgvector <----+
                            |
                            v
                      symbol graph (Neo4j / kuzu)
                            |
  query --> LangGraph agent (retrieve -> rerank -> synth)
                            |
                            v
                 Claude Sonnet 4.7 1M context
                            |
                            v
                 answer + file:line citations
```

## Stack

- Phân tích: tree-sitter với 17 ngữ pháp ngôn ngữ (Python, TS, Rust, Go, Java, C++, v.v.)
- Embedding dày đặc: Voyage-code-3 (hosted) hoặc nomic-embed-code-v1.5 (self-host), dự phòng bge-code-v1
- Chỉ mục thưa: Tantivy (Rust) với BM25F, trọng số trường trên tên ký hiệu so với nội dung
- Vector DB: Qdrant 1.12 với tìm kiếm lai, hoặc pgvector + pgvectorscale cho các nhóm dưới 50 triệu vector
- Mô hình tóm tắt đoạn mã: Claude Haiku 4.5 hoặc Gemini 2.5 Flash, prompt-cached
- Bộ tái xếp hạng: Cohere rerank-3 hoặc bge-reranker-v2-gemma-2b tự lưu trữ
- Điều phối: LlamaIndex Workflows cho nạp dữ liệu, LangGraph cho tác nhân truy vấn
- Bộ tổng hợp: Claude Sonnet 4.7 (1M context) với prompt caching
- Đồ thị ký hiệu: Neo4j (managed) hoặc kuzu (embedded) cho các cạnh import và lời gọi
- Khả năng quan sát: Langfuse spans cho mỗi bước truy xuất + tổng hợp

```figure
ce-hybrid-retrieval
```

## Xây dựng

1. **Trình duyệt nạp dữ liệu.** Lặp lại lịch sử git trên mỗi webhook push. Thu thập các tệp đã thay đổi. Với mỗi tệp, phân tích bằng tree-sitter, trích xuất các nút hàm và lớp cùng với phạm vi nguồn đầy đủ của chúng. Phát hành các bản ghi đoạn mã `{repo, path, start_line, end_line, symbol, body}`.

2. **Bộ tóm tắt đoạn mã.** Gom các đoạn mã vào các lệnh gọi Haiku 4.5 với prompt caching trên phần mở đầu hệ thống. Prompt: "Tóm tắt hàm này trong một câu, nêu tên hợp đồng công khai và các tác dụng phụ của nó." Lưu trữ bản tóm tắt cùng với đoạn mã.

3. **Nhóm Embedding.** Hai hàng đợi song song: dày đặc (Voyage-code-3 batch 128) và tóm tắt (cùng mô hình, nhưng trên chuỗi tóm tắt). Ghi vector vào Qdrant với payload `{repo, path, start_line, end_line, symbol, kind}`.

4. **Chỉ mục BM25.** Chỉ mục Tantivy có trọng số trường: trọng số tên ký hiệu 4, trọng số nội dung ký hiệu 1, trọng số tóm tắt 2. Cho phép các truy vấn "tìm hàm tên X" cùng với "tìm hàm thực hiện X".

5. **Đồ thị ký hiệu.** Với mỗi đoạn mã, ghi lại các cạnh: import (tệp này sử dụng ký hiệu Y từ repo Z), lời gọi (hàm này gọi phương thức M trên lớp C), kế thừa. Lưu trữ trong kuzu. Được sử dụng tại thời điểm truy vấn để mở rộng truy xuất qua các ranh giới repo.

6. **Tác nhân truy vấn.** LangGraph với ba nút. `retrieve` kích hoạt dày đặc + BM25 song song, loại bỏ trùng lặp theo (repo, path, symbol). `rerank` chạy cross-encoder trên top-50 và giữ lại top-10. `synth` gọi Claude Sonnet 4.7 với các đoạn mã đã tái xếp hạng trong ngữ cảnh, cache prompt hệ thống, yêu cầu trích dẫn tệp:dòng.

7. **Thực thi trích dẫn.** Phân tích đầu ra của mô hình; bất kỳ khẳng định nào không có neo `(repo/path:start-end)` sẽ bị gắn cờ để hỏi lại hoặc loại bỏ. Trả về câu trả lời chỉ chứa trích dẫn cho người dùng.

8. **Tái lập chỉ mục tăng dần.** Trên mỗi webhook, tính toán diff ở cấp độ ký hiệu. Chỉ nhúng lại các đoạn mã có văn bản thay đổi. Tính toán lại các cạnh ký hiệu cho các đoạn mã có import thay đổi. Đo lường: một lần push 50 tệp được tái lập chỉ mục trong dưới 60 giây cho một đội ngũ 2 triệu LOC.

9. **Đánh giá.** Gán nhãn 100 câu hỏi liên repo với câu trả lời tệp:dòng chuẩn. Đo lường MRR@10, nDCG@10, độ trung thực của trích dẫn (tỷ lệ các khẳng định có neo xác minh được) và độ trễ p50/p99.

## Sử dụng

```
$ code-rag ask "how is S3 multipart abort wired into our retry budget?"
[retrieve]  12 chunks dense + 7 chunks bm25, 16 unique after dedup
[rerank]    top-5 kept (cohere rerank-3)
[synth]     claude-sonnet-4.7, cache hit rate 68%, 2.1s
answer:
  Multipart aborts are triggered by `AbortMultipartOnFail` in
  services/uploader/retry.go:122-148, which decrements the per-bucket
  retry budget defined in config/budgets.yaml:34-51 ...
  citations: [services/uploader/retry.go:122-148, config/budgets.yaml:34-51,
              libs/s3client/multipart.ts:44-61]
```

## Triển khai

Kỹ năng cần đạt `outputs/skill-codebase-rag.md`. Với một tập hợp các repo, hệ thống thiết lập pipeline nạp dữ liệu, chỉ mục lai và tác nhân truy vấn, đồng thời trả về câu trả lời có trích dẫn cho bất kỳ câu hỏi liên repo nào. Rubric:

| Trọng số | Tiêu chí | Cách đo lường |
|:-:|---|---|
| 25 | Chất lượng truy xuất | MRR@10 và nDCG@10 trên tập 100 câu hỏi kiểm chứng |
| 20 | Độ trung thực của trích dẫn | Tỷ lệ các khẳng định trong câu trả lời có neo tệp:dòng xác minh được |
| 20 | Độ trễ và quy mô | Độ trễ truy vấn p95 tại 10k QPS trên quy mô corpus đã lập chỉ mục |
| 20 | Độ chính xác của tái lập chỉ mục tăng dần | Thời gian từ git push đến khi có thể tìm kiếm trên một commit 50 tệp |
| 15 | UX và định dạng câu trả lời | Khả năng nhấp vào trích dẫn, xem trước đoạn mã, khả năng theo dõi |
| **100** | | |

## Bài tập

1. Thay thế Voyage-code-3 bằng nomic-embed-code tự lưu trữ. Đo lường sự thay đổi MRR@10. Báo cáo liệu khoảng cách có thu hẹp khi bật tái xếp hạng hay không.

2. Chèn 20% mã được tạo tự động (boilerplate do LLM tạo) vào corpus và đánh giá lại. Quan sát sự nhiễu trong truy xuất. Thêm cờ "generated" vào payload và giảm trọng số các kết quả đó.

3. Benchmark tìm kiếm lai của Qdrant so với pgvector + pgvectorscale ở quy mô corpus của bạn. Báo cáo p99 tại batch size 1.

4. Thêm kiểm tra độ lệch dựa trên lấy mẫu: hàng tuần, chạy lại đánh giá 100 câu hỏi. Cảnh báo nếu MRR@10 giảm > 5%.

5. Mở rộng sang giải quyết ký hiệu đa ngôn ngữ: một hàm Python gọi dịch vụ Go qua gRPC. Sử dụng đồ thị ký hiệu để liên kết chúng.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|------------------------|
| AST-aware chunking | "Chia cấp độ hàm" | Cắt mã tại ranh giới nút tree-sitter thay vì cửa sổ token cố định |
| Hybrid search | "Dày đặc + thưa" | Chạy tìm kiếm BM25 và vector song song, hợp nhất top-k, tái xếp hạng |
| Cross-encoder rerank | "Xếp hạng giai đoạn hai" | Mô hình chấm điểm cặp (truy vấn, ứng viên) cùng nhau, chính xác hơn cosine |
| Prompt caching | "Cached system prompt" | Tính năng Claude / OpenAI 2026 giảm giá token tiền tố lặp lại tới 90% |
| Symbol graph | "Đồ thị mã" | Các cạnh cho import, lời gọi, kế thừa qua các tệp và repo |
| Citation faithfulness | "Tỷ lệ câu trả lời có căn cứ" | Tỷ lệ các khẳng định người dùng có thể xác minh bằng cách nhấp vào neo và đọc phạm vi tham chiếu |
| Incremental re-index | "Thời gian push-to-search" | Thời gian thực từ git push đến khi các ký hiệu thay đổi có thể truy vấn được |

## Đọc thêm

- [Sourcegraph Amp](https://ampcode.com) — trí tuệ mã nguồn liên repo trong sản xuất
- [Kiến trúc RAG của Sourcegraph Cody](https://sourcegraph.com/blog/how-cody-understands-your-codebase) — tài liệu tham khảo chuyên sâu cho capstone này
- [Aider repo-map](https://aider.chat/docs/repomap.html) — chế độ xem repo được xếp hạng bằng tree-sitter
- [Augment Code enterprise graph](https://www.augmentcode.com) — RAG đồ thị ký hiệu thương mại
- [Tài liệu tìm kiếm lai Qdrant](https://qdrant.tech/documentation/concepts/hybrid-queries/) — triển khai tham khảo
- [Voyage AI code embeddings](https://docs.voyageai.com/docs/embeddings) — chi tiết về Voyage-code-3
- [Cohere rerank-3](https://docs.cohere.com/reference/rerank) — tham khảo cross-encoder
- [Tìm kiếm nội bộ Pinterest MCP](https://medium.com/pinterest-engineering) — tham khảo nền tảng nội bộ