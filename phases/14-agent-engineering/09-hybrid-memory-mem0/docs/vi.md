# Hybrid Memory: Vector + Graph + KV

> Hybrid memory vận hành ba kho lưu trữ song song — vector cho độ tương đồng ngữ nghĩa, KV cho tra cứu dữ kiện nhanh, đồ thị (graph) cho suy luận thực thể-mối quan hệ — với một lớp tính điểm (scoring layer) để hợp nhất chúng khi truy xuất. Đây là mô hình sản xuất được sử dụng rộng rãi cho bộ nhớ ngoài; Mem0 (Chhikara và cộng sự, 2025) là một bản triển khai tham chiếu.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 07 (MemGPT), Phase 14 · 08 (Letta Blocks)
**Time:** ~75 phút

## Mục tiêu học tập

- Giải thích lý do tại sao một kho lưu trữ đơn lẻ (chỉ vector, chỉ đồ thị, hoặc chỉ KV) là không đủ cho bộ nhớ của tác nhân (agent).
- Kể tên ba kho lưu trữ song song của Mem0 và mục tiêu tối ưu hóa của từng kho.
- Mô tả cơ chế tính điểm hợp nhất của Mem0 — mức độ liên quan, tầm quan trọng, tính thời điểm — và tại sao đó là tổng có trọng số chứ không phải phân cấp.
- Triển khai một bộ nhớ ba kho lưu trữ đơn giản bằng stdlib với `add()` ghi vào cả ba kho và `search()` hợp nhất các kết quả.

## Vấn đề

Một kho lưu trữ duy nhất sẽ không phù hợp với một trong ba loại truy vấn sau:

- **Tương đồng ngữ nghĩa** — "tuần trước chúng ta đã thảo luận gì về agent drift?" Vector thắng; KV và đồ thị bỏ lỡ.
- **Tra cứu dữ kiện** — "số điện thoại của người dùng là gì?" KV thắng; vector gây lãng phí, đồ thị là quá mức cần thiết.
- **Suy luận mối quan hệ** — "những khách hàng nào dùng chung một thực thể thanh toán?" Đồ thị thắng; vector và KV không thể trả lời.

Các tác nhân trong môi trường sản xuất thực hiện cả ba loại truy vấn trong một phiên làm việc. Bộ nhớ đơn kho luôn sai với hai trong số đó. Đóng góp của Mem0 là kết nối cả ba phía sau một giao diện `add`/`search` duy nhất với hàm tính điểm để hợp nhất chúng.

## Khái niệm

### Ba kho lưu trữ song song

Mem0 (arXiv:2504.19413, tháng 4 năm 2025) trên `add(text, user_id, metadata)`:

1. Trích xuất các dữ kiện ứng viên từ văn bản (bước điều khiển bởi LLM).
2. Ghi từng dữ kiện vào kho vector (embedding) để tìm kiếm ngữ nghĩa.
3. Ghi từng dữ kiện vào kho KV với khóa là (user_id, fact_type, entity) để tra cứu O(1).
4. Ghi từng dữ kiện vào kho đồ thị (Mem0g) dưới dạng các cạnh có kiểu (typed edges) cho các truy vấn mối quan hệ.

Trên `search(query, user_id)`:

1. Kho vector trả về top-k theo cosine embedding.
2. Kho KV trả về các kết quả khớp trực tiếp dựa trên (user_id, type, entity) từ truy vấn.
3. Kho đồ thị trả về đồ thị con có thể tiếp cận từ các thực thể truy vấn.
4. Một lớp tính điểm hợp nhất cả ba.

### Tính điểm hợp nhất

```
score = w_relevance * relevance(q, record)
      + w_importance * importance(record)
      + w_recency * recency(record)
```

- **Mức độ liên quan (Relevance)** — cosine vector, khớp chính xác KV, trọng số đường đi đồ thị.
- **Tầm quan trọng (Importance)** — được gắn thẻ tại thời điểm ghi hoặc được học (một số dữ kiện quan trọng hơn: tên, ID, chính sách).
- **Tính thời điểm (Recency)** — suy giảm theo hàm mũ theo thời gian kể từ lần ghi hoặc đọc cuối cùng.

Trọng số được tinh chỉnh theo từng sản phẩm. `w_recency` cao hơn cho các tác nhân trò chuyện; `w_importance` cao hơn cho các tác nhân tuân thủ; `w_relevance` cao hơn cho các tác nhân truy xuất.

### Mem0g và suy luận thời gian

Mem0g bổ sung bộ phát hiện mâu thuẫn. Khi một dữ kiện mới mâu thuẫn với một cạnh hiện có, cạnh cũ sẽ được đánh dấu là không hợp lệ nhưng không bị xóa. Các truy vấn thời gian ("thành phố của người dùng vào tháng 3 là gì?") sẽ duyệt qua đồ thị con hợp lệ tại thời điểm đó.

Đây là hành vi đạt chuẩn tuân thủ mà mô hình vô hiệu hóa của Letta đã khái quát hóa.

### Các con số benchmark

Bài báo Mem0 báo cáo (2025):

- **LoCoMo** (bộ nhớ hội thoại dài): 91.6
- **LongMemEval** (bộ nhớ tình tiết dài hạn): 93.4
- **BEAM 1M** (benchmark bộ nhớ 1 triệu token): 64.1

Các baseline so sánh (LLM full-context 128k, kho vector phẳng, KV phẳng) đều thua kém từ 10 điểm trở lên. Các con số benchmark không chỉ đơn thuần biện minh cho lựa chọn — mà là hình thái vận hành — nhưng chúng cho thấy thiết kế hợp nhất không phải là sai số làm tròn.

### Phân loại phạm vi (Scope)

Mem0 chia bộ nhớ theo phạm vi:

- **Bộ nhớ người dùng (User memory)** — tồn tại qua các phiên, khóa theo `user_id`.
- **Bộ nhớ phiên (Session memory)** — tồn tại trong một luồng hội thoại.
- **Bộ nhớ tác nhân (Agent memory)** — trạng thái cho mỗi instance tác nhân.

Mỗi lần ghi sẽ chọn một phạm vi. Truy xuất có thể thực hiện trên nhiều phạm vi với trọng số riêng cho từng phạm vi. Việc trộn lẫn các phạm vi mà không suy tính kỹ là nguyên nhân dẫn đến các sự cố như "trợ lý tiết lộ dự án của Bob cho Alice".

### Nơi mô hình này đi chệch hướng

- **Embedding drift.** Kết quả vector trông có vẻ đúng ở hàng trăm truy vấn đầu tiên nhưng sẽ suy giảm khi kho dữ liệu lớn dần. Hãy thêm bước re-embedding định kỳ cho các bản ghi được sử dụng nhiều nhất (top-N).
- **KV schema creep.** `(user_id, type, entity)` trông có vẻ đơn giản cho đến khi mọi đội ngũ thêm vào `type` của riêng họ. Hãy kiểm tra tập hợp kiểu dữ liệu hàng quý.
- **Graph explosion.** Một bộ trích xuất nhiễu sẽ thêm 50 cạnh mỗi tin nhắn. Hãy giới hạn số lần ghi đồ thị trên mỗi lệnh `add`; loại bỏ các cạnh có độ tin cậy thấp.

```figure
ae-memory-fusion
```

## Xây dựng

`code/main.py` triển khai mô hình ba kho lưu trữ trong stdlib:

- `VectorStore` — độ tương đồng chồng lấp token ngây thơ thay thế cho embedding.
- `KVStore` — dict khóa theo `(user_id, fact_type, entity)`.
- `GraphStore` — các cạnh có kiểu (chủ thể, quan hệ, đối tượng, hợp lệ).
- `Mem0` — facade cấp cao với `add()`, `search()`, tính điểm hợp nhất và truy xuất nhận biết phạm vi.
- Một trace thực tế trên cuộc hội thoại đa người dùng, đa phiên.

Chạy nó:

```
python3 code/main.py
```

Kết quả đầu ra hiển thị ba đường truy xuất riêng biệt cộng với top-k đã hợp nhất. Hãy thay đổi trọng số tính điểm ở đầu `main()` và quan sát thứ hạng thay đổi.

## Sử dụng

- **Mem0 (Apache 2.0)** — sẵn sàng cho sản xuất. Tự lưu trữ với Postgres + Qdrant + Neo4j, hoặc sử dụng dịch vụ đám mây được quản lý.
- **Letta** — ba tầng core/recall/archival; tự mang theo backend vector và đồ thị của riêng bạn.
- **Zep** — giải pháp thương mại thay thế với KG thời gian và trích xuất dữ kiện.
- **Custom builds** — khi bạn cần kiểm soát chính xác bộ trích xuất (tuân thủ) hoặc trọng số hợp nhất (tác nhân giọng nói nơi tính thời điểm chiếm ưu thế).

## Triển khai

`outputs/skill-hybrid-memory.md` tạo ra một khung bộ nhớ ba kho với bộ tính điểm hợp nhất, phân loại phạm vi và vô hiệu hóa theo thời gian được tích hợp sẵn.

## Bài tập

1. Thay thế độ tương đồng vector đơn giản bằng một mô hình embedding thực tế (sentence-transformers, Ollama, OpenAI embeddings). Đo lường recall@10 trên một cuộc hội thoại dài tổng hợp. Thứ hạng có bị trôi (drift) sau 1000 lần ghi không?
2. Thêm truy vấn thời gian: `search(query, as_of=timestamp)`. Chỉ trả về các bản ghi hợp lệ tại hoặc trước thời điểm đó. Kho lưu trữ nào cần nhiều công sức nhất?
3. Triển khai bộ phát hiện mâu thuẫn: nếu một dữ kiện mới mâu thuẫn với một cạnh đồ thị, hãy vô hiệu hóa cạnh cũ và ghi log cả hai. Kiểm tra với "người dùng sống ở Berlin" -> "người dùng sống ở Lisbon."
4. Chuyển đổi bộ tính điểm hợp nhất để bao gồm chiều `user_feedback` (thích/không thích trên các bản ghi được truy xuất). Làm thế nào để ngăn chặn việc thao túng (tác nhân chỉ trả về các bản ghi mà nó đã "thích")?
5. Đọc tài liệu Mem0 (`docs.mem0.ai`). Chuyển đổi bản demo sang các lệnh gọi client `mem0`. So sánh chất lượng truy xuất trên cùng 20 truy vấn kiểm thử.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|------------------------|
| Hybrid memory | "Vector cộng đồ thị cộng KV" | Ba kho lưu trữ được ghi song song, hợp nhất khi truy xuất |
| Fact extraction | "Nạp bộ nhớ" | Bước LLM chia nhỏ văn bản thành các bộ (thực thể, quan hệ, dữ kiện) |
| Fusion scoring | "Xếp hạng mức độ liên quan" | Tổng có trọng số của mức độ liên quan, tầm quan trọng, tính thời điểm |
| Scope | "Không gian tên bộ nhớ" | user / session / agent — xác định ai thấy cái gì |
| Mem0g | "Đồ thị bộ nhớ" | Các cạnh có kiểu với tính hợp lệ thời gian cho truy vấn mối quan hệ |
| Temporal invalidation | "Xóa mềm" | Đánh dấu các cạnh mâu thuẫn là không hợp lệ; không bao giờ xóa |
| Embedding drift | "Suy giảm truy xuất" | Chất lượng vector giảm khi kho dữ liệu lớn dần; re-embed định kỳ |

## Đọc thêm

- [Chhikara và cộng sự, Mem0 (arXiv:2504.19413)](https://arxiv.org/abs/2504.19413) — bài báo gốc
- [Tài liệu Mem0](https://docs.mem0.ai/platform/overview) — API sản xuất, SDK, đám mây được quản lý
- [Packer và cộng sự, MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560) — tiền thân của ngữ cảnh ảo
- [Letta, Blog Memory Blocks](https://www.letta.com/blog/memory-blocks) — thiết kế ba tầng tương tự