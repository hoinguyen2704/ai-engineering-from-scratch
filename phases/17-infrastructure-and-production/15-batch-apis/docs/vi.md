# Batch APIs — Giảm giá 50% như một tiêu chuẩn ngành

> Mọi nhà cung cấp lớn đều cung cấp Batch API bất đồng bộ với mức giảm giá 50% và thời gian hoàn thành khoảng 24 giờ. OpenAI, Anthropic, Google và hầu hết các nền tảng suy luận (như batch tier của Fireworks, batch của Together) đều triển khai cùng một mô hình này. Kết hợp batch với prompt caching và các pipeline chạy qua đêm có thể giảm chi phí xuống còn khoảng 10% so với chi phí đồng bộ không cache. Quy tắc rất đơn giản: nếu không cần tương tác thời gian thực, hãy dùng batch. Các pipeline tạo nội dung, phân loại tài liệu, trích xuất dữ liệu, tạo báo cáo, gắn nhãn hàng loạt, gắn thẻ danh mục — bất cứ thứ gì chấp nhận được độ trễ 24 giờ mà không dùng batch đều là đang lãng phí tiền bạc. Mô hình sản xuất năm 2026 là phân loại mọi workload LLM mới vào ba luồng: tương tác (đồng bộ với caching), bán tương tác (hàng đợi bất đồng bộ với dự phòng), và batch (chạy qua đêm, input được cache). Những workload giả vờ là tương tác nhưng thực tế chấp nhận độ trễ vài phút là những thứ gây lãng phí nhất.

**Type:** Learn
**Languages:** Python (stdlib, toy batch-vs-sync cost simulator)
**Prerequisites:** Phase 17 · 14 (Prompt & Semantic Caching)
**Time:** ~45 minutes

## Mục tiêu học tập

- Nêu tên ba Batch API của các nhà cung cấp (OpenAI, Anthropic, Google) và các cam kết chung về giảm giá 50% + thời gian hoàn thành 24 giờ.
- Tính toán chi phí cho việc kết hợp batch + cached-input trên một workload phân loại chạy qua đêm và so sánh với baseline đồng bộ không cache.
- Phân loại workload vào nhóm tương tác / bán tương tác / batch và giải thích lý do.
- Nêu tên hai cái bẫy: tương tác một phần (người dùng mong đợi nhanh hơn 24 giờ) và trôi dạt schema đầu ra (định dạng file batch khác nhau tùy nhà cung cấp).

## Vấn đề

Nhóm của bạn vận hành một pipeline tạo báo cáo hàng đêm. Với 50.000 tài liệu, tóm tắt từng tài liệu, phân cụm các bản tóm tắt, và soạn thảo bản tóm tắt điều hành. Chạy đồng bộ mất 4 giờ với chi phí $2.000/đêm. Bạn nghe nói về Batch API.

Batch giúp bạn giảm 50% chi phí. Bạn cũng bật prompt caching trên system prompt (dùng chung cho tất cả 50k cuộc gọi). Khi kết hợp lại, hóa đơn giảm xuống còn $180/đêm — khoảng 9% so với baseline. Cùng một pipeline, chỉ cần ba thay đổi cấu hình.

Batch là đòn bẩy rẻ nhất trong bộ công cụ chi phí LLM mà ít người tận dụng. Lý do chủ yếu là về mặt tổ chức: các nhóm thường nghĩ đến "thời gian thực" trong khi SLA thực tế chỉ là "trước sáng mai". Bài học này là về việc không để lãng phí 90% ngân sách.

## Khái niệm

### Ba Batch API chính

**OpenAI Batch API**: Tải lên file JSONL với danh sách các yêu cầu. Cam kết hoàn thành trong 24 giờ (thực tế thường là ~2-8 giờ). Giảm giá 50% trên token đầu vào và đầu ra. `/v1/batches` endpoint. Các input đủ điều kiện cache cũng nhận được mức giá cached-input bổ sung.

**Anthropic Message Batches**: Tải lên JSONL. Thời gian hoàn thành 24 giờ. Giảm giá 50%. Hỗ trợ `cache_control` — việc ghi cache là rõ ràng, việc đọc diễn ra tự động trong batch.

**Google Vertex AI Batch Prediction**: Input từ BigQuery hoặc GCS. Mức giảm giá 50% tương tự cho Gemini. Tích hợp với các pipeline của Vertex.

### Ý nghĩa: bất đồng bộ, không phải chậm

Batch có nghĩa là "Tôi hứa sẽ trả kết quả trong vòng 24 giờ" — không phải "việc này sẽ mất 24 giờ". P50 điển hình là 2-6 giờ. Nhà cung cấp sẽ lên lịch cho batch của bạn vào các khung giờ thấp điểm khi tài nguyên GPU chưa được sử dụng hết.

### Kết hợp với caching

Một tác vụ tóm tắt 50k tài liệu với cùng một system prompt 4K-token:

- Đồng bộ không cache: 50000 × ($input × 4000 + $output × 200) với giá đầy đủ.
- Đồng bộ có cache: system prompt được cache sau lần ghi đầu tiên; 49999 lần còn lại nhận giá input rẻ hơn 10 lần.
- Batch có cache: tất cả các điều trên cộng với giảm giá 50% cho cả đọc và ghi.

Kết hợp: batch + cache = ~10% hóa đơn đồng bộ không cache. Bất kỳ workload nào chạy qua đêm và có system prompt dùng chung đều nên sử dụng cách này.

### Phân loại workload

**Tương tác (Interactive)** — người dùng chờ phản hồi. TTFT (Time To First Token) rất quan trọng. Gọi đồng bộ với prompt caching. Không thể batch.

**Bán tương tác (Semi-interactive)** — người dùng gửi tác vụ, kiểm tra lại sau vài phút. Hàng đợi bất đồng bộ với dự phòng sang đồng bộ nếu batch không khả dụng. Ví dụ: lập chỉ mục RAG với lưu lượng vừa phải.

**Batch** — người dùng mong đợi kết quả "vào sáng mai" hoặc "trong giờ tới". Các pipeline nội dung, phân loại quy mô lớn, phân tích ngoại tuyến. Luôn dùng batch, luôn kết hợp caching.

Sai lầm phổ biến: phân loại mọi thứ là tương tác vì pipeline đó thuộc môi trường production. Production không phải là thông số kỹ thuật về độ trễ — SLA mới là yếu tố quyết định.

### Cái bẫy tương tác một phần

Một số tính năng trông có vẻ tương tác nhưng thực tế chấp nhận độ trễ 5-10 phút. Ví dụ: báo cáo sức khỏe khách hàng hàng đêm với nút "làm mới". Người dùng nhấn làm mới; chờ 10 phút là ổn. Nhóm triển khai nó dưới dạng đồng bộ. 50 lượt làm mới đồng thời tốn kém gấp 10 lần so với việc batch và gửi qua email.

Câu hỏi cần đặt ra: "24 giờ có ý nghĩa gì với người dùng này?" Nếu câu trả lời là "họ sẽ không nhận ra", hãy batch nó.

### Cái bẫy schema đầu ra

Định dạng file batch khác nhau tùy nhà cung cấp:

- OpenAI: JSONL, mỗi dòng một yêu cầu.
- Anthropic: JSONL, mỗi dòng một tin nhắn; định dạng phản hồi được nhúng.
- Vertex: Bảng BigQuery hoặc tiền tố GCS với TFRecord.

Việc viết "một batch client" cho nhiều nhà cung cấp đồng nghĩa với việc phải viết code adapter cho từng bên. Các gateway quảng cáo batch đa nhà cung cấp (Portkey, LiteLLM ở một số gói) vẫn chỉ là lớp bao bọc mỏng trên định dạng gốc.

### Các con số cần ghi nhớ

- Giảm giá batch giữa các nhà cung cấp: giảm phẳng 50% trên input + output.
- SLA hoàn thành: cam kết 24 giờ, P50 điển hình là 2-6 giờ.
- Kết hợp batch + cached input: ~10% chi phí đồng bộ không cache.
- Quy tắc phân loại workload: nếu độ trễ 24 giờ là chấp nhận được, luôn luôn dùng batch.

```figure
batch-lane-triage
```

## Sử dụng

`code/main.py` tính toán chi phí giữa đồng bộ, đồng bộ+cache, batch, và batch+cache cho workload 50k tài liệu. Báo cáo mức tiết kiệm bằng $ và phần trăm.

## Triển khai

Bài học này tạo ra `outputs/skill-batch-triager.md`. Dựa trên đặc điểm workload, phân loại thành tương tác/bán tương tác/batch và ước tính mức tiết kiệm.

## Bài tập

1. Chạy `code/main.py`. Đối với pipeline 100k tài liệu với system prompt 3K-token và output 500-token, hãy tính toán mức tiết kiệm của toàn bộ stack (batch + cache) so với baseline đồng bộ.
2. Chọn ba tính năng trong một sản phẩm thực tế mà bạn biết. Phân loại từng tính năng vào nhóm tương tác/bán tương tác/batch.
3. Một người dùng phàn nàn rằng báo cáo của họ mất 3 giờ. Đó là do phân loại sai batch hay là tương tác hợp lệ? Hãy viết tiêu chí quyết định.
4. SLA trả về Batch API của bạn là 24 giờ nhưng P99 là 20 giờ. Bạn truyền đạt điều này với người dùng như thế nào — hành vi của hệ thống hạ nguồn trong trường hợp biên là gì?
5. Tính điểm hòa vốn: ở độ dài tiền tố dùng chung nào thì batch + cache trở nên rẻ hơn so với việc chạy qua đêm trên GPU dự phòng của riêng bạn?

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|------------------------|
| Batch API | "giảm giá bất đồng bộ" | Giảm 50% với thời gian hoàn thành 24h |
| JSONL | "định dạng batch" | Một yêu cầu JSON mỗi dòng; tiêu chuẩn OpenAI/Anthropic |
| Message Batches | "batch của Anthropic" | Tên sản phẩm Batch API của Anthropic |
| Batch prediction | "batch của Vertex" | Sản phẩm Batch API của Vertex AI |
| Turnaround SLA | "cam kết 24h" | Là cam kết, không phải điển hình; điển hình là 2-6h |
| Workload triage | "quyết định tương tác" | Quyết định định tuyến tương tác / bán tương tác / batch |
| Output schema | "định dạng phản hồi" | Bố cục JSONL theo từng nhà cung cấp; không di động |
| Stacked discount | "batch + cache" | ~10% hóa đơn đồng bộ không cache khi áp dụng cả hai |

## Đọc thêm

- [OpenAI Batch API](https://platform.openai.com/docs/guides/batch) — định dạng JSONL và ngữ nghĩa `/v1/batches`.
- [Anthropic Message Batches](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing) — định dạng batch và tương tác `cache_control`.
- [Vertex AI Batch Prediction](https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/batch-prediction-gemini) — ngữ nghĩa batch của Gemini.
- [Finout — So sánh giá API OpenAI vs Anthropic 2026](https://www.finout.io/blog/openai-vs-anthropic-api-pricing-comparison)
- [Zen Van Riel — So sánh chi phí API LLM 2026](https://zenvanriel.com/ai-engineer-blog/llm-api-cost-comparison-2026/)