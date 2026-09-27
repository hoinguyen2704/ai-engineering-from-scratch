# Kinh tế học về Prompt Caching và Semantic Caching

> **Ảnh chụp nhanh giá cước ngày 2026-04.** Các tuyên bố về số liệu dưới đây phản ánh bảng giá của nhà cung cấp tại thời điểm xuất bản bài học này; hãy xác minh lại với các tài liệu được liên kết trước khi trích dẫn cho các mục đích sử dụng sau này.

> Caching diễn ra ở hai lớp. Lớp L2 (cấp nhà cung cấp) prompt/prefix caching tái sử dụng attention KV cho các tiền tố lặp lại — tài liệu về prompt-caching của Anthropic quảng cáo mức giảm chi phí lên tới 90% và giảm độ trễ 85% cho các prompt dài; đối với Claude 3.5 Sonnet, chi phí đọc từ cache là $0.30/M vs $3.00/M token mới với TTL 5 phút và phí ghi cao gấp 2 lần cho tùy chọn TTL 1 giờ (docs.anthropic.com, 2026-04). OpenAI áp dụng prompt caching tự động cho các prompt ≥1024 token và định giá input được cache ở mức giảm giá khoảng 90% so với input mới (platform.openai.com, 2026-04); mức giá chính xác cho mỗi model phụ thuộc vào bảng giá thực tế. Lớp L1 (cấp ứng dụng) semantic caching bỏ qua hoàn toàn LLM khi có kết quả khớp về độ tương đồng embedding. Tuyên bố "độ chính xác 95%" của nhà cung cấp đề cập đến tính đúng đắn của kết quả khớp, không phải tỷ lệ hit (hit rate) — tỷ lệ hit thực tế được báo cáo dao động từ 10% (chat mở) lên đến 70% (FAQ có cấu trúc); không nhà cung cấp nào công bố mức cơ sở chính thức, vì vậy hãy coi đây là dữ liệu đo lường từ cộng đồng thay vì các cam kết. Những cạm bẫy trong sản xuất: song song hóa (parallelization) sẽ phá hủy caching (N yêu cầu song song được gửi trước khi lần ghi cache đầu tiên hoàn tất có thể làm tăng chi phí lên gấp nhiều lần), và nội dung động bên trong tiền tố sẽ ngăn chặn hoàn toàn việc hit cache. ProjectDiscovery báo cáo đã tăng tỷ lệ hit từ 7% lên 74% (2025-11) bằng cách di chuyển văn bản động ra khỏi tiền tố có thể cache.

**Type:** Learn
**Languages:** Python (stdlib, toy two-layer cache simulator)
**Prerequisites:** Phase 17 · 04 (Serving Engine Internals), Phase 17 · 06 (SGLang RadixAttention)
**Time:** ~60 phút

## Mục tiêu học tập

- Phân biệt L2 prompt/prefix caching (tái sử dụng KV tại nhà cung cấp) với L1 semantic caching (bỏ qua LLM khi có prompt tương tự).
- Giải thích đánh dấu `cache_control` rõ ràng của Anthropic và hai tùy chọn TTL (5 phút so với 1 giờ) cùng với các hệ số nhân giá của chúng.
- Tính toán mức tiết kiệm hàng tháng dự kiến dựa trên tỷ lệ hit, hỗn hợp prompt/response và giá token.
- Gọi tên anti-pattern song song hóa gây lãng phí chi phí gấp 5-10 lần và anti-pattern nội dung động làm sụp đổ tỷ lệ hit.

## Vấn đề

Bạn thêm prompt caching vào dịch vụ RAG của mình. Hóa đơn không thay đổi. Bạn đo lường tỷ lệ hit; nó là 7%. Các prompt của bạn trông có vẻ tĩnh nhưng thực tế không phải vậy — system prompt bao gồm ngày hiện tại được định dạng đến từng phút, một request ID và một ví dụ được sắp xếp ngẫu nhiên để tăng tính đa dạng. Mỗi yêu cầu đều ghi một mục cache mới, đọc bằng không.

Đồng thời, agent của bạn chạy mười lệnh gọi công cụ song song cho mỗi câu hỏi của người dùng. Tất cả mười yêu cầu đều đến nhà cung cấp trước khi lần ghi cache đầu tiên hoàn tất. Mười lần ghi, không lần đọc nào. Hóa đơn của bạn cao gấp 5-10 lần so với chi phí dự kiến khi "có caching".

Caching là một giao thức, không phải là một flag. Hai lớp, hai chế độ lỗi khác nhau.

## Khái niệm

### L2 — provider prompt/prefix caching

Nhà cung cấp lưu trữ attention KV cho một tiền tố có thể cache và tái sử dụng nó cho yêu cầu tiếp theo khớp với tiền tố đó. Bạn trả phí ghi một lần, phí đọc gần như bằng không.

**Anthropic (dòng Claude 3.5 / 3.7 / 4)**: đánh dấu `cache_control` rõ ràng trong yêu cầu. Bạn gắn thẻ các khối nào có thể cache. TTL: 5 phút (phí ghi gấp 1.25x cơ bản) hoặc 1 giờ (phí ghi gấp 2x cơ bản). Đọc từ cache: $0.30/M on Claude 3.5 Sonnet vs $3.00/M token mới — rẻ hơn 10 lần (docs.anthropic.com, tính đến 2026-04). Mức giá khác nhau tùy theo model (Opus/Haiku được công bố riêng); luôn kiểm tra chéo trang giá thực tế.

**OpenAI**: caching tự động cho các prompt ≥1024 token (platform.openai.com, 2026-04). Không có flag rõ ràng. Input được cache rẻ hơn khoảng 10 lần so với input mới trên bảng giá gpt-4o/gpt-5 hiện tại. Cả tài liệu lẫn ghi chú phát hành đều không công bố mức cơ sở tỷ lệ hit chính thức; các báo cáo cộng đồng tập trung ở mức 30–60% với thiết kế prompt cẩn thận. Theo dõi `usage.cached_tokens` để tự đo lường.

**Google (Gemini)**: context caching thông qua API rõ ràng; context 1M-token có nghĩa là caching mang lại lợi ích lớn hơn nữa.

**Self-hosted (vLLM, SGLang)**: Phase 17 · 06 đề cập đến RadixAttention — cùng một mô hình trên hạ tầng tính toán của riêng bạn.

### L1 — app-level semantic caching

Trước khi gọi LLM, hãy băm (hash) prompt, nhúng (embed) nó và tìm kiếm một yêu cầu đã cache tương tự (độ tương đồng cosine trên ngưỡng, thường là 0.95+). Khi có hit, trả về phản hồi đã cache. Khi miss, gọi LLM và cache kết quả.

Mã nguồn mở: Redis Vector Similarity, GPTCache, Qdrant. Thương mại: Portkey Cache, Helicone Cache.

Các tuyên bố về độ chính xác của nhà cung cấp đề cập đến tần suất phản hồi được cache trả về phù hợp về mặt ngữ nghĩa — không phải tần suất bạn hit cache. Tỷ lệ hit trong sản xuất:

- Chat mở: 10-15%.
- FAQ / hỗ trợ có cấu trúc: 40-70%.
- Câu hỏi về code: 20-30% (các biến thể nhỏ làm mất hit).
- Voice agents lặp lại prompt: 50-80% (chuẩn hóa giọng nói tạo ra tập hợp cố định).

### Anti-pattern song song hóa

Agent của bạn thực hiện 10 lệnh gọi công cụ song song. Tất cả 10 lệnh đều có cùng một system prompt 4K-token. Việc ghi cache của Anthropic diễn ra theo từng yêu cầu; lần ghi cache đầu tiên hoàn tất khoảng 300 ms sau khi nhà cung cấp nhận được prompt. Các yêu cầu 2-10 đến trong cùng một cửa sổ mili giây và mỗi yêu cầu đều thấy cache miss. Bạn trả phí ghi 10 lần, nhận 0 lần giảm giá đọc.

Cách khắc phục: batch với tuần tự trước — thực hiện yêu cầu 1 một mình, sau đó gửi 2-10 khi cache của 1 đã được điền. Thêm 300 ms vào lệnh gọi công cụ đầu tiên; tiết kiệm 5-10 lần chi phí hóa đơn.

### Anti-pattern nội dung động

System prompt của bạn trông như thế này:

```
You are a helpful assistant. The current time is 14:32:17.
User ID: abc123. Today is Tuesday...
```

Mỗi yêu cầu là duy nhất. Mỗi yêu cầu đều ghi. Không có hit nào.

Cách khắc phục: di chuyển mọi thứ thực sự tĩnh vào tiền tố có thể cache; nối nội dung động sau ranh giới cache:

```
[cacheable]
You are a helpful assistant. [rules, examples, instructions]
[/cacheable]
[dynamic, not cached]
Current time: 14:32:17. User: abc123.
```

ProjectDiscovery đã tăng tỷ lệ hit từ 7% lên 74% theo cách này và đã công bố cấu trúc thực hiện.

### Kết hợp batch + cache cho khối lượng công việc qua đêm

Batch API (Phase 17 · 15) giảm giá 50% với thời gian hoàn thành 24 giờ. Input được cache bên trên giúp bạn tiết kiệm thêm ~10 lần nữa. Các khối lượng công việc phân loại, dán nhãn và tạo báo cáo qua đêm có thể giảm xuống ~10% chi phí so với việc không cache đồng bộ bằng cách kết hợp cả hai.

### Các con số bạn nên nhớ

Các điểm giá được ghi nhận vào 2026-04 từ tài liệu nhà cung cấp được liên kết và thay đổi vài tháng một lần — hãy kiểm tra lại trước khi dựa vào chúng.

- Anthropic cached read: $0.30/M trên Claude 3.5 Sonnet, rẻ hơn khoảng 10 lần so với input mới (docs.anthropic.com).
- Phí ghi cache Anthropic: 1.25x (TTL 5 phút) hoặc 2x (TTL 1 giờ).
- OpenAI auto-cache: áp dụng cho prompt ≥1024 token; input được cache có giá khoảng 10% so với input mới trên bảng giá hiện tại (platform.openai.com).
- Tỷ lệ hit semantic cache (báo cáo cộng đồng): ~10% chat mở; lên đến ~70% FAQ có cấu trúc. Không phải là mức cơ sở do nhà cung cấp tài liệu.
- ProjectDiscovery: 7% → 74% tỷ lệ hit bằng cách di chuyển nội dung động ra khỏi tiền tố (blog dự án, 2025-11).
- Anti-pattern song song hóa: các báo cáo điển hình về việc hóa đơn tăng 5–10 lần khi N yêu cầu song song miss lần ghi cache đầu tiên.

```figure
semantic-cache-hit
```

## Sử dụng

`code/main.py` mô phỏng caching L1 + L2 trên các khối lượng công việc hỗn hợp. Báo cáo tỷ lệ hit, hóa đơn và hiển thị hình phạt do song song hóa.

## Triển khai

Bài học này tạo ra `outputs/skill-cache-auditor.md`. Với template prompt và lưu lượng truy cập, nó kiểm tra khả năng cache và đề xuất tái cấu trúc.

## Bài tập

1. Chạy `code/main.py`. Bật/tắt flag song song hóa. Hóa đơn thay đổi bao nhiêu?
2. System prompt của bạn có ngày tháng. Hãy di chuyển nó ra ngoài. Hiển thị toán học tỷ lệ hit trước/sau.
3. Tính điểm hòa vốn cho TTL 1 giờ (ghi 2x) so với TTL 5 phút (ghi 1.25x) dựa trên tốc độ đến của yêu cầu.
4. Semantic cache ở ngưỡng 0.95 đạt tỷ lệ hit 20%. Ở 0.85 nó đạt 50% nhưng bạn thấy các phản hồi cache không chính xác. Chọn ngưỡng phù hợp và biện minh.
5. Bạn batch 10 truy vấn con song song cho mỗi câu hỏi người dùng. Viết lại để thân thiện với cache mà không làm tăng độ trễ end-to-end.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|------------------------|
| L2 prompt cache | "prefix cache" | Nhà cung cấp lưu trữ KV cho tiền tố lặp lại |
| `cache_control` | "Anthropic cache marker" | Thuộc tính rõ ràng đánh dấu các khối có thể cache |
| Cache write premium | "write tax" | Chi phí bổ sung cho lần miss-to-cache đầu tiên (1.25x hoặc 2x) |
| L1 semantic cache | "embedding cache" | Hash-và-embed ở cấp ứng dụng trước khi gọi LLM |
| GPTCache | "LLM caching lib" | Thư viện cache L1 OSS phổ biến |
| Cache hit rate | "hits / total" | Tỷ lệ yêu cầu được phục vụ từ cache |
| Parallelization anti-pattern | "the N-write trap" | N yêu cầu song song miss cache N lần |
| Dynamic content trap | "the time-in-prompt trap" | Byte động trong tiền tố làm hỏng tỷ lệ hit |
| RadixAttention | "intra-replica cache" | Triển khai prefix-cache của SGLang |

## Đọc thêm

- [Anthropic Prompt Caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — ngữ nghĩa `cache_control` chính thức và các TTL.
- [OpenAI Prompt Caching](https://platform.openai.com/docs/guides/prompt-caching) — hành vi caching tự động và điều kiện đủ.
- [TianPan — Semantic Caching for LLMs Production](https://tianpan.co/blog/2026-04-10-semantic-caching-llm-production)
- [ProjectDiscovery — Cut LLM Costs 59% With Prompt Caching](https://projectdiscovery.io/blog/how-we-cut-llm-cost-with-prompt-caching)
- [DigitalOcean / Anthropic — Prompt Caching](https://www.digitalocean.com/blog/prompt-caching-with-digital-ocean)