# LLM Routing Layer — LiteLLM, OpenRouter, Portkey

> Việc phụ thuộc vào một nhà cung cấp (provider lock-in) rất tốn kém. Các tác vụ gọi công cụ (tool-calling) khác nhau sẽ phù hợp với các model khác nhau. Các routing gateway cung cấp một giao diện API thống nhất, hỗ trợ thử lại (retry), chuyển đổi dự phòng (failover), theo dõi chi phí và các bộ lọc an toàn (guardrails). Ba mô hình chủ đạo trong năm 2026 là: LiteLLM (mã nguồn mở, tự lưu trữ), OpenRouter (SaaS được quản lý), Portkey (cấp độ sản xuất, mã nguồn mở từ tháng 3/2026). Bài học này xác định các tiêu chí ra quyết định và hướng dẫn xây dựng một routing gateway tiêu chuẩn.

**Type:** Learn
**Languages:** Python (stdlib, routing + failover + cost tracker)
**Prerequisites:** Phase 13 · 02 (function calling), Phase 13 · 17 (gateways)
**Time:** ~45 minutes

## Mục tiêu học tập

- Phân biệt các tùy chọn routing: tự lưu trữ, được quản lý và cấp độ sản xuất.
- Triển khai chuỗi dự phòng (fallback chain) tự động thử lại khi nhà cung cấp gặp lỗi theo thứ tự ưu tiên xác định.
- Theo dõi chi phí mỗi yêu cầu và mức sử dụng token trên các nhà cung cấp.
- Đưa ra quyết định chọn giữa LiteLLM, OpenRouter và Portkey cho một ràng buộc sản xuất cụ thể.

## Vấn đề

Các kịch bản mà routing nhà cung cấp trở nên quan trọng:

1. **Chi phí.** Claude Sonnet có chi phí gấp 3 lần Haiku. Đối với tác vụ phân loại (triage), Haiku là đủ; đối với tác vụ tổng hợp (synthesis), Sonnet xứng đáng hơn. Hãy route theo từng yêu cầu.

2. **Failover.** OpenAI gặp sự cố trong một giờ. Mọi yêu cầu đều thất bại. Bạn muốn tự động chuyển sang Anthropic mà không cần triển khai lại (redeploy).

3. **Độ trễ (Latency).** Giao diện chat trực tiếp cần thời gian phản hồi token đầu tiên (time-to-first-token) nhanh. Trình tóm tắt theo lô (batch summarizer) thì không. Hãy route theo SLA độ trễ.

4. **Tuân thủ (Compliance).** Người dùng EU phải ở trong khu vực EU. Hãy route theo khu vực.

5. **Thử nghiệm.** A/B test hai model trên cùng một khối lượng công việc. Hãy route theo nhóm thử nghiệm (test bucket).

Việc viết code thủ công cho tất cả những điều này ở mỗi tích hợp là rất lặp lại. Một routing gateway cung cấp một API tương thích với OpenAI và xử lý phần còn lại.

## Khái niệm

### Cấu trúc proxy tương thích với OpenAI

Mọi thứ đều sử dụng cấu trúc OpenAI. Routing gateway hiển thị `/v1/chat/completions`, chấp nhận schema của OpenAI và proxy nội bộ đến Anthropic / Gemini / Cohere / Ollama / bất kỳ thứ gì. Client không cần quan tâm đến điều này.

### Model aliases

Thay vì một id snapshot cố định, code của bạn sử dụng `our_smart_model`. Gateway sẽ ánh xạ các alias này tới các model thực tế. Khi nhà cung cấp ra mắt thế hệ mới, bạn chỉ cần thay đổi alias ở phía server; code của bạn không cần thay đổi gì cả.

### Chuỗi dự phòng (Fallback chains)

```
primary: openai/gpt-4o
on 5xx: anthropic/claude-3-5-sonnet
on 5xx: google/gemini-1.5-pro
on 5xx: refuse
```

Các gateway định nghĩa điều này trong cấu hình. Các lần thử lại được tính vào ngân sách để các chuỗi dự phòng không làm tăng chi phí đột biến.

### Semantic caching

Các prompt giống hệt hoặc gần như giống hệt nhau sẽ truy cập vào cache thay vì nhà cung cấp. Việc tiết kiệm chi phí trên các vòng lặp agent lặp đi lặp lại có thể đạt từ 30 đến 60 phần trăm. Các khóa (key) dựa trên embedding; các prompt gần như giống hệt nhau sẽ chia sẻ một slot cache.

### Guardrails

Ở cấp độ gateway:

- **Ẩn danh hóa PII (PII redaction).** Sử dụng Regex hoặc ML để lọc trước khi gửi prompt.
- **Vi phạm chính sách.** Từ chối các prompt có nội dung bị cấm.
- **Bộ lọc đầu ra.** Quét các kết quả hoàn thiện để tránh rò rỉ dữ liệu.

Cả Portkey và Kong đều cung cấp các guardrails chuyên biệt. LiteLLM để chúng ở dạng tùy chọn.

### Giới hạn tốc độ theo key (Per-key rate limits)

Một API key = một team. Ngân sách theo key ngăn chặn một team tiêu thụ hết hạn ngạch chung. Hầu hết các gateway đều hỗ trợ điều này.

### Đánh đổi giữa tự lưu trữ và được quản lý

| Yếu tố | LiteLLM (tự lưu trữ) | OpenRouter (được quản lý) | Portkey (sản xuất) |
|--------|----------------------|----------------------|----------------------|
| Code | Mã nguồn mở, Python | SaaS được quản lý | Mã nguồn mở (tháng 3/2026) + quản lý |
| Thiết lập | Triển khai proxy | Đăng ký | Cả hai |
| Nhà cung cấp | 100+ | 300+ | 100+ |
| Thanh toán | Key của riêng bạn | Tín dụng OpenRouter | Key của riêng bạn |
| Khả năng quan sát | OpenTelemetry | Dashboard | Full OTel + ẩn danh PII |
| Tốt nhất cho | Team muốn toàn quyền kiểm soát | Tạo mẫu nhanh | Sản xuất cần tuân thủ |

LiteLLM thắng thế khi bạn có đội ngũ SRE và muốn chủ quyền dữ liệu. OpenRouter thắng thế khi bạn muốn đăng ký một lần và không cần hạ tầng. Portkey thắng thế khi bạn cần guardrails và tuân thủ ngay lập tức.

### Theo dõi chi phí

Mỗi yêu cầu mang theo `provider`, `model`, `input_tokens`, `output_tokens`. Nhân với giá mỗi token của từng model (được lấy từ bảng giá mà gateway duy trì). Tổng hợp theo người dùng / theo team / theo dự án.

### MCP kết hợp routing

Một gateway có thể route cả các cuộc gọi LLM VÀ các yêu cầu lấy mẫu MCP (sampling requests). Khi modelPreferences của một yêu cầu lấy mẫu ưu tiên một model cụ thể, gateway sẽ dịch sang backend phù hợp. Đây là nơi Phase 13 · 17 (MCP gateway) và routing gateway của bài học này đôi khi hợp nhất thành một dịch vụ.

### Chiến lược định tuyến (Routing strategies)

- **Ưu tiên tĩnh.** Đầu tiên trong danh sách; dự phòng khi có lỗi.
- **Cân bằng tải.** Round-robin hoặc có trọng số.
- **Nhận thức chi phí.** Chọn model rẻ nhất đáp ứng độ trễ / chất lượng.
- **Nhận thức độ trễ.** Chọn model nhanh nhất trong N phút gần nhất.
- **Nhận thức tác vụ.** Bộ phân loại prompt route các tác vụ code đến một model, tóm tắt đến một model khác.

```figure
tp-router-failover
```

## Sử dụng

`code/main.py` triển khai một routing gateway trong khoảng 150 dòng: chấp nhận các yêu cầu theo cấu trúc OpenAI, dịch sang các stub của từng nhà cung cấp, chạy chuỗi dự phòng ưu tiên, theo dõi chi phí mỗi yêu cầu và áp dụng bộ lọc ẩn danh PII trên đầu vào. Chạy nó với ba kịch bản: yêu cầu bình thường, nhà cung cấp chính gặp sự cố kích hoạt dự phòng, rò rỉ PII bị chặn bởi bộ lọc.

Những điểm cần chú ý:

- Dict `ROUTES`: alias -> danh sách các nhà cung cấp cụ thể theo thứ tự ưu tiên.
- Vòng lặp dự phòng thử lại trên lỗi 5xx.
- Trình theo dõi chi phí nhân mức sử dụng token với giá của từng model.
- Bộ lọc PII xóa các mẫu có dạng SSN trước khi chuyển tiếp.

## Triển khai

Bài học này tạo ra `outputs/skill-routing-config-designer.md`. Với một hồ sơ khối lượng công việc (độ trễ, chi phí, tuân thủ), kỹ năng này sẽ chọn LiteLLM / OpenRouter / Portkey và tạo ra một cấu hình routing.

## Bài tập

1. Chạy `code/main.py`. Kích hoạt kịch bản sự cố; xác nhận dự phòng chuyển sang nhà cung cấp thứ hai và chi phí được ghi nhận chính xác.

2. Thêm semantic caching: SHA256 của prompt là khóa tra cứu; cache hit trả về ngay lập tức. Đo lường mức tiết kiệm chi phí trên một cuộc gọi lặp lại.

3. Thêm bộ phân loại prompt để route các prompt "code ..." đến một alias ưu tiên trí tuệ và các prompt "summarize ..." đến một alias ưu tiên tốc độ.

4. Thiết kế ngân sách theo team: mỗi team có giới hạn chi tiêu hàng tháng; gateway từ chối yêu cầu khi đạt giới hạn. Chọn mức độ thực thi (theo yêu cầu hoặc theo cửa sổ thời gian).

5. Đọc tài liệu của LiteLLM, OpenRouter và Portkey song song. Nêu tên một tính năng mà mỗi bên cung cấp mà hai bên còn lại không có.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|------------------------|
| Routing gateway | "LLM proxy" | Lớp giao diện API duy nhất phía trước nhiều nhà cung cấp |
| Tương thích OpenAI | "Nói schema OpenAI" | Chấp nhận cấu trúc `/v1/chat/completions`, dịch sang bất kỳ backend nào |
| Model alias | "our_smart_model" | Tên trong code của bạn mà gateway ánh xạ tới model cụ thể |
| Chuỗi dự phòng | "Retry list" | Danh sách nhà cung cấp theo thứ tự được thử khi thất bại |
| Semantic caching | "Prompt-embedding cache" | Khóa là embedding của prompt; các bản sao gần giống chia sẻ cache hit |
| Guardrails | "Input/output filters" | Ẩn danh PII, từ chối vi phạm chính sách |
| Giới hạn tốc độ theo key | "Team budget" | Hạn ngạch giới hạn cho một API key |
| Theo dõi chi phí | "Per-request spend" | Tổng mức sử dụng token x giá mỗi model |
| LiteLLM | "The open proxy" | Routing gateway mã nguồn mở có thể tự lưu trữ |
| OpenRouter | "The managed SaaS" | Gateway được lưu trữ với thanh toán dựa trên tín dụng |
| Portkey | "The production option" | Mã nguồn mở + quản lý với guardrails tích hợp sẵn |

## Đọc thêm

- [LiteLLM — tài liệu](https://docs.litellm.ai/) — routing gateway tự lưu trữ
- [OpenRouter — quickstart](https://openrouter.ai/docs/quickstart) — SaaS routing được quản lý
- [Portkey — tài liệu](https://portkey.ai/docs) — routing sản xuất với guardrails
- [TrueFoundry — LiteLLM vs OpenRouter](https://www.truefoundry.com/blog/litellm-vs-openrouter) — hướng dẫn ra quyết định
- [Relayplane — so sánh LLM gateway 2026](https://relayplane.com/blog/llm-gateway-comparison-2026) — khảo sát nhà cung cấp