# AI Gateways — LiteLLM, Portkey, Kong AI Gateway, Bifrost

> Một gateway nằm giữa các ứng dụng của bạn và các nhà cung cấp mô hình. Các tính năng cốt lõi bao gồm định tuyến nhà cung cấp (provider routing), dự phòng (fallback), thử lại (retries), giới hạn tốc độ (rate limiting), tham chiếu bí mật (secret references), khả năng quan sát (observability) và các rào cản bảo mật (guardrails). Phân khúc thị trường năm 2026: **LiteLLM** là mã nguồn mở MIT với hơn 100 nhà cung cấp, tương thích với OpenAI, nhưng gặp sự cố ở mức khoảng ~2000 RPS (8 GB bộ nhớ, lỗi dây chuyền trong các bài kiểm tra hiệu năng đã công bố); phù hợp nhất cho Python, <500 RPS, phát triển/tạo mẫu. **Portkey** được định vị là control-plane (guardrails, ẩn danh PII, phát hiện jailbreak, nhật ký kiểm toán), đã chuyển sang mã nguồn mở Apache 2.0 vào tháng 3 năm 2026, độ trễ tăng thêm 20-40 ms, giá $49/mo production tier. **Kong AI Gateway** built on Kong Gateway — Kong's own benchmark on same 12 CPUs: 228% faster than Portkey, 859% faster than LiteLLM; $100/mô hình/tháng (tối đa 5 trên gói Plus); phù hợp cho doanh nghiệp nếu bạn đã sử dụng Kong. **Bifrost** (Maxim AI) — tự động thử lại với cơ chế backoff có thể cấu hình, dự phòng sang Anthropic khi OpenAI trả về 429. **Cloudflare / Vercel AI Gateways** — được quản lý, không cần vận hành (zero-ops), thử lại cơ bản. Quyền lưu trú dữ liệu (data residency) là yếu tố thúc đẩy quyết định tự lưu trữ (self-host); Portkey và Kong nằm ở giữa với tùy chọn OSS + quản lý.

**Type:** Learn
**Languages:** Python (stdlib, toy gateway-routing simulator)
**Prerequisites:** Phase 17 · 01 (Managed LLM Platforms), Phase 17 · 16 (Model Routing)
**Time:** ~60 phút

## Mục tiêu học tập

- Liệt kê sáu tính năng cốt lõi của gateway (định tuyến, dự phòng, thử lại, giới hạn tốc độ, bí mật, khả năng quan sát, guardrails).
- Ánh xạ bốn gateway năm 2026 (LiteLLM, Portkey, Kong AI, Bifrost) với giới hạn quy mô và trường hợp sử dụng.
- Trích dẫn bài kiểm tra hiệu năng của Kong (228% so với Portkey, 859% so với LiteLLM) và giải thích tại sao nó quan trọng đối với >500 RPS.
- Lựa chọn giữa tự lưu trữ (self-hosted) và được quản lý (managed) dựa trên quyền lưu trú dữ liệu và ngân sách vận hành.

## Vấn đề

Sản phẩm của bạn gọi OpenAI, Anthropic và một Llama tự lưu trữ. Mỗi nhà cung cấp có SDK, mô hình lỗi, giới hạn tốc độ và cơ chế xác thực khác nhau. Bạn muốn có tính năng chuyển đổi dự phòng (nếu OpenAI trả về 429, hãy thử Anthropic), một kho lưu trữ thông tin xác thực duy nhất, khả năng quan sát thống nhất và giới hạn tốc độ cho mỗi khách hàng (tenant).

Việc xây dựng lại điều này ở tầng ứng dụng sẽ gắn chặt mọi dịch vụ với mọi nhà cung cấp. Một tầng gateway hợp nhất nó thành một quy trình với một API duy nhất (thường tương thích với OpenAI) để phân phối đến các nhà cung cấp.

## Khái niệm

### Sáu tính năng cốt lõi

1. **Provider routing** — OpenAI, Anthropic, Gemini, tự lưu trữ, v.v. nằm sau một API duy nhất.
2. **Fallback** — khi gặp lỗi 429, 5xx hoặc lỗi chất lượng, hãy thử lại ở nơi khác.
3. **Retries** — exponential backoff, giới hạn số lần thử.
4. **Rate limits** — theo từng tenant, theo từng key, theo từng mô hình.
5. **Secret references** — lấy thông tin xác thực từ vault tại thời điểm chạy (không bao giờ để trong ứng dụng).
6. **Observability** — OTel + các thuộc tính GenAI (Phase 17 · 13) + phân bổ chi phí.
7. **Guardrails** — ẩn danh PII, phát hiện jailbreak, bộ lọc chủ đề cho phép.

### LiteLLM — MIT OSS, Python

- Hơn 100 nhà cung cấp, tương thích OpenAI, cấu hình router, dự phòng, khả năng quan sát cơ bản.
- Gặp sự cố ở mức khoảng 2000 RPS trong bài kiểm tra của Kong; chiếm dụng 8 GB bộ nhớ, lỗi dây chuyền dưới tải trọng duy trì.
- Phù hợp nhất: Ứng dụng Python, <500 RPS, gateway cho dev/staging, định tuyến thử nghiệm.
- Chi phí: $0 cho OSS; có gói miễn phí trên cloud.

### Portkey — định vị control plane

- Mã nguồn mở Apache 2.0 từ tháng 3 năm 2026. Guardrails, ẩn danh PII, phát hiện jailbreak, nhật ký kiểm toán.
- Độ trễ tăng thêm 20-40 ms mỗi yêu cầu.
- $49/tháng cho gói production với lưu trữ dữ liệu + SLA.
- Phù hợp nhất: các ngành được quản lý cần tích hợp sẵn guardrails + khả năng quan sát.

### Kong AI Gateway — giải pháp quy mô

- Được xây dựng trên Kong Gateway (sản phẩm API gateway trưởng thành, lua+OpenResty).
- Bài kiểm tra của Kong trên cấu hình tương đương 12-CPU: nhanh hơn 228% so với Portkey, 859% so với LiteLLM.
- Giá: $100/mô hình/tháng, tối đa 5 trên gói Plus.
- Phù hợp nhất: đã sử dụng Kong; >1000 RPS; sẵn sàng trả phí bản quyền.

### Bifrost (Maxim AI)

- Tự động thử lại với backoff có thể cấu hình.
- Dự phòng sang Anthropic khi OpenAI trả về 429 là một công thức kinh điển.
- Người chơi mới; thương mại.

### Cloudflare AI Gateway / Vercel AI Gateway

- Được quản lý, không cần vận hành. Thử lại và khả năng quan sát cơ bản.
- Phù hợp nhất: Ứng dụng JavaScript chạy tại Edge trên Cloudflare/Vercel.
- Hạn chế so với Kong/Portkey về guardrails và giới hạn tốc độ.

### Tự lưu trữ vs được quản lý

Quyền lưu trú dữ liệu là yếu tố quyết định. Y tế và tài chính thường chọn tự lưu trữ (LiteLLM hoặc Portkey OSS hoặc Kong). Sản phẩm tiêu dùng thường chọn được quản lý (Cloudflare AI Gateway) hoặc phân khúc trung bình (Portkey managed). Lai (Hybrid): tự lưu trữ cho tenant được quản lý, được quản lý cho các tenant khác.

### Ngân sách độ trễ

- LiteLLM: độ trễ tăng thêm 5-15 ms.
- Portkey: độ trễ tăng thêm 20-40 ms.
- Kong: độ trễ tăng thêm 3-8 ms.
- Cloudflare/Vercel: độ trễ tăng thêm 1-3 ms (lợi thế tại Edge).

Độ trễ của Gateway cộng trực tiếp vào TTFT. Với SLA TTFT P99 < 100 ms, hãy chọn Kong hoặc Cloudflare. Với P99 < 500 ms, chọn bất kỳ cái nào.

### Ngữ nghĩa của giới hạn tốc độ rất quan trọng

Token-bucket đơn giản hoạt động tốt ở quy mô vừa phải. Đa tenant yêu cầu sliding-window + cho phép burst + phân tầng theo tenant. LiteLLM sử dụng token-bucket; Kong sử dụng sliding-window; Portkey sử dụng phân tầng.

### Gateway + khả năng quan sát + định tuyến kết hợp

Phase 17 · 13 (khả năng quan sát) + 16 (định tuyến mô hình) + 19 (gateways) là cùng một tầng trong môi trường production. Hãy chọn một công cụ bao phủ cả ba hoặc kết hợp chúng cẩn thận: hầu hết các triển khai năm 2026 kết hợp Helicone (khả năng quan sát) hoặc Portkey (guardrails) với Kong (quy mô) để phân chia vai trò.

### Các con số bạn cần nhớ

- LiteLLM: gặp sự cố ở ~2000 RPS, 8 GB bộ nhớ.
- Portkey: độ trễ tăng thêm 20-40 ms; Apache 2.0 từ tháng 3 năm 2026.
- Kong: nhanh hơn 228% so với Portkey, 859% so với LiteLLM.
- Giá Kong: $100/mô hình/tháng, tối đa 5 trên gói Plus.
- Cloudflare/Vercel: độ trễ tăng thêm 1-3 ms tại Edge.

```figure
mx-gateway-fallback
```

## Sử dụng

`code/main.py` mô phỏng định tuyến gateway với dự phòng qua 3 nhà cung cấp dưới sự tiêm nhiễm lỗi 429/5xx. Báo cáo độ trễ, tỷ lệ thử lại và tỷ lệ dự phòng thành công.

## Triển khai

Bài học này tạo ra `outputs/skill-gateway-picker.md`. Dựa trên quy mô, trạng thái vận hành, tuân thủ, ngân sách độ trễ, hãy chọn một gateway.

## Bài tập

1. Chạy `code/main.py`. Cấu hình dự phòng từ OpenAI→Anthropic→tự lưu trữ. Tỷ lệ thành công dự kiến là bao nhiêu ở mức 5% lỗi nhà cung cấp?
2. SLA của bạn là TTFT P99 < 200 ms trên nền tảng 300 ms. Những gateway nào nằm trong ngân sách?
3. Một khách hàng y tế yêu cầu tự lưu trữ + ẩn danh PII + kiểm toán. Chọn Portkey OSS hay Kong?
4. So sánh LiteLLM và Kong: ở ngưỡng RPS nào một đội ngũ nên chuyển đổi?
5. Thiết kế chính sách giới hạn tốc độ cho SaaS đa tenant: gói miễn phí, gói dùng thử, gói trả phí. Token-bucket hay sliding-window?

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|------------------------|
| Gateway | "API broker" | Quy trình nằm giữa ứng dụng và nhà cung cấp |
| LiteLLM | "cái MIT" | Python OSS, 100+ nhà cung cấp, gặp sự cố ở 2K RPS |
| Portkey | "gateway guardrails" | Control plane + khả năng quan sát, Apache 2.0 |
| Kong AI Gateway | "cái cho quy mô" | Xây dựng trên Kong Gateway, dẫn đầu về hiệu năng |
| Bifrost | "gateway của Maxim" | Thử lại + công thức dự phòng Anthropic |
| Cloudflare AI Gateway | "edge managed" | Gateway được quản lý tại Edge, không cần vận hành |
| Ẩn danh PII | "làm sạch dữ liệu" | Regex + NER mask trước khi gửi đến mô hình |
| Phát hiện jailbreak | "bảo vệ prompt injection" | Bộ phân loại trên đầu vào người dùng |
| Nhật ký kiểm toán | "log quy định" | Bản ghi bất biến của mọi cuộc gọi LLM |
| Token-bucket | "giới hạn tốc độ đơn giản" | Bộ giới hạn tốc độ dựa trên việc nạp lại |
| Sliding-window | "giới hạn tốc độ chính xác" | Bộ giới hạn tốc độ theo cửa sổ thời gian; công bằng hơn |

## Đọc thêm

- [Kong AI Gateway Benchmark](https://konghq.com/blog/engineering/ai-gateway-benchmark-kong-ai-gateway-portkey-litellm)
- [TrueFoundry — AI Gateways 2026 Comparison](https://www.truefoundry.com/blog/a-definitive-guide-to-ai-gateways-in-2026-competitive-landscape-comparison)
- [Techsy — Top LLM Gateway Tools 2026](https://techsy.io/en/blog/best-llm-gateway-tools)
- [LiteLLM GitHub](https://github.com/BerriAI/litellm)
- [Portkey GitHub](https://github.com/Portkey-AI/gateway)
- [Kong AI Gateway docs](https://docs.konghq.com/gateway/latest/ai-gateway/)