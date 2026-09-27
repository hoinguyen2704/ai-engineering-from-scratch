# Agent Observability: Langfuse, Phoenix, Opik

> Ba nền tảng observability cho agent mã nguồn mở thống trị năm 2026. Langfuse (MIT) — 6 triệu+ lượt cài đặt/tháng, tracing + quản lý prompt + evals + phát lại phiên (session replay). Arize Phoenix (Elastic 2.0) — đánh giá chuyên sâu cho agent, độ liên quan của RAG, tự động đo lường (auto-instrumentation) OpenInference. Comet Opik (Apache 2.0) — tối ưu hóa prompt tự động, guardrails, phát hiện ảo giác bằng LLM-judge.

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 23 (OTel GenAI)
**Time:** ~45 phút

## Mục tiêu học tập

- Kể tên ba nền tảng observability cho agent hàng đầu và giấy phép của chúng.
- Phân biệt thế mạnh của từng nền tảng: Langfuse (quản lý prompt + phiên), Phoenix (RAG + tự động đo lường), Opik (tối ưu hóa + guardrails).
- Giải thích lý do tại sao 89% tổ chức báo cáo đã triển khai observability cho agent vào năm 2026.
- Triển khai pipeline trace-to-dashboard bằng stdlib với đánh giá LLM-judge.

## Vấn đề

OTel GenAI (Bài 23) cung cấp cho bạn schema. Bạn vẫn cần nền tảng để tiếp nhận các span, chạy đánh giá, lưu trữ phiên bản prompt và hiển thị các lỗi hồi quy (regressions). Ba ứng cử viên này đều nhấn mạnh vào các phần khác nhau của vòng đời phát triển.

## Khái niệm

### Langfuse (MIT)

- 6 triệu+ lượt cài đặt SDK/tháng, 19k+ sao trên GitHub.
- Tính năng: tracing, quản lý prompt với phiên bản + playground, đánh giá (LLM-as-judge, phản hồi người dùng, tùy chỉnh), phát lại phiên.
- Tháng 6 năm 2025: các module thương mại trước đây (LLM-as-a-judge, hàng đợi chú thích, thử nghiệm prompt, Playground) đã được mở mã nguồn theo giấy phép MIT.
- Thế mạnh: observability toàn diện với vòng lặp quản lý prompt chặt chẽ.

### Arize Phoenix (Elastic License 2.0)

- Đánh giá chuyên sâu cho agent: phân cụm trace, phát hiện bất thường, độ liên quan của truy xuất cho RAG.
- Tự động đo lường OpenInference gốc.
- Kết hợp với Arize AX được quản lý cho môi trường production.
- Không có quản lý phiên bản prompt — được định vị là công cụ phát hiện drift/hồi quy hành vi bên cạnh các nền tảng rộng hơn.
- Thế mạnh: độ liên quan của RAG, drift hành vi, phát hiện bất thường.

### Comet Opik (Apache 2.0)

- Tối ưu hóa prompt tự động thông qua thử nghiệm A/B.
- Guardrails (xóa thông tin PII, ràng buộc chủ đề).
- Phát hiện ảo giác bằng LLM-judge.
- Benchmark từ phép đo của chính Comet: Opik ghi log + đánh giá trong 23.44 giây so với Langfuse 327.15 giây (khoảng cách ~14 lần) — hãy coi các benchmark từ nhà cung cấp chỉ mang tính tham khảo.
- Thế mạnh: vòng lặp tối ưu hóa, thử nghiệm tự động, thực thi guardrail.

### Dữ liệu ngành

Theo Maxim (phân tích thực địa năm 2026): 89% tổ chức đã triển khai observability cho agent; các vấn đề về chất lượng là rào cản lớn nhất trong production (32% người được hỏi trích dẫn điều này).

### Lựa chọn nền tảng

| Nhu cầu | Lựa chọn |
|------|------|
| Tất cả trong một với quản lý prompt | Langfuse |
| Đánh giá RAG chuyên sâu + drift | Phoenix |
| Tối ưu hóa tự động + guardrails | Opik |
| Giấy phép mở, không dùng ELv2 | Langfuse (MIT) hoặc Opik (Apache 2.0) |
| Tích hợp Datadog / New Relic | Bất kỳ cái nào — tất cả đều xuất OTel |

### Những sai lầm thường gặp

- **Không có chiến lược đánh giá.** Tracing mà không có đánh giá chỉ là ghi log tốn kém.
- **Tự xây dựng LLM-judge mà không có grounding.** Mô hình CRITIC (Bài 05) được áp dụng — các judge cần công cụ bên ngoài để xác minh thực tế.
- **Phiên bản prompt không gắn liền với trace.** Khi production bị lỗi, bạn không thể truy ngược lại prompt nào đã gây ra lỗi đó.

```figure
wb-trace-ingest
```

## Xây dựng

`code/main.py` triển khai một bộ thu thập trace stdlib + trình đánh giá LLM-judge:

- Tiếp nhận các span theo định dạng GenAI.
- Nhóm theo phiên, gắn thẻ các lần chạy thất bại (vượt quá guardrail, đánh giá độ tin cậy thấp).
- Một LLM-judge được viết kịch bản để chấm điểm phản hồi của agent dựa trên một rubric.
- Tóm tắt dạng dashboard: tỷ lệ thất bại, lý do thất bại hàng đầu, phân phối điểm đánh giá.

Chạy nó:

```
python3 code/main.py
```

Đầu ra: điểm đánh giá theo phiên và phân loại thất bại tương tự như những gì Langfuse/Phoenix/Opik sẽ hiển thị.

## Sử dụng

- **Langfuse** tự lưu trữ hoặc cloud; kết nối qua OTel hoặc SDK của họ.
- **Arize Phoenix** tự lưu trữ; tự động đo lường OpenInference.
- **Comet Opik** tự lưu trữ hoặc cloud; vòng lặp tối ưu hóa tự động.
- **Datadog LLM Observability** cho các nhóm vận hành + ML hỗn hợp đã sử dụng Datadog.

## Triển khai

`outputs/skill-obs-platform-wiring.md` chọn một nền tảng và kết nối các trace + đánh giá + phiên bản prompt vào một agent hiện có.

## Bài tập

1. Xuất một tuần trace OTel sang Langfuse cloud (gói miễn phí). Những phiên nào đã thất bại? Tại sao?
2. Viết một rubric LLM-judge cho lĩnh vực của bạn (độ chính xác thực tế, giọng điệu, tuân thủ phạm vi). Kiểm tra trên 50 trace.
3. So sánh quản lý phiên bản prompt của Langfuse với phân cụm trace của Phoenix. Cái nào giúp bạn biết cái gì bị hỏng nhanh hơn?
4. Đọc tài liệu về guardrail của Opik. Kết nối một guardrail xóa PII vào một trong các lần chạy agent của bạn.
5. Benchmark cả ba trên tập dữ liệu của bạn. Bỏ qua các con số do nhà cung cấp công bố; hãy tự đo lường.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|------------------------|
| Tracing | "Bộ thu thập span" | Tiếp nhận span OTel / SDK; lập chỉ mục theo phiên |
| Quản lý prompt | "Prompt CMS" | Các prompt có phiên bản gắn liền với trace |
| LLM-as-judge | "Đánh giá tự động" | LLM riêng biệt chấm điểm đầu ra của agent dựa trên rubric |
| Phát lại phiên | "Phát lại trace" | Xem lại các lần chạy trước để gỡ lỗi |
| Độ liên quan RAG | "Chất lượng truy xuất" | Ngữ cảnh được truy xuất có khớp với truy vấn không |
| Phân cụm trace | "Nhóm hành vi" | Nhóm các lần chạy tương tự để phát hiện drift |
| Thực thi guardrail | "Chính sách khi ghi log" | Kiểm tra PII/độc hại/phạm vi trên nội dung được ghi log |

## Đọc thêm

- [Tài liệu Langfuse](https://langfuse.com/) — tracing, đánh giá, quản lý prompt
- [Tài liệu Arize Phoenix](https://docs.arize.com/phoenix) — tự động đo lường, drift
- [Comet Opik](https://www.comet.com/site/products/opik/) — tối ưu hóa + guardrails
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — schema mà cả ba nền tảng đều sử dụng