# Production Agent Runtimes — Fast Instantiation and Typed Workflows

> Một production agent runtime tối ưu hóa những gì mà các framework tạo mẫu (prototyping) thường bỏ qua: chi phí khởi tạo (instantiation cost), bề mặt workflow có kiểu dữ liệu (typed workflow surfaces) và backend sẵn sàng để phục vụ (serving-ready). Cặp đôi của năm 2026: Agno (Python) hướng tới việc khởi tạo agent trong micro giây và các stateless FastAPI backend. Mastra cung cấp các agent, tool, workflow, định tuyến model thống nhất (unified model routing) và lưu trữ tổng hợp (composite storage) trên nền tảng Vercel AI SDK.

**Type:** Learn
**Languages:** Python, TypeScript
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 13 (LangGraph)
**Time:** ~45 minutes

## Learning Objectives

- Xác định các mục tiêu hiệu năng của Agno và thời điểm chúng trở nên quan trọng.
- Nêu tên ba nguyên thủy (primitives) của Mastra — Agents, Tools, Workflows — và các server adapter được hỗ trợ.
- Giải thích lý do tại sao stateless FastAPI backend có phạm vi session là con đường production được khuyến nghị cho Agno.
- Lựa chọn giữa Agno và Mastra cho một stack cụ thể (ưu tiên Python hoặc ưu tiên TypeScript).

## The Problem

LangGraph, AutoGen, CrewAI là các framework khá nặng nề. Các đội ngũ muốn "chỉ cần vòng lặp agent, nhanh chóng, trong runtime của tôi" thường tìm đến Agno (Python) hoặc Mastra (TypeScript). Cả hai đều đánh đổi một số nguyên thủy do framework sở hữu để lấy tốc độ thô và sự phù hợp chặt chẽ hơn với stack xung quanh.

## The Concept

### Agno

- Python runtime, trước đây là Phi-data.
- "Không đồ thị, không chuỗi, không các mô hình phức tạp — chỉ thuần Python."
- Các mục tiêu hiệu năng từ tài liệu của họ: ~2μs để khởi tạo agent, ~3.75 KiB bộ nhớ mỗi agent, ~23 nhà cung cấp model.
- Con đường production: stateless FastAPI backend có phạm vi session. Mỗi request bắt đầu một agent mới; trạng thái session nằm trong DB.
- Hỗ trợ đa phương thức (văn bản, hình ảnh, âm thanh, video, tệp) và agentic RAG.

Các mục tiêu về tốc độ trở nên quan trọng khi bạn có hàng nghìn agent tồn tại trong thời gian ngắn mỗi giây (chat fan-in, các pipeline đánh giá). Chúng ít quan trọng hơn khi một agent chạy trong 10 phút.

### Mastra

- TypeScript, được xây dựng trên Vercel AI SDK.
- Ba nguyên thủy: **Agents**, **Tools** (được định kiểu bằng Zod), **Workflows**.
- Unified Model Router — hơn 3.300 model từ 94 nhà cung cấp (tháng 3 năm 2026).
- Lưu trữ tổng hợp: bộ nhớ, workflow, khả năng quan sát (observability) tới các backend khác nhau; ClickHouse được khuyến nghị cho khả năng quan sát ở quy mô lớn.
- Apache 2.0 với các thư mục `ee/` theo giấy phép doanh nghiệp source-available.
- Server adapter cho Express, Hono, Fastify, Koa; tích hợp hạng nhất với Next.js và Astro.
- Cung cấp Mastra Studio (localhost:4111) để gỡ lỗi.
- Hơn 22k sao trên GitHub, 300k lượt tải xuống npm hàng tuần ở phiên bản 1.0 (tháng 1 năm 2026).

### Positioning

Không bên nào cố gắng trở thành LangGraph. Họ cạnh tranh dựa trên:

- **Sự phù hợp về ngôn ngữ.** Agno cho các đội ngũ ưu tiên Python; Mastra cho các đội ngũ ưu tiên TypeScript.
- **Công thái học của Runtime.** Agno = overhead gần bằng 0; Mastra = tích hợp với hệ sinh thái Vercel.
- **Khả năng quan sát.** Cả hai đều tích hợp với Langfuse/Phoenix/Opik (Bài 24) nhưng Mastra Studio là công cụ chính chủ.

### When to pick each

- **Agno** — Backend Python, nhiều agent tồn tại trong thời gian ngắn, yêu cầu hiệu năng cao, sử dụng FastAPI.
- **Mastra** — Backend TypeScript, triển khai trên Next.js / Vercel, định tuyến model đa nhà cung cấp thống nhất, tool được định kiểu bằng Zod.
- **LangGraph** (Bài 13) — khi trạng thái bền vững (durable state) và suy luận đồ thị rõ ràng quan trọng hơn tốc độ thô.
- **OpenAI / Claude Agent SDK** — khi bạn muốn hình thái sản phẩm của chính nhà cung cấp (Bài 16–17).

### Where this pattern goes wrong

- **Hiệu năng vì mục đích hiệu năng.** Chọn Agno vì "2μs" nghe có vẻ hay trong khi workload chỉ là một lệnh gọi agent chậm mỗi request. Overhead không phải là nút thắt cổ chai.
- **Sự phụ thuộc vào hệ sinh thái.** Tích hợp theo phong cách Vercel của Mastra là một điểm cộng trên Vercel, nhưng là điểm trừ ở nơi khác.
- **Sự nhầm lẫn về giấy phép doanh nghiệp.** Các thư mục `ee/` của Mastra là source-available, không phải Apache 2.0. Hãy đọc kỹ giấy phép nếu bạn định fork.

```figure
wb-runtime-spawn
```

## Build It

Bài học này chủ yếu mang tính so sánh — không có một đoạn mã đơn lẻ nào có thể đại diện công bằng cho cả hai framework. Xem `code/main.py` để thấy một ví dụ so sánh song song: một luồng "chạy agent, stream đầu ra, lưu trữ session" tối giản được triển khai hai lần (một theo kiểu Agno, một theo kiểu Mastra).

Chạy nó:

```
python3 code/main.py
```

Hai dấu vết (trace) khác nhau về cấu trúc nhưng tương đương về chức năng.

## Use It

- **Agno** — Backend Python cần tốc độ và cấu trúc FastAPI.
- **Mastra** — Backend TypeScript với nhiều nhà cung cấp và các nguyên thủy workflow.
- Cả hai đều cung cấp các hook quan sát chính chủ. Cả hai đều tích hợp với Langfuse.

## Ship It

`outputs/skill-runtime-picker.md` chọn Agno, Mastra, LangGraph hoặc SDK của nhà cung cấp dựa trên stack, ngân sách độ trễ và hình thái vận hành.

## Exercises

1. Đọc tài liệu của Agno. Chuyển đổi vòng lặp ReAct tiêu chuẩn (Bài 01) sang Agno. Những gì đã biến mất? Những gì được giữ lại?
2. Đọc tài liệu của Mastra. Chuyển đổi cùng vòng lặp đó sang Mastra. Điều gì đã thay đổi trong việc định kiểu tool (Zod so với không có gì)?
3. Benchmark: đo độ trễ khởi tạo agent trên stack của bạn. Liệu 2μs của Agno có quan trọng đối với workload của bạn không?
4. Thiết kế một quá trình di chuyển: nếu bạn đang chạy CrewAI bằng Python, điều gì sẽ bị hỏng nếu bạn chuyển sang Agno?
5. Đọc các điều khoản giấy phép `ee/` của Mastra. Những hạn chế nào sẽ ảnh hưởng đến một bản fork mã nguồn mở?

## Key Terms

| Thuật ngữ | Mọi người nói gì | Ý nghĩa thực sự |
|------|----------------|------------------------|
| Agno | "Agent Python nhanh" | Runtime agent stateless có phạm vi session |
| Mastra | "Agent TypeScript trên Vercel AI SDK" | Agents + Tools + Workflows + Model Router |
| Unified Model Router | "Truy cập đa nhà cung cấp" | Client duy nhất cho hơn 3.300 model từ 94 nhà cung cấp |
| Composite storage | "Nhiều backend" | Bộ nhớ/workflow/khả năng quan sát tới các kho lưu trữ khác nhau |
| Mastra Studio | "Trình gỡ lỗi cục bộ" | Giao diện localhost:4111 để kiểm tra agent |
| Source-available | "Không phải OSS" | Giấy phép cho phép đọc mã nguồn nhưng hạn chế sử dụng thương mại |

## Further Reading

- [Tài liệu Agno Agent Framework](https://www.agno.com/agent-framework) — mục tiêu hiệu năng, tích hợp FastAPI
- [Tài liệu Mastra](https://mastra.ai/docs) — các nguyên thủy, server adapter, Model Router
- [Tổng quan về LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) — giải pháp thay thế đồ thị có trạng thái
- [Comet Opik](https://www.comet.com/site/products/opik/) — so sánh khả năng quan sát được trích dẫn bởi các tích hợp của Mastra