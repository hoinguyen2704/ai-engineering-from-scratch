# Failure Modes: Why Agents Break

> MASFT (Berkeley, 2025) phân loại 14 kiểu lỗi (failure modes) trong hệ thống đa tác nhân (multi-agent) thành 3 nhóm. Taxonomy của Microsoft ghi lại cách các lỗi AI hiện tại khuếch đại trong môi trường agentic. Dữ liệu thực tế từ ngành công nghiệp hội tụ vào năm kiểu lỗi thường gặp: hành động ảo tưởng (hallucinated actions), lan man ngoài phạm vi (scope creep), lỗi dây chuyền (cascading errors), mất ngữ cảnh (context loss), và sử dụng sai công cụ (tool misuse).

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 05 (Self-Refine and CRITIC), Phase 14 · 24 (Observability)
**Time:** ~60 phút

## Mục tiêu học tập

- Nêu tên ba nhóm lỗi của MASFT và ít nhất bốn kiểu lỗi cụ thể trong mỗi nhóm.
- Giải thích lý do tại sao lỗi agentic làm khuếch đại các kiểu lỗi AI hiện có (bias, hallucination).
- Mô tả năm kiểu lỗi thường gặp trong ngành và các biện pháp giảm thiểu.
- Triển khai một bộ phát hiện bằng stdlib để gắn nhãn các trace của agent với các kiểu lỗi.

## Vấn đề

Các đội ngũ triển khai agent hoạt động tốt trên 90% các trace. 10% lỗi còn lại không phải là nhiễu ngẫu nhiên — chúng rơi vào một số lượng nhỏ các danh mục lặp đi lặp lại. Một khi bạn có thể gọi tên chúng, bạn có thể giám sát và khắc phục chúng.

## Khái niệm

### MASFT (Berkeley, arXiv:2503.13657)

Multi-Agent System Failure Taxonomy. 14 kiểu lỗi được gom thành 3 nhóm. Chỉ số Cohen's Kappa giữa các người chú giải là 0.88 — các nhóm này có thể phân biệt một cách đáng tin cậy.

Khẳng định cốt lõi: các lỗi là những khiếm khuyết thiết kế cơ bản trong hệ thống đa tác nhân, không phải là hạn chế của LLM cần khắc phục bằng các base model tốt hơn.

### Microsoft Taxonomy of Failure Mode in Agentic AI Systems

- Các lỗi AI hiện có (bias, hallucination, data leakage) khuếch đại trong môi trường agentic.
- Các lỗi mới nảy sinh từ tính tự chủ: hành động ngoài ý muốn ở quy mô lớn, sử dụng sai công cụ, chệch hướng nhiệm vụ (mission drift).
- Whitepaper này đóng vai trò là sổ đăng ký rủi ro cho các sản phẩm agentic.

### Characterizing Faults in Agentic AI (arXiv:2603.06847)

- Lỗi phát sinh từ việc điều phối (orchestration), sự tiến hóa của trạng thái nội bộ và tương tác với môi trường.
- Không chỉ là "code tồi" hay "output model tồi".

### LLM Agent Hallucinations Survey (arXiv:2509.18970)

Hai biểu hiện chính:

1. **Lệch hướng tuân thủ chỉ dẫn (Instruction-following Deviation)** — agent không tuân theo system prompt.
2. **Sử dụng sai ngữ cảnh dài hạn (Long-range Contextual Misuse)** — agent quên hoặc áp dụng sai ngữ cảnh từ các lượt trước đó.

Các lỗi về ý định phụ (sub-intention): Bỏ sót (omission - bỏ lỡ bước), Dư thừa (redundancy - lặp lại bước), Rối loạn (disorder - sai thứ tự các bước).

### Năm kiểu lỗi thường gặp trong ngành

Các phân tích thực tế từ Arize, Galileo, NimbleBrain giai đoạn 2024-2026 hội tụ vào:

1. **Hành động ảo tưởng (Hallucinated actions).** Agent gọi một công cụ không tồn tại hoặc bịa đặt các đối số.
2. **Lan man ngoài phạm vi (Scope creep).** Agent mở rộng nhiệm vụ vượt quá yêu cầu của người dùng (tạo thêm PR, gửi thêm email).
3. **Lỗi dây chuyền (Cascading errors).** Một lệnh gọi sai kích hoạt các hiệu ứng hạ nguồn. Một SKU ảo tưởng kích hoạt bốn lệnh gọi API — một sự cố đa hệ thống.
4. **Mất ngữ cảnh (Context loss).** Các nhiệm vụ dài hạn quên đi các ràng buộc từ những lượt đầu.
5. **Sử dụng sai công cụ (Tool misuse).** Gọi đúng công cụ với đối số sai, hoặc gọi sai hoàn toàn công cụ.

Lỗi dây chuyền là nguy hiểm nhất. Agent không thể phân biệt được "tôi đã thất bại" với "nhiệm vụ là bất khả thi" và thường ảo tưởng ra một thông báo thành công đối với các lỗi 400 để đóng vòng lặp.

### Giảm thiểu: các cổng kiểm soát tại mỗi bước

Các cổng xác minh tự động tại mỗi bước của chuỗi suy luận, kiểm tra tính xác thực dựa trên trạng thái môi trường. Cụ thể:

- Bộ phân loại an toàn theo từng bước (Bài 21).
- Xác thực đối số lệnh gọi công cụ (Bài 06).
- Đối chiếu nội dung được truy xuất với các sự thật đã biết (Bài 05, CRITIC).
- Phát hiện ảo tưởng thành công bằng cách kiểm tra lại trạng thái (tệp có thực sự được tạo không?).

### Những sai lầm trong giám sát lỗi

- **Chỉ gắn nhãn các lỗi crash.** Hầu hết các lỗi của agent tạo ra output trông có vẻ hợp lệ. Cần các kiểm tra ở cấp độ nội dung.
- **Không có baseline.** Phát hiện drift cần một trạng thái "tốt cuối cùng đã biết"; nếu không có nó, bạn không thể nói "điều này đang trở nên tồi tệ hơn".
- **Cảnh báo quá mức.** Mọi lỗi đều tạo ra một thông báo. Hãy gom nhóm và giới hạn tần suất.

```figure
failure-cascade
```

## Xây dựng

`code/main.py` triển khai một bộ gắn nhãn kiểu lỗi bằng stdlib:

- Một tập dữ liệu trace tổng hợp bao gồm năm kiểu lỗi.
- Các hàm phát hiện cho từng kiểu lỗi (các mẫu chữ ký trên lệnh gọi công cụ, output, hành động lặp lại).
- Một bộ gắn nhãn dán nhãn cho từng trace và báo cáo phân phối kiểu lỗi.

Chạy nó:

```
python3 code/main.py
```

Output: nhãn cho từng trace + phân phối tổng hợp, một bản sao giá trị thấp của những gì tính năng gom nhóm trace của Phoenix hiển thị.

## Sử dụng

- **Phoenix** để gom nhóm drift trong sản xuất (Bài 24).
- **Langfuse** để phát lại phiên (session replay) + chú giải.
- **Custom** cho các chữ ký đặc thù của miền mà nền tảng quan sát của bạn không thể phát hiện.

## Triển khai

`outputs/skill-failure-detector.md` tạo ra các bộ phát hiện kiểu lỗi được tùy chỉnh cho miền của bạn, kết nối với kho lưu trữ trace.

## Bài tập

1. Thêm bộ phát hiện cho "ảo tưởng thành công": agent trả về thành công nhưng trạng thái mục tiêu không thay đổi.
2. Gắn nhãn 100 trace thực tế từ một sản phẩm bạn đã xây dựng. Kiểu lỗi nào chiếm ưu thế? Chi phí để khắc phục nó là bao nhiêu?
3. Triển khai chỉ số "bán kính dây chuyền" (cascade radius): nếu có lỗi ở bước N, nó ảnh hưởng đến bao nhiêu bước hạ nguồn?
4. Đọc 14 kiểu lỗi của MASFT. Chọn ba kiểu áp dụng cho sản phẩm của bạn. Viết các bộ phát hiện.
5. Kết nối một bộ phát hiện vào công việc CI: làm thất bại bản build nếu >=5% số trace bị gắn nhãn một kiểu lỗi.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|------------------------|
| MASFT | "Phân loại lỗi đa tác nhân" | Phân loại 14 kiểu lỗi của Berkeley |
| Cascading error | "Lỗi gợn sóng" | Một sai lầm sớm lan truyền qua N bước |
| Context loss | "Quên ràng buộc" | Lượt dài hạn làm mất các sự thật từ lượt đầu |
| Tool misuse | "Sai công cụ / sai đối số" | Lệnh gọi hợp lệ, nhưng cách gọi sai |
| Success hallucination | "Hoàn thành giả" | Agent báo thành công trên lỗi 400; trạng thái không đổi |
| Scope creep | "Vượt quá phạm vi" | Agent làm nhiều hơn yêu cầu |
| Instruction-following deviation | "Không tuân lệnh" | Phớt lờ system prompt hoặc ràng buộc người dùng |
| Sub-intention errors | "Lỗi kế hoạch" | Bỏ sót, dư thừa, rối loạn trong thực thi kế hoạch |

## Đọc thêm

- [Cemri et al., MASFT (arXiv:2503.13657)](https://arxiv.org/abs/2503.13657) — 14 kiểu lỗi, 3 nhóm
- [Microsoft, Taxonomy of Failure Mode in Agentic AI Systems](https://cdn-dynmedia-1.microsoft.com/is/content/microsoftcorp/microsoft/final/en-us/microsoft-brand/documents/Taxonomy-of-Failure-Mode-in-Agentic-AI-Systems-Whitepaper.pdf) — sổ đăng ký rủi ro
- [Arize Phoenix](https://docs.arize.com/phoenix) — gom nhóm drift trong thực tế
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — khi nào các mẫu đơn giản tránh được hoàn toàn các kiểu lỗi