# Multi-Agent Debate and Collaboration

> Du et al. (ICML 2024, "Society of Minds") vận hành N instance của model để đưa ra các đề xuất độc lập, sau đó lặp lại việc phê bình lẫn nhau qua R vòng để hội tụ. Phương pháp này cải thiện tính xác thực, khả năng tuân thủ quy tắc và suy luận. Cấu trúc liên kết thưa (sparse topology) mang lại hiệu quả tốt hơn cấu trúc lưới đầy đủ (full mesh) về chi phí token.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 12 (Workflow Patterns), Phase 14 · 05 (Self-Refine and CRITIC)
**Time:** ~60 phút

## Mục tiêu học tập

- Giải thích giao thức tranh luận: N người đề xuất, R vòng, hội tụ về một câu trả lời chung.
- Mô tả lý do tại sao tranh luận cải thiện tính xác thực, khả năng tuân thủ quy tắc và suy luận.
- Giải thích cấu trúc liên kết thưa: không phải mọi người tranh luận đều cần thấy tất cả những người khác.
- Triển khai tranh luận bằng stdlib trên một LLM kịch bản với các biến thể full-mesh và sparse; đo lường chi phí token so với độ chính xác.

## Vấn đề

Self-Refine (Bài 05) là việc một model tự phê bình chính nó — có nguy cơ dẫn đến tư duy nhóm (groupthink). CRITIC (Bài 05) dựa trên các công cụ bên ngoài để phê bình — không phải lúc nào cũng khả dụng. Tranh luận giới thiệu một phương thức thứ ba: nhiều instance, phê bình chéo, hội tụ thông qua sự bất đồng.

## Khái niệm

### Society of Minds (Du et al., ICML 2024)

- N instance của model độc lập đưa ra các đề xuất cho cùng một câu hỏi.
- Qua R vòng, mỗi model đọc các đề xuất của những model khác và phê bình chúng.
- Các model cập nhật câu trả lời của mình dựa trên các phê bình.
- Sau R vòng, trả về câu trả lời hội tụ.

Các thí nghiệm gốc sử dụng N=3, R=2 do chi phí. Độ chính xác cải thiện khi có nhiều agent và nhiều vòng hơn đối với các bài toán khó (MMLU, GSM8K, tính hợp lệ của nước đi cờ vua, tạo tiểu sử).

Sự kết hợp giữa các model khác nhau mang lại kết quả tốt hơn tranh luận đơn model: ChatGPT + Bard cùng nhau > bất kỳ model nào đứng một mình.

### Cấu trúc liên kết thưa (Sparse topology)

"Improving Multi-Agent Debate with Sparse Communication Topology" (arXiv:2406.11776, 2024-2025) cho thấy tranh luận full-mesh không phải lúc nào cũng tối ưu. Các cấu trúc liên kết thưa (hình sao, vòng, hub-and-spoke) có thể đạt được độ chính xác tương đương với chi phí token thấp hơn. Mỗi người tranh luận chỉ thấy một tập con các đồng nghiệp.

Hệ quả:

- Full mesh N=5, R=3 = 5 × 3 = 15 đề xuất, mỗi đề xuất đọc 4 đồng nghiệp = 60 thao tác phê bình.
- Star N=5, R=3 (một hub + 4 spoke) = 15 đề xuất, các spoke chỉ đọc hub = 12 thao tác phê bình.

### Khi nào tranh luận hữu ích

- **Tính xác thực.** N đề xuất độc lập, kiểm tra chéo giúp giảm thiểu ảo giác (hallucination).
- **Tuân thủ quy tắc.** Tính hợp lệ của nước đi cờ vua — một model bỏ sót quy tắc, những model khác sẽ phát hiện ra.
- **Suy luận mở.** Nhiều cách đặt vấn đề giúp thu hẹp phạm vi để tìm ra câu trả lời đúng.

### Khi nào tranh luận gây hại

- **UX nhạy cảm với độ trễ.** N × R vòng tuần tự là độ trễ mà bạn có thể không có.
- **Quy mô nhạy cảm với chi phí.** N × R token cho mỗi câu hỏi.
- **Tra cứu thực tế đơn giản.** Một lần tra cứu rẻ hơn năm lần tranh luận.

### Các triển khai thực tế năm 2026

- **Anthropic orchestrator-workers** (Bài 12) — một biến thể của tranh luận với bước tổng hợp.
- **LangGraph supervisor** (Bài 13) — bộ định tuyến trung tâm + các agent chuyên gia có thể triển khai tranh luận như một node.
- **OpenAI Agents SDK** (Bài 16) — các agent bàn giao qua lại để phê bình lặp đi lặp lại.
- **Đánh giá đa agent (Multi-agent evals)** — kết hợp tranh luận + trình tối ưu hóa đánh giá để lấy tín hiệu đánh giá.

### Nơi mô hình này đi chệch hướng

- **Sụp đổ hội tụ (Convergence collapse).** Tất cả các agent hội tụ về câu trả lời sai đầu tiên. Giảm thiểu bằng cách yêu cầu các vòng bất đồng.
- **Lỗi hub.** Trong cấu trúc hình sao, một hub tồi sẽ làm hỏng tất cả mọi người. Hãy xoay vòng hoặc sử dụng nhiều hub.
- **Đồng nhất hóa prompt.** Tất cả các agent sử dụng cùng một prompt; chúng tạo ra các câu trả lời giống nhau. Hãy sử dụng các prompt và/hoặc model đa dạng.

```figure
debate-converge
```

## Xây dựng

`code/main.py` triển khai tranh luận bằng stdlib:

- Lớp `Debater` (LLM kịch bản với sự trôi dạt ý kiến theo từng người tranh luận).
- Các runner `FullMeshDebate` và `SparseDebate`.
- Ba câu hỏi: một câu về thực tế, một câu dựa trên quy tắc, một câu về suy luận.
- Các chỉ số: câu trả lời hội tụ, số vòng để hội tụ, tổng số thao tác phê bình.

Chạy nó:

```
python3 code/main.py
```

Đầu ra: độ chính xác và chi phí theo từng giao thức; cấu trúc thưa khớp với full mesh trên 2/3 câu hỏi với chi phí thấp hơn.

## Sử dụng

- **Anthropic orchestrator-workers** cho các cuộc tranh luận đơn giản với 2-3 worker.
- **LangGraph** cho tranh luận đa vòng có trạng thái với checkpointing.
- **Tùy chỉnh** cho nghiên cứu hoặc các đảm bảo tính đúng đắn chuyên biệt.

## Triển khai

`outputs/skill-debate.md` tạo khung cho một cuộc tranh luận đa agent với cấu trúc liên kết, N, R có thể cấu hình và quy tắc hội tụ.

## Bài tập

1. Triển khai quy tắc "bắt buộc bất đồng": ở vòng 1, mỗi người tranh luận phải đưa ra một đề xuất khác biệt. Đo lường ảnh hưởng đến tốc độ hội tụ.
2. Thêm tổng hợp có trọng số tin cậy: các người tranh luận trả về (câu trả lời, độ tin cậy); bộ tổng hợp tính trọng số theo độ tin cậy. Nó có giúp ích không?
3. Thay thế một "agent" bằng một LLM kịch bản khác với các ý kiến khác nhau. Tính không đồng nhất có cải thiện độ chính xác không?
4. Đo chi phí token cho full mesh so với sparse trên 3 câu hỏi của bạn. Vẽ biểu đồ chi phí so với độ chính xác.
5. Đọc bài báo Society of Minds. Chuyển đổi mã nguồn của bạn sang N=5, R=3. Điều gì bị hỏng? Điều gì trở nên tốt hơn?

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|------------------------|
| Debate | "Phê bình đa agent" | N người đề xuất, R vòng phê bình chéo, hội tụ |
| Full mesh | "Mọi người đọc mọi người" | Mỗi người tranh luận đọc mọi đồng nghiệp mỗi vòng |
| Sparse topology | "Góc nhìn đồng nghiệp hạn chế" | Người tranh luận chỉ đọc một tập con các đồng nghiệp |
| Hub-and-spoke | "Cấu trúc hình sao" | Một người tranh luận trung tâm, N-1 spoke chỉ đọc hub |
| Convergence | "Sự đồng thuận" | Các người tranh luận hội tụ về một câu trả lời chung |
| Society of Minds | "Bài báo tranh luận của Du et al." | Phương pháp tranh luận đa agent ICML 2024 |

## Đọc thêm

- [Du et al., Society of Minds (arXiv:2305.14325)](https://arxiv.org/abs/2305.14325) — bài báo tranh luận đa agent kinh điển
- [Sparse Communication Topology (arXiv:2406.11776)](https://arxiv.org/abs/2406.11776) — kết quả về cấu trúc liên kết thưa
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — orchestrator-workers như một biến thể tranh luận
- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) — đối trọng tự phê bình đơn model