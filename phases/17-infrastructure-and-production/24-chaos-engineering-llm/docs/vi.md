# Chaos Engineering cho LLM Production

> Chaos engineering cho LLM là một chuyên ngành riêng biệt vào năm 2026. Các điều kiện tiên quyết trước khi chạy thử nghiệm trong môi trường production: SLI/SLO được xác định rõ, khả năng quan sát (observability) bao gồm trace+metric+log, khả năng rollback tự động, runbook và đội ngũ on-call. Kiến trúc bao gồm bốn mặt phẳng (plane): control (lập lịch thử nghiệm), target (dịch vụ, hạ tầng, kho dữ liệu), safety (cơ chế bảo vệ + hủy bỏ + bộ lọc lưu lượng), observability (metric + trace + log), và feedback (phản hồi để điều chỉnh SLO). Các chốt chặn an toàn (guardrails) là bắt buộc: cảnh báo burn-rate sẽ tạm dừng thử nghiệm nếu mức tiêu thụ ngân sách lỗi (error-budget) hàng ngày > 2 lần dự kiến; các cửa sổ chặn (suppression windows) + tương quan trace-ID giúp loại bỏ nhiễu cảnh báo. Nhịp độ: hàng tuần thực hiện canary nhỏ + đánh giá SLO; hàng tháng tổ chức game day + postmortem; hàng quý kiểm toán khả năng phục hồi liên phòng ban + lập bản đồ phụ thuộc. Các thử nghiệm đặc thù cho LLM: quá tải bộ nhớ, lỗi mạng, gián đoạn nhà cung cấp, prompt sai định dạng, bão giải phóng KV cache. Công cụ: Harness Chaos Engineering (đề xuất dựa trên LLM, giảm thiểu blast-radius, tích hợp công cụ MCP); LitmusChaos (CNCF); Chaos Mesh (CNCF Kubernetes-native).

**Type:** Learn
**Languages:** Python (stdlib, toy chaos experiment runner)
**Prerequisites:** Phase 17 · 23 (SRE for AI), Phase 17 · 13 (Observability)
**Time:** ~60 phút

## Mục tiêu học tập

- Nêu tên năm điều kiện tiên quyết của chaos engineering (SLI/SLO, observability, rollback, runbook, on-call) và giải thích tại sao việc bỏ qua bất kỳ yếu tố nào cũng làm hỏng quy trình.
- Vẽ sơ đồ bốn mặt phẳng (control, target, safety, observability) và vòng lặp phản hồi vào SLO.
- Liệt kê năm thử nghiệm đặc thù cho LLM (quá tải bộ nhớ, lỗi mạng, gián đoạn nhà cung cấp, prompt sai định dạng, bão giải phóng KV cache).
- Chọn một công cụ — Harness, LitmusChaos, Chaos Mesh — dựa trên stack hiện có.

## Vấn đề

Chaos testing trong các stack truyền thống đã rất phổ biến. Các stack LLM bổ sung thêm những chế độ lỗi mới. Một prompt 4K-token với ký tự độc hại có thể làm treo tokenizer trong 12 giây. Một nhà cung cấp upstream trả về lỗi 429; gateway của bạn thử lại; dịch vụ của bạn bị OOM do sự gia tăng đồng thời khi thử lại. Một cơn bão giải phóng KV cache dưới tải đột biến gây ra các chuỗi re-prefill làm bão hòa tài nguyên tính toán.

Không có lỗi nào trong số này xuất hiện trong unit test. Chaos engineering là cách bạn khám phá chúng trước khi người dùng gặp phải.

## Khái niệm

### Điều kiện tiên quyết

Đừng chạy chaos trong production nếu thiếu:

1. **SLI/SLO** — các chỉ số và mục tiêu cấp độ dịch vụ đã được xác định.
2. **Observability** — trace, metric, log được kết nối với dashboard.
3. **Automated rollback** — chính sách rollback theo Phase 17 · 20.
4. **Runbooks** — tài liệu hướng dẫn vận hành, Phase 17 · 23.
5. **On-call** — nhân sự sẵn sàng phản ứng.

Thiếu bất kỳ yếu tố nào, chaos sẽ trở thành sự cố thực sự.

### Bốn mặt phẳng + phản hồi

**Control plane** — bộ lập lịch thử nghiệm (quy trình Litmus, lịch trình Chaos Mesh, giao diện Harness).

**Target plane** — các dịch vụ, pod, node, load balancer, kho dữ liệu.

**Safety plane** — công tắc ngắt khẩn cấp (kill switch), cửa sổ chặn, giới hạn phạm vi ảnh hưởng (blast-radius), cổng kiểm soát ngân sách lỗi.

**Observability plane** — các metric thông thường + tương quan trace-ID để phân biệt lỗi do chaos gây ra với lỗi tự nhiên.

**Feedback loop** — các phát hiện được phản hồi để điều chỉnh SLO, cập nhật runbook, sửa lỗi code.

### Các chốt chặn an toàn (Guardrails) là bắt buộc

- **Cảnh báo burn-rate**: tạm dừng thử nghiệm nếu mức tiêu thụ ngân sách lỗi hàng ngày vượt quá 2 lần dự kiến.
- **Cửa sổ chặn (Suppression windows)**: tắt các cảnh báo không liên quan đến thử nghiệm trong phạm vi ảnh hưởng.
- **Tương quan trace-ID**: tất cả lỗi do thử nghiệm gây ra đều mang thẻ tag để đội ngũ on-call có thể lọc nhiễu.

### Năm thử nghiệm đặc thù cho LLM

1. **Quá tải bộ nhớ** — ép buộc bão giải phóng KV cache bằng cách gửi các yêu cầu context dài với độ đồng thời cao. Quan sát: dịch vụ có xử lý duyên dáng hay bị crash?

2. **Lỗi mạng** — cắt kết nối giữa inference gateway và nhà cung cấp. Quan sát: cơ chế fallback có kích hoạt trong SLA không? (Phase 17 · 19)

3. **Mô phỏng gián đoạn nhà cung cấp** — trả về 100% lỗi 429 từ OpenAI. Quan sát: routing có chuyển hướng sang Anthropic không? (Phase 17 · 16, 19)

4. **Prompt sai định dạng** — chèn payload làm treo tokenizer (ví dụ: unicode lồng nhau sâu, codepoint UTF-8 khổng lồ). Quan sát: một yêu cầu có làm treo worker không?

5. **Bão giải phóng KV cache** — ép buộc giải phóng bằng cách làm bão hòa ngân sách block của vLLM. Quan sát: LMCache có phục hồi hay dịch vụ bị suy giảm hiệu năng?

### Nhịp độ

- **Hàng tuần** — các thử nghiệm canary nhỏ trong staging, có thể 5% ở prod.
- **Hàng tháng** — game day theo lịch trình cho một kịch bản cụ thể; có sự tham gia liên phòng ban; postmortem.
- **Hàng quý** — kiểm toán khả năng phục hồi liên phòng ban; cập nhật bản đồ phụ thuộc.

### Công cụ

- **Harness Chaos Engineering** — thương mại; đề xuất thử nghiệm dựa trên AI; giảm thiểu blast-radius; tích hợp công cụ MCP.
- **LitmusChaos** — dự án tốt nghiệp CNCF; dựa trên quy trình Kubernetes.
- **Chaos Mesh** — dự án sandbox CNCF; phong cách CRD Kubernetes-native.
- **Gremlin** — thương mại; hỗ trợ rộng rãi.
- **AWS FIS** / **Azure Chaos Studio** — các dịch vụ cloud được quản lý.

### Bắt đầu nhỏ

Thử nghiệm đầu tiên: kill một pod decode dưới lưu lượng ổn định. Quan sát việc định tuyến lại và phục hồi. Nếu thành công và an toàn, hãy nâng cấp lên chaos mạng.

Thử nghiệm đặc thù LLM đầu tiên: chèn một lỗi 429 từ nhà cung cấp trong 5 phút. Quan sát cơ chế fallback. Hầu hết các đội ngũ đều phát hiện ra cơ chế fallback của họ chưa được kiểm thử đầy đủ.

### Các con số cần ghi nhớ

- Bốn mặt phẳng: control, target, safety, observability.
- Tạm dừng do burn-rate: 2 lần ngân sách lỗi hàng ngày dự kiến.
- Nhịp độ: canary hàng tuần, game day hàng tháng, kiểm toán hàng quý.
- Năm thử nghiệm LLM: bộ nhớ, mạng, nhà cung cấp, prompt sai định dạng, bão KV.

```figure
i4-chaos-guard
```

## Sử dụng

`code/main.py` mô phỏng ba thử nghiệm chaos với các cổng an toàn của safety plane. Báo cáo những thử nghiệm nào sẽ kích hoạt cơ chế hủy bỏ do burn-rate.

## Triển khai

Bài học này tạo ra `outputs/skill-chaos-plan.md`. Dựa trên stack và độ trưởng thành, chọn ra ba thử nghiệm đầu tiên và công cụ phù hợp.

## Bài tập

1. Chạy `code/main.py`. Thử nghiệm nào kích hoạt cổng burn-rate và tại sao?
2. Thiết kế năm thử nghiệm chaos đầu tiên cho dịch vụ RAG dựa trên vLLM. Bao gồm tiêu chí thành công.
3. Cảnh báo burn-rate của bạn đã tạm dừng một thử nghiệm. Làm thế nào để xác định nguyên nhân gốc rễ — do chaos hay tự nhiên?
4. Tranh luận xem chaos nên chạy trong production hay chỉ staging. Khi nào production là câu trả lời đúng?
5. Nêu tên ba chế độ lỗi đặc thù của LLM mà chaos mạng thông thường không thể tái tạo.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|------------------------|
| SLI / SLO | "mục tiêu dịch vụ" | Chỉ số + mục tiêu; điều kiện tiên quyết bắt buộc |
| Blast radius | "phạm vi" | Tập hợp các dịch vụ / người dùng bị ảnh hưởng bởi thử nghiệm |
| Burn-rate alert | "cổng ngân sách" | Kích hoạt khi tốc độ tiêu thụ ngân sách lỗi > 2 lần dự kiến |
| Game day | "diễn tập hàng tháng" | Bài tập chaos liên phòng ban theo lịch trình |
| LitmusChaos | "quy trình CNCF" | Công cụ chaos Kubernetes tốt nghiệp CNCF |
| Chaos Mesh | "CRD CNCF" | Công cụ chaos Kubernetes-native sandbox CNCF |
| Harness CE | "AI hỗ trợ thương mại" | Chaos của Harness với các đề xuất từ AI |
| Malformed prompt | "bom tokenizer" | Input làm treo quá trình token hóa |
| KV eviction storm | "chuỗi giải phóng" | Giải phóng hàng loạt kích hoạt re-prefill |

## Đọc thêm

- [DevSecOps School — Hướng dẫn Chaos Engineering 2026](https://devsecopsschool.com/blog/chaos-engineering/)
- [Ankush Sharma — Observability cho LLM (sách)](https://www.amazon.com/Observability-Large-Language-Models-Engineering-ebook/dp/B0DJSR65TR)
- [LitmusChaos (CNCF)](https://litmuschaos.io/)
- [Chaos Mesh (CNCF)](https://chaos-mesh.org/)
- [Harness Chaos Engineering](https://www.harness.io/products/chaos-engineering)
- [AWS FIS](https://aws.amazon.com/fis/)