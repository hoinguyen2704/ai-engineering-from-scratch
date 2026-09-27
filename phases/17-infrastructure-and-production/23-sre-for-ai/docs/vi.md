# SRE cho AI — Phản ứng sự cố đa tác nhân (Multi-Agent), Runbook, Phát hiện dự đoán

> AI SRE sử dụng các LLM được tiếp đất (grounded) trên dữ liệu hạ tầng (logs, runbooks, cấu trúc liên kết dịch vụ) thông qua RAG để tự động hóa các giai đoạn điều tra, lập tài liệu và phối hợp. Mô hình kiến trúc năm 2026 là điều phối đa tác nhân (multi-agent orchestration) — các tác nhân chuyên biệt (logs, metrics, runbooks) được điều phối bởi một tác nhân giám sát (supervisor); AI đề xuất các giả thuyết và truy vấn, con người phê duyệt các quyết định quan trọng. Datadog Bits AI và Azure SRE Agent cung cấp điều này dưới dạng các sản phẩm được quản lý. Runbook đang phát triển: NeuBird Hawkeye sử dụng đánh giá đối nghịch (hai mô hình phân tích cùng một sự cố; đồng thuận = tin cậy, bất đồng = không chắc chắn); bộ nhớ vận hành được duy trì xuyên suốt các thay đổi nhân sự trong nhóm. Tự động khắc phục (auto-remediation) vẫn giữ thái độ thận trọng: AI đề xuất, con người phê duyệt. Hành động hoàn toàn tự chủ chỉ giới hạn trong phạm vi hẹp (khởi động lại pod, rollback một bản deploy cụ thể) với các rào cản chặt chẽ — bất kỳ ai quảng cáo "thiết lập xong rồi quên đi" đều là nói quá. Biên giới mới nổi: dự đoán trước sự cố. Nghiên cứu của MIT báo cáo rằng một LLM được huấn luyện trên logs lịch sử + nhiệt độ GPU + các mẫu lỗi API đã dự đoán được 89% các sự cố trước 10-15 phút. Dự báo: 95% các LLM doanh nghiệp sẽ có tính năng tự động chuyển đổi dự phòng (automated failover) vào cuối năm 2026.

**Type:** Học tập
**Languages:** Python (stdlib, trình mô phỏng phân loại sự cố đa tác nhân đơn giản)
**Prerequisites:** Phase 17 · 13 (Khả năng quan sát - Observability), Phase 17 · 24 (Chaos Engineering)
**Time:** ~60 phút

## Mục tiêu học tập

- Vẽ sơ đồ kiến trúc AI SRE đa tác nhân: supervisor + các tác nhân chuyên biệt (logs, metrics, runbooks) + cổng phê duyệt của con người.
- Giải thích lý do tại sao tự động khắc phục lại giới hạn trong phạm vi hẹp (khởi động lại pod, hoàn tác deploy) thay vì phạm vi rộng (tái cấu trúc dịch vụ).
- Gọi tên mô hình đánh giá đối nghịch (NeuBird Hawkeye): hai mô hình đồng thuận = tin cậy; bất đồng = leo thang.
- Trích dẫn kết quả phát hiện sớm 89% của MIT và ràng buộc vận hành: các dự đoán mà không có hành động thực thi chỉ là các bảng điều khiển (dashboards).

## Vấn đề

Một kỹ sư trực ca bị gọi lúc 3 giờ sáng. "Tỷ lệ lỗi cao trong quá trình thanh toán." Họ kiểm tra Datadog, Loki, ba runbook, nhật ký deploy. 30 phút sau, họ nhận ra nguyên nhân gốc rễ là vLLM bị OOM do đột biến KV cache. Họ khởi động lại pod; lỗi được khắc phục.

Vào năm 2026, 20 phút đầu tiên của cuộc điều tra đó có thể tự động hóa. Nhóm logs theo dịch vụ, tương quan với các bản deploy gần đây, đối chiếu với runbook — tất cả đều là RAG + sử dụng công cụ. Một tác nhân có giám sát có thể thực hiện phân loại sơ bộ và đưa ra giả thuyết trước khi con người mở Datadog.

Tự động khắc phục hoàn toàn là một vấn đề khác. Khởi động lại pod: an toàn. Mở rộng quy mô cụm GPU: an toàn nếu chính sách cho phép. Tái cấu trúc dịch vụ: tuyệt đối không. Kỷ luật ở đây là vạch ra ranh giới hẹp đó.

## Khái niệm

### Kiến trúc đa tác nhân

```
          Incident
             │
             ▼
        Supervisor
        /    |    \
       ▼     ▼     ▼
  Log agent  Metric agent  Runbook agent
       │     │     │
       └─────┴─────┘
             │
             ▼
        Hypothesis + evidence
             │
             ▼
        Human approval
             │
             ▼
        Action (narrow set)
```

Supervisor chia sự cố thành các truy vấn con. Các tác nhân chuyên biệt có quyền truy cập công cụ (tìm kiếm log, PromQL, truy xuất tài liệu). Supervisor tổng hợp, trình bày giả thuyết + bằng chứng cho con người. Con người phê duyệt hoặc điều hướng lại.

### Phạm vi tự động khắc phục

**An toàn (hẹp)**: khởi động lại pod, hoàn tác một bản deploy cụ thể, mở rộng quy mô trong giới hạn đã phê duyệt trước, bật cờ tính năng (feature flag) đã phê duyệt trước.

**Không an toàn (rộng)**: thay đổi cấu trúc dịch vụ, sửa đổi giới hạn tài nguyên, deploy mã mới, thay đổi IAM, thay đổi cơ sở dữ liệu.

Bất kỳ ai quảng cáo "thiết lập xong rồi quên đi" đều là nói quá. Tập hợp an toàn sẽ mở rộng khi AI SRE trưởng thành, nhưng ranh giới đó là có thật.

### Đánh giá đối nghịch (NeuBird Hawkeye)

Hai mô hình độc lập phân tích cùng một sự cố. Nếu chúng đồng thuận về nguyên nhân gốc rễ, độ tin cậy sẽ cao. Nếu chúng bất đồng, hãy leo thang lên con người với cả hai giả thuyết được hiển thị. Mô hình đơn giản, bộ lọc hiệu quả chống lại các nguyên nhân gốc rễ bị ảo giác (hallucinated).

### Bộ nhớ vận hành

Sự thay đổi nhân sự trong nhóm là kẻ giết chết thầm lặng của SRE truyền thống — kiến thức bộ lạc (tribal knowledge) bị mất đi. AI SRE lưu trữ runbooks + post-mortems trong một vector DB; các tác nhân truy xuất chúng trong mỗi sự cố mới. Khi các kỹ sư mới gia nhập, AI đã có toàn bộ lịch sử.

### Dự đoán trước sự cố

Nghiên cứu của MIT năm 2025: LLM được huấn luyện trên logs lịch sử, nhiệt độ GPU, các mẫu lỗi API đã dự đoán được 89% các sự cố trước 10-15 phút trên tập kiểm tra.

Kiểm chứng thực tế: dự đoán mà không có hành động thực thi chỉ là các bảng điều khiển. Câu hỏi vận hành là "khi chúng ta dự đoán, chúng ta làm gì?". Xả tải trước (pre-emptive drain)? Pager? Tự động mở rộng? Câu trả lời phụ thuộc vào chính sách.

### Các sản phẩm năm 2026

- **Datadog Bits AI** — copilot SRE được quản lý bên trong Datadog.
- **Azure SRE Agent** — bản địa hóa trên Azure.
- **NeuBird Hawkeye** — đánh giá đối nghịch + bộ nhớ vận hành.
- **PagerDuty AIOps** — phân loại + khử trùng lặp.
- **Incident.io Autopilot** — chỉ huy sự cố + phối hợp.

### Runbook dưới dạng mã (Runbooks as code)

Runbook phát triển từ các trang Confluence thành markdown có phiên bản với các phần cấu trúc (triệu chứng, giả thuyết, xác minh, hành động). Các runbook có cấu trúc giúp việc truy xuất RAG tốt hơn. Hãy bắt đầu bất kỳ quá trình triển khai AI-SRE nào bằng cách chuyển đổi các runbook phi cấu trúc thành có cấu trúc.

### Các con số bạn nên nhớ

- Phát hiện sớm của MIT: 89% sự cố, thời gian dẫn trước 10-15 phút.
- Phân loại đa tác nhân: supervisor + (logs, metrics, runbooks) + con người.
- Tập hợp tự động khắc phục an toàn: khởi động lại pod, hoàn tác deploy, mở rộng trong giới hạn.
- Đánh giá đối nghịch: hai mô hình độc lập; đồng thuận = tin cậy.

```figure
i4-incident-agents
```

## Sử dụng

`code/main.py` mô phỏng một quá trình phân loại đa tác nhân: tác nhân log tìm thấy lỗi, tác nhân metric tìm thấy đột biến CPU, tác nhân runbook khớp với vấn đề đã biết. Supervisor xếp hạng các giả thuyết.

## Triển khai

Bài học này tạo ra `outputs/skill-ai-sre-plan.md`. Dựa trên tình hình trực ca, khối lượng sự cố, sự trưởng thành của nhóm hiện tại, hãy thiết kế một kế hoạch triển khai AI SRE.

## Bài tập

1. Chạy `code/main.py`. Điều gì xảy ra nếu tác nhân log và metric bất đồng? Supervisor giải quyết như thế nào?
2. Xác định ba hành động tự động khắc phục "an toàn" cho dịch vụ của bạn. Biện minh cho từng hành động.
3. Viết một mẫu runbook có cấu trúc: các phần, các trường bắt buộc, các lệnh xác minh.
4. Dự đoán phát hiện sớm kích hoạt trước 12 phút. Chính sách của bạn là gì — pager, xả tải trước, hay cả hai?
5. Tranh luận xem một nhóm 3 người có nên áp dụng AI SRE vào năm 2026 hay chờ đợi. Cân nhắc sự trưởng thành, khối lượng công việc, rủi ro.

## Thuật ngữ chính

| Thuật ngữ | Mọi người nói | Ý nghĩa thực tế |
|------|----------------|------------------------|
| AI SRE | "tác nhân trực ca" | Điều tra + phối hợp sự cố dựa trên LLM |
| Tác nhân Supervisor | "người điều phối" | Tác nhân cấp cao nhất chia sự cố thành các truy vấn con |
| Tác nhân chuyên biệt | "tác nhân miền" | Tác nhân phụ có quyền truy cập công cụ (logs, metrics, runbooks) |
| Tự động khắc phục | "AI tự sửa" | Hành động hẹp đã phê duyệt trước; KHÔNG PHẢI tái cấu trúc rộng |
| Bộ nhớ vận hành | "vector runbooks" | Post-mortems + runbooks trong vector DB cho RAG |
| Đánh giá đối nghịch | "kiểm tra hai mô hình" | Phân tích độc lập; đồng thuận = tin cậy |
| NeuBird Hawkeye | "cái đối nghịch" | Sản phẩm với mô hình đánh giá đối nghịch + bộ nhớ |
| Bits AI | "tác nhân SRE của Datadog" | AI SRE do Datadog quản lý |
| Dự đoán trước sự cố | "phát hiện sớm" | Thời gian dẫn trước 10-15 phút khi dự đoán sự cố |

## Đọc thêm

- [incident.io — Hướng dẫn toàn diện về AI SRE 2026](https://incident.io/blog/what-is-ai-sre-complete-guide-2026)
- [InfoQ — AI lấy con người làm trung tâm cho SRE](https://www.infoq.com/news/2026/01/opsworker-ai-sre/)
- [DZone — AI trong SRE 2026](https://dzone.com/articles/ai-in-sre-whats-actually-coming-in-2026)
- [Datadog Bits AI](https://www.datadoghq.com/product/bits-ai/)
- [NeuBird Hawkeye](https://www.neubird.ai/)
- [awesome-ai-sre](https://github.com/agamm/awesome-ai-sre)