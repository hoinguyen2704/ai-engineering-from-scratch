# Shadow Traffic, Canary Rollout, và Progressive Deployment cho LLM

> Việc triển khai LLM kết hợp những phần khó nhất của triển khai phần mềm: không có unit test, các chế độ lỗi phân tán, và tín hiệu phản hồi chậm. Quy trình bao gồm: (1) shadow mode — sao chép các request từ production sang model ứng viên, ghi log và so sánh mà không ảnh hưởng đến người dùng; giúp phát hiện các vấn đề phân phối rõ ràng nhưng không đảm bảo chất lượng; (2) canary rollout — chuyển hướng lưu lượng truy cập dần dần 10% → 25% → 50% → 75% → 100% với các cổng kiểm soát (gates) tại mỗi bước; theo dõi các phân vị độ trễ (latency percentiles), chi phí/request, tỷ lệ lỗi/từ chối, phân phối độ dài đầu ra, tỷ lệ phản hồi của người dùng; (3) A/B testing cho các phương án thay thế khác biệt sau khi đã xác nhận tính ổn định. Tính không xác định (non-determinism) là không thể loại bỏ — sai số độ chính xác lên tới 15% giữa các lần chạy với cùng đầu vào do tính không kết hợp của GPU FP và sự thay đổi kích thước batch. Chi phí là một biến số, không phải hằng số — một model tốt hơn 20% có thể đắt gấp 3 lần mỗi lần gọi. Tốc độ rollback là yếu tố quyết định: nếu rollback yêu cầu redeploy, bạn đã quá chậm. Chính sách nằm trong config/flags; model nằm trong registry với các digest được ghim (pinned); rollback = lật chính sách + hoàn nguyên ngưỡng + ghim model cũ trong vài giây.

**Type:** Learn
**Languages:** Python (stdlib, toy canary-progression simulator)
**Prerequisites:** Phase 17 · 13 (Observability), Phase 17 · 21 (A/B Testing)
**Time:** ~60 phút

## Mục tiêu học tập

- Phân biệt shadow mode (so sánh không ảnh hưởng), canary (triển khai dần dần trên lưu lượng thực tế), và A/B (so sánh sau khi đã xác nhận ổn định).
- Liệt kê năm chỉ số canary dành riêng cho LLM (độ trễ, chi phí/request, lỗi/từ chối, phân phối độ dài đầu ra, phản hồi người dùng).
- Giải thích tại sao tính không xác định của LLM (lên tới 15%) làm thay đổi định nghĩa về "ổn định" trong một đợt rollout.
- Thiết kế lộ trình rollback chỉ mất vài giây (lật chính sách) thay vì hàng giờ (redeploy).

## Vấn đề

Bạn phát hành một model mới. Các đánh giá ngoại tuyến (offline evals) cho thấy độ chính xác tăng 3%. Bạn bật nó trên production. Trong vòng 24 giờ, chi phí tăng 40%, tỷ lệ người dùng nhấn "không thích" tăng 8%, ba khách hàng báo cáo "câu trả lời kỳ lạ". Bạn rollback. Việc redeploy mất 3 giờ. Cuối tuần của bạn coi như bỏ đi.

Mọi phần trong đó đều có thể tránh được. Shadow mode sẽ phát hiện mức tăng chi phí 40% trước khi bất kỳ người dùng nào nhìn thấy. Canary sẽ dừng lại ở mức 10% khi tỷ lệ "không thích" tăng lên. Rollback bằng policy-flag sẽ chỉ mất 30 giây. Kỷ luật chính là thứ lấp đầy khoảng trống giữa "đánh giá ngoại tuyến tốt" và "người dùng thực sự hài lòng".

## Khái niệm

### Shadow mode

Model ứng viên nhận các request giống như production; đầu ra được ghi log, không trả về cho người dùng. Không ảnh hưởng đến người dùng. Log bao gồm:

- Nội dung đầu ra (so sánh với production).
- Số lượng token (chênh lệch chi phí).
- Độ trễ.
- Từ chối và lỗi.

Phát hiện: chi phí tăng vọt, hồi quy về độ dài, thay đổi rõ rệt trong việc từ chối, lỗi nghiêm trọng. KHÔNG phát hiện: chênh lệch chất lượng mà người dùng có thể cảm nhận được. Shadow là smoke test, không phải quality test.

### Canary rollout

Chuyển hướng lưu lượng truy cập dần dần với các cổng kiểm soát. Tiến trình điển hình: 1% → 10% → 25% → 50% → 75% → 100%. Kiểm soát 5 chỉ số tại mỗi bước:

1. **Phân vị độ trễ** — P50, P95, P99. Vi phạm: canary có P99 > 1.5x baseline.
2. **Chi phí mỗi request** — $ trung bình. Vi phạm: >20% so với baseline.
3. **Tỷ lệ lỗi / từ chối** — 5xx cộng với các từ chối rõ ràng. Vi phạm: 2x baseline.
4. **Phân phối độ dài đầu ra** — trung bình + P99. Vi phạm: dịch chuyển phân phối.
5. **Tỷ lệ phản hồi của người dùng** — nhấn không thích / gửi ticket. Vi phạm: 1.5x baseline.

### Tính không xác định là phương sai mới

Các đầu vào giống hệt nhau tạo ra các đầu ra không giống nhau. Lý do:

- Tính không kết hợp của GPU FP (thứ tự giảm dấu phẩy động thay đổi theo batch).
- Sự thay đổi kích thước batch (cùng một prompt trong batch 128 so với batch 16).
- Lấy mẫu (temperature > 0).

Đo lường: sai số độ chính xác lên tới 15% giữa các lần chạy trên cùng tập eval. "Ổn định" trong một đợt rollout có nghĩa là các chỉ số nằm trong phương sai dự kiến, không phải giống hệt baseline. Thiết lập các cổng kiểm soát trên mức nhiễu nền.

### Chi phí là một biến số

Một model tốt hơn 20% có thể đắt gấp 3 lần mỗi lần gọi. Chi phí/request là một trong năm cổng kiểm soát. Việc phát hành một model "tốt hơn" nhưng phá vỡ kinh tế đơn vị là một trường hợp cần rollback.

### Rollback là vũ khí

- Policy flag (hệ thống feature flag): lật tỷ lệ trong config; mất vài giây.
- Model pinning (registry digest): model được ghim không tự động nâng cấp.
- Rollback = hoàn nguyên flag + đặt digest được ghim về phiên bản trước. Tính bằng giây, không phải giờ.

Nếu stack của bạn yêu cầu redeploy để rollback, hãy sửa điều đó trước khi triển khai.

### Công cụ

**Argo Rollouts** / **Flagger** — Các bộ điều khiển phân phối lũy tiến trên Kubernetes. Tích hợp với định tuyến trọng số Istio/Linkerd.

**Istio weighted routing** — phân chia lưu lượng ở cấp độ service-mesh.

**KServe / Seldon Core** — phục vụ model với canary tích hợp sẵn.

**Feature flags** — LaunchDarkly, Flagsmith, Unleash. Lật ở cấp độ chính sách, không cần redeploy.

### Nhịp độ chỉ số

Các cổng canary kiểm tra mỗi 5-15 phút tùy thuộc vào lưu lượng. 1% lưu lượng với 10 req/phút cho 50-150 điểm dữ liệu mỗi cửa sổ — đủ cho độ trễ nhưng nhiễu cho phản hồi người dùng. 10% cho nhiều hơn ~10 lần. Các tiến trình nên tạm dừng đủ lâu để tích lũy đủ mẫu tại mỗi bước.

### Bước A/B là tùy chọn

Nếu model mới hoàn toàn khác biệt (hành vi khác, đường cong chi phí khác, tông giọng khác), hãy A/B test ở mức 50% sau khi canary vượt qua. Nếu chỉ là phiên bản cải tiến, hãy chuyển thẳng lên 100% khi các cổng canary vượt qua.

### Các con số bạn nên nhớ

- Tiến trình canary: 1% → 10% → 25% → 50% → 75% → 100%.
- Giới hạn tính không xác định: sai số lên tới 15% giữa các lần chạy trên cùng đầu vào.
- Năm chỉ số canary: độ trễ, chi phí, lỗi/từ chối, độ dài đầu ra, phản hồi người dùng.
- Cổng chi phí: >20% so với baseline là vi phạm.
- Rollback: vài giây, không phải hàng giờ.

```figure
i4-canary-ramp
```

## Sử dụng

`code/main.py` mô phỏng một đợt canary rollout với các hồi quy được tiêm vào. Báo cáo giai đoạn nào rollout dừng lại và cổng nào đã kích hoạt.

## Phát hành

Bài học này tạo ra `outputs/skill-rollout-runbook.md`. Với model ứng viên, baseline và mức độ chấp nhận rủi ro, thiết kế kế hoạch shadow→canary→100%.

## Bài tập

1. Chạy `code/main.py`. Tiêm một hồi quy chi phí 25%. Canary dừng lại ở giai đoạn nào?
2. Model mới của bạn có độ chính xác tăng 3% ngoại tuyến nhưng chi phí/request tăng +18%. Có nên phát hành không? Phụ thuộc vào chính sách — hãy viết cả hai lộ trình.
3. Thiết kế một quy trình rollback mất dưới 60 giây từ đầu đến cuối. Liệt kê cơ sở hạ tầng cần thiết.
4. Tính không xác định cho thấy ±7% trên eval của bạn. Thiết lập các cổng canary để không báo động giả. Bạn sử dụng hệ số nhân nào?
5. Shadow mode phát hiện mức tăng chi phí 40% trước khi canary. Viết quy tắc cảnh báo kích hoạt trong shadow mode.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|------------------------|
| Shadow mode | "sao chép sang mới" | Gửi đến ứng viên không ảnh hưởng để ghi log |
| Canary | "lưu lượng lũy tiến" | Triển khai dần dần cho người dùng với các cổng kiểm soát |
| Gates | "kiểm tra rollout" | Ngưỡng chỉ số chặn tiến trình |
| Non-determinism | "phương sai LLM" | Sự khác biệt không thể loại bỏ giữa các lần chạy |
| Policy flag | "rollback lật flag" | Rollback cấp cấu hình, tính bằng giây |
| Model pin | "registry digest" | Tham chiếu bất biến đến một phiên bản model |
| Argo Rollouts | "K8s lũy tiến" | Bộ điều khiển canary/rollback gốc của Kubernetes |
| KServe | "inference K8s" | Phục vụ model với các nguyên hàm canary |
| Istio weighted | "chia mesh" | Bộ chia lưu lượng service-mesh |

## Đọc thêm

- [TianPan — Releasing AI Features Without Breaking Production](https://tianpan.co/blog/2026-04-09-llm-gradual-rollout-shadow-canary-ab-testing)
- [MarkTechPost — Safely Deploying ML Models](https://www.marktechpost.com/2026/03/21/safely-deploying-ml-models-to-production-four-controlled-strategies-a-b-canary-interleaved-shadow-testing/)
- [APXML — Advanced LLM Deployment Patterns](https://apxml.com/courses/mlops-for-large-models-llmops/chapter-4-llm-deployment-serving-optimization/advanced-llm-deployment-patterns)
- [Argo Rollouts docs](https://argo-rollouts.readthedocs.io/)
- [Flagger docs](https://docs.flagger.app/)