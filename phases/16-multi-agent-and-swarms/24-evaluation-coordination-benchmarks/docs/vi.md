# Các Benchmark Đánh giá và Điều phối

> Năm 2025-2026 có năm benchmark bao phủ không gian đánh giá đa tác nhân (multi-agent). **MultiAgentBench / MARBLE** (ACL 2025, arXiv:2503.01935) đánh giá các cấu trúc liên kết (topology) dạng sao/chuỗi/cây/đồ thị với các KPI dựa trên cột mốc; **đồ thị là tốt nhất cho nghiên cứu**, lập kế hoạch nhận thức (cognitive planning) giúp tăng khoảng 3% khả năng đạt cột mốc. **COMMA** đánh giá sự điều phối đa phương thức với thông tin bất đối xứng; các mô hình tiên tiến nhất bao gồm GPT-4o vẫn chật vật để vượt qua baseline ngẫu nhiên. **MedAgentBoard** (arXiv:2505.12371) bao gồm bốn danh mục tác vụ y tế và thường cho thấy đa tác nhân không vượt trội hơn so với một LLM đơn lẻ. **AgentArch** (arXiv:2509.10769) benchmark các kiến trúc tác nhân doanh nghiệp kết hợp sử dụng công cụ + bộ nhớ + điều phối. **SWE-bench Pro** ([arXiv:2509.16941](https://arxiv.org/abs/2509.16941)) có 1865 vấn đề trên 41 kho lưu trữ bao gồm các ứng dụng kinh doanh, dịch vụ B2B và công cụ dành cho nhà phát triển; các mô hình tiên phong đạt khoảng 23% trên Pro so với 70%+ trên Verified — một sự kiểm chứng thực tế về vấn đề nhiễm dữ liệu (contamination). Claude Opus 4.7 (tháng 4 năm 2026) được báo cáo đạt **64.3%** trên Pro với sự điều phối nhóm tác nhân rõ ràng (chưa có nguồn chính thức từ Anthropic — hãy coi đây là dữ liệu sơ bộ); Verdent (agent scaffold) đạt **76.1% pass@1** trên Verified ([báo cáo kỹ thuật Verdent](https://www.verdent.ai/blog/swe-bench-verified-technical-report)). **AAAI 2026 Bridge Program WMAC** (https://multiagents.org/2026/) là tâm điểm của cộng đồng trong năm 2026. Bài học này xây dựng dựa trên các chỉ số của MARBLE, chạy một lượt quét topology-vs-metric, và xác lập quy tắc "chỉ vượt qua SWE-bench Verified không phải là bằng chứng của sự tổng quát hóa".

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 15 (Voting and Debate Topology), Phase 16 · 23 (Failure Modes)
**Time:** ~75 phút

## Vấn đề

Khi một bài báo tuyên bố "hệ thống đa tác nhân của chúng tôi tốt hơn", câu hỏi đặt ra là: tốt hơn cái gì, trên tác vụ nào, và đo lường bằng cách nào? Kỷ nguyên 2023-2024 của đánh giá đa tác nhân là một sự hỗn loạn — mọi người tự chọn chỉ số, baseline và tập tác vụ riêng. Các benchmark 2025-2026 đã áp đặt cấu trúc vào đó.

Nếu không có các benchmark chung, bạn không thể so sánh hai hệ thống đa tác nhân một cách có ý nghĩa. Tệ hơn nữa, nếu không có các benchmark hold-out, các mô hình tiên phong có thể bị nhiễm dữ liệu. SWE-bench Verified đã bị nhiễm một phần trong các tập dữ liệu huấn luyện vào giữa năm 2025; điểm số của các mô hình tiên phong bị thổi phồng; Pro được thiết kế như một bài kiểm tra thực tế không bị nhiễm dữ liệu.

Bài học này liệt kê năm benchmark chuẩn tắc của năm 2026, nêu rõ mỗi benchmark đo lường điều gì và dạy bạn cách đọc các tuyên bố về benchmark một cách hoài nghi.

## Khái niệm

### MultiAgentBench (MARBLE) — ACL 2025

arXiv:2503.01935. Đánh giá bốn cấu trúc liên kết điều phối (sao, chuỗi, cây, đồ thị) trên các tác vụ nghiên cứu, lập trình và lập kế hoạch. Các KPI dựa trên cột mốc theo dõi tiến độ từng phần thay vì chỉ kết quả thành công cuối cùng.

Kết quả đo lường:

- **Đồ thị (Graph)**: Tốt nhất cho các kịch bản nghiên cứu; hỗ trợ phê bình bất kỳ-đến-bất kỳ.
- **Chuỗi (Chain)**: Tốt nhất cho lập trình tinh chỉnh từng bước.
- **Sao (Star)**: Tốt nhất cho việc hợp nhất thông tin thực tế nhanh chóng.
- **Thuế điều phối (Coordination tax)**: Xuất hiện sau khoảng 4 tác nhân trên đồ thị.
- **Lập kế hoạch nhận thức (Cognitive planning)**: Tăng khoảng 3% khả năng đạt cột mốc trên các cấu trúc liên kết.

Sử dụng khi: bạn muốn so sánh các cấu trúc liên kết điều phối một cách công bằng. Kho lưu trữ MARBLE (https://github.com/ulab-uiuc/MARBLE) cung cấp trình đánh giá.

### COMMA — thông tin bất đối xứng đa phương thức

Bao gồm các tác vụ mà các tác nhân có các phương thức quan sát khác nhau và phải điều phối mà không chia sẻ toàn bộ thông tin. Kết quả được báo cáo khá đáng ngại: các mô hình tiên phong bao gồm GPT-4o chật vật để vượt qua **baseline ngẫu nhiên** trong việc cộng tác giữa các tác nhân trên COMMA. Tín hiệu ở đây là các phương thức đa tác nhân đang được huấn luyện và đánh giá chưa đầy đủ — LLM xử lý sự hợp tác đơn phương thức khá tốt; nhưng sự điều phối đa phương thức lại sụp đổ.

Sử dụng khi: hệ thống của bạn có sự điều phối đa phương thức hoặc thông tin bất đối xứng. Kết quả null từ COMMA là một lời cảnh báo hãy đo lường trước khi đưa ra tuyên bố.

### MedAgentBoard — kiểm tra áp lực miền

arXiv:2505.12371. Bốn danh mục tác vụ y tế: chẩn đoán, lập kế hoạch điều trị, tạo báo cáo, giao tiếp với bệnh nhân. So sánh đa tác nhân vs LLM đơn lẻ vs các hệ thống dựa trên quy tắc truyền thống.

Phát hiện: đa tác nhân KHÔNG vượt trội hơn LLM đơn lẻ ở hầu hết các danh mục. Lợi thế của đa tác nhân rất hẹp — phân rã tác vụ giúp ích khi các tác vụ con có thể tách biệt rõ ràng (chẩn đoán + điều trị); nó gây hại khi chi phí điều phối vượt quá lợi ích của sự chuyên môn hóa (tạo báo cáo).

Sử dụng khi: miền của bạn có các baseline LLM đơn lẻ rõ ràng. Nếu bài học từ MedAgentBoard có tính tổng quát, nhiều hệ thống đa tác nhân được đề xuất hiện nay đang bị thiết kế quá mức (over-engineered).

### AgentArch — kiến trúc doanh nghiệp

arXiv:2509.10769. Các thiết lập doanh nghiệp với việc sử dụng công cụ, bộ nhớ và điều phối được xếp lớp cùng nhau. Benchmark này cô lập đóng góp của từng lớp: việc thêm công cụ giúp ích bao nhiêu? Thêm bộ nhớ? Thêm điều phối đa tác nhân?

Sử dụng khi: bạn đang thiết kế một stack tác nhân doanh nghiệp và cần biện minh cho từng lớp. AgentArch giúp tránh việc mua các tính năng mà bạn không thể đo lường giá trị.

### SWE-bench Pro — bài kiểm tra thực tế

arXiv:2509.16941. 1865 vấn đề trên 41 kho lưu trữ bao gồm các ứng dụng kinh doanh, dịch vụ B2B và công cụ dành cho nhà phát triển. Được thiết kế để **không bị nhiễm dữ liệu** với các mốc cắt huấn luyện sau này. Các mô hình tiên phong đạt khoảng 23% trên Pro so với 70%+ trên Verified. Khoảng cách này chính là tín hiệu của sự nhiễm dữ liệu.

Điểm số tháng 4 năm 2026:
- Claude Opus 4.7 trên Pro: **64.3%** (được báo cáo với sự điều phối nhóm tác nhân rõ ràng; chưa có nguồn chính thức từ Anthropic — hãy coi đây là dữ liệu sơ bộ).
- Verdent (agent scaffold) trên Verified: **76.1% pass@1** ([báo cáo kỹ thuật](https://www.verdent.ai/blog/swe-bench-verified-technical-report)).
- Điểm thô của các mô hình tiên phong trên Pro không có scaffolding tác nhân: ~23-35% ([bài báo SWE-bench Pro](https://arxiv.org/abs/2509.16941)).

Kết luận: "chúng tôi đã đánh bại SWE-bench Verified" không còn là bằng chứng của năng lực. Pro hiện là bài kiểm tra sàng lọc hiện tại. Scaffolding nhóm tác nhân tạo ra những lợi ích có thể đo lường được trên Pro (chênh lệch ~30-40 điểm), đây là một trong những lập luận thực nghiệm mạnh mẽ nhất cho sự điều phối đa tác nhân vào năm 2026.

### AAAI 2026 WMAC

AAAI 2026 Bridge Program — Workshop về Điều phối Đa tác nhân (https://multiagents.org/2026/). Tâm điểm của cộng đồng năm 2026 cho nghiên cứu AI đa tác nhân. Các bài báo được chấp nhận và kỷ yếu hội thảo là nơi chuẩn tắc để đánh giá các phương pháp mới; hãy ưu tiên các tuyên bố được chấp nhận tại WMAC hơn các bản in trước (preprint) trên arXiv khi đưa ra quyết định sản xuất.

### Đọc các tuyên bố về benchmark một cách hoài nghi — danh sách kiểm tra năm 2026

Khi ai đó tuyên bố một kết quả đa tác nhân:

1. **Benchmark nào, split nào?** SWE-bench Verified vs Pro rất quan trọng. Một con số được báo cáo trên split sai là vô giá trị.
2. **Kiểm tra nhiễm dữ liệu.** Benchmark có được phát hành sau mốc cắt huấn luyện của mô hình không? Nếu không, hãy thận trọng.
3. **So sánh baseline.** So với baseline LLM đơn lẻ, so với ngẫu nhiên, so với các công trình đa tác nhân trước đó. Không phải "so với phiên bản chưa được tinh chỉnh của cùng một hệ thống".
4. **Ý nghĩa thống kê.** Số lần thử nghiệm (N trials), p-value, khoảng tin cậy. Các mô hình tiên phong có phương sai cao; các lần chạy đơn lẻ gây hiểu lầm.
5. **Sự đa dạng của tác vụ.** Một tác vụ hay nhiều tác vụ? Sự tổng quát hóa rất quan trọng đối với sản xuất.
6. **Công khai chi phí.** Token trên mỗi tác vụ, thời gian thực (wall-clock). Một giải pháp đạt 90% với chi phí gấp 20 lần là một quyết định kinh doanh, không phải là tuyên bố về năng lực.

### Những gì không có benchmark nào đo lường tốt

- **Điều phối dài hạn.** Nhiều ngày tương tác thực tế. Tất cả các benchmark hiện tại đều chạy trong thời gian ngắn.
- **Khả năng phục hồi trước đối thủ.** Điều gì xảy ra khi một tác nhân độc hại hoặc bị xâm nhập?
- **Sự trôi dạt (drift) khi triển khai.** Các benchmark là tĩnh; phân phối trong sản xuất luôn thay đổi.
- **Hiệu suất chuẩn hóa theo chi phí.** Hầu hết các benchmark báo cáo độ chính xác thô, không phải độ chính xác trên mỗi đô la.

Xây dựng benchmark nội bộ của riêng bạn cho trục mà bạn thực sự quan tâm thường là bước đi đúng đắn.

```figure
a5-bench-gap
```

## Xây dựng

`code/main.py` là một hướng dẫn không tương tác:

- Mô phỏng 3 hệ thống đa tác nhân trên một tác vụ đồ chơi.
- Tính toán các chỉ số cột mốc kiểu MARBLE cho từng hệ thống.
- Chạy kiểm tra nhiễm dữ liệu bằng cách giữ lại các tác vụ từ tập "huấn luyện".
- So sánh rõ ràng với một baseline ngẫu nhiên.
- In ra bảng điểm các tuyên bố benchmark.

Chạy:

```bash
python3 code/main.py
```

Kết quả mong đợi: bảng điểm hệ thống với độ chính xác thô, khả năng đạt cột mốc, chi phí trên mỗi tác vụ, chênh lệch so với baseline ngẫu nhiên và ghi chú kiểm tra nhiễm dữ liệu.

## Sử dụng

`outputs/skill-benchmark-reader.md` đọc bất kỳ tuyên bố benchmark đa tác nhân nào và áp dụng danh sách kiểm tra sự giám sát. Đầu ra: một điểm số và các lưu ý.

## Triển khai

Kỷ luật đánh giá sản xuất:

- **Xây dựng benchmark nội bộ** phản ánh phân phối sản xuất thực tế của bạn. Các benchmark công khai chỉ mang tính thông tin, không thay thế được.
- **Bao gồm baseline ngẫu nhiên** trong mọi so sánh. Nếu bạn không thể vượt qua ngẫu nhiên với biên độ lớn trên một tác vụ điều phối, tác vụ đó có thể đã được đặt sai cách.
- **Báo cáo chi phí cùng với độ chính xác.** Chi phí token và thời gian thực. Các đội ngũ vận hành cần cả hai.
- **Xây dựng lại benchmark hàng quý.** Phân phối sản xuất thay đổi; các benchmark cũ gây hiểu lầm.
- **Tránh overfitting các benchmark đã xuất bản.** Nếu nhóm của bạn đang tối ưu hóa cụ thể cho các con số SWE-bench Pro, bạn sẽ bị thoái lui trong sản xuất.

## Bài tập

1. Chạy `code/main.py`. Xác định hệ thống nào trong ba hệ thống được mô phỏng có chi phí trên mỗi cột mốc tốt nhất. Nó có khớp với hệ thống có độ chính xác thô cao nhất không?
2. Đọc MultiAgentBench (arXiv:2503.01935). Đối với miền tác vụ của riêng bạn, hãy quyết định cấu trúc liên kết nào trong bốn cấu trúc mà MARBLE sẽ đề xuất. Biện minh từ kết quả của bài báo.
3. Đọc bài báo SWE-bench Pro. Điều gì cụ thể làm cho nó có khả năng chống nhiễm dữ liệu? Kỹ thuật tương tự có thể áp dụng cho các benchmark khác mà bạn quan tâm không?
4. Đọc phát hiện của COMMA về điều phối đa phương thức. Thiết kế một tác vụ điều phối đa phương thức đơn giản mà bạn có thể thêm vào benchmark nội bộ của mình. Điều gì sẽ được coi là một tín hiệu hữu ích?
5. Áp dụng danh sách kiểm tra tuyên bố benchmark cho kết quả tiêu đề của một bài báo đa tác nhân gần đây. Bạn sẽ cho tuyên bố đó điểm số nào?

## Thuật ngữ chính

| Thuật ngữ | Mọi người nói | Ý nghĩa thực sự |
|------|----------------|------------------------|
| MARBLE | "MultiAgentBench" | ACL 2025; các cấu trúc sao/chuỗi/cây/đồ thị với KPI cột mốc. |
| COMMA | "Benchmark đa phương thức" | Điều phối thông tin bất đối xứng đa phương thức; các mô hình tiên phong chật vật so với ngẫu nhiên. |
| MedAgentBoard | "Kiểm tra áp lực miền" | Bốn danh mục y tế; thường thấy đa tác nhân không vượt trội hơn LLM đơn lẻ. |
| AgentArch | "Benchmark doanh nghiệp" | Công cụ + bộ nhớ + điều phối được xếp lớp. |
| SWE-bench Pro | "Chống nhiễm dữ liệu" | 1865 vấn đề, 41 kho lưu trữ; ~23% vs 70%+ trên Verified (tín hiệu nhiễm dữ liệu). |
| Đạt cột mốc | "Điểm một phần" | Các benchmark thưởng cho tiến độ, không chỉ thành công cuối cùng. |
| Nhiễm dữ liệu | "Benchmark rò rỉ vào huấn luyện" | Sau khi phát hành, các benchmark trôi vào tập dữ liệu huấn luyện; điểm số bị thổi phồng. |
| WMAC | "AAAI 2026 Bridge Program" | Workshop về Điều phối Đa tác nhân; tâm điểm của cộng đồng. |

## Đọc thêm

- [MultiAgentBench / MARBLE](https://arxiv.org/abs/2503.01935) — benchmark cấu trúc liên kết với KPI cột mốc
- [Kho lưu trữ MARBLE](https://github.com/ulab-uiuc/MARBLE) — triển khai tham chiếu
- [MedAgentBoard](https://arxiv.org/abs/2505.12371) — kiểm tra áp lực miền; đa tác nhân thường không vượt trội
- [AgentArch](https://arxiv.org/abs/2509.10769) — kiến trúc tác nhân doanh nghiệp
- [Bảng xếp hạng SWE-bench](https://www.swebench.com/) — điểm số Verified và Pro cho các mô hình tiên phong
- [AAAI 2026 WMAC](https://multiagents.org/2026/) — tâm điểm của cộng đồng năm 2026