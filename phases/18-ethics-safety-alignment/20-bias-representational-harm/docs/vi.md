# Bias và Tác hại Đại diện trong LLM

> Gallegos, Rossi, Barrow, Tanjim, Kim, Dernoncourt, Yu, Zhang, Ahmed (Computational Linguistics 2024, arXiv:2309.00770). Khảo sát nền tảng năm 2024 phân biệt giữa tác hại đại diện (định kiến, xóa bỏ) và tác hại phân bổ (phân phối tài nguyên không công bằng), đồng thời phân loại các chỉ số đánh giá thành: dựa trên embedding, dựa trên xác suất, hoặc dựa trên văn bản được tạo. Nghiên cứu thực nghiệm năm 2024-2025: An và cộng sự (PNAS Nexus, tháng 3 năm 2025) đo lường định kiến giao thoa giới tính x chủng tộc trên GPT-3.5 Turbo, GPT-4o, Gemini 1.5 Flash, Claude 3.5 Sonnet, Llama 3-70B trong việc đánh giá sơ yếu lý lịch tự động cho 20 công việc cấp độ đầu vào. WinoIdentity (COLM 2025, arXiv:2508.07111) giới thiệu đánh giá công bằng dựa trên sự không chắc chắn cho các danh tính giao thoa. Yu & Ananiadou 2025 xác định các "gender neurons" trong các lớp MLP; Ahsan & Wallace 2025 sử dụng SAE để tiết lộ định kiến chủng tộc trong y tế; Zhou và cộng sự 2024 (UniBias) thao tác các attention head để khử định kiến. Meta-critique (arXiv:2508.11067): Tài liệu trong 10 năm qua tập trung quá mức vào định kiến giới tính nhị phân.

**Type:** Build
**Languages:** Python (stdlib, toy embedding-based bias probe)
**Prerequisites:** Phase 05 (word embeddings), Phase 18 · 01 (instruction following)
**Time:** ~60 phút

## Mục tiêu học tập

- Định nghĩa tác hại đại diện so với tác hại phân bổ và đưa ra một ví dụ cho mỗi loại trong triển khai LLM.
- Kể tên ba danh mục chỉ số đánh giá từ Gallegos và cộng sự 2024 và mô tả một chỉ số từ mỗi danh mục.
- Mô tả tính giao thoa (intersectionality) và lý do tại sao phép đo công bằng dựa trên sự không chắc chắn của WinoIdentity giải quyết được các lỗ hổng trong đánh giá định kiến đơn trục.
- Mô tả hai phương pháp diễn giải cơ học (mechanistic interpretability) đối với định kiến (gender neurons, SAE features, thao tác attention-head).

## Vấn đề

Các bài học trước bao gồm tác hại cố ý (jailbreak, âm mưu) và quản trị an toàn. Định kiến là tác hại nảy sinh không có chủ đích — từ phân phối dữ liệu huấn luyện, từ cách đặt câu lệnh (prompt framing), từ các lựa chọn thiết kế tích lũy. Việc đo lường và giảm thiểu nó là một thách thức phương pháp luận khác biệt so với tính bền vững trước các cuộc tấn công (adversarial robustness).

## Khái niệm

### Tác hại đại diện so với tác hại phân bổ

- **Tác hại đại diện (Representational harm).** Định kiến, xóa bỏ, mô tả hạ thấp. Một LLM mô tả y tá chỉ toàn là nữ giới đang tạo ra tác hại đại diện.
- **Tác hại phân bổ (Allocational harm).** Kết quả vật chất không bình đẳng. Một LLM đánh giá sơ yếu lý lịch của ứng viên da màu thấp hơn một cách hệ thống đang tạo ra tác hại phân bổ.

Đây không phải là những thứ giống nhau. Một mô hình có thể "không có định kiến đại diện" (tạo ra các mô tả đa dạng) trong khi vẫn "có định kiến phân bổ" (đưa ra các khuyến nghị không bình đẳng). Các đánh giá cần đo lường cả hai.

### Ba danh mục chỉ số đánh giá (Gallegos và cộng sự 2024)

- **Dựa trên embedding.** Các bài kiểm tra kiểu WEAT trên các embedding trước RLHF. Đo lường mối liên hệ thống kê giữa các thuật ngữ danh tính và thuật ngữ thuộc tính. Hạn chế: đo lường sự đại diện, không phải hành vi.
- **Dựa trên xác suất.** Log-likelihood của các phần hoàn thiện xác nhận định kiến so với vi phạm định kiến. Đo lường phía bộ giải mã (decoder). Ghi lại một phần định kiến hành vi.
- **Dựa trên văn bản được tạo.** Đo lường tác vụ hạ nguồn trên văn bản được tạo. Đánh giá sơ yếu lý lịch, viết khuyến nghị, đối thoại. Có giá trị thực tế nhất; khó tái lập nhất.

### Tính giao thoa (Intersectionality)

Đánh giá định kiến về "giới tính" bỏ lỡ định kiến chỉ xuất hiện trên các cặp (giới tính, chủng tộc). An và cộng sự 2025 phát hiện GPT-4o phạt phụ nữ da màu trong việc đánh giá sơ yếu lý lịch nặng nề hơn so với nam giới da màu và phụ nữ da trắng khi xét riêng lẻ. Đánh giá đơn trục không thể nắm bắt được điều này.

WinoIdentity (COLM 2025) giới thiệu sự công bằng giao thoa dựa trên sự không chắc chắn. Nó đo lường liệu sự không chắc chắn của mô hình về kết quả có khác biệt giữa các bộ danh tính giao thoa hay không — không chỉ là dự đoán điểm. Điều này bắt được các trường hợp mô hình sai lệch như nhau giữa các nhóm nhưng lại không chắc chắn hơn đối với một số nhóm, dẫn đến hành vi phân bổ hạ nguồn khác nhau.

### Các phương pháp cơ học

Công trình diễn giải năm 2024-2025 mở ra khả năng can thiệp cơ học vào định kiến:

- **Gender neurons (Yu & Ananiadou 2025).** Các neuron MLP cụ thể tương quan với các hành vi đặc thù về giới tính. Loại bỏ (ablate) các neuron này giúp giảm các chỉ số chênh lệch giới tính với chi phí năng lực hạn chế.
- **Định kiến chủng tộc trong y tế thông qua SAE (Ahsan & Wallace 2025).** Các đặc trưng Sparse Autoencoder phân tách biểu diễn nội bộ thành các chiều có thể diễn giải; các đặc trưng tương quan với chủng tộc có thể được xác định và triệt tiêu.
- **UniBias (Zhou và cộng sự 2024).** Thao tác attention-head để khử định kiến zero-shot. Các head cụ thể khuếch đại độ nhạy với lớp danh tính; việc đưa về 0 hoặc điều chỉnh trọng số các head này giúp giảm định kiến mà không cần tinh chỉnh (fine-tuning).

### Meta-critique

Đánh giá tài liệu 10 năm (arXiv:2508.11067, 2025) cho thấy lĩnh vực này tập trung quá mức vào định kiến giới tính nhị phân. Các trục khác — khuyết tật, tôn giáo, tình trạng di cư, danh tính đa ngôn ngữ — nhận được ít sự chú ý hơn nhiều. Meta-critique lập luận rằng sự tập trung hẹp có thể gây hại cho các nhóm yếu thế do bị bỏ quên: một mô hình đã khử định kiến tốt về giới tính nhị phân có thể vẫn bị định kiến nặng nề trên các khía cạnh không ai kiểm tra.

### Vị trí trong Phase 18

Các bài học 20-21 bao quát định kiến và sự công bằng một cách chính thức. Bài học 22 bao quát quyền riêng tư. Bài học 23 bao quát đóng dấu bản quyền (watermarking). Đây là lớp tác hại đối với người dùng, bổ sung cho lớp lừa đảo/an toàn trước đó.

```figure
an-bias-two-harms
```

## Sử dụng

`code/main.py` xây dựng một công cụ thăm dò định kiến dựa trên embedding đơn giản: đo khoảng cách kiểu WEAT giữa các thuật ngữ danh tính và thuật ngữ thuộc tính trong một embedding đồng xuất hiện đơn giản. Bạn có thể tiêm một định kiến và quan sát chỉ số kích hoạt; áp dụng một thao tác khử định kiến đơn giản và quan sát sự phục hồi một phần.

## Triển khai

Bài học này tạo ra `outputs/skill-bias-eval.md`. Với một thẻ mô hình (model card) hoặc tuyên bố về sự công bằng, nó kiểm toán việc đánh giá trên ba danh mục chỉ số (embedding, xác suất, văn bản được tạo), phạm vi bao phủ tính giao thoa và cơ chế của bất kỳ sự can thiệp khử định kiến nào.

## Bài tập

1. Chạy `code/main.py`. Báo cáo điểm định kiến kiểu WEAT trước và sau bước khử định kiến. Giải thích tại sao chỉ số không giảm xuống bằng 0.

2. Mở rộng công cụ thăm dò với một bài kiểm tra giao thoa: (giới tính, chủng tộc) x (sự nghiệp, gia đình). Báo cáo điểm định kiến chéo trục.

3. Đọc An và cộng sự 2025 (PNAS Nexus). Xác định hai hiệu ứng giao thoa mà họ báo cáo mà đánh giá giới tính đơn trục sẽ bỏ lỡ.

4. Yu & Ananiadou 2025 xác định các gender neurons. Phác thảo một thí nghiệm bác bỏ (falsification experiment) để phân biệt giữa "các neuron này gây ra định kiến giới tính" và "các neuron này tương quan với định kiến giới tính."

5. Meta-critique lập luận rằng lĩnh vực này tập trung quá hẹp vào giới tính nhị phân. Chọn một trục ít được nghiên cứu và mô tả một giao thức đo lường tác hại đại diện cho nó.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|------------------------|
| Tác hại đại diện | "định kiến / xóa bỏ" | Mô tả thiên kiến về một nhóm |
| Tác hại phân bổ | "quyết định không bình đẳng" | Kết quả vật chất thiên kiến cho một nhóm |
| WEAT | "bài kiểm tra embedding" | Word Embedding Association Test; công cụ thăm dò định kiến dựa trên đồng xuất hiện |
| Tính giao thoa | "hiệu ứng danh tính kết hợp" | Định kiến nảy sinh tại điểm giao của nhiều trục danh tính |
| Gender neurons | "neuron định kiến MLP" | Các neuron cụ thể có kích hoạt tương quan với hành vi đặc thù về giới tính |
| SAE feature | "chiều có thể diễn giải" | Đặc trưng được xác định bởi sparse-autoencoder; hữu ích cho phân tích định kiến cơ học |
| UniBias | "khử định kiến attention-head" | Khử định kiến zero-shot bằng cách điều chỉnh trọng số các attention head |

## Đọc thêm

- [Gallegos và cộng sự — Bias and Fairness in LLMs: A Survey (arXiv:2309.00770, Computational Linguistics 2024)](https://arxiv.org/abs/2309.00770) — khảo sát kinh điển
- [An và cộng sự — Intersectional resume-evaluation bias (PNAS Nexus, tháng 3 năm 2025)](https://academic.oup.com/pnasnexus/article/4/3/pgaf089/8111343) — nghiên cứu giao thoa trên năm mô hình
- [WinoIdentity — uncertainty-based intersectional fairness (arXiv:2508.07111, COLM 2025)](https://arxiv.org/abs/2508.07111) — benchmark mới
- [UniBias — attention-head manipulation (Zhou và cộng sự 2024, ACL)](https://arxiv.org/abs/2405.20612) — khử định kiến zero-shot