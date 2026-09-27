# EAGLE-3 Speculative Decoding in Production

> Speculative decoding kết hợp một draft model nhanh với target model. Draft model đề xuất K token; target model xác thực trong một lần forward pass duy nhất; các token được chấp nhận sẽ không tốn chi phí tính toán. Vào năm 2026, EAGLE-3 là biến thể đạt chuẩn sản xuất (production-grade) — nó huấn luyện một draft head dựa trên các hidden states của target model thay vì trên các token thô, đẩy tỷ lệ chấp nhận alpha lên ngưỡng 0.6-0.8 đối với các tác vụ chat thông thường. Câu hỏi đúng không phải là "draft model nhanh đến mức nào" mà là "alpha trên lưu lượng truy cập của tôi là bao nhiêu?". Nếu alpha giảm xuống dưới ~0.55, speculative decoding sẽ gây tác động tiêu cực ở mức độ đồng thời (concurrency) cao vì mỗi draft bị từ chối sẽ tốn thêm một lần target forward pass thứ hai. Bài học này dạy bạn cách đo lường alpha trước và bật cấu hình sau.

**Type:** Learn
**Languages:** Python (stdlib, toy acceptance-rate simulator)
**Prerequisites:** Phase 17 · 04 (Serving Engine Internals), Phase 10 · 18 (Multi-Token Prediction)
**Time:** ~60 minutes

## Learning Objectives

- Nêu tên ba thế hệ của speculative decoding và giải thích những thay đổi của EAGLE-3 so với EAGLE-2 và so với một draft model cổ điển.
- Định nghĩa tỷ lệ chấp nhận alpha, tính toán tốc độ tăng tốc kỳ vọng từ alpha và K (độ dài draft), và xác định alpha hòa vốn cho mức độ đồng thời mục tiêu của bạn.
- Giải thích lý do tại sao speculative decoding là tính năng tùy chọn (không phải mặc định) trong vLLM 2026 và tại sao việc bật nó mà không đo lường alpha là một anti-pattern trong sản xuất.
- Viết kế hoạch đo lường: benchmark nào, phân phối prompt nào, điểm đồng thời nào, và metric nào để làm ngưỡng chặn (gate).

## The Problem

Decode bị giới hạn bởi bộ nhớ (memory-bound). Trên một GPU H100 chạy Llama 3.3 70B FP8, mỗi token được decode sẽ đọc ~140 GB/s trọng số và xuất ra một token. GPU compute gần như nhàn rỗi trong quá trình decode — nút thắt cổ chai là băng thông HBM, không phải thông lượng matmul.

Speculative decoding khai thác khoảng trống này. Tạo K token ứng viên với một draft model giá rẻ, sau đó yêu cầu target model xác thực tất cả K token trong một forward pass duy nhất. Mỗi token được xác thực về cơ bản là miễn phí (được phân bổ vào batch-of-K forward mà target model dù sao cũng phải thực hiện).

Cách tiếp cận draft-model cổ điển sử dụng một model nhỏ hơn cùng họ (Llama 3.2 1B làm draft cho Llama 3.3 70B). Nó hoạt động nhưng tỷ lệ chấp nhận ở mức trung bình — phân phối của model nhỏ hơn bị lệch so với target model. EAGLE, sau đó là EAGLE-2, và EAGLE-3 huấn luyện một draft head nhẹ trực tiếp trên các internal states của target model, vì vậy phân phối của draft bám sát target model hơn nhiều. Đó là lý do tại sao alpha tăng từ 0.4 với draft-model lên 0.6-0.8 với EAGLE-3.

Điểm cần lưu ý: EAGLE-3 là tính năng tùy chọn trong vLLM 2026. `speculative_config` phải được thiết lập một cách rõ ràng. Không có flag, không có tăng tốc. Các đội ngũ bật tính năng này mà không đo lường alpha trên lưu lượng thực tế thường thấy tail latency trở nên tệ hơn thay vì tốt hơn.

## The Concept

### Speculative decoding thực sự mang lại điều gì

Nếu không có spec decode, chi phí mỗi token là một target forward. Với spec decode ở độ dài draft K và tỷ lệ chấp nhận alpha, số token kỳ vọng trên mỗi target forward là `1 + K * alpha`. Tốc độ tăng tốc là `(1 + K * alpha) / (1 + epsilon)`, trong đó epsilon là chi phí overhead của việc draft-cộng-xác thực. Với K=5, alpha=0.7: `(1 + 5*0.7) / (1 + 0.1) = 4.5 / 1.1 = 4.1x`. Các con số thực tế thường dao động quanh mức 2-3x vì alpha hiếm khi cao như vậy trên lưu lượng sản xuất và epsilon tăng lên khi batch size lớn.

### Tại sao alpha là metric duy nhất quan trọng

Các token bị từ chối không biến mất — chúng buộc phải thực hiện một target forward thứ hai cho token bị từ chối đầu tiên. Trên một khối lượng công việc mà alpha giảm xuống 0.4, bạn phải trả chi phí draft overhead cộng với xác thực và reroll. Ở mức độ đồng thời cao (ví dụ: 256 concurrent), decode batch đã đủ lớn để khoảng cách băng thông bộ nhớ giữa "target đơn lẻ" và "target với xác thực" bị thu hẹp. Dưới mức alpha 0.55 trên hầu hết phần cứng năm 2026, spec decode gây tác động tiêu cực.

Alpha thay đổi tùy theo khối lượng công việc. Trên các tác vụ chat thông thường kiểu ShareGPT, EAGLE-3 được huấn luyện trên ShareGPT đạt 0.6-0.8. Trên lưu lượng truy cập chuyên biệt (code, y tế, pháp lý), draft head được huấn luyện trên dữ liệu chung sẽ giảm xuống 0.4-0.6. Việc huấn luyện một draft head chuyên biệt cho lĩnh vực đó sẽ khôi phục alpha — đây là một công việc huấn luyện nhẹ và nhanh so với việc finetuning target model.

### Các thế hệ EAGLE tóm tắt

- **Classic draft model**: model nhỏ cùng họ. Alpha 0.3-0.5. Hạ tầng đơn giản — tải hai model, draft chạy K forward cho mỗi target forward.
- **EAGLE-1 (2024)**: draft head đơn lẻ được huấn luyện trên hidden states của target model (lớp cuối). Alpha ~0.5-0.6. Overhead tham số nhỏ trên target model.
- **EAGLE-2 (2025)**: độ dài draft thích ứng và draft dựa trên cây (xác thực nhiều nhánh trong một lần target pass). Alpha ~0.6-0.7. Bộ lập lịch draft phức tạp hơn.
- **EAGLE-3 (2025-2026)**: draft head được huấn luyện trên nhiều lớp của target model (không chỉ lớp cuối), căn chỉnh tốt hơn. Alpha ~0.6-0.8 trên chat thông thường.

### Công thức sản xuất năm 2026

1. Triển khai target model ở dạng thuần. Đo lường baseline TTFT, ITL, thông lượng tại mức đồng thời mục tiêu.
2. Bật EAGLE-3 draft thông qua vLLM `speculative_config`. Chạy lại benchmark.
3. Ghi lại tỷ lệ chấp nhận alpha. vLLM V1 báo cáo chỉ số này dưới dạng `spec_decode_metrics.accepted_tokens_per_request`. Chia cho độ dài draft yêu cầu để có alpha.
4. Nếu alpha < 0.55 trên phân phối lưu lượng sản xuất, hãy tắt spec decode hoặc huấn luyện một EAGLE-3 draft head chuyên biệt.
5. Tại mức đồng thời sản xuất, chạy lại. Xác nhận P99 ITL không bị tệ đi.

### Cạm bẫy sản xuất: P99 tail

Mean ITL giảm với spec decode. P99 có thể tệ hơn nếu bạn không tinh chỉnh. Các draft bị từ chối kích hoạt chuỗi hai bước (draft + verify-fail + reroll). Dưới batch đầy đủ, hai bước đó được tuần tự hóa. Hãy theo dõi P99 ITL, không phải P50.

### Nơi EAGLE-3 đã được triển khai

Google đã triển khai speculative decoding trong AI Overviews vào năm 2025 (chất lượng tương đương, phản hồi nhanh hơn). vLLM V1 cung cấp `speculative_config` như giao diện được tài liệu hóa; N-gram GPU speculative decoding trong V1 là biến thể tương thích với chunked prefill. SGLang hỗ trợ EAGLE-3 như đường dẫn draft được khuyến nghị cho các khối lượng công việc có nhiều prefix.

### Toán học hòa vốn trong một dòng

Tốc độ tăng tốc kỳ vọng: `S(alpha, K) = (1 + K*alpha) / (1 + verify_overhead)`. Thiết lập `S = 1` để giải cho alpha: `alpha_breakeven = verify_overhead / K`. Với verify_overhead điển hình ~0.15 và K=5: `alpha_breakeven = 0.03`. Nhưng đó là toán học decode thô. Ở mức đồng thời cao, verify overhead tăng lên và decode batch đã phân bổ việc đọc bộ nhớ trên các chuỗi, vì vậy alpha_breakeven hiệu dụng tăng lên ~0.45-0.55 trong thực tế.

### Khi nào không nên sử dụng speculative decoding

- Offline generation với batch-1 nơi độ trễ không quan trọng. Hãy sử dụng target model thuần.
- Các đầu ra rất ngắn (dưới 50 token). Draft overhead và chi phí xác thực sẽ chiếm ưu thế.
- Các lĩnh vực chuyên biệt mà không có draft head được huấn luyện chuyên biệt. Alpha quá thấp.
- vLLM v0.18.0 cộng với draft-model spec decode cộng với `--enable-chunked-prefill`. Sự kết hợp này không biên dịch được. Ngoại lệ được tài liệu hóa là N-gram GPU spec decode trong V1.

```figure
mx-speculative-tree
```

## Use It

`code/main.py` mô phỏng một vòng lặp decode có và không có speculative decoding trên một dải các giá trị alpha và độ dài draft K. Nó in ra alpha hòa vốn, tốc độ tăng tốc đo được, và hành vi tail. Chạy nó trên một vài tổ hợp (alpha, K) để thấy chính xác nơi speculative decoding ngừng mang lại hiệu quả.

## Ship It

Bài học này tạo ra `outputs/skill-eagle3-rollout.md`. Với một target model, mô tả phân phối lưu lượng, và mục tiêu đồng thời, nó tạo ra một kế hoạch triển khai EAGLE-3 theo giai đoạn — benchmark baseline, bật cấu hình, đo lường alpha, chặn ở mức alpha >= 0.55, theo dõi P99 ITL.

## Exercises

1. Chạy `code/main.py`. Tại K=5, bạn cần alpha là bao nhiêu để đạt tốc độ tăng tốc 2x? Để đạt 3x? Nó nhạy cảm như thế nào với verify_overhead?
2. Hãy tưởng tượng lưu lượng sản xuất chia 70% chat thông thường, 30% code. Chat thông thường đạt alpha 0.7 với EAGLE-3 huấn luyện trên ShareGPT; code đạt alpha 0.4. Alpha hỗn hợp là bao nhiêu và spec decode có mang lại lợi ích ròng không?
3. Đọc tài liệu vLLM `speculative_config`. Nêu tên ba chế độ (draft model, EAGLE, N-gram) và chế độ nào tương thích với chunked prefill.
4. Bạn thấy mean ITL giảm 25% sau khi bật EAGLE-3 nhưng P99 ITL tăng 15%. Hãy chẩn đoán và đề xuất biện pháp giảm thiểu.
5. Tính chi phí bộ nhớ của EAGLE-3 draft head cho Llama 3.3 70B. Nó so sánh thế nào với việc chạy Llama 3.2 1B như một draft cổ điển?

## Key Terms

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|------------------------|
| Speculative decoding | "draft plus verify" | Đề xuất K token với model rẻ, xác thực tất cả K trong một target forward |
| Acceptance rate alpha | "spec accept rate" | Tỷ lệ các token draft được target chấp nhận; metric duy nhất quan trọng |
| Draft length K | "spec k" | Số lượng token draft đề xuất mỗi target forward; thường là 4-8 |
| Verify overhead epsilon | "spec overhead" | Chi phí thêm để xác thực-và-reroll so với target forward thuần; tăng theo batch |
| EAGLE-3 | "latest EAGLE" | Biến thể 2025-2026; huấn luyện draft head trên nhiều lớp target; alpha 0.6-0.8 trên chat thông thường |
| `speculative_config` | "vLLM spec config" | Cấu hình opt-in rõ ràng trong vLLM V1; không có mặc định nghĩa là không có tăng tốc |
| N-gram spec decode | "N-gram draft" | Draft phía GPU sử dụng tra cứu N-gram trong prompt; tương thích chunked-prefill |
| Break-even alpha | "no-op alpha" | Alpha mà tại đó spec decode cho tốc độ tăng tốc bằng 0; theo dõi chỉ số này ở mức đồng thời sản xuất |
| Rejected-draft two-pass | "reroll cost" | Hai target forward khi draft bị từ chối; đẩy P99 tail lên cao |

## Further Reading

- [vLLM — Speculative Decoding docs](https://docs.vllm.ai/en/latest/features/spec_decode/) — nguồn chính thống về `speculative_config` và khả năng tương thích chunked-prefill trong V1.
- [vLLM Speculative Config API](https://docs.vllm.ai/en/latest/api/vllm/config/speculative/) — tập hợp các trường cấu hình chính xác.
- [EAGLE paper (arXiv:2401.15077)](https://arxiv.org/abs/2401.15077) — công thức draft-head EAGLE gốc.
- [EAGLE-2 paper (arXiv:2406.16858)](https://arxiv.org/abs/2406.16858) — draft thích ứng và cây.
- [UC Berkeley EECS-2025-224](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-224.html) — hệ thống LLM hiệu quả với speculative decoding.
- [BentoML — Speculative Decoding](https://bentoml.com/llm/inference-optimization/speculative-decoding) — danh sách kiểm tra triển khai sản xuất.