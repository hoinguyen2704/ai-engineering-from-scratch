# A/B Testing LLM Features — GrowthBook, Statsig và Vấn đề "Vibe"

> A/B testing truyền thống không được xây dựng cho các LLM không tất định (non-deterministic). Sự khác biệt cốt lõi: các eval trả lời câu hỏi "mô hình có làm được việc không?", còn A/B test trả lời "người dùng có quan tâm không?". Cả hai đều cần thiết; thời đại của việc triển khai dựa trên cảm tính (vibe check) đã kết thúc. Những gì cần kiểm thử vào năm 2026: prompt engineering (cách diễn đạt), lựa chọn mô hình (GPT-4 vs GPT-3.5 vs OSS; độ chính xác vs chi phí vs độ trễ), các tham số tạo văn bản (temperature, top-p). Các trường hợp thực tế: một biến thể reward-model cho chatbot mang lại +70% độ dài hội thoại và +30% tỷ lệ giữ chân; các thử nghiệm tiêu đề email của Nextdoor AI mang lại +1% CTR sau khi tinh chỉnh hàm thưởng; Khanmigo của Khan Academy đã lặp lại thử nghiệm trên trục độ trễ vs độ chính xác toán học. Phân khúc nền tảng: **Statsig** (được OpenAI mua lại với giá 1,1 tỷ USD vào tháng 9 năm 2025) — kiểm thử tuần tự, CUPED, tất cả trong một. **GrowthBook** — mã nguồn mở, warehouse-native, các engine Bayesian + Frequentist + Sequential, CUPED, kiểm tra SRM, hiệu chỉnh Benjamini-Hochberg + Bonferroni. Bạn lựa chọn dựa trên ưu tiên về warehouse-SQL và việc liệu "được OpenAI mua lại" có quan trọng với tổ chức của bạn hay không.

**Type:** Learn
**Languages:** Python (stdlib, toy sequential test simulator)
**Prerequisites:** Phase 17 · 13 (Observability), Phase 17 · 20 (Progressive Deployment)
**Time:** ~60 phút

## Mục tiêu học tập

- Phân biệt giữa evals ("mô hình có làm được việc không") và A/B tests ("người dùng có quan tâm không").
- Liệt kê ba trục có thể kiểm thử (prompt, mô hình, tham số) và chọn chỉ số (metric) cho từng trục.
- Giải thích CUPED, kiểm thử tuần tự (sequential testing) và hiệu chỉnh so sánh đa nhóm Benjamini-Hochberg.
- Lựa chọn Statsig hoặc GrowthBook dựa trên hạ tầng warehouse-SQL và quan điểm về việc mua lại doanh nghiệp.

## Vấn đề

Bạn đã tinh chỉnh một system prompt. Bạn cảm thấy nó tốt hơn. Bạn triển khai nó. Tỷ lệ chuyển đổi thay đổi do nhiễu. Bạn đổ lỗi cho chỉ số. Hoặc bạn đã triển khai một mô hình mới nhưng tỷ lệ chuyển đổi không thay đổi — liệu mô hình có bị suy giảm hay thay đổi quá nhỏ để phát hiện? Bạn không biết, vì bạn đã triển khai mà không thực hiện A/B test.

Evals trả lời liệu mô hình có thể thực hiện một tác vụ trên tập dữ liệu được gán nhãn hay không. Chúng không trả lời liệu người dùng có thích kết quả đầu ra đó hay không. Chỉ có một thử nghiệm trực tuyến có kiểm soát mới trả lời được điều đó, và chỉ khi thử nghiệm đó đủ độ mạnh (power), kiểm soát được tính không tất định và hiệu chỉnh được các so sánh đa nhóm.

## Khái niệm

### Evals vs A/B tests

**Evals** — ngoại tuyến (offline), tập dữ liệu được gán nhãn, người đánh giá (rubric hoặc LLM-as-judge hoặc con người). Trả lời: "Kết quả đầu ra có đúng / hữu ích / an toàn trên phân phối cố định này không?"

**A/B test** — trực tuyến (online), người dùng thực tế, ngẫu nhiên hóa. Trả lời: "Biến thể mới có làm thay đổi chỉ số cấp người dùng quan trọng không?"

Cả hai đều cần thiết. Evals bắt lỗi hồi quy trước khi triển khai; A/B test xác nhận tác động sản phẩm sau khi triển khai.

### Những gì cần kiểm thử

1. **Prompt engineering** — cách diễn đạt, cấu trúc system-prompt, các ví dụ. Chỉ số: thành công của tác vụ, tỷ lệ giữ chân người dùng, chi phí/yêu cầu.
2. **Lựa chọn mô hình** — GPT-4 vs GPT-3.5-Turbo vs Llama-OSS. Chỉ số: độ chính xác (tác vụ) + chi phí/yêu cầu + độ trễ P99. Đa mục tiêu.
3. **Tham số tạo văn bản** — temperature, top-p, max_tokens. Chỉ số: đặc thù tác vụ (độ đa dạng đầu ra vs tính tất định).

### CUPED — giảm phương sai

Controlled-experiments Using Pre-Experiment Data (Thử nghiệm có kiểm soát sử dụng dữ liệu tiền thử nghiệm). Loại bỏ phương sai giai đoạn trước khi so sánh giai đoạn sau. Mức giảm phương sai điển hình: 30-70%. Kích thước mẫu hiệu dụng tăng lên miễn phí.

Triển khai: cả Statsig và GrowthBook đều hỗ trợ.

### Kiểm thử tuần tự (Sequential testing)

A/B test cổ điển giả định kích thước mẫu cố định. Kiểm thử tuần tự ("peek-and-decide") kiểm soát tỷ lệ dương tính giả khi kiểm tra nhiều lần. Các quy trình tuần tự luôn hợp lệ (mSPRT, Howard's confidence sequences) cho phép bạn dừng sớm khi đã có người chiến thắng rõ ràng.

### Hiệu chỉnh so sánh đa nhóm

Chạy 20 A/B test ở độ tin cậy 95% sẽ tạo ra một kết quả dương tính giả do ngẫu nhiên. Hiệu chỉnh Bonferroni thắt chặt α cho mỗi bài kiểm tra; Benjamini-Hochberg kiểm soát tỷ lệ khám phá sai (false-discovery rate). GrowthBook triển khai cả hai.

### SRM — sai lệch tỷ lệ mẫu (sample ratio mismatch)

Hash phân bổ người dùng vào các biến thể. Nếu chia 50/50 mà kết quả là 47/53, có gì đó bị hỏng — kiểm tra SRM sẽ gắn cờ điều này. Cả hai nền tảng đều triển khai.

### Statsig vs GrowthBook

**Statsig**:
- Được OpenAI mua lại với giá 1,1 tỷ USD (tháng 9 năm 2025). Dịch vụ SaaS được lưu trữ.
- Kiểm thử tuần tự, CUPED, các nhóm dân số được giữ lại (held-out).
- Tất cả trong một: feature flags + thử nghiệm + quan sát (observability).
- Phù hợp nhất: đội ngũ muốn một sản phẩm trọn gói, không quan tâm đến việc sở hữu của OpenAI.

**GrowthBook**:
- Mã nguồn mở (MIT); warehouse-native (đọc trực tiếp từ Snowflake/BigQuery/Redshift).
- Nhiều engine: Bayesian, Frequentist, Sequential.
- CUPED, SRM, Bonferroni, hiệu chỉnh BH.
- Tự lưu trữ hoặc cloud được quản lý.
- Phù hợp nhất: đội ngũ chuyên về warehouse-SQL, đội dữ liệu kiểm soát lớp chỉ số, muốn dùng OSS.

### Tính không tất định làm phức tạp độ mạnh (power)

Cùng một prompt tạo ra các kết quả đầu ra khác nhau. Các tính toán độ mạnh truyền thống giả định các quan sát IID. Với tính không tất định của LLM, kích thước mẫu hiệu dụng thấp hơn kích thước danh nghĩa. Hãy nhân kích thước mẫu yêu cầu với khoảng 1,3-1,5x làm biên độ an toàn.

### Kết quả thực tế

- Biến thể reward model cho chatbot: +70% độ dài hội thoại, +30% tỷ lệ giữ chân.
- Tiêu đề email Nextdoor: +1% CTR sau khi tinh chỉnh hàm thưởng.
- Khanmigo của Khan Academy: đánh đổi lặp lại giữa độ trễ và độ chính xác toán học.

### Anti-pattern: triển khai dựa trên cảm tính (vibes)

Mọi kỹ sư cấp cao đều có thể kể tên một tính năng được triển khai vì "cảm thấy tốt hơn" mà không có A/B test. Hầu hết chúng đều làm giảm các chỉ số sản phẩm mà đội ngũ không nhận ra trong nhiều tháng. A/B test là cơ chế bắt buộc.

### Các con số bạn nên nhớ

- Statsig được OpenAI mua lại: 1,1 tỷ USD, tháng 9 năm 2025.
- GrowthBook: mã nguồn mở MIT; Bayesian + Frequentist + Sequential.
- Giảm phương sai CUPED: 30-70%.
- Tính không tất định của LLM → +30-50% bộ đệm kích thước mẫu.

```figure
mx-sequential-test
```

## Sử dụng

`code/main.py` mô phỏng một A/B test tuần tự với các ranh giới cố định và tuần tự. Cho thấy cách kiểm thử tuần tự cho phép bạn dừng sớm.

## Triển khai

Bài học này tạo ra `outputs/skill-ab-plan.md`. Dựa trên thay đổi tính năng, khối lượng công việc, đường cơ sở (baseline), chọn nền tảng, cổng (gates), kích thước mẫu.

## Bài tập

1. Chạy `code/main.py`. Với mức tăng kỳ vọng 5% và tỷ lệ chuyển đổi cơ sở 3%, cần kích thước mẫu bao nhiêu để đạt độ mạnh 80%?
2. Chọn Statsig hoặc GrowthBook cho một khách hàng on-prem trong lĩnh vực y tế được quản lý chặt chẽ.
3. Thiết kế một A/B test so sánh GPT-4 vs GPT-3.5 về chi phí trên mỗi ticket được giải quyết. Chỉ số chính, chỉ số bảo vệ (guardrail), chỉ số phụ là gì?
4. Canary của bạn vượt qua nhưng A/B test cho thấy tỷ lệ chuyển đổi -1,2%. Bạn có triển khai không? Viết các tiêu chí leo thang.
5. Áp dụng CUPED cho giai đoạn trước với 60% phương sai của giai đoạn sau. Tính mức tăng kích thước mẫu hiệu dụng.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|------------------------|
| Eval | "kiểm tra ngoại tuyến" | Đánh giá khả năng mô hình trên tập dữ liệu gán nhãn |
| A/B test | "thử nghiệm" | So sánh ngẫu nhiên trực tiếp trên người dùng |
| CUPED | "giảm phương sai" | Hồi quy giai đoạn trước để giảm phương sai |
| Sequential test | "kiểm tra peek-ok" | Quy trình luôn hợp lệ cho phép dừng sớm |
| Multiple comparison | "lỗi gia đình" | Chạy nhiều bài kiểm tra làm tăng dương tính giả |
| Bonferroni | "hiệu chỉnh chặt" | Chia α cho số lượng bài kiểm tra |
| Benjamini-Hochberg | "BH FDR" | Kiểm soát tỷ lệ khám phá sai, ít bảo thủ hơn |
| SRM | "chia sai" | Sai lệch tỷ lệ mẫu; lỗi phân bổ |
| Statsig | "của OpenAI" | Nền tảng thương mại tất cả trong một, mua lại 2025 |
| GrowthBook | "bản OSS" | Nền tảng warehouse-native giấy phép MIT |
| mSPRT | "kiểm tra tỷ lệ xác suất tuần tự" | Quy trình tuần tự cổ điển |

## Đọc thêm

- [GrowthBook — Cách thực hiện A/B Test cho AI](https://blog.growthbook.io/how-to-a-b-test-ai-a-practical-guide/)
- [Statsig — Vượt xa các Prompt: Tối ưu hóa LLM dựa trên dữ liệu](https://www.statsig.com/blog/llm-optimization-online-experimentation)
- [So sánh Statsig vs GrowthBook](https://www.statsig.com/perspectives/ab-testing-feature-flags-comparison-tools)
- [Deng et al. — CUPED](https://www.exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf)
- [Howard — Chuỗi tin cậy (Confidence Sequences)](https://arxiv.org/abs/1810.08240)