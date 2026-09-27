# STaR, V-STaR, Quiet-STaR — Self-Taught Reasoning

> Vòng lặp tự cải thiện nhỏ nhất có thể nằm ngay bên trong rationale (lập luận). Một mô hình tạo ra một chuỗi suy nghĩ (chain of thought), giữ lại những chuỗi dẫn đến câu trả lời đúng và fine-tune dựa trên đó. Đó chính là STaR. V-STaR bổ sung thêm một verifier để việc lựa chọn tại thời điểm inference hiệu quả hơn. Quiet-STaR đẩy việc tạo lập luận xuống từng token. Cả ba đều hoạt động. Không có phương pháp nào là phép màu cả — vòng lặp này bảo toàn bất kỳ lối tắt (shortcut) nào vô tình dẫn đến câu trả lời đúng.

**Type:** Learn
**Languages:** Python (stdlib, bootstrap-loop simulator)
**Prerequisites:** Phase 13 · 01-03 (Reasoning and CoT), Phase 15 · 01 (long-horizon framing)
**Time:** ~60 minutes

## Vấn đề

Cách trực diện nhất để dạy mô hình suy luận là thu thập các chuỗi suy luận do con người viết. Cách này tốn kém, chậm chạp và bị giới hạn bởi số lượng chuỗi suy luận (chain-of-thought) chất lượng cao mà con người sẵn sàng viết.

STaR (Self-Taught Reasoner, Zelikman et al., 2022) đặt câu hỏi: điều gì sẽ xảy ra nếu mô hình tự viết các lập luận của riêng nó và tự chấm điểm dựa trên các câu trả lời đã biết? Vòng lặp như sau:

1. Lấy mẫu một chuỗi suy luận kèm câu trả lời.
2. Nếu câu trả lời cuối cùng đúng, giữ lại chuỗi suy luận đó.
3. Fine-tune mô hình trên tập hợp các chuỗi được giữ lại.
4. Lặp lại.

Cách này hiệu quả. Cả GSM8K và CommonsenseQA đều cải thiện mà không cần thêm dữ liệu chú thích từ con người. Tuy nhiên, vòng lặp này có một thiên kiến tích hợp: bất kỳ lập luận nào tạo ra câu trả lời đúng đều được giữ lại, bất kể bản thân lập luận đó có logic hay không. V-STaR (Hosseini et al., 2024) khắc phục điều này bằng một verifier đã qua huấn luyện; Quiet-STaR (Zelikman et al., 2024) tổng quát hóa ý tưởng này thành các lập luận nội bộ cho từng token.

## Khái niệm

### STaR: bootstrap dựa trên những gì hiệu quả

Bắt đầu từ một mô hình cơ sở (base model) với khả năng suy luận yếu. Với mỗi bài toán huấn luyện, lấy mẫu một lập luận kèm câu trả lời. Nếu câu trả lời khớp với nhãn, giữ lại bộ ba (bài toán, lập luận, câu trả lời). Fine-tune mô hình trên tập hợp đã giữ lại. Lặp lại.

Có một điểm mấu chốt. Nếu mô hình không bao giờ giải đúng một bài toán, vòng lặp không thể học từ nó. STaR bổ sung **rationalization** (hợp lý hóa): đối với những bài toán mô hình thất bại, hãy chèn câu trả lời đúng như một gợi ý và yêu cầu mô hình tạo lại một lập luận dẫn đến câu trả lời đó. Các lập luận đã được hợp lý hóa sẽ được thêm vào tập huấn luyện.

Kết quả trong bài báo gốc (Zelikman et al., 2022): một mô hình cơ sở GPT-J đã cải thiện trên GSM8K từ 5.8% lên 10.7% thông qua các vòng STaR lặp lại với rationalization — tăng khoảng 5 điểm phần trăm tuyệt đối. Trên CommonsenseQA, mô hình GPT-J 6B được huấn luyện bằng STaR đạt 72.5%, tương đương với GPT-3 175B đã fine-tune (~73%) — một mô hình lớn hơn khoảng 30 lần và được huấn luyện trên các lập luận do con người chú thích.

### V-STaR: huấn luyện verifier bằng DPO

STaR loại bỏ các lập luận sai. Hosseini et al. (2024) nhận thấy rằng đó cũng là dữ liệu: mỗi cặp (lập luận, "đây có phải là câu trả lời đúng không") có thể dùng để huấn luyện một verifier. Họ sử dụng Direct Preference Optimization (DPO) trên cả lời giải đúng và sai để xây dựng một bộ xếp hạng (ranker). Tại thời điểm inference, lấy mẫu N lập luận và chọn lựa chọn hàng đầu của verifier.

Delta được báo cáo: +4 đến +17 điểm phần trăm so với các baseline tự cải thiện trước đó trên GSM8K và MATH, với phần lớn mức tăng đến từ việc sử dụng verifier để lựa chọn tại thời điểm inference thay vì fine-tune thêm cho generator.

### Quiet-STaR: lập luận nội bộ cho từng token

Zelikman et al. (2024) đặt câu hỏi: điều gì sẽ xảy ra nếu mô hình học cách tạo ra một lập luận nội bộ ngắn tại mỗi vị trí token, thay vì chỉ giữa bài toán và câu trả lời? Quiet-STaR huấn luyện mô hình phát ra một "suy nghĩ" ẩn trước mỗi token được dự đoán, sau đó trộn dự đoán có tính đến suy nghĩ đó với dự đoán baseline thông qua một trọng số đã học.

Kết quả: Mistral 7B đạt mức cải thiện zero-shot tuyệt đối trên GSM8K từ 5.9% lên 10.9% và CommonsenseQA từ 36.3% lên 47.2% mà không cần fine-tune theo tác vụ cụ thể. Mô hình đã học được "khi nào cần suy nghĩ" — các token khó sẽ có lập luận nội bộ dài hơn; các token dễ gần như không có.

### Tại sao cả ba đều chia sẻ mối lo ngại về an toàn

Cả ba phương pháp đều sử dụng câu trả lời cuối cùng làm tín hiệu gradient. Một lập luận dẫn đến câu trả lời đúng thông qua suy luận sai lầm — khai thác lối tắt, đoán mò hoặc sử dụng một mẫu không có tính tổng quát — sẽ được củng cố tích cực. Trên các bài toán cùng phân phối (in-distribution), lối tắt này hoạt động. Trên các bài toán ngoài phân phối (out-of-distribution), nó sẽ thất bại một cách âm thầm.

Verifier của V-STaR giảm thiểu điều này bằng cách học cách xếp hạng các lập luận, nhưng verifier lại được huấn luyện trên cùng tập nhãn. Nó có thể học cách ưu tiên các lập luận sai được định dạng tốt hơn là sự không chắc chắn trung thực. Thiết kế an toàn hơn là kết hợp dữ liệu kiểu STaR với (a) các mô hình phần thưởng được giám sát theo quy trình (process-supervised reward models - thưởng cho các bước trung gian, không chỉ câu trả lời) và (b) đánh giá OOD (out-of-distribution) tách biệt để phá vỡ các lối tắt đơn giản.

### So sánh

| Phương pháp | Tín hiệu huấn luyện | Chi phí inference | Lãng phí dữ liệu | Chế độ thất bại đã biết |
|---|---|---|---|---|
| STaR | giữ (lập luận, câu trả lời) nếu đúng | 1x | loại bỏ tất cả lập luận sai | lập luận lối tắt |
| STaR + rationalization | như trên + thử lại với gợi ý câu trả lời đúng | 1x | ít hơn | lập luận hợp lý hóa có thể không đáng tin |
| V-STaR | STaR + DPO verifier từ cả hai lớp | Nx (best-of-N) | tối thiểu | verifier có thể củng cố sự sai lầm tự tin |
| Quiet-STaR | lập luận từng token + trọng số trộn | 1.5-3x | tối thiểu | vẫn là gradient phụ thuộc vào câu trả lời |

### Vị trí trong stack công nghệ 2026

STaR đã cũ. Nhưng mô hình này xuất hiện trở lại ở khắp mọi nơi trong giai đoạn 2025-2026. RL trên các bài toán có thể kiểm chứng (DeepSeek-R1, Kimi-k1.5, o1) chính là tín hiệu gradient phụ thuộc vào câu trả lời của STaR, được mở rộng quy mô. Các mô hình phần thưởng quy trình (Lightman et al., 2023; "Let's verify step by step" của OpenAI) là giải pháp thay thế được giám sát theo quy trình. AlphaEvolve (Bài 3) là STaR dành cho code, với một trình đánh giá chương trình thay vì nhãn. Darwin Godel Machine (Bài 4) là STaR dành cho việc tự xây dựng khung tác nhân (agent scaffolding).

Hiểu về STaR giúp làm sáng tỏ tất cả những điều này. Đó là vòng lặp tự cải thiện tối thiểu khả thi.

```figure
reflection-loop
```

## Sử dụng

`code/main.py` chạy một vòng lặp STaR mô phỏng trên một tác vụ số học đơn giản. Bạn có thể quan sát:

- Độ chính xác tăng dần qua các vòng bootstrap.
- Cách các lối tắt len lỏi vào: trình mô phỏng bao gồm một lớp lập luận "lười biếng" có câu trả lời đúng 40% thời gian nhưng khả năng tổng quát hóa kém. Hãy quan sát xem STaR có giữ lại chúng hay không.
- Cách một verifier (kiểu V-STaR) hỗ trợ tại thời điểm inference nhưng không thể loại bỏ hoàn toàn các lối tắt được đưa vào trong quá trình huấn luyện.

## Triển khai

`outputs/skill-star-loop-reviewer.md` giúp bạn kiểm tra một pipeline suy luận tự học được đề xuất trước khi huấn luyện trên đó.

## Bài tập

1. Chạy trình mô phỏng. Đặt tần suất lối tắt về 0, sau đó là 0.4. Độ chính xác cuối cùng khác biệt bao nhiêu giữa hai lần chạy, mặc dù cả hai đều đạt >90% trên phân phối huấn luyện?

2. Thêm một bài kiểm tra OOD tách biệt vào trình mô phỏng. Lấy các bài toán từ một phân phối khác và đánh giá mô hình đã bootstrap trên cả tập in-distribution và OOD. Định lượng khoảng cách này.

3. Đọc bài báo Quiet-STaR (arXiv:2403.09629) Phần 3. Giải thích token "end-of-thought" và head trọng số trộn (mixing-weight head), mỗi phần trong ba câu.

4. So sánh bộ lọc keep-if-correct của STaR với một giải pháp thay thế được giám sát theo quy trình (process-supervised) thưởng cho từng bước lập luận một cách độc lập. Xác định sự khác biệt về chi phí dán nhãn và sự khác biệt về chất lượng có thể đạt được.

5. Thiết kế một đánh giá có thể phát hiện các lập luận lối tắt trong một mô hình đã triển khai. Nó không cần phải hoàn hảo — nó chỉ cần phá vỡ các lối tắt đơn giản nhất mà vòng lặp STaR sẽ củng cố.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|---|---|---|
| STaR | "Self-Taught Reasoner" | Fine-tune trên các lập luận do mô hình tạo ra dẫn đến câu trả lời đúng; lặp lại |
| Rationalization | "Thử lại với gợi ý" | Chèn câu trả lời đúng và yêu cầu tạo lại lập luận cho các bài toán mô hình cơ sở thất bại |
| V-STaR | "Verifier STaR" | Huấn luyện DPO verifier trên cả lập luận đúng và sai, dùng để chọn lựa tại thời điểm inference |
| Quiet-STaR | "Lập luận từng token" | Tạo suy nghĩ ẩn tại mỗi vị trí token; trộn với dự đoán baseline |
| Answer-conditioned gradient | "Tín hiệu dựa trên kết quả" | Vòng lặp huấn luyện thưởng cho câu trả lời cuối cùng, không phải các bước suy luận |
| Process reward model | "Verifier cấp bước" | Mô hình phần thưởng được huấn luyện trên độ chính xác từng bước, không phải kết quả — đối lập với STaR |
| Shortcut rationale | "Câu trả lời đúng, lập luận sai" | Lập luận dẫn đến nhãn thông qua mẫu không tổng quát; STaR giữ lại những cái này |

## Đọc thêm

- [Zelikman et al. (2022). STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465) — bài báo gốc.
- [Hosseini et al. (2024). V-STaR: Training Verifiers for Self-Taught Reasoners](https://arxiv.org/abs/2402.06457) — bổ sung DPO verifier cho việc lựa chọn tại thời điểm inference.
- [Zelikman et al. (2024). Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking](https://arxiv.org/abs/2403.09629) — lập luận nội bộ cho từng token.
- [Lightman et al. (2023). Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) — mô hình phần thưởng quy trình, tín hiệu gradient thay thế.
- [DeepSeek-R1 paper (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — RL trên các tác vụ có thể kiểm chứng, STaR được mở rộng quy mô cho huấn luyện frontier.