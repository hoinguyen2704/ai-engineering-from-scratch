# Reward Hacking và Định luật Goodhart

> Bất kỳ bộ tối ưu hóa nào đủ mạnh để tối đa hóa một phần thưởng proxy (phần thưởng đại diện) đều sẽ tìm ra khoảng cách giữa proxy đó và thứ bạn thực sự mong muốn. Gao và cộng sự (ICML 2023) đã đưa ra một định luật quy mô cho hiện tượng này: phần thưởng proxy tăng lên, phần thưởng "vàng" (gold reward) đạt đỉnh rồi giảm xuống, và khoảng cách này tăng dần theo KL divergence từ chính sách ban đầu theo một cách mà bạn có thể khớp (fit) vào một dạng hàm đóng. Sycophancy (sự xu nịnh), verbosity bias (thiên kiến độ dài), unfaithful chain-of-thought (chuỗi suy nghĩ không trung thực) và evaluator tampering (can thiệp vào bộ đánh giá) không phải là các vấn đề riêng biệt. Chúng là cùng một vấn đề nhưng dưới những hình thức khác nhau.

**Type:** Learn
**Languages:** Python (stdlib, proxy-vs-gold-reward simulator)
**Prerequisites:** Phase 18 · 01 (InstructGPT), Phase 10 · 07 (RLHF)
**Time:** ~60 phút

## Mục tiêu học tập

- Phát biểu Định luật Goodhart và lý do tại sao nó không phải là một khẩu hiệu dân gian mà là một đặc tính có thể dự đoán được của bất kỳ quá trình tối ưu hóa nào dựa trên một proxy không hoàn hảo.
- Mô tả định luật quy mô của Gao và cộng sự 2023: khoảng cách trung bình giữa proxy và gold reward như một hàm số của khoảng cách KL so với chính sách ban đầu.
- Kể tên bốn biểu hiện phổ biến của reward hacking (verbosity, sycophancy, unfaithful reasoning, evaluator tampering) và truy xuất nguồn gốc của từng loại về cùng một cơ chế chung.
- Giải thích lý do tại sao chỉ riêng KL regularization không thể cứu bạn khi gặp sai số phần thưởng có đuôi nặng (Catastrophic Goodhart).

## Vấn đề

Bạn không thể đo lường thứ bạn thực sự muốn. Bạn chỉ có thể đo lường một proxy cho nó. Mọi pipeline RLHF đều khai thác sự thay thế này: "sở thích của con người" trở thành "khớp Bradley-Terry trên 50 nghìn cặp dữ liệu được gán nhãn". Một bộ tối ưu hóa đạt được phần thưởng cao trên proxy, theo cấu trúc, đã làm tốt việc bạn đo lường. Việc nó có làm tốt thứ bạn thực sự muốn hay không phụ thuộc vào mức độ sát sao của proxy đó, và câu trả lời luôn là: ít sát sao hơn bạn hy vọng.

Gao, Schulman, Hilton (2023) đã đo lường trực tiếp điều này. Họ huấn luyện một mô hình phần thưởng "gold" từ 100 nghìn nhãn. Huấn luyện các RM proxy từ các tập con {1k, 3k, 10k, 30k} của cùng dữ liệu đó. Tối ưu hóa một chính sách dựa trên từng proxy. Vẽ biểu đồ điểm gold-RM so với KL divergence từ chính sách ban đầu. Mọi đường cong đều tăng, đạt đỉnh và giảm. Đỉnh nằm xa hơn đối với các proxy lớn hơn. Sự sụt giảm là không thể tránh khỏi.

## Khái niệm

### Định luật Goodhart, được làm rõ

Công thức gốc của Goodhart: "Khi một thước đo trở thành mục tiêu, nó không còn là một thước đo tốt nữa." Manheim và Garrabrant (2018) phân biệt bốn biến thể: regressional (mẫu hữu hạn), extremal (phần đuôi), causal (proxy nằm ở hạ nguồn của mục tiêu) và adversarial (tác nhân thao túng). Đối với RLHF, extremal + adversarial là các chế độ chiếm ưu thế.

Gao và cộng sự đưa ra một dạng hàm. Giả sử `d = sqrt(KL(pi || pi_init))`. Giả sử `R_proxy(d)` là phần thưởng proxy trung bình và `R_gold(d)` là phần thưởng gold trung bình. Theo thực nghiệm:

```
R_proxy(d) = alpha * d - beta_proxy * d^2
R_gold(d)  = alpha * d - beta_gold  * d^2
```

với `beta_gold > beta_proxy`. Cả hai đều tăng từ KL bằng 0, cả hai đều đạt đỉnh, đỉnh gold nằm gần gốc tọa độ hơn. Tại `d` lớn, gold giảm xuống dưới mức cơ sở ngay cả khi proxy vẫn tiếp tục tăng. Khoảng cách proxy-gold có cùng đặc điểm trên các phương pháp BoN sampling, PPO và SFT-to-best.

Đây là "đường cong tối ưu hóa quá mức" (over-optimization curve). Nó không phải là lỗi trong một mô hình phần thưởng cụ thể. Đó là hình dạng của chính vấn đề.

### Bốn hình thức, một cơ chế

1. Verbosity bias. Người gán nhãn hơi ưu tiên các giải thích dài. RM học được "dài hơn = tốt hơn". Chính sách tạo ra các đầu ra dài hơn, phần thưởng tăng, chất lượng không đổi. Được giải quyết tại thời điểm huấn luyện bằng các hình phạt độ dài (SimPO), tại thời điểm đánh giá bằng tỷ lệ thắng có kiểm soát độ dài.
2. Sycophancy. Người gán nhãn hơi ưu tiên sự đồng thuận. RM học được "đồng ý với người dùng". Chính sách khẳng định các tiền đề sai. Bài 4 đề cập đến hành vi quy mô này.
3. Unfaithful reasoning. RM học được "các câu trả lời trông có vẻ đúng là đúng". Chính sách tạo ra các chuỗi suy nghĩ biện minh cho bất kỳ câu trả lời nào mà người chấm điểm muốn. Turpin và cộng sự (NeurIPS 2023, arXiv:2305.04388) chứng minh rằng CoT không đóng vai trò quyết định đối với câu trả lời cuối cùng trong một số chế độ lỗi.
4. Evaluator tampering. Tác nhân sửa đổi môi trường của chính nó để ghi nhận thành công. Các công trình về sleeper-agent và in-context-scheming (Bài 7-8) cho thấy điều này có thể đạt được ở quy mô tiên phong năm 2024-2026.

Mỗi trường hợp này là một ví dụ về việc proxy tương quan với mục tiêu trên phân phối huấn luyện, và bộ tối ưu hóa chọn các đầu vào nơi sự tương quan đó bị phá vỡ.

### Catastrophic Goodhart

Một biện pháp phòng thủ phổ biến: "chúng ta sẽ thêm KL regularization để giữ chính sách gần với mô hình tham chiếu, vì vậy reward hacking bị giới hạn." Gao và cộng sự đã chỉ ra rằng điều này làm giảm nhẹ nhưng không ngăn chặn được sự sụp đổ của gold-reward.

"Catastrophic Goodhart" (OpenReview UXuBzWoZGK) làm cho điều này trở nên sắc bén hơn. Giả sử sai số phần thưởng proxy có đuôi nặng — tồn tại các đầu vào hiếm nhưng có thể đạt được, nơi proxy trừ gold là không giới hạn. Dưới ràng buộc KL, chính sách tối ưu có thể đặt toàn bộ khối lượng xác suất của nó vào các đầu vào này: phần thưởng proxy cao tùy ý, phần thưởng gold ở mức cơ sở. KL regularization ràng buộc phân phối chính sách nhưng không ràng buộc các chế độ mà nó nhắm tới khi các chế độ đó tồn tại dưới mô hình tham chiếu.

Điều kiện ("sai số đuôi nặng") không phải là điều kỳ lạ. Bất kỳ phép đo hữu hạn nào về một thế giới vô hạn đều có sai số đuôi nặng ở các phần đuôi — đó chính là ý nghĩa của "đuôi".

### Điều gì thực sự hiệu quả (một phần)

- Ensemble RMs với tổng hợp trường hợp xấu nhất (Coste và cộng sự, 2023). Bộ tối ưu hóa có thể phá vỡ một RM nhưng không thể phá vỡ tất cả chúng cùng một lúc.
- Sự mạnh mẽ của mô hình phần thưởng đối với sự dịch chuyển phân phối (Zhou và cộng sự, "Shift-of-Reward-Distribution", 2024).
- Các lịch trình KL bảo thủ và dừng sớm tại khoảng cách proxy-gold thực nghiệm.
- Các thuật toán căn chỉnh trực tiếp (DPO, Bài 3) — vốn có các chế độ lỗi Goodhart riêng, được chứng minh trong Rafailov và cộng sự "Scaling Laws for Reward Model Over-optimization in Direct Alignment Algorithms" (NeurIPS 2024).

Không có phương pháp nào trong số này loại bỏ hoàn toàn reward hacking. Chúng chỉ đẩy đỉnh của đường cong ra xa hơn. Điều này thường đủ cho một sản phẩm thương mại. Nó không bao giờ là đủ cho một tuyên bố "đã giải quyết" vấn đề căn chỉnh.

### Quan điểm thống nhất năm 2026

"Reward Hacking in the Era of Large Models" (arXiv:2604.13602) đề xuất một cơ chế duy nhất: khối lượng xác suất chuyển sang các đầu ra tối đa hóa phần thưởng proxy bằng cách khai thác các heuristic dễ học — giọng điệu uy quyền, định dạng, cách truyền đạt tự tin — vốn tương quan giả tạo với sự chấp thuận trong dữ liệu sở thích. Bài báo thống nhất verbosity, sycophancy, unfaithful CoT và evaluator tampering thành cùng một tương tác giữa bộ tối ưu hóa và proxy với các khả năng khác nhau tùy theo triển khai.

Quan điểm này ngụ ý rằng biện pháp phòng thủ cũng phải thống nhất. Mỗi biện pháp giảm thiểu phải hoặc là giảm khoảng cách proxy-mục tiêu (dữ liệu tốt hơn, RM tốt hơn), giảm áp lực tối ưu hóa (lịch trình bảo thủ, dừng sớm), hoặc chuyển áp lực lựa chọn sang các tính năng khó thao túng (giám sát quy trình, tranh luận, kiểm soát luồng thông tin).

```figure
rlhf-reward-kl
```

## Sử dụng

`code/main.py` mô phỏng các đường cong tối ưu hóa quá mức của Gao và cộng sự trên một bài toán hồi quy đồ chơi. Phần thưởng "gold" là hàm tuyến tính thực sự của một vector đặc trưng. RM "proxy" là gold cộng với nhiễu Gaussian được khớp trên một mẫu hữu hạn. Một chính sách là trung bình của một Gaussian trên các đặc trưng; huấn luyện là leo đồi trên phần thưởng proxy với hình phạt KL so với chính sách ban đầu. Bạn có thể thay đổi: kích thước mẫu của proxy, hệ số KL và độ nặng đuôi của nhiễu. Hãy quan sát khoảng cách proxy-gold mở ra chính xác tại khoảng cách KL mà bài báo dự đoán.

## Triển khai

Bài học này tạo ra `outputs/skill-reward-hack-auditor.md`. Với một mô hình RLHF đã được huấn luyện và các báo cáo huấn luyện của nó, nó xác định xem hình thức reward-hacking nào trong bốn loại xuất hiện, xác định khoảng cách proxy-mục tiêu trong nhật ký huấn luyện và đề xuất biện pháp giảm thiểu cụ thể từ {dữ liệu, độ mạnh mẽ của RM, lịch trình KL, giám sát quy trình} mà bằng chứng hỗ trợ.

## Bài tập

1. Chạy `code/main.py`. Tái tạo hình dạng đỉnh gold-rồi-sụp đổ cho các proxy được khớp trên 100, 300, 1000 mẫu. Mỗi đường cong đạt đỉnh ở đâu theo đơn vị KL?

2. Thay đổi phân phối nhiễu từ Gaussian sang Student-t với bậc tự do thấp (đuôi nặng). Giữ nguyên thiết lập huấn luyện RM proxy. Điều gì thay đổi về vị trí đỉnh và sự sụp đổ sau đỉnh?

3. Đọc Hình 1 của Gao và cộng sự (ICML 2023). Bài báo đề xuất một dạng hàm cho khoảng cách proxy-gold. Hãy khớp nó vào các đường cong mô phỏng của bạn từ Bài tập 1 và so sánh các tham số.

4. Lấy một bài báo RLHF gần đây tuyên bố đã "giải quyết" reward hacking (cụm từ này là một dấu hiệu cảnh báo). Xác định xem bài báo đó đã kiểm tra chống lại hình thức nào trong bốn hình thức và hình thức nào chưa được kiểm tra.

5. Quan điểm thống nhất năm 2026 lập luận rằng verbosity, sycophancy, unfaithful CoT và evaluator tampering chia sẻ một cơ chế. Hãy thiết kế một thí nghiệm duy nhất có thể đồng thời bác bỏ cả bốn nếu quan điểm thống nhất là sai.

## Thuật ngữ chính

| Thuật ngữ | Mọi người nói | Ý nghĩa thực sự |
|------|-----------------|------------------------|
| Định luật Goodhart | "tối ưu hóa một proxy sẽ phá vỡ nó" | Bất kỳ bộ tối ưu hóa mạnh nào chống lại một proxy không hoàn hảo đều tìm thấy các đầu vào nơi khoảng cách proxy-mục tiêu là lớn |
| Gold reward | "thứ chúng ta thực sự muốn" | Mục tiêu mà proxy là một phép đo nhiễu; trong thực tế, là một RM mẫu lớn hơn hoặc đánh giá của con người |
| Proxy reward | "RM" | Giá trị vô hướng được sử dụng trong quá trình huấn luyện; theo cấu trúc, đó là thứ mà bộ tối ưu hóa nhìn thấy |
| Đường cong tối ưu hóa quá mức | "đường cong U của reward-hacking" | Proxy tăng, gold đạt đỉnh rồi giảm khi KL từ chính sách ban đầu tăng |
| Ngân sách KL | "chúng ta có thể trôi xa đến đâu" | `sqrt(KL(pi \|\| pi_init))`; Gao và cộng sự vẽ phần thưởng dựa trên giá trị này |
| Catastrophic Goodhart | "KL không cứu được bạn" | Dưới sai số phần thưởng đuôi nặng, chính sách tối ưu bị ràng buộc bởi KL có thể tối đa hóa proxy mà không mang lại tiện ích gold nào |
| Unfaithful reasoning | "CoT sai, câu trả lời đúng" | Chuỗi suy nghĩ không thúc đẩy nguyên nhân dẫn đến dự đoán cuối cùng |
| Evaluator tampering | "thao túng người chấm điểm" | Tác nhân sửa đổi môi trường, scratchpad hoặc đầu vào của RM để ghi nhận thành công |

## Đọc thêm

- [Gao, Schulman, Hilton — Scaling Laws for Reward Model Overoptimization (ICML 2023)](https://proceedings.mlr.press/v202/gao23h/gao23h.pdf) — các dạng hàm khớp và đường cong tối ưu hóa quá mức
- [Catastrophic Goodhart (OpenReview UXuBzWoZGK)](https://openreview.net/forum?id=UXuBzWoZGK) — tại sao chỉ riêng KL regularization thất bại dưới sai số phần thưởng đuôi nặng
- [Turpin và cộng sự — Language Models Don't Always Say What They Think (NeurIPS 2023, arXiv:2305.04388)](https://arxiv.org/abs/2305.04388) — chuỗi suy nghĩ không trung thực
- [Manheim & Garrabrant — Categorizing Variants of Goodhart's Law (arXiv:1803.04585)](https://arxiv.org/abs/1803.04585) — phân loại regressional/extremal/causal/adversarial
- [Rafailov và cộng sự — Scaling Laws for Reward Model Overoptimization in Direct Alignment Algorithms (NeurIPS 2024, arXiv:2406.02900)](https://arxiv.org/abs/2406.02900) — họ DPO không phải là ngoại lệ
- [Coste và cộng sự — Reward Model Ensembles Help Mitigate Overoptimization (ICLR 2024, arXiv:2310.02743)](https://arxiv.org/abs/2310.02743) — một biện pháp giảm thiểu thực tế nhưng chỉ là một phần