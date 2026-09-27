# Scalable Oversight và Weak-to-Strong Generalization

> Burns và cộng sự (OpenAI Superalignment, "Weak-to-Strong Generalization", 2023) đã đề xuất một đại diện (proxy) cho bài toán superalignment: tinh chỉnh (fine-tune) một mô hình mạnh bằng cách sử dụng các nhãn do một mô hình yếu hơn tạo ra. Nếu mô hình mạnh có thể tổng quát hóa chính xác từ sự giám sát yếu không hoàn hảo, thì các phương pháp alignment ở quy mô con người hiện nay có thể mở rộng sang các hệ thống siêu thông minh. Scalable oversight và W2SG là các khái niệm bổ trợ cho nhau. Scalable oversight (debate, recursive reward modeling, task decomposition) làm tăng năng lực hiệu dụng của người giám sát để họ có thể theo kịp mô hình đang được giám sát. W2SG đảm bảo rằng mô hình mạnh tổng quát hóa chính xác từ bất kỳ sự giám sát không hoàn hảo nào mà người giám sát cung cấp. Debate Helps W2SG (arXiv:2501.13124, tháng 1 năm 2025) là nghiên cứu kết hợp cả hai phương pháp này.

**Type:** Learn
**Languages:** Python (stdlib, W2SG gap simulator)
**Prerequisites:** Phase 18 · 01 (instruction-following), Phase 18 · 10 (AI Control), Phase 09 (RL foundations)
**Time:** ~60 phút

## Mục tiêu học tập

- Định nghĩa scalable oversight và weak-to-strong generalization, đồng thời giải thích cách chúng bổ trợ cho nhau.
- Mô tả thiết lập thực nghiệm của Burns và cộng sự năm 2023: tinh chỉnh GPT-4 bằng cách sử dụng nhãn từ GPT-2.
- Giải thích chỉ số Performance Gap Recovered (PGR) và ý nghĩa của nó.
- Nêu ba cơ chế scalable-oversight chính (debate, recursive reward modeling, task decomposition) và một ưu điểm của mỗi cơ chế.

## Vấn đề

Mọi kỹ thuật alignment cho đến nay trong Phase 18 đều giả định rằng người giám sát có thể đánh giá hành vi của mô hình. Khi mô hình đạt trình độ siêu thông minh, người giám sát trở thành mắt xích yếu nhất. Câu hỏi về superalignment là: liệu một người giám sát yếu hơn có thể tạo ra một mô hình mạnh hơn và được căn chỉnh (aligned) một cách đáng tin cậy hay không?

Burns và cộng sự đã rút gọn vấn đề này thành một thiết lập thực nghiệm có thể vận hành: giám sát mô hình mạnh bằng mô hình yếu, đo lường xem bao nhiêu năng lực của mô hình mạnh được giữ lại sau quá trình giám sát yếu. Đây không phải là lời giải cho superalignment — mà là một đại diện cho phép chúng ta đo lường tiến độ.

## Khái niệm

### W2SG: thiết lập của Burns và cộng sự

- Mô hình yếu: Lớp GPT-2.
- Mô hình mạnh: Lớp GPT-4.
- Mục tiêu: Ngưỡng hiệu năng của GPT-4 mạnh trên tác vụ đó.

Quy trình:
1. Lấy các dự đoán zero-shot của mô hình yếu trên một tác vụ.
2. Tinh chỉnh mô hình mạnh trên dữ liệu đã được gán nhãn bởi mô hình yếu.
3. Đo lường độ chính xác của mô hình mạnh sau khi tinh chỉnh.

Các đường cơ sở (baselines): độ chính xác của chính mô hình yếu, ngưỡng giám sát của mô hình mạnh trên các nhãn chuẩn (gold labels).

Chỉ số khoảng cách: Performance Gap Recovered (PGR) = (fine-tuned - weak) / (ceiling - weak). PGR bằng 1.0 nghĩa là sự giám sát yếu đã khỏa lấp hoàn toàn khoảng cách; PGR bằng 0 nghĩa là sự giám sát yếu không mang lại hiệu quả.

### Các phát hiện thực nghiệm của Burns và cộng sự

Trên các tác vụ NLP, giải đố cờ vua và reward-modeling, PGR luôn dương: mô hình mạnh đã tổng quát hóa vượt ra ngoài những sai lầm của người giám sát yếu. PGR đặc thù cho từng tác vụ dao động từ ~20% đến ~80%. Mô hình mạnh "hiểu" được tác vụ dự định từ các nhãn không hoàn hảo, bù đắp bằng các priors (tiền giả định) đã được huấn luyện trước của nó.

Các hạn chế mà Burns và cộng sự lưu ý:
- Khoảng cách weak-to-strong ở đây là về năng lực, không phải về alignment. Mô hình yếu đã được căn chỉnh; một mô hình thực sự siêu thông minh có thể được căn chỉnh bởi con người nhưng với khoảng cách năng lực lớn hơn nhiều.
- Sự tổng quát hóa của mô hình mạnh có thể thiên về các priors của tác vụ hơn là khôi phục ý định thực sự (ground-truth intent).
- Khai thác tri thức tiềm ẩn của mô hình mạnh mới là vấn đề thực sự; PGR chỉ đo lường một cách vận hành cụ thể.

### Scalable oversight: ba cơ chế

- **Debate (Irving và cộng sự 2018).** Hai thực thể U tranh luận về các mặt đối lập; thẩm phán yếu sẽ đưa ra quyết định. Giả định: việc tìm ra một câu trả lời đúng thuyết phục dễ hơn tìm ra một câu trả lời sai thuyết phục. Các nghiên cứu gần đây (2024-2025) của Khan và cộng sự, Michael và cộng sự cho thấy debate đôi khi có hiệu quả, đôi khi không, và việc nó có hiệu quả hay không phụ thuộc vào cấu trúc tác vụ.
- **Recursive Reward Modeling (Leike và cộng sự 2018).** U giúp con người huấn luyện reward model cho U+1. Năng lực hiệu dụng của người giám sát tăng dần theo mô hình.
- **Task Decomposition (Christiano, Shlegeris, Amodei 2018).** Phân rã một tác vụ khó thành các tác vụ con mà con người có thể kiểm tra, một cách đệ quy. Giả định rằng tác vụ có khả năng phân rã.

Mỗi cơ chế đều giả định một điều gì đó về cấu trúc của tác vụ hoặc sự căn chỉnh của các thành phần trung gian.

### Tại sao scalable oversight và W2SG bổ trợ cho nhau

Scalable oversight làm tăng chất lượng tín hiệu hiệu dụng của người giám sát.
W2SG khỏa lấp khoảng cách từ bất kỳ tín hiệu không hoàn hảo nào mà người giám sát có thể cung cấp.

Lang và cộng sự — Debate Helps Weak-to-Strong Generalization (arXiv:2501.13124) kết hợp cả hai: giao thức debate cung cấp các nhãn yếu tốt hơn, và mô hình mạnh được huấn luyện trên các nhãn đó. Kết quả cho thấy PGR tăng trên các tác vụ NLP.

### Biến động tổ chức

Nhóm Superalignment của OpenAI đã giải thể vào tháng 5 năm 2024 sau khi Jan Leike rời sang Anthropic. Chương trình nghị sự (scalable oversight, W2SG, nghiên cứu alignment tự động) vẫn tiếp tục tại Anthropic và các phòng thí nghiệm học thuật — MATS (Bài 28), Redwood (Bài 10), Apollo (Bài 8), METR (Bài 28). Cấu trúc tổ chức thay đổi; nhưng các câu hỏi nghiên cứu thì không.

### Vị trí của bài học này trong Phase 18

Các bài 6-10 mô tả mối đe dọa và mô hình phòng thủ với giả định U không đáng tin cậy. Bài 11 là mô hình tấn công: làm cho người giám sát đủ mạnh để xác minh sự căn chỉnh của U. Các bài 12-16 sau đó chuyển sang các công cụ thực tế của đánh giá đối kháng (adversarial evaluation).

```figure
scalable-oversight
```

## Sử dụng

`code/main.py` mô phỏng một quá trình tinh chỉnh W2SG trên một tác vụ tổng hợp. Người gán nhãn yếu có độ chính xác 70% với các lỗi có cấu trúc; mô hình mạnh có ngưỡng 95% trên các nhãn chuẩn. Bạn tinh chỉnh mô hình mạnh trên các nhãn yếu, đo lường PGR, và so sánh với strong-on-gold và weak-alone.

## Triển khai

Bài học này tạo ra `outputs/skill-w2sg-pgr.md`. Với mô tả về thiết lập giám sát, nó xác định người giám sát yếu, mô hình mạnh, chất lượng giám sát, và tính toán (hoặc yêu cầu) PGR. Nó gắn cờ liệu tuyên bố đó là "yếu có thể giám sát mạnh" hay "yếu + cơ chế giám sát có thể giám sát mạnh".

## Bài tập

1. Chạy `code/main.py`. Báo cáo PGR cho weak_accuracy = 0.60, 0.70, 0.80. Giải thích hình dạng của đường cong PGR.

2. Sửa đổi người gán nhãn yếu để có lỗi có cấu trúc (ví dụ: luôn sai trên một lớp đầu vào cụ thể). PGR tăng, giảm hay giữ nguyên? Giải thích.

3. Đọc Burns và cộng sự 2023 Mục 4.3 (các tác vụ NLP). Tái hiện trực giác về "confidence auxiliary loss": khi mô hình mạnh tự tin hơn các nhãn yếu, ai sẽ thắng?

4. Thiết kế một giao thức scalable-oversight kết hợp debate và task decomposition cho một tác vụ kỹ thuật phần mềm. Nêu một chế độ thất bại (failure mode) của mỗi thành phần và giải thích cách sự kết hợp giải quyết hoặc không giải quyết được từng chế độ đó.

5. Trình bày rõ điều gì sẽ bác bỏ tuyên bố "weak-to-strong generalization là một con đường khả thi dẫn đến superalignment". Hãy cụ thể về dấu hiệu thực nghiệm mà bạn cần thấy.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|------------------------|
| Scalable oversight | "làm cho người giám sát mạnh hơn" | Các cơ chế làm tăng khả năng đánh giá mô hình có năng lực cao hơn của người giám sát |
| W2SG | "yếu giám sát mạnh" | Tinh chỉnh mô hình mạnh trên các nhãn yếu và đo lường năng lực được khôi phục |
| PGR | "khoảng cách hiệu năng được khôi phục" | (fine-tuned - weak) / (ceiling - weak); 1.0 = khỏa lấp hoàn toàn, 0 = không giúp ích |
| Debate | "hai thực thể U tranh luận" | Cơ chế scalable oversight nơi thẩm phán yếu chọn giữa hai bên bảo vệ U |
| RRM | "recursive reward modeling" | U giúp huấn luyện reward model cho U+1; năng lực người giám sát theo kịp U |
| Task decomposition | "các tác vụ con mà con người kiểm tra" | Chia nhỏ tác vụ khó thành các tác vụ con mà con người có thể xác minh, một cách đệ quy |
| Superalignment | "căn chỉnh AI siêu thông minh" | Chương trình nghiên cứu liên quan đến việc căn chỉnh các mô hình mà con người không thể trực tiếp đánh giá |

## Đọc thêm

- [Burns và cộng sự — Weak-to-Strong Generalization (OpenAI 2023)](https://openai.com/index/weak-to-strong-generalization/) — bài báo về W2SG
- [Irving, Christiano, Amodei — AI safety via debate (arXiv:1805.00899)](https://arxiv.org/abs/1805.00899) — cơ chế debate
- [Leike và cộng sự — Scalable agent alignment via reward modeling (arXiv:1811.07871)](https://arxiv.org/abs/1811.07871) — recursive reward modeling
- [Khan và cộng sự — Debating with More Persuasive LLMs Leads to More Truthful Answers (arXiv:2402.06782)](https://arxiv.org/abs/2402.06782) — nghiên cứu thực nghiệm năm 2024 về debate với các debater mạnh hơn
- [Lang và cộng sự — Debate Helps Weak-to-Strong Generalization (arXiv:2501.13124)](https://arxiv.org/abs/2501.13124) — sự kết hợp giữa debate + W2SG năm 2025