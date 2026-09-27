# AI Scientist v2 — Workshop-Level Autonomous Research

> AI Scientist v2 của Sakana (Yamada và cộng sự, arXiv:2504.08066) thực hiện toàn bộ vòng lặp nghiên cứu: giả thuyết, mã nguồn, thực nghiệm, biểu đồ, viết bài và nộp bài. Đây là hệ thống đầu tiên có bài báo được tạo tự động vượt qua vòng bình duyệt tại một workshop của ICLR 2025. Đánh giá độc lập (Beel và cộng sự) cho thấy 42% các thực nghiệm thất bại do lỗi lập trình và việc rà soát tài liệu thường xuyên gắn nhãn sai các khái niệm đã có thành khái niệm mới. Tài liệu của chính Sakana cảnh báo rằng codebase thực thi mã do LLM viết và khuyến nghị sử dụng Docker để cô lập. Cả hai khía cạnh này đều là những điểm mấu chốt cần lưu ý.

**Type:** Learn
**Languages:** Python (stdlib, research-loop state-machine toy)
**Prerequisites:** Phase 15 · 03 (AlphaEvolve), Phase 15 · 04 (DGM)
**Time:** ~60 minutes

## Vấn đề

Nghiên cứu là một tác vụ mở. Không giống như tìm kiếm thuật toán của AlphaEvolve hay khả năng tự sửa đổi trong phạm vi benchmark của DGM, kết quả nghiên cứu không có tiêu chí đúng sai có thể kiểm tra bằng máy. Một bài báo được đánh giá bởi các phản biện (reviewer), không phải bởi các unit test. Điều đó làm cho việc đóng vòng lặp trở nên khó khăn hơn — nhưng cũng giá trị hơn nếu thực hiện được, vì nghiên cứu chính là nơi tạo ra sự tiến bộ tích lũy.

AI Scientist v1 (Sakana, 2024) đã đóng vòng lặp bằng cách bắt đầu từ các template do con người soạn thảo. LLM điền các thực nghiệm vào một khung sườn cố định. AI Scientist v2 (Yamada và cộng sự, 2025) loại bỏ yêu cầu về template bằng cách sử dụng tìm kiếm cây (tree search) tác tử với vòng lặp phê bình từ mô hình ngôn ngữ-hình ảnh (vision-language model). Hệ thống tạo ra ý tưởng, triển khai thực nghiệm, tạo biểu đồ, viết bài báo và lặp lại dựa trên phản hồi của người phản biện.

Kết quả bình duyệt: một bài báo do v2 tạo ra đã được chấp nhận tại một workshop của ICLR 2025 (có công khai nguồn gốc). Kết quả đánh giá độc lập: hệ thống còn lâu mới đạt độ tin cậy cao. Cả hai nhận định trên đều đúng.

## Khái niệm

### Kiến trúc

1. **Tạo ý tưởng.** LLM đề xuất các ý tưởng nghiên cứu dựa trên một chủ đề và tài liệu tham khảo trước đó. v1 sử dụng template; v2 sử dụng tìm kiếm tác tử trên không gian các giả thuyết.
2. **Kiểm tra tính mới.** Một bước truy xuất tài liệu kiểm tra xem ý tưởng đã được công bố hay chưa. Đây là bước mà đánh giá của Beel và cộng sự phát hiện lỗi gắn nhãn sai — các phương pháp đã có thường xuyên bị phân loại là mới.
3. **Kế hoạch thực nghiệm.** Tác tử soạn thảo giao thức thực nghiệm và viết mã.
4. **Thực thi.** Mã chạy trong một sandbox. Các lỗi được phản hồi ngược lại vào vòng lặp thử lại. Theo đo lường của Beel và cộng sự, 42% thực nghiệm thất bại do lỗi lập trình ở giai đoạn này.
5. **Tạo biểu đồ.** Một mô hình ngôn ngữ-hình ảnh đọc các biểu đồ được tạo ra và viết lại chúng để rõ ràng hơn. Đây là bổ sung kỹ thuật quan trọng của v2.
6. **Viết bài.** LLM soạn thảo bài báo và lặp lại với một người phản biện nội bộ.
7. **Tùy chọn: nộp bài.** Bài báo được gửi đến một hội nghị/tạp chí.

### Ý nghĩa của kết quả được chấp nhận tại workshop

Một bài báo do v2 tạo ra đã vượt qua vòng bình duyệt tại một workshop của ICLR 2025. Các tác giả đã công khai nguồn gốc bài báo với ban chương trình. Sự chấp nhận này là một điểm dữ liệu; nó không phải là giấy phép để khẳng định hệ thống "thực hiện nghiên cứu".

Bối cảnh quan trọng: các bài báo workshop có tiêu chuẩn thấp hơn so với các bài báo tại hội nghị chính. Bình duyệt là một quá trình nhiễu; một phần nhỏ các bài nộp được chấp nhận mỗi ngày. Một thành công là bằng chứng về khái niệm (proof of concept), không phải là tuyên bố về độ tin cậy. Bài báo Nature 2026 ghi lại vòng lặp end-to-end và chính nó cũng được đồng tác giả bởi các nhà nghiên cứu con người; không phải là "hệ thống tự viết một bài báo Nature".

### Kết quả đánh giá độc lập

Beel và cộng sự (arXiv:2502.14297) đã thực hiện một đánh giá bên ngoài. Các phát hiện chính:

- **Thất bại thực nghiệm.** 42% thực nghiệm thất bại do lỗi lập trình (import sai, không khớp shape, biến chưa định nghĩa). Vòng lặp thử lại đã bắt được một số lỗi, nhưng không phải tất cả.
- **Gắn nhãn sai tính mới.** Bước truy xuất tài liệu thường xuyên gắn nhãn các khái niệm đã có là mới. Đây là sự tương đương của "ảo giác" (hallucination) trong nghiên cứu.
- **Khoảng cách về chất lượng trình bày.** Việc phê bình biểu đồ bằng mô hình ngôn ngữ-hình ảnh đã tạo ra các hình ảnh đạt chuẩn xuất bản, che đậy những điểm yếu thực nghiệm bên dưới.

Phát hiện cuối cùng là quan trọng nhất đối với giai đoạn này. Một hệ thống tạo ra các kết quả thuyết phục mà không thực hiện nghiên cứu thuyết phục thì nguy hiểm hơn, chứ không an toàn hơn, so với một hệ thống thất bại rõ ràng. Việc đánh giá phải đi sâu vào các tuyên bố cốt lõi, không dừng lại ở biểu đồ.

### Mối lo ngại về thoát khỏi sandbox

README trong repository của Sakana cảnh báo:

> Do bản chất của phần mềm này, vốn thực thi mã do LLM tạo ra, chúng tôi không thể đảm bảo an toàn. Có những rủi ro về các gói độc hại, truy cập web không kiểm soát và tạo ra các tiến trình ngoài ý muốn. Hãy tự chịu rủi ro khi sử dụng và cân nhắc việc cô lập bằng Docker.

Đây là hình thái vận hành của tính tự chủ trong một lĩnh vực chưa được kiểm chứng. LLM viết mã; mã chạy; mã có thể làm bất cứ điều gì mà tiến trình được phép làm. Nếu không có một sandbox giới hạn chặt chẽ hệ thống tệp, mạng và các hành động tiến trình, bất kỳ tác tử nghiên cứu tự định hướng nào cũng có thể đánh cắp dữ liệu, tiêu tốn tài nguyên tính toán hoặc tự viết lại chính nó.

Câu chuyện sandbox của AlphaEvolve dễ dàng hơn vì bộ đánh giá của nó rất chặt chẽ. Vòng lặp của AI Scientist v2 chạy mã mở với các mục tiêu mở. Đó là lý do tại sao nó cần sự cô lập mạnh mẽ hơn (tối thiểu là Docker; ưu tiên seccomp / gVisor) và sự xem xét thủ công đối với mọi bài nộp trước khi nó rời khỏi hệ thống.

### Vị trí của v2 trong hệ sinh thái tiên phong

| Hệ thống | Mục tiêu | Loại đầu ra | Bộ đánh giá | Lỗi đã biết |
|---|---|---|---|---|
| AlphaEvolve | thuật toán | mã nguồn | unit + benchmark | bị giới hạn bởi độ nghiêm ngặt của bộ đánh giá |
| DGM | khung sườn tác tử | mã nguồn | SWE-bench | reward hacking |
| AI Scientist v2 | bài báo nghiên cứu | văn bản + mã + biểu đồ | bình duyệt (yếu) | thất bại thực nghiệm, gắn nhãn sai, che đậy điểm yếu |

v2 có bộ đánh giá tự động yếu nhất trong ba hệ thống, bề mặt đầu ra rộng nhất và con đường ngắn nhất đến các sản phẩm công khai. Các kiểm soát vận hành (sandbox, xem xét, công khai) đang thực hiện phần lớn công việc đảm bảo an toàn.

```figure
mx-research-loop
```

## Sử dụng

`code/main.py` mô phỏng vòng lặp v2 như một máy trạng thái: ý tưởng → kiểm tra tính mới → thực nghiệm → biểu đồ → viết bài → bình duyệt → chấp nhận hoặc lặp lại. Mỗi trạng thái có một xác suất thất bại có thể cấu hình được lấy từ các phát hiện của Beel và cộng sự. Chạy trình mô phỏng cho N vòng lặp và đếm:

- Có bao nhiêu ý tưởng đạt đến giai đoạn nộp bài.
- Có bao nhiêu bài nộp có lỗi thực nghiệm nghiêm trọng mà bài báo đã đánh bóng che đậy.
- Ngân sách thử lại đánh đổi giữa chất lượng và hiệu suất như thế nào.

## Triển khai

`outputs/skill-ai-scientist-sandbox-review.md` là một danh sách kiểm tra đánh giá hai cổng cho bất kỳ thứ gì được tạo ra bởi một tác tử vòng lặp nghiên cứu trước khi nó rời khỏi sandbox.

## Bài tập

1. Chạy `code/main.py` với các tham số mặc định. Tỷ lệ bao nhiêu vòng lặp tạo ra một bài báo "sạch"? Tỷ lệ bao nhiêu tạo ra một bài báo có lỗi thất bại thực nghiệm mà việc phê bình biểu đồ đã che đậy?

2. Các giá trị mặc định đã sử dụng tỷ lệ 42% / 25% của Beel và cộng sự. Chạy lại với `--experiment-failure 0.20 --novelty-mislabel 0.10` và sau đó với `--experiment-failure 0.60 --novelty-mislabel 0.40`. Tỷ lệ bài báo được đánh bóng nhưng có lỗi thay đổi như thế nào giữa hai lần chạy?

3. Đọc README của repo AI Scientist v2 của Sakana về các yêu cầu sandbox. Nêu hai hạn chế bổ sung (ngoài Docker) mà bạn sẽ áp dụng cho một lần chạy tự động kéo dài nhiều ngày.

4. Đọc phần 4 của Beel và cộng sự về khoảng cách chất lượng trình bày. Thiết kế thêm một bộ đánh giá có thể phát hiện các bài báo trông có vẻ bóng bẩy nhưng bị lỗi về thực nghiệm.

5. Đề xuất một quy trình xem xét của con người đối với đầu ra của tác tử nghiên cứu sao cho có khả năng mở rộng tốt hơn là "một tiến sĩ đọc mọi bài báo". Xác định nút thắt cổ chai và thiết kế giải pháp xung quanh nó.

## Thuật ngữ chính

| Thuật ngữ | Mọi người nói gì | Ý nghĩa thực tế |
|---|---|---|
| AI Scientist v1 | "Tác tử nghiên cứu dựa trên template của Sakana" | Điền thực nghiệm vào một khung sườn cố định |
| AI Scientist v2 | "Tác tử nghiên cứu không cần template" | Tìm kiếm cây tác tử với phê bình biểu đồ VLM |
| Agentic tree search | "Tác tử nghiên cứu phân nhánh" | Mở rộng nhiều kế hoạch thực nghiệm song song; cắt tỉa bởi nhà phê bình nội bộ |
| Vision-language critique | "VLM đánh bóng biểu đồ" | Mô hình đa phương thức đọc biểu đồ và viết lại để rõ ràng hơn |
| Literature retrieval | "Kiểm tra tính mới" | Tìm kiếm công trình trước đó để xác nhận tính mới của ý tưởng — được ghi nhận là gắn nhãn sai |
| Polish masking | "Bài báo đẹp, nghiên cứu hỏng" | Chất lượng trình bày vượt quá chất lượng thực nghiệm; che giấu điểm yếu |
| Sandbox escape | "Mã LLM thoát ra ngoài" | Mã do tác tử thực thi thực hiện các hành động mà người thiết kế vòng lặp không dự định |

## Đọc thêm

- [Yamada và cộng sự (2025). The AI Scientist-v2](https://arxiv.org/abs/2504.08066) — bài báo.
- [Blog của Sakana về ấn phẩm Nature 2026](https://sakana.ai/ai-scientist-nature/) — tóm tắt từ nhà cung cấp với bối cảnh bình duyệt.
- [Beel và cộng sự (2025). Đánh giá độc lập về The AI Scientist](https://arxiv.org/abs/2502.14297) — các con số đánh giá bên ngoài.
- [Bài báo AI Scientist v1 của Sakana](https://arxiv.org/abs/2408.06292) — tiền thân dựa trên template.
- [Anthropic — Đo lường tính tự chủ của tác tử AI](https://www.anthropic.com/research/measuring-agent-autonomy) — khung khái niệm rộng hơn về các tác tử nghiên cứu mở.