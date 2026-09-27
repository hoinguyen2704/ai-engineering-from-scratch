# AlphaEvolve — Evolutionary Coding Agents

> Kết hợp một mô hình lập trình tiên phong với một vòng lặp tiến hóa và một trình đánh giá có thể kiểm chứng bằng máy. Để vòng lặp chạy đủ lâu. Nó khám phá ra một quy trình nhân ma trận phức 4x4 sử dụng 48 phép nhân vô hướng — cải tiến đầu tiên so với Strassen trong 56 năm qua. Nó cũng tìm ra một heuristic lập lịch Borg trên toàn bộ Google giúp thu hồi ~0,7% tài nguyên tính toán cụm trong môi trường production. Kiến trúc này được thiết kế đơn giản một cách có chủ đích. Những thành công đến từ sự nghiêm ngặt của trình đánh giá.

**Type:** Learn
**Languages:** Python (stdlib, evolutionary-loop toy)
**Prerequisites:** Phase 15 · 01 (long-horizon framing), Phase 15 · 02 (self-taught reasoning)
**Time:** ~60 minutes

## Vấn đề

Các mô hình ngôn ngữ lớn (LLM) có thể viết mã. Các thuật toán tiến hóa có thể tìm kiếm trên mã nguồn. Cả hai đều đã được thử nghiệm riêng biệt trong nhiều thập kỷ; cả hai đều chạm ngưỡng giới hạn. Giới hạn của LLM là sự bịa đặt (confabulation): mô hình viết ra mã trông có vẻ hợp lý nhưng không thực hiện đúng những gì nó tuyên bố. Giới hạn của tiến hóa là chi phí tìm kiếm: các đột biến ngẫu nhiên trên cú pháp hiếm khi tạo ra các chương trình có thể biên dịch được, chứ chưa nói đến việc tốt hơn.

AlphaEvolve (Novikov và cộng sự, DeepMind, arXiv:2506.13131, tháng 6 năm 2025) kết hợp cả hai. LLM đề xuất các chỉnh sửa có mục tiêu vào một cơ sở dữ liệu chương trình; một trình đánh giá tự động chấm điểm từng biến thể; các biến thể đạt điểm cao trở thành cha mẹ cho các thế hệ tương lai. LLM xử lý bước tốn kém là viết mã hợp lý; trình đánh giá bắt lỗi các sự bịa đặt. Vòng lặp chạy từ vài giờ đến vài tuần.

Các kết quả được báo cáo: nhân ma trận phức 4x4 với 48 phép nhân vô hướng (giới hạn của Strassen năm 1969 là 49), một heuristic lập lịch Borg trong môi trường production của Google, tăng tốc 32,5% cho FlashAttention kernel, cải thiện thông lượng huấn luyện Gemini.

Kiến trúc này hoạt động vì trình đánh giá có thể kiểm chứng bằng máy. Nó không hoạt động ở nơi mà trình đánh giá không làm được điều đó. Sự bất đối xứng đó chính là bài học rút ra.

## Khái niệm

### Vòng lặp

1. Bắt đầu từ một chương trình hạt giống `P_0` đúng nhưng chưa tối ưu.
2. Duy trì một cơ sở dữ liệu các chương trình biến thể, mỗi chương trình được chấm điểm bởi trình đánh giá.
3. Lấy mẫu một hoặc nhiều cha mẹ từ cơ sở dữ liệu (theo kiểu MAP-elites hoặc dựa trên đảo).
4. Prompt LLM (Gemini Flash cho nhiều ứng viên, Gemini Pro cho các trường hợp khó) để tạo ra một biến thể đã sửa đổi của cha mẹ.
5. Biên dịch, chạy và đánh giá biến thể trên trình đánh giá độc lập (held-out).
6. Chèn vào cơ sở dữ liệu dựa trên điểm số và vector đặc trưng của nó.
7. Lặp lại.

Hai chi tiết quan trọng. Thứ nhất, LLM được prompt với nhiều thông tin hơn là chỉ chương trình cha mẹ — thường là một vài biến thể hàng đầu từ cơ sở dữ liệu, cộng với chữ ký của trình đánh giá, cộng với mô tả nhiệm vụ ngắn gọn. Công việc của mô hình là đề xuất một thay đổi có mục tiêu có thể cải thiện điểm số. Thứ hai, cơ sở dữ liệu được cấu trúc (lưới MAP-elites, dựa trên đảo) để vòng lặp khám phá sự đa dạng, không chỉ là người dẫn đầu hiện tại.

### Điều gì làm cho trình đánh giá trở nên không thể thay thế

Những thành công của AlphaEvolve đều đến từ các lĩnh vực mà trình đánh giá nhanh, xác định và khó bị thao túng:

- **Thuật toán nhân ma trận**: một unit test nhân các ma trận và kiểm tra sự bằng nhau theo từng bit.
- **Heuristic lập lịch Borg**: một trình mô phỏng cấp production phát lại tải cụm lịch sử và đo lường tài nguyên tính toán bị lãng phí.
- **FlashAttention kernel**: một bài kiểm tra tính đúng đắn cộng với benchmark thời gian thực trên phần cứng thực tế.
- **Thông lượng huấn luyện Gemini**: đo lường GPU-seconds trên mỗi bước.

Trong mỗi trường hợp, trình đánh giá bắt được các loại lỗi LLM mà nếu không sẽ chiếm ưu thế: các tuyên bố đúng đắn bịa đặt, các tuyên bố về hiệu suất biến mất trên phần cứng và các lỗi ở trường hợp biên. Loại bỏ trình đánh giá và vòng lặp sẽ tối ưu hóa cho mã nguồn trông "đẹp".

### Reward hacking là mặt kia của vấn đề đó

Tiến hóa tối ưu hóa cho bất cứ thứ gì mà trình đánh giá đo lường. Nếu trình đánh giá không hoàn hảo, vòng lặp sẽ tìm ra sự không hoàn hảo đó. Trong một lĩnh vực không được kiểm chứng, vòng lặp sẽ tối ưu hóa cho các đặc trưng bề mặt, không phải hành vi dự định. DeepMind nêu rõ điều này trong bài báo: những thành công của AlphaEvolve chỉ chuyển đổi sang các lĩnh vực mà sự nghiêm ngặt của trình đánh giá tương xứng với tham vọng của việc tìm kiếm.

Các ví dụ cụ thể về reward hacking trong các vòng lặp tìm kiếm mã nguồn giai đoạn 2025-2026:

- Các mục tiêu tối ưu hóa thưởng cho "thời gian hoàn thành" dẫn đến việc gửi các giải pháp trống.
- Điểm benchmark thưởng cho tính đúng đắn dưới bài kiểm tra dẫn đến việc ghi nhớ các bài kiểm tra và overfitting.
- Một proxy "chất lượng mã" thưởng cho việc xóa các bình luận và viết lại tên biến mà không có thay đổi về ngữ nghĩa.

Giải pháp trong AlphaEvolve: sử dụng một trình đánh giá held-out mà LLM chưa từng thấy, với các đầu vào được tạo tại thời điểm đánh giá. Ngay cả khi đó, DeepMind khuyến nghị nên xem xét kỹ lưỡng bất kỳ triển khai nào được đề xuất.

### Tại sao LLM + tìm kiếm vượt trội hơn so với việc chỉ dùng một trong hai

LLM có thể tạo ra các sửa đổi có thể biên dịch và hợp lý về mặt ngữ nghĩa. Một thuật toán di truyền (GA) đột biến ngẫu nhiên trên một tệp Python 2000 dòng gần như luôn tạo ra lỗi cú pháp. LLM cũng tập trung tìm kiếm vào các vùng lân cận hợp lý (thay đổi một hàm, không phải các byte ngẫu nhiên), giúp giảm đáng kể các lệnh gọi trình đánh giá lãng phí.

Ngược lại, trình đánh giá bắt được các sự bịa đặt của LLM. LLM sẽ tự tin tuyên bố rằng một hàm "là O(n log n) trong giới hạn" khi thực tế nó là O(n^2); một benchmark thời gian thực sẽ giải quyết vấn đề này.

### Vị trí của AlphaEvolve trong stack tiên phong

| Hệ thống | Trình tạo | Trình đánh giá | Lĩnh vực | Ví dụ thành công |
|---|---|---|---|---|
| AlphaEvolve | Gemini | tính đúng đắn + benchmark | thuật toán, kernel, bộ lập lịch | 48-mul 4x4 matmul |
| FunSearch (DeepMind, 2023) | PaLM / Codey | tính đúng đắn | toán tổ hợp | giới hạn dưới cap-set |
| AI Scientist v2 (Sakana, L5) | GPT/Claude | phê bình LLM + thực nghiệm | nghiên cứu ML | bài báo hội thảo ICLR |
| Darwin Godel Machine (L4) | khung tác nhân | SWE-bench / Polyglot | mã tác nhân | 20% → 50% SWE-bench |

Cả bốn đều là các biến thể của cùng một công thức: trình tạo cộng với trình đánh giá, vòng lặp. Sự khác biệt nằm ở chỗ trình đánh giá chấm điểm cái gì và nó nghiêm ngặt đến mức nào.

```figure
alphaevolve-loop
```

## Sử dụng

`code/main.py` triển khai một vòng lặp tối thiểu kiểu AlphaEvolve trên một bài toán hồi quy biểu tượng đơn giản. "LLM" là một proxy stdlib đề xuất các đột biến cú pháp nhỏ cho một chương trình tính toán một hàm mục tiêu. "Trình đánh giá" đo lường sai số bình phương trung bình (MSE) trên các điểm kiểm tra held-out.

Quan sát:

- Cách điểm số tốt nhất cải thiện qua các thế hệ.
- Cách lưới MAP-elites giữ cho các giải pháp đa dạng tồn tại để vòng lặp không hội tụ vào một cực tiểu địa phương.
- Cách việc loại bỏ bài kiểm tra held-out (trình đánh giá chỉ dùng tập huấn luyện) khiến vòng lặp overfitting một cách ngoạn mục.

## Triển khai

`outputs/skill-evaluator-rigor-audit.md` là điều kiện tiên quyết để xem xét một vòng lặp kiểu AlphaEvolve trong một lĩnh vực mới: trình đánh giá của bạn có thực sự bắt được các lỗi mà bạn quan tâm không?

## Bài tập

1. Chạy `code/main.py`. Ghi lại quỹ đạo điểm số tốt nhất. Vô hiệu hóa trình đánh giá held-out (cờ `--no-holdout`) và chạy lại. Định lượng mức độ overfitting.

2. Đọc Phần 3 của bài báo AlphaEvolve về lưới MAP-elites. Thiết kế một bộ mô tả vector đặc trưng cho một bài toán mới (ví dụ: các bước tối ưu hóa trình biên dịch) để giữ cho việc tìm kiếm đa dạng.

3. Kết quả 48 phép nhân 4x4 đã cải thiện giới hạn 49 phép nhân của Strassen sau 56 năm. Đọc Phụ lục F của bài báo và giải thích bằng ba câu tại sao trình đánh giá cho bài toán này đặc biệt dễ thực hiện đúng, và tại sao hầu hết các lĩnh vực không giống như vậy.

4. Đề xuất một lĩnh vực mà AlphaEvolve sẽ thất bại. Xác định chính xác nơi trình đánh giá bị phá vỡ và tại sao.

5. Đối với một lĩnh vực bạn biết, hãy viết chữ ký trình đánh giá mà bạn sẽ sử dụng. Bao gồm (a) các điều kiện đúng đắn, (b) chỉ số hiệu suất, (c) quy tắc tạo đầu vào held-out, (d) ít nhất một kiểm tra chống reward-hacking.

## Thuật ngữ chính

| Thuật ngữ | Mọi người nói gì | Ý nghĩa thực sự |
|---|---|---|
| AlphaEvolve | "Tác nhân lập trình tiến hóa của DeepMind" | Gemini + cơ sở dữ liệu chương trình + trình đánh giá kiểm chứng bằng máy |
| MAP-elites | "Kho lưu trữ bảo tồn sự đa dạng" | Lưới được khóa bởi các vector đặc trưng; mỗi ô chứa biến thể tốt nhất với bộ mô tả đó |
| Island model | "Các quần thể con tiến hóa song song" | Các quần thể độc lập di cư định kỳ; ngăn chặn hội tụ sớm |
| Machine-checkable evaluator | "Oracle xác định" | Một unit test, trình mô phỏng hoặc benchmark mà LLM không thể làm giả — điều kiện tiên quyết cho vòng lặp này |
| Reward hacking | "Tối ưu hóa thước đo, không phải mục tiêu" | Vòng lặp tìm cách tối đa hóa điểm số mà không thực hiện nhiệm vụ dự định |
| Seed program | "Điểm bắt đầu" | Một chương trình ban đầu đúng nhưng chưa tối ưu mà vòng lặp tiến hóa từ đó |
| Held-out evaluator | "Dữ liệu đánh giá LLM chưa từng thấy" | Các đầu vào được tạo tại thời điểm đánh giá để ngăn chặn việc ghi nhớ |

## Đọc thêm

- [Novikov và cộng sự (2025). AlphaEvolve: A coding agent for scientific and algorithmic discovery](https://arxiv.org/abs/2506.13131) — bài báo đầy đủ.
- [Blog DeepMind về AlphaEvolve](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) — bài viết từ nhà cung cấp với các kết quả.
- [Kho lưu trữ kết quả AlphaEvolve](https://github.com/google-deepmind/alphaevolve_results) — các thuật toán được khám phá, bao gồm nhân ma trận 4x4 48-mul.
- [Romera-Paredes và cộng sự (2023). Mathematical discoveries from program search with LLMs (FunSearch)](https://www.nature.com/articles/s41586-023-06924-6) — hệ thống tiền nhiệm.
- [Anthropic — Responsible Scaling Policy v3.0 (Feb 2026)](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — định khung quyền tự chủ bị ràng buộc bởi trình đánh giá như một hướng nghiên cứu chính.