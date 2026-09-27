# Async and Hogwild! Inference

> Speculative decoding (Phase 10 · 15) song song hóa các token trong một chuỗi. Các framework đa tác nhân (multi-agent) song song hóa trên toàn bộ các chuỗi nhưng yêu cầu sự phối hợp rõ ràng (bỏ phiếu, chia nhỏ tác vụ). Hogwild! Inference (Rodionov và cộng sự, arXiv:2504.06261) thực hiện một cách khác: chạy N instance của cùng một LLM song song trên một KV cache CHUNG. Mỗi worker nhìn thấy ngay lập tức các token do các worker khác tạo ra. Các mô hình suy luận hiện đại — QwQ, DeepSeek-R1 — có thể tự phối hợp thông qua cache chung đó mà không cần bất kỳ quá trình fine-tuning nào. Phương pháp này vẫn đang trong giai đoạn thử nghiệm nhưng nó mở ra một trục song song hóa suy luận hoàn toàn mới, nằm trực giao với spec decode. Bài học này triển khai một trình mô phỏng Hogwild! hai worker bằng Python stdlib và giải thích lý do tại sao sự cộng tác qua cache chung lại xuất hiện từ khả năng suy luận sẵn có của mô hình.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 10 · 12 (inference optimization), Phase 10 · 15 (speculative decoding)
**Time:** ~60 phút

## Mục tiêu học tập

- Mô tả ba cấu trúc liên kết LLM song song phổ biến (bỏ phiếu, chia nhỏ tác vụ, Hogwild!) và nêu tên các vấn đề mà mỗi cấu trúc nhắm tới.
- Nêu thiết lập cốt lõi của Hogwild!: nhiều worker, một KV cache chung, sự phối hợp tự phát thông qua self-prompting.
- Tính toán tốc độ tăng tốc (wall-time speedup) của Hogwild! dưới dạng hàm số của số lượng worker `N`, mức độ song song hóa ở cấp độ tác vụ `p`, và chi phí phối hợp `c`.
- Triển khai trình mô phỏng Hogwild! hai worker trên một bài toán đơn giản và quan sát sự phân chia tác vụ tự phát.

## Vấn đề

Các LLM hiện đại giải quyết các bài toán khó bằng cách tạo ra các chuỗi suy luận dài — 5000 token logic từng bước là chuyện bình thường, hàng chục nghìn token xảy ra trên các bài toán toán học chuyên sâu. Với tốc độ giải mã 35 token/giây trên mô hình 70B, 50k token sẽ mất 24 phút. Mô hình không còn tính tương tác nữa.

Speculative decoding (Phase 10 · 15) mang lại tốc độ nhanh gấp 3-5 lần bằng cách song song hóa trong một chuỗi. Ngoài mức đó ra, sự phụ thuộc tuần tự của giải mã tự hồi quy (autoregressive decoding) là giới hạn cứng. Mỗi token mới phụ thuộc vào mọi token trước đó.

Câu hỏi hiển nhiên là: chúng ta có thể song song hóa giữa các chuỗi không? Chạy nhiều bản sao của cùng một mô hình trên cùng một bài toán, để chúng hợp tác, để chúng chia nhau công việc?

Các công trình trước đây: ensemble bỏ phiếu (chạy N mô hình, chọn câu trả lời đa số), tree-of-thought (phân nhánh các đường dẫn suy luận và kết hợp lại), và các framework đa tác nhân (gán cho mỗi tác nhân một tác vụ con, sử dụng một điều phối viên). Tất cả những cách này đều giúp ích trong các lĩnh vực tác vụ cụ thể. Chúng cũng đều đưa vào các cơ chế phối hợp rõ ràng — quy tắc bỏ phiếu, logic phân nhánh và cắt tỉa, giao thức nhắn tin giữa các tác nhân.

Hogwild! Inference thực hiện một cách tiếp cận khác. N worker chia sẻ một KV cache duy nhất. Mỗi worker nhìn thấy các token do các worker khác tạo ra ngay lập tức, như thể đó là ngữ cảnh của chính nó. Các worker — mà không cần bất kỳ quá trình đào tạo hay fine-tuning nào — tự tìm cách chia việc. Các mô hình suy luận hiện đại (QwQ, DeepSeek-R1, chế độ suy luận của dòng Claude) có thể đọc cache chung và đưa ra các câu như "Tôi thấy worker 2 đã xử lý xong trường hợp cơ sở, vì vậy tôi sẽ làm bước quy nạp."

Tốc độ tăng tốc phụ thuộc vào khối lượng công việc và vẫn đang trong giai đoạn thử nghiệm tính đến tháng 4 năm 2026. Nhưng ý tưởng này đáng để biết vì nó mở ra một trục song song hóa suy luận mới.

## Khái niệm

### Thiết lập

Khởi tạo N tiến trình worker, tất cả đều chạy cùng một LLM. Thay vì mỗi worker có một KV cache riêng, hãy duy trì MỘT cache chung. Khi worker `i` tạo ra token `t_j`, token đó được ghi vào cache chung tại vị trí tiếp theo. Khi worker `k` thực hiện bước tiếp theo, nó đọc trạng thái hiện tại của cache (bao gồm mọi thứ mà tất cả N worker đã tạo ra cho đến nay).

Tại thời điểm bước, các worker chạy đua để ghi token. Không có chỉ số vị trí theo từng worker — cache là một chuỗi duy nhất đang phát triển. Thứ tự được xác định bởi thời điểm ghi đến nơi.

### Tại sao sự phối hợp lại xuất hiện

Các worker chia sẻ một prompt. Thông thường là kiểu "Bạn là một trong N instance cùng làm việc trên bài toán này. Mỗi instance đọc bộ nhớ chung và có thể thấy những gì các instance khác đã viết. Tránh làm việc trùng lặp." Prompt cộng với cache chung là đủ. Các mô hình suy luận đọc cache, nhận thấy những phần nào của bài toán đã được thử nghiệm, và (thường xuyên nhưng không phải lúc nào cũng vậy) chuyển hướng sang các phần chưa được khám phá.

Bài báo Hogwild! (Rodionov và cộng sự, 2025) báo cáo các quan sát như:

- Các worker lập kế hoạch và truyền đạt chúng cho các worker khác thông qua cache.
- Các worker nhận thấy lỗi trong suy luận của các worker khác và chỉ ra chúng.
- Các worker thích nghi khi một kế hoạch thất bại và đề xuất các phương án thay thế.
- Khi được nhắc kiểm tra sự trùng lặp, các worker phát hiện ra nó và chuyển hướng.

Không điều nào trong số này yêu cầu fine-tuning. Hành vi tự phát đến từ khả năng suy luận mà mô hình đã có sẵn.

### Tên gọi

Tên của bài báo lấy cảm hứng từ Hogwild! SGD (Recht và cộng sự, 2011), một trình tối ưu hóa cập nhật không đồng bộ. Sự tương tự: các worker không đồng bộ của SGD đều ghi vào một vector tham số chung; các worker của Hogwild! Inference đều ghi vào một KV cache chung. Cả hai đều dựa vào sự hội tụ thực nghiệm thay vì các đảm bảo đồng bộ hóa.

### RoPE làm cho điều này trở nên khả thi

Rotary Position Embeddings (RoPE, Su và cộng sự 2021) mã hóa thông tin vị trí thông qua phép quay trong các vector Q và K. Vì các vị trí là các phép quay chứ không phải là các độ lệch cố định, vị trí của một token có thể thay đổi mà không cần tính toán lại mục trong KV cache. Khi worker `i` ghi vào cache chung tại vị trí `p`, các worker khác đọc vị trí đó có thể sử dụng trực tiếp mục đã lưu trong cache — không cần xoay lại.

Trong một mô hình vị trí đã học hoặc vị trí tuyệt đối, Hogwild! sẽ cần vô hiệu hóa cache (cache invalidation) trên mỗi lần ghi đồng thời. RoPE cho phép cache duy trì ổn định.

### Toán học về wall-time

Gọi `T_serial` là thời gian để một worker tự giải quyết bài toán. Gọi `p` là phần có thể song song hóa ở cấp độ tác vụ. Gọi `c` là chi phí phối hợp mỗi bước (đọc cache mở rộng, quyết định những gì cần viết).

Thời gian một worker: `T_serial`.
Thời gian Hogwild! N-worker, nếu phối hợp là miễn phí: `T_serial * ((1 - p) + p / N)`. Định luật Amdahl cổ điển.
Với chi phí phối hợp: `T_serial * ((1 - p) + p / N) + c * steps_per_worker`.

Để một worker đạt hiệu quả, `c` phải nhỏ so với thời gian giải mã mỗi bước. Trên các mô hình suy luận tạo ra hơn 5k token, các worker có thể chịu được hàng trăm token chi phí phối hợp mà vẫn có lợi. Trên các tác vụ chat ngắn, chi phí phối hợp chiếm ưu thế và Hogwild! tệ hơn so với tuần tự.

### Ví dụ cụ thể

Bài toán suy luận: 10k token chuỗi suy luận (chain-of-thought). Giả sử bài toán có `p = 0.7` nội dung có thể song song hóa (các chiến lược chứng minh khác nhau, các phân tích trường hợp khác nhau) và `c = 200` token chi phí phối hợp mỗi worker. Với `N = 4` worker:

- Thời gian tuần tự: 10000 bước giải mã.
- Thời gian Hogwild!: 10000 * (0.3 + 0.7 / 4) + 200 * 4 = 10000 * 0.475 + 800 = 5550 bước giải mã.
- Tăng tốc: 10000 / 5550 = 1.8x.

Con số đó khá khiêm tốn. Nhưng trên các bài toán suy luận dài hơn (50k token), chi phí phối hợp được khấu hao và tốc độ tăng tốc đạt 2.5-3x. Hogwild! là phiên bản suy luận tương đương với song song hóa cấp luồng (thread-level parallelism) trong một ngôn ngữ cho phép bạn viết mã đa luồng một cách tự nhiên.

### Khi nào nên dùng Hogwild!

- Các bài toán suy luận dài (hàng nghìn token) nơi tác vụ có thể được song song hóa giữa các mục tiêu con độc lập.
- Các mô hình suy luận đã được đào tạo để suy nghĩ từng bước. Các mô hình không suy luận không tự phối hợp tốt.
- Triển khai trên một node với đủ VRAM để chứa cache chung cộng với N tiến trình worker. Cache được chia sẻ, nhưng mỗi worker có bộ nhớ kích hoạt (activation memory) riêng.

### Khi nào không nên dùng

- Chat tương tác ngắn. Chi phí phối hợp chiếm ưu thế.
- Các tác vụ không thể song song hóa (chứng minh tuyến tính đơn lẻ, biên dịch đơn lẻ). N=1 là tối đa.
- Các mô hình không suy luận. Không có sự phối hợp tự phát nào xuất hiện.
- Triển khai đa node. Cache chung cần đồng bộ hóa giữa các worker cực nhanh. Trong cùng một node thì ổn; giữa các node là một thảm họa về độ trễ.

### Trạng thái thử nghiệm

Tính đến tháng 4 năm 2026, Hogwild! là một phương pháp nghiên cứu với bản triển khai PyTorch mã nguồn mở. Việc áp dụng trong sản xuất vẫn chưa xảy ra. Ba rào cản:

1. Quản lý KV cache chung giữa các tiến trình đồng thời là một kỹ thuật không hề đơn giản.
2. Sự phối hợp tự phát phụ thuộc vào tác vụ; các benchmark vẫn đang được xây dựng.
3. Tốc độ tăng tốc khá khiêm tốn so với những gì speculative decoding đã mang lại, và cả hai có thể kết hợp nhưng kỹ thuật kết hợp lại là một lớp phức tạp khác.

Đáng để biết. Đáng để thử nghiệm. Chưa đáng để đặt cược vào một sản phẩm.

```figure
continuous-batching
```

## Xây dựng

`code/main.py` triển khai một trình mô phỏng Hogwild! đơn giản:

- Hai tiến trình worker, mỗi tiến trình là một "LLM" tất định tạo ra một trong số các loại token (token công việc, token quan sát, token phối hợp) với xác suất đã biết.
- Một cache chung (chỉ là một danh sách các token) mà cả hai worker đều đọc và ghi.
- Một logic phối hợp đơn giản: khi một worker thấy worker kia đã tạo đủ token công việc trong một danh mục, nó sẽ chọn một danh mục khác.

Trình mô phỏng chạy trong một ngân sách bước cố định và báo cáo:

- Tổng số token công việc được tạo ra.
- Tổng thời gian thực (số bước của worker).
- Tốc độ tăng tốc hiệu quả so với một worker đơn lẻ.
- Dấu vết của việc worker nào đã viết token nào.

### Bước 1: cache chung

Một danh sách mà cả hai worker đều thêm vào. Khóa đơn giản (Python `threading.Lock`) trong một bản triển khai thực tế; chúng ta mô phỏng bằng một bộ đếm.

### Bước 2: vòng lặp worker

Mỗi worker, tại mỗi bước:

- Đọc cache chung hiện tại.
- Quyết định loại token nào cần viết dựa trên những gì đã có ở đó.
- Viết một token.

### Bước 3: heuristic phối hợp

Nếu danh mục X đã có K token trong cache và danh mục dự định của worker là X, worker chuyển sang danh mục Y. Đây là một mô hình thay thế đơn giản cho hành vi của mô hình suy luận là "nhận thấy phần này đã được bao phủ, hãy làm việc khác thay thế."

### Bước 4: đo lường tốc độ tăng tốc

Chạy trình mô phỏng với N=1 worker và với N=2 worker, cùng ngân sách bước tổng cộng. Đếm số token công việc được tạo ra. N=2 sẽ tạo ra nhiều hơn khoảng 1.5-1.8x token công việc nhờ sự phân chia tác vụ dựa trên phối hợp.

### Bước 5: kiểm tra sự phối hợp

Giảm độ nhạy của heuristic phối hợp. Chạy lại. Quan sát thấy rằng nếu không có sự phối hợp tốt, N=2 sẽ tạo ra các token trùng lặp và tốc độ tăng tốc giảm xuống dưới 1. Điều này khớp với quan sát của bài báo: thủ thuật chỉ hoạt động nếu các worker có khả năng suy luận để tự phối hợp.

## Sử dụng

Việc tích hợp Hogwild! vào sản xuất tính đến tháng 4 năm 2026 vẫn ở mức nghiên cứu. Bản triển khai tham chiếu từ Yandex/HSE/IST dựa trên PyTorch và nhắm vào các thiết lập đa tiến trình trên một node cho các mô hình DeepSeek-R1 và QwQ.

Lộ trình áp dụng thực tế:

1. Profile khối lượng công việc tác vụ suy luận của bạn. Đo lường tỷ lệ các token mang tính khám phá (nhiều chiến lược, phân tích trường hợp, tìm kiếm) so với tuyến tính.
2. Nếu khám phá chiếm ưu thế, hãy chạy thử nghiệm Hogwild! hai worker. Đo lường sự cải thiện về thời gian thực.
3. Nếu sự cải thiện dưới 1.3x, bạn đang ở chế độ bị chi phối bởi phối hợp. Hãy quay lại worker đơn lẻ.
4. Nếu sự cải thiện trên 1.5x, hãy tăng lên N=4 và đo lại. Lợi nhuận giảm dần thường đạt mức N=4-8.

Kết hợp với speculative decoding: mỗi worker Hogwild! có thể sử dụng độc lập spec decode. Hai tốc độ tăng tốc nhân với nhau (xấp xỉ), mang lại 3x từ spec decode và 1.8x từ Hogwild! thành 5.4x hiệu quả so với giải mã worker đơn lẻ thông thường.

## Triển khai

Bài học này tạo ra `outputs/skill-parallel-inference-router.md`. Với một profile khối lượng công việc suy luận (ngân sách token, profile song song hóa tác vụ, dòng mô hình, mục tiêu triển khai), nó định tuyến giữa các chiến lược bỏ phiếu, tree-of-thought, đa tác nhân, Hogwild!, và speculative decoding.

## Bài tập

1. Chạy `code/main.py` với các cài đặt mặc định. Xác nhận cấu hình Hogwild! N=2 tạo ra nhiều token công việc hơn so với baseline N=1 trong cùng một thời gian thực.

2. Giảm cường độ của heuristic phối hợp (đặt `coordination_weight=0.1`). Chạy lại. Cho thấy tốc độ tăng tốc sụp đổ. Giải thích tại sao: các worker trùng lặp nỗ lực khi chúng không thể phối hợp.

3. Tính toán tốc độ tăng tốc Hogwild! dự kiến cho một tác vụ suy luận 50k-token với `p=0.8, c=500` và N=4 worker. Làm tương tự cho một tác vụ chat 1k-token với `p=0.3, c=200` và N=4. Tại sao một cái là thắng lợi và cái kia là thua lỗ?

4. Đọc Phần 4 của bài báo Hogwild! (đánh giá sơ bộ). Xác định hai chế độ thất bại mà các tác giả báo cáo. Mô tả cách một prompt phối hợp tốt hơn có thể giảm thiểu từng chế độ.

5. Kết hợp Hogwild! với speculative decoding trong trình mô phỏng: mỗi worker sử dụng spec-decode 2-token nội bộ. Báo cáo tốc độ tăng tốc nhân. Vấn đề sổ sách nào nảy sinh khi hai worker cùng muốn mở rộng cùng một tiền tố cache chung?

## Thuật ngữ chính

| Thuật ngữ | Mọi người nói | Ý nghĩa thực sự |
|------|----------------|------------------------|
| Hogwild! | "Worker song song, cache chung" | N instance của cùng một LLM chạy đồng thời với một KV cache chung; phối hợp tự phát thông qua self-prompting |
| Shared KV cache | "Môi trường phối hợp" | Một bộ đệm KV duy nhất đang phát triển mà tất cả các worker đều đọc và ghi; cho phép hiển thị token tức thì giữa các worker |
| Emergent coordination | "Không cần đào tạo" | Các LLM có khả năng suy luận có thể đọc cache chung và chia việc mà không cần fine-tuning hoặc giao thức rõ ràng |
| Coordination overhead (c) | "Token dành cho việc định hướng" | Chi phí mỗi worker để đọc cache mở rộng và quyết định việc cần làm; phải giữ nhỏ so với tổng thời gian giải mã |
| Parallelizable fraction (p) | "Cái gì có thể chạy song song" | Song song hóa cấp tác vụ: phần công việc tổng thể không mang tính tuần tự nội tại |
| RoPE enables Hogwild! | "Vị trí quay là bất biến với dịch chuyển" | Vì các vị trí là các phép quay, việc ghi vào cache chung không yêu cầu tính toán lại các token trước đó |
| Voting ensemble | "Chạy N, chọn đa số" | Cấu trúc liên kết suy luận song song đơn giản nhất; hữu ích cho phân loại, ít hữu ích cho suy luận dài |
| Tree of thought | "Phân nhánh và cắt tỉa" | Chiến lược suy luận khám phá nhiều nhánh và cắt tỉa; logic phối hợp rõ ràng |
| Multi-agent framework | "Gán tác vụ con" | Mỗi tác nhân có một vai trò; một điều phối viên điều phối; chi phí giao thức nặng nề |

## Đọc thêm

- [Rodionov và cộng sự — Hogwild! Inference: Parallel LLM Generation via Concurrent Attention (arXiv:2504.06261)](https://arxiv.org/abs/2504.06261) — bài báo Hogwild!, đánh giá sơ bộ trên các mô hình QwQ và DeepSeek-R1
- [Recht, Re, Wright, Niu — Hogwild!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent (arXiv:1106.5730, NeurIPS 2011)](https://arxiv.org/abs/1106.5730) — Hogwild! gốc, nguồn gốc tên gọi
- [Su và cộng sự — RoFormer: Enhanced Transformer with Rotary Position Embedding (arXiv:2104.09864)](https://arxiv.org/abs/2104.09864) — RoPE, đặc tính làm cho suy luận cache chung trở nên khả thi
- [Yao và cộng sự — Tree of Thoughts: Deliberate Problem Solving with Large Language Models (arXiv:2305.10601)](https://arxiv.org/abs/2305.10601) — chiến lược suy luận tree-of-thought mà Hogwild! nằm trực giao với nó
- [Leviathan và cộng sự — Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192)](https://arxiv.org/abs/2211.17192) — speculative decoding, sự song song hóa trong chuỗi mà Hogwild! kết hợp cùng
- [Bản triển khai PyTorch tham chiếu Hogwild!](https://github.com/eqimp/hogwild_llm) — nguồn sự thật duy nhất cho các thí nghiệm của bài báo