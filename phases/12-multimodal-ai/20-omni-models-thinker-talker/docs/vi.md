# Omni Models: Qwen2.5-Omni và Thinker-Talker Split

> Bản demo sản phẩm của GPT-4o vào tháng 5 năm 2024 gây chấn động không phải vì mô hình nền tảng, mà vì hình thái sản phẩm — một giao diện giọng nói nơi bạn trò chuyện, mô hình nhìn thấy những gì camera thấy và phản hồi lại trong dưới 250ms. Hệ sinh thái mã nguồn mở đã dành phần còn lại của năm 2024 và 2025 để chạy đua đạt tới trải nghiệm sản phẩm đó. Qwen2.5-Omni (tháng 3 năm 2025) là thiết kế mở tham chiếu: một Thinker (transformer tạo văn bản lớn) cộng với một Talker (transformer tạo giọng nói song song), được liên kết bởi các token giọng nói streaming. Mini-Omni đã đơn giản hóa nó, Moshi bắt kịp độ trễ của nó, GLM-4-Voice mở rộng nó sang tiếng Trung. Bài học này tìm hiểu kiến trúc Thinker-Talker và ngân sách độ trễ giúp việc đối thoại thời gian thực qua streaming trở nên khả thi.

**Type:** Build
**Languages:** Python (stdlib, streaming pipeline latency simulator + VAD loop)
**Prerequisites:** Phase 12 · 19 (audio-LLMs), Phase 12 · 16 (any-to-any)
**Time:** ~180 phút

## Mục tiêu học tập

- Chia pipeline suy luận thành Thinker (lập luận văn bản) và Talker (tổng hợp giọng nói) và giải thích tại sao streaming song song lại hiệu quả.
- Tính toán ngân sách thời gian đến byte âm thanh đầu tiên (TTFAB) cho một tương tác đối thoại, theo từng thành phần.
- Mô tả TMRoPE, phương pháp mã hóa vị trí căn chỉnh theo thời gian trên vision, audio và text bên trong Thinker.
- Gọi tên ba mô hình đối thoại thời gian thực: half-duplex, turn-taking, full-duplex.

## Vấn đề

Một trợ lý giọng nói thời gian thực phải thực hiện rất nhiều việc, với tốc độ cao:

1. Nghe người dùng. Token hóa giọng nói thời gian thực, phát hiện hoạt động giọng nói (VAD) để biết khi nào họ ngừng nói.
2. Tùy chọn nhìn. Đầu vào camera ở mức 2-4 FPS, được stream vào Thinker cùng với âm thanh.
3. Suy nghĩ. Soạn thảo phản hồi dựa trên lịch sử cuộc trò chuyện.
4. Nói. Tổng hợp các token giọng nói, giải mã thành dạng sóng, stream đến loa của người dùng.

Mỗi bước đều làm tăng độ trễ. Cảm giác đối thoại tự nhiên đòi hỏi tổng thời gian khứ hồi < 500ms — dưới mức đó, người dùng sẽ không nhận thấy độ trễ. GPT-4o tuyên bố ~250ms. Moshi ~160ms. Qwen2.5-Omni ~350-500ms.

Mỗi thành phần cần phải stream. Không thể có chuyện "batch mọi thứ rồi mới giải mã".

## Khái niệm

### Thinker và Talker

Sự phân tách của Qwen2.5-Omni:

- Thinker: một transformer tạo văn bản từ 7B-80B tham số. Tiêu thụ các token văn bản + hình ảnh + âm thanh xen kẽ. Xuất ra các token văn bản đại diện cho những gì cần nói.
- Talker: một transformer tạo giọng nói nhỏ hơn (200M-1B). Tiêu thụ các token đầu ra văn bản của Thinker cộng với các token ngữ cảnh giọng nói gần đây. Xuất ra các token giọng nói rời rạc (chỉ số residual-VQ).
- Bộ giải mã giọng nói: một bộ giải mã dạng sóng streaming (họ SNAC, MoVQGAN) chuyển đổi các token giọng nói thành các mẫu âm thanh trong thời gian thực.

Sự phân tách này rất quan trọng. Thinker phải lớn để có khả năng lập luận tốt. Talker có thể nhỏ vì công việc của nó mang tính cục bộ — chuyển đổi văn bản thành token giọng nói. Talker lớn hơn không có nghĩa là biểu cảm hơn; nó chỉ chậm hơn.

Chạy cả hai song song:

1. Thinker phát ra token văn bản t_i.
2. Talker tiêu thụ t_i (thông qua streaming) và phát ra các token giọng nói s_i, s_{i+1}, ..., s_{i+k}.
3. Bộ giải mã giọng nói tiêu thụ các token giọng nói ngay khi chúng đến và phát ra các mẫu âm thanh.
4. Khi Thinker đang ở token văn bản t_{i+3}, Talker đã stream xong âm thanh cho t_0..t_{i+2}.

### TMRoPE — vị trí đa phương thức căn chỉnh theo thời gian

Thinker cần tích hợp các khung hình ảnh (đến với tốc độ, ví dụ, 4 FPS), các khung âm thanh (đến với tốc độ 50 khung hình/giây) và văn bản từ lịch sử cuộc trò chuyện. Thứ tự chuỗi ngây thơ (tất cả hình ảnh, sau đó là tất cả âm thanh, rồi đến văn bản) sẽ làm mất sự căn chỉnh thời gian.

TMRoPE gán dấu thời gian tuyệt đối cho mọi token. Token hình ảnh tại t=2.3s. Token âm thanh tại t=2.32s. Token văn bản từ người dùng "stop" tại t=2.35s. RoPE xoay attention theo dấu thời gian; mô hình nhìn thấy chúng như đang diễn ra đồng thời về mặt thời gian.

Đây là cơ sở hạ tầng để tính năng "anh ấy vẫy tay trong khi nói xin chào" hoạt động — mô hình nhìn thấy khung hình video và âm thanh tại cùng một thời điểm khái niệm.

### Tổng hợp giọng nói streaming

Các token giọng nói phải được stream. Mini-Omni (Xie & Wu, 2024) đã giới thiệu "các mô hình ngôn ngữ có thể nghe, nói trong khi suy nghĩ theo kiểu streaming": các token đầu ra của Thinker và các token đầu ra của Talker xen kẽ trong cùng một chuỗi. Talker kích hoạt ngay khi Thinker cam kết token văn bản tiếp theo. Không có ranh giới batch.

Moshi (Défossez et al., tháng 10 năm 2024) là triển khai mở nhanh nhất. 160ms TTFAB trên một card A100 đơn lẻ. Kiến trúc: một transformer 7B duy nhất phát ra các token văn bản và giọng nói ở các vị trí xen kẽ, với một "độc thoại nội tâm" (inner monologue) tách biệt luồng suy nghĩ khỏi luồng nói. Đây thực chất là Thinker + Talker được hợp nhất thành một mô hình với quá trình huấn luyện cẩn thận.

### VAD và lượt đối thoại (turn-taking)

Phát hiện hoạt động giọng nói (VAD) chạy ở phía đầu vào. Hai mô hình:

- Half-duplex: người dùng nói, mô hình nghe. Mô hình nói, người dùng nghe. Chuyển giao rõ ràng thông qua phát hiện im lặng của VAD (~200ms).
- Full-duplex: cả hai có thể nói đồng thời. Mô hình có thể phản hồi phụ ("uh-huh") hoặc ngắt lời. Khó hơn nhiều. Moshi hỗ trợ điều này.

Qwen2.5-Omni hỗ trợ half-duplex theo mặc định, với việc chuyển lượt thông qua ngưỡng im lặng. Full-duplex đòi hỏi xử lý ở tầng ứng dụng.

### Qwen3-Omni (tháng 11 năm 2025)

Người kế nhiệm. Thinker Qwen3-80B, Talker lớn hơn, TMRoPE-v2 cải tiến. Độ trễ gần với mức 250ms của GPT-4o. Trọng số mở. Các benchmark trên OmniBench cạnh tranh với Gemini 2.0 Live.

### Ngân sách độ trễ sản xuất

Đối với một tương tác streaming điển hình:

- Mic -> token âm thanh: 40-80ms.
- Prefill (prompt + lịch sử): 100-200ms ở 7B, nhiều hơn đáng kể ở 70B.
- Token văn bản Thinker đầu tiên: 40ms.
- Talker xử lý token văn bản đầu tiên: 20ms.
- Các token giọng nói đầu tiên được cam kết: 40ms.
- Giải mã Residual-VQ: 30ms.
- Giải mã dạng sóng giọng nói: 50-80ms.

Tổng TTFAB: 320-510ms ở 7B, 600-900ms ở 70B. Chất lượng frontier thường có nghĩa là 70B+; do đó có khoảng cách độ trễ frontier.

### Toán học về tốc độ token

Ở giọng nói 16kHz với các token giọng nói cơ sở 50 Hz, bạn cần 50 token giọng nói mỗi giây đầu ra. Talker phải phát ra ≥50 tok/s để theo kịp. Ở thông lượng LLM điển hình là 30-80 tok/s trên H100, một Talker nhỏ (200-300M) là đủ nhanh; một Talker 7B sẽ bị tụt lại phía sau.

Đây là lý do tại sao các mô hình Talker chuyên dụng nhỏ tồn tại thay vì "chỉ sử dụng mô hình chính".

```figure
l5-thinker-talker
```

## Sử dụng

`code/main.py`:

- Mô phỏng pipeline Thinker-Talker với các tốc độ phát token giả lập.
- Tính toán TTFAB cho các kích thước mô hình và tốc độ lấy mẫu mic có thể cấu hình.
- Minh họa việc chuyển lượt half-duplex với ngưỡng im lặng VAD.

## Triển khai

Bài học này tạo ra `outputs/skill-omni-streaming-budget.md`. Với mục tiêu TTFAB và bộ tính năng của một sản phẩm giọng nói thời gian thực (vision-in, song ngữ, full-duplex), hãy chọn Qwen2.5-Omni, Qwen3-Omni, Moshi hoặc Mini-Omni và định cỡ Thinker/Talker.

## Bài tập

1. Mục tiêu TTFAB của bạn là 300ms. Với Thinker 7B và Talker 300M, hãy viết ra độ trễ của từng thành phần.

2. Qwen2.5-Omni sử dụng TMRoPE. Mô tả những gì mô hình nhìn thấy đối với một prompt mà người dùng bắt đầu nói tại t=1s và camera bắt được một cử chỉ tại t=1.2s.

3. Hỗ trợ full-duplex đòi hỏi mô hình phải phát ra âm thanh trong khi đang nghe. Hãy đề xuất một định dạng dữ liệu huấn luyện để dạy điều này.

4. Đọc Mục 4 của bài báo Moshi. Mô tả sự tách biệt "độc thoại nội tâm" và tại sao nó tránh được việc chia tách Thinker-Talker.

5. Tính toán ngân sách thông lượng: Talker phải phát ra token nhanh đến mức nào để theo kịp giọng nói 16kHz ở 50 token/giây lớp cơ sở?

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|------------------------|
| Thinker | "Bộ não lập luận" | Transformer tạo văn bản lớn sản xuất nội dung cần nói |
| Talker | "Miệng tạo giọng nói" | Transformer nhỏ sản xuất các token giọng nói rời rạc từ văn bản của Thinker |
| TTFAB | "Ngân sách độ trễ" | Time-to-first-audio-byte: từ khi người dùng ngừng nói đến mẫu âm thanh đầu tiên được xuất ra |
| TMRoPE | "RoPE căn chỉnh thời gian" | Mã hóa vị trí sử dụng dấu thời gian tuyệt đối trên vision, audio, text |
| Half-duplex | "Chuyển lượt" | Người dùng và mô hình luân phiên; im lặng VAD phát hiện người dùng đã nói xong |
| Full-duplex | "Đồng thời" | Mô hình có thể nói và nghe cùng lúc; có khả năng phản hồi phụ |
| Inner monologue | "Sự tách biệt Moshi" | Thiết kế mô hình đơn lẻ nơi luồng suy nghĩ và luồng nói xen kẽ |

## Đọc thêm

- [Xu et al. — Qwen2.5-Omni (arXiv:2503.20215)](https://arxiv.org/abs/2503.20215)
- [Qwen Team — Qwen3-Omni (arXiv:2509.17765)](https://arxiv.org/html/2509.17765v1)
- [Xie & Wu — Mini-Omni (arXiv:2408.16725)](https://arxiv.org/abs/2408.16725)
- [Défossez et al. — Moshi (arXiv:2410.00037)](https://arxiv.org/abs/2410.00037)
- [Zeng et al. — GLM-4-Voice (arXiv:2412.02612)](https://arxiv.org/abs/2412.02612)