# Video-Language Models: Temporal Tokens and Grounding

> Video không phải là một chồng ảnh. Một đoạn clip 5 giây có thứ tự nhân quả, các động từ chỉ hành động và thời điểm sự kiện mà một mô hình hình ảnh không thể biểu diễn được. Video-LLaMA (Zhang và cộng sự, tháng 6 năm 2023) đã ra mắt video-LLM mã nguồn mở đầu tiên có khả năng grounding âm thanh-hình ảnh. VideoChat và Video-LLaVA đã mở rộng mô hình này. Đến năm 2025, TMRoPE của Qwen2.5-VL đã thu hẹp khoảng cách với các mô hình độc quyền tiên tiến. Mỗi hệ thống giải quyết các temporal token theo cách khác nhau — Q-former cho mỗi clip, concat-pool cho mỗi khung hình, TMRoPE cho mỗi token. Bài học này sẽ tìm hiểu các mô hình đó, xây dựng bộ lấy mẫu khung hình (frame sampler) từ đồng nhất đến động, và đánh giá trên các tác vụ temporal grounding.

**Type:** Build
**Languages:** Python (stdlib, frame sampler + temporal-grounding evaluator)
**Prerequisites:** Phase 12 · 08 (LLaVA-OneVision)
**Time:** ~180 minutes

## Mục tiêu học tập

- Giải thích lý do tại sao mã hóa vị trí theo thời gian (temporal positional encoding) làm thay đổi hiệu suất của video VLM một cách độc lập với vision encoder.
- So sánh việc lấy mẫu khung hình đồng nhất (uniform), FPS động (dynamic-FPS) và dựa trên sự kiện (event-driven) về số lượng token trên giây so với độ chính xác của grounding.
- Mô tả các thiết kế Q-former-cho-mỗi-clip (Video-LLaMA) so với pooled-cho-mỗi-khung-hình (Video-LLaVA) so với M-RoPE-cho-mỗi-token (Qwen2.5-VL).
- Kể tên bốn bộ benchmark video: VideoMME, TempCompass, EgoSchema, Video-MMMU.

## Vấn đề

Một video dài 1 phút ở tốc độ 30 FPS tương đương với 1800 khung hình. Với 196 visual token mỗi khung hình (ViT-B ở 224), đó là 352k token — lớn hơn bất kỳ ngữ cảnh LLM nào vào thời điểm năm 2024.

Có ba chiến lược giảm thiểu:

1. Lấy mẫu phụ (subsample) khung hình (1-8 FPS tùy thuộc vào nội dung).
2. Gộp (pool) các patch token của mỗi khung hình một cách mạnh mẽ (bilinear pool 3x3 hoặc 4x4).
3. Nén thông qua Q-former nhận một clip 16 khung hình và xuất ra 64 token.

Mỗi sự đánh đổi đều khác nhau. Lấy mẫu phụ làm mất chi tiết thời gian. Gộp làm mất chi tiết không gian. Q-former làm mất một chút cả hai nhưng tiết kiệm token.

Mã hóa vị trí thời gian là trục còn lại: làm thế nào mô hình biết khung hình 5 đến trước khung hình 6? Các tùy chọn bao gồm 1D temporal RoPE đơn giản (Video-LLaMA), embedding thời gian được học (Video-LLaVA), và TMRoPE (Qwen2.5-VL, 3D đầy đủ).

## Khái niệm

### Video-LLaMA: Q-former cho mỗi clip + nhánh âm thanh

Video-LLaMA (2023) là video-LLM mã nguồn mở đầu tiên. Kiến trúc:

- Clip 16 khung hình ở 2 FPS (tức là 8 giây).
- Đặc trưng ViT mỗi khung hình -> Video Q-former thực hiện cross-attention trên tất cả 16 khung hình -> 32 truy vấn đã học -> LLM.
- Nhánh âm thanh song song: dạng sóng -> ImageBind audio encoder -> Audio Q-former -> 32 truy vấn -> LLM.

Điểm mạnh: suy luận kết hợp âm thanh-hình ảnh. Điểm yếu: độ dài clip cố định, không có grounding thời gian tùy ý.

### VideoChat và Video-LLaVA

VideoChat giữ ý tưởng của Video-LLaMA nhưng loại bỏ âm thanh và đơn giản hóa. Video-LLaVA (Lin và cộng sự, 2023) đã huấn luyện một visual encoder duy nhất trên cả hình ảnh và khung hình video ("alignment before projection"), tạo ra một biểu diễn thống nhất. Cả hai đều là frozen-CLIP-encoder + MLP + LLM.

Không mô hình nào xử lý được video dài. Cả hai đều là hệ thống 8-16 khung hình.

### Qwen2.5-VL và TMRoPE

Qwen2.5-VL giới thiệu TMRoPE — Temporal-Modality Rotary Position Embedding. Mỗi patch token mang một vị trí (t, h, w) trong đó t là dấu thời gian thực (không phải chỉ số khung hình).

Sự khác biệt chính so với embedding thời gian đơn giản:

- Thời gian tuyệt đối, không phải chỉ số. Mô hình nhìn thấy "tại 4.2 giây" chứ không phải "tại khung hình 15".
- Xoay mỗi token, không phải mỗi clip. Mỗi visual token xoay độc lập theo dấu thời gian của nó.
- Tương thích với FPS động. Nếu bạn lấy mẫu ở 2 FPS ở đây và 4 FPS ở kia, TMRoPE xử lý khoảng cách không đều một cách tự nhiên.

TMRoPE cho phép các truy vấn "con mèo nhảy vào giây thứ mấy?". Mô hình có thể xuất ra "tại 4.2 giây". Video-LLaMA chỉ có thể nói "đầu clip".

### Chiến lược lấy mẫu khung hình

Đồng nhất (Uniform): lấy mẫu N khung hình đều nhau trong suốt thời lượng. Đơn giản, làm mất các đỉnh chuyển động.

FPS động (Dynamic FPS): lấy mẫu thích ứng dựa trên cường độ chuyển động. Optical flow hoặc chênh lệch khung hình chọn các phân đoạn chuyển động cao để lấy mẫu dày đặc hơn. Qwen2.5-VL huấn luyện trên phương pháp này.

Dựa trên sự kiện (Event-driven): chạy một detector nhẹ, lấy mẫu nhiều hơn ở nơi hành động xảy ra. Được sử dụng bởi VideoAgent.

Keyframe + ngữ cảnh: lấy mẫu tại các ranh giới cảnh + một vài khung hình lân cận. Được sử dụng cho nội dung điện ảnh.

### Gộp (Pooling) cho mỗi khung hình

Ở 1 FPS và 576 token mỗi khung hình, một video 5 phút là 172,800 token. Có thể thực hiện được với ngữ cảnh 128k của Qwen2.5-VL-72B nhưng rất tốn kém.

Bilinear pool 3x3 giảm xuống còn 64 token mỗi khung hình -> 19,200 token cho 5 phút. Điểm tối ưu cho hầu hết các tác vụ.

Gộp mạnh hơn (6x6 -> 16 token mỗi khung hình) cho các quy trình làm việc của agent nơi chi tiết không gian ít quan trọng hơn.

### Bốn bộ benchmark video

- VideoMME: hiểu video toàn diện, ngắn + trung bình + dài.
- TempCompass: suy luận thời gian chi tiết, các câu hỏi "trước" / "sau".
- EgoSchema: video góc nhìn thứ nhất dài.
- Video-MMMU: câu hỏi video đa phương thức, đa lĩnh vực.

Một đánh giá video-VLM đầy đủ sẽ bao gồm cả bốn. Chúng nhấn mạnh các trục khác nhau — TempCompass tập trung vào thứ tự, EgoSchema tập trung vào suy luận trên 3 phút, VideoMME trải dài các thời lượng.

### Định dạng đầu ra Grounding

Định dạng đầu ra cho temporal grounding:

- Văn bản tự do: "Con mèo nhảy vào khoảng giây thứ 4." Dễ phân tích nhưng không chính xác.
- JSON có cấu trúc: `{"event": "jump", "start": 4.1, "end": 4.3}`. Qwen2.5-VL huấn luyện định dạng này.
- Dựa trên token: các token đặc biệt `<time>4.1</time>` xen kẽ với câu trả lời. Định dạng nội bộ của Qwen2.5-VL.

Dựa trên token là chính xác nhất cho việc sử dụng hạ nguồn. Định dạng đầu ra JSON của Qwen2.5-VL có thể phân tích trực tiếp.

### Thực tiễn tốt nhất năm 2026

Đối với video VLM vào năm 2026:

- Encoder: SigLIP 2 với M-RoPE hoặc TMRoPE (Qwen2.5-VL).
- Lấy mẫu khung hình: FPS động (1-4 tùy thuộc vào chuyển động) với giới hạn khung hình tối đa.
- Gộp mỗi khung hình: bilinear 3x3.
- Đầu ra: JSON có cấu trúc với các trường thời gian + sự kiện.
- Benchmark: VideoMME + TempCompass cho tổng quát; EgoSchema cho dài hạn.

```figure
video-temporal-patches
```

## Sử dụng

`code/main.py` bao gồm:

- Các bộ lấy mẫu khung hình đồng nhất và FPS động.
- Một bộ đánh giá temporal-grounding đơn giản: với một sự kiện "ground truth" tại thời điểm T và đầu ra của mô hình, tính điểm độ chính xác với sai số cho phép.
- So sánh giữa Video-LLaMA (16 khung hình, Q-former), Video-LLaVA (8 khung hình, MLP), Qwen2.5-VL (FPS động + TMRoPE).

## Triển khai

Bài học này tạo ra `outputs/skill-video-vlm-frame-planner.md`. Với một tác vụ video (giám sát, nhận dạng hành động, temporal grounding, tóm tắt), nó chọn bộ lấy mẫu khung hình, hệ số gộp, định dạng đầu ra và mức độ chính xác kỳ vọng.

## Bài tập

1. Đối với một video hướng dẫn nấu ăn dài 3 phút, hãy chọn giữa FPS đồng nhất và FPS động. Giải thích bằng số lượng token.

2. TMRoPE bổ sung những gì cụ thể mà một bảng embedding thời gian đơn giản không thể làm được?

3. Viết một JSON schema cho temporal grounding mà một VLM có thể học cách xuất ra. Bao gồm các trường hợp lỗi.

4. Đọc Phần 3 của Video-LLaVA về "Alignment Before Projection". Tại sao điều này tốt hơn việc huấn luyện các encoder hình ảnh và video riêng biệt?

5. Dựa trên bảng xếp hạng VideoMME, khoảng cách giữa mô hình mã nguồn mở hàng đầu và mô hình độc quyền hàng đầu tính đến năm 2026 là bao nhiêu? Bao nhiêu phần trăm khoảng cách đó là do mã hóa thời gian so với quy mô LLM cơ sở?

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|------------------------|
| Temporal grounding | "Câu trả lời định vị thời gian" | VLM xuất ra một khoảng thời gian cụ thể cho thời điểm sự kiện xảy ra |
| TMRoPE | "Time-Multimodal RoPE" | Vị trí xoay 3D với dấu thời gian tuyệt đối, được Qwen2.5-VL sử dụng |
| Dynamic FPS | "Lấy mẫu nhận biết chuyển động" | Lấy mẫu nhiều khung hình hơn ở các phân đoạn chuyển động cao, ít hơn ở các phân đoạn tĩnh |
| Frame pooling | "Nén không gian mỗi khung hình" | Giảm các patch mỗi khung hình bằng nội suy bilinear trước khi đưa vào LLM |
| Video Q-former | "Bộ nén clip" | Nút thắt cross-attention ánh xạ N khung hình thành K truy vấn đã học |
| VideoMME | "Benchmark video" | Benchmark video toàn diện ngắn/trung bình/dài, hơn 2500 mẫu |

## Đọc thêm

- [Zhang và cộng sự — Video-LLaMA (arXiv:2306.02858)](https://arxiv.org/abs/2306.02858)
- [Li và cộng sự — VideoChat (arXiv:2305.06355)](https://arxiv.org/abs/2305.06355)
- [Lin và cộng sự — Video-LLaVA (arXiv:2311.10122)](https://arxiv.org/abs/2311.10122)
- [Qwen Team — Qwen2.5-VL (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
- [Lin và cộng sự — VILA-1.5 (arXiv:2312.07533)](https://arxiv.org/abs/2312.07533)