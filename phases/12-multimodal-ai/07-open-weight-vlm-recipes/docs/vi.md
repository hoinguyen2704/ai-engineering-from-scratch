# Open-Weight VLM Recipes: What Actually Matters

> Tài liệu về open-weight VLM giai đoạn 2024-2026 là một rừng các bảng ablation (bảng phân tích thành phần). MM1 của Apple đã thử nghiệm 13 tổ hợp giữa image encoder, connector và data mix. Molmo của Allen AI đã chứng minh rằng các chú thích chi tiết từ con người (detailed human captions) vượt trội hơn so với việc chưng cất (distillation) từ GPT-4V. Cambrian-1 đã thực hiện hơn 20 so sánh về encoder. Idefics2 đã chính thức hóa không gian thiết kế năm trục. Prismatic VLMs đã so sánh 27 công thức huấn luyện trên một benchmark được kiểm soát. Trong tất cả những nhiễu loạn đó, một tập hợp nhỏ các kết quả vẫn giữ nguyên giá trị qua các bài báo: image encoder quan trọng hơn kiến trúc connector, data mixture quan trọng hơn cả hai, và các chú thích chi tiết từ con người vượt trội hơn dữ liệu tổng hợp được chưng cất. Bài học này sẽ đọc các bảng đó thay cho bạn.

**Type:** Learn + lab
**Languages:** Python (stdlib, ablation table parser + recipe picker)
**Prerequisites:** Phase 12 · 05 (LLaVA baseline)
**Time:** ~180 phút

## Mục tiêu học tập

- Gọi tên không gian thiết kế VLM năm trục: image encoder, connector, LLM, data mix, resolution schedule.
- Đọc bảng ablation của MM1 / Idefics2 / Cambrian-1 và dự đoán nút điều chỉnh nào sẽ tác động đến một benchmark nhất định.
- Chọn một công thức (encoder, connector, data, resolution) cho một VLM mới dựa trên ngân sách tính toán và tập hợp tác vụ.
- Giải thích tại sao các chú thích chi tiết từ con người lại vượt trội hơn việc chưng cất từ GPT-4V ở cùng số lượng token.

## Vấn đề

Hiện có hàng trăm open-weight VLM. Phần lớn khoảng cách giữa "tốt" và "state-of-the-art" không nằm ở kiến trúc. Nó nằm ở dữ liệu, lịch trình độ phân giải (resolution schedule) và lựa chọn encoder. Biết được nút nào cần xoay trước khi mô hình của bạn hoạt động kém hiệu quả sẽ giúp bạn tránh được sai lầm tốn kém 5 triệu giờ GPU.

Làn sóng 2023 (LLaVA-1.5, InstructBLIP, MiniGPT-4) chạy trên tiền huấn luyện cặp chú thích + LLaVA-Instruct-150k. Baseline tốt. Đạt đỉnh khoảng 35% trên MMMU.

Làn sóng 2024 (MM1, Idefics2, Molmo, Cambrian-1, Prismatic VLMs) đã thực hiện các thử nghiệm ablation toàn diện. Kết quả rất đáng ngạc nhiên và thực tế.

## Khái niệm

### Không gian thiết kế năm trục

Idefics2 (Laurençon et al., 2024) đã đặt tên cho các trục:

1. Image encoder. CLIP ViT-L/14, SigLIP SO400m/14, DINOv2 ViT-g/14, InternViT-6B. Các encoder khác nhau về kích thước patch, độ phân giải và mục tiêu tiền huấn luyện.
2. Connector. MLP (2-4 lớp), Q-Former (32 queries + cross-attn), Perceiver Resampler (64 queries), C-Abstractor (convolutional + bilinear pooling).
3. Language model. Llama-3 8B / 70B, Mistral 7B, Phi-3, Gemma-2, Qwen2.5. Kích thước LLM là chi phí tham số chiếm ưu thế.
4. Training data. Cặp chú thích (CC3M, LAION), interleaved (OBELICS, MMC4), instruction (LLaVA-Instruct, ShareGPT4V, PixMo, Cauldron).
5. Resolution schedule. Cố định 224/336/448, AnyRes, native dynamic. Tăng dần trong quá trình huấn luyện hoặc giữ cố định.

Mỗi VLM thương mại đều đưa ra lựa chọn trên từng trục. Phần lớn sự biến thiên trong điểm số MMMU được giải thích bởi các trục 1, 4 và 5 — chứ không phải bởi connector bạn chọn.

### Trục 1: encoder > connector

MM1 Phần 3.2 cho thấy: chuyển từ CLIP ViT-L/14 sang SigLIP SO400m/14 giúp tăng hơn 3 điểm MMMU. Chuyển connector từ MLP sang Perceiver Resampler chỉ tăng chưa đầy 1 điểm. Idefics2 đã tái lập kết quả: SigLIP > CLIP, Q-Former ≈ MLP ≈ Perceiver ở cùng số lượng token.

"Cambrian Vision Encoders Match-Up" của Cambrian-1 (Tong et al., 2024) đã chạy hơn 20 encoder trên một benchmark tập trung vào thị giác (CV-Bench). Đứng đầu bảng xếp hạng là sự kết hợp giữa DINOv2 và SigLIP; CLIP nằm ở giữa; ImageBind và ViT-MAE thấp hơn. Khoảng cách từ CLIP ViT-L đến DINOv2 ViT-g/14 là khoảng 5-7 điểm trên CV-Bench.

Encoder mặc định năm 2026 cho các open VLM là SigLIP 2 SO400m/14 cho các đặc trưng ngữ nghĩa + dày đặc, đôi khi được kết hợp với các đặc trưng DINOv2 ViT-g/14 ("Spatial Vision Aggregator" của Cambrian thực hiện điều này).

### Trục 2: thiết kế connector không tạo ra khác biệt lớn

MM1, Idefics2, Prismatic và MM-Interleaved đều đi đến cùng một kết luận: ở một số lượng token thị giác cố định, kiến trúc connector hầu như không quan trọng. Một MLP 2 lớp trên các patch được mean-pooled có hiệu suất chênh lệch trong vòng 1 điểm so với Q-Former 32-query ở cùng ngân sách token.

Điều quan trọng là số lượng token. Nhiều token thị giác hơn = nhiều tính toán LLM hơn = hiệu suất tốt hơn đến một mức nhất định, sau đó là lợi nhuận giảm dần. 64 token mỗi ảnh là quá ít cho OCR. 576-1024 token là điểm ngọt (sweet spot) cho hầu hết các open VLM. 2048+ chỉ giúp ích cho tài liệu và biểu đồ.

Q-Former so với MLP là câu hỏi về chi phí, không phải câu hỏi về chất lượng: Q-Former giới hạn token ở mức 32-64 bất kể độ phân giải ảnh; MLP phát ra tất cả các token patch. Đối với đầu vào độ phân giải cao, Q-Former tiết kiệm ngữ cảnh LLM; đối với độ phân giải thấp, sự khác biệt chỉ là nhiễu.

### Trục 3: kích thước LLM thiết lập trần hiệu suất

Việc tăng gấp đôi LLM từ 7B lên 13B giúp tăng ổn định 2-4 điểm trên MMMU trong mọi bài báo VLM. Ở mức 70B, bạn bão hòa hầu hết các benchmark. Trần lý luận đa phương thức của VLM chính là trần lý luận văn bản của LLM — vision encoder chỉ có thể cung cấp dữ liệu cho nó, chứ không thể lý luận thay cho nó.

Đây là lý do tại sao Qwen2.5-VL-72B và Claude Opus 4.7 đè bẹp MMMU-Pro và ScreenSpot-Pro: bộ não ngôn ngữ rất lớn. Một VLM 7B không thể thay thế cho một VLM 70B thông qua thiết kế connector thông minh.

### Trục 4: dữ liệu — chú thích chi tiết từ con người vượt trội hơn chưng cất

Molmo + PixMo (Deitke et al., 2024) là kết quả năm 2024 mà mọi người nên đọc. Allen AI đã thuê các annotator con người mô tả hình ảnh trong các lượt chuyển đổi giọng nói thành văn bản dày đặc kéo dài 1-3 phút, tạo ra 712K hình ảnh được chú thích dày đặc. Không có sự chưng cất GPT-4V nào trong dữ liệu huấn luyện.

Molmo-72B đã đánh bại Llama-3.2-90B-Vision trên 11/11 benchmark. Sự khác biệt không nằm ở kiến trúc — mà ở chất lượng chú thích. Các chú thích chi tiết từ con người chứa thông tin nhiều gấp 5-10 lần mỗi ảnh so với các chú thích web ngắn và duy trì tính xác thực, trong khi dữ liệu chưng cất từ GPT-4V thường gây ra ảo giác.

ShareGPT4V (Chen et al., 2023) và Cauldron (Idefics2) đã đi theo cùng một kịch bản với các chú thích hỗn hợp từ con người + GPT-4V. Xu hướng rất rõ ràng: cho giai đoạn 2026, mật độ chú thích > số lượng chú thích > sự tiện lợi của chưng cất.

### Trục 5: độ phân giải và lịch trình của nó

Các thử nghiệm ablation của Idefics2: 384 -> 448 tăng 1-2 điểm. 448 -> 980 với chia nhỏ ảnh (AnyRes) tăng thêm 3-5 điểm trên các benchmark OCR. Huấn luyện độ phân giải phẳng sẽ đạt ngưỡng bão hòa ở độ chính xác trung bình; tăng dần độ phân giải (bắt đầu 224, kết thúc 448 hoặc native) giúp huấn luyện nhanh hơn và đạt kết quả cao hơn.

Cambrian-1 đã thực hiện đánh đổi độ phân giải so với token: ở mức tính toán cố định, bạn có thể có nhiều token hơn ở độ phân giải thấp hơn hoặc ít token hơn ở độ phân giải cao hơn. Độ phân giải cao hơn thắng cho OCR; độ phân giải thấp hơn với nhiều token hơn thắng cho hiểu biết cảnh quan chung.

Công thức sản xuất năm 2026: huấn luyện Giai đoạn 1 ở mức 384 cố định, Giai đoạn 2 với độ phân giải động lên đến 1280 cho các tác vụ nặng về OCR.

### So sánh có kiểm soát của Prismatic

Prismatic VLMs (Karamcheti et al., 2024) là bài báo kiểm soát tất cả các trục. Cùng một LLM 13B, cùng dữ liệu instruction, cùng đánh giá — chỉ một trục thay đổi tại một thời điểm. Kết quả:

- Số lượng token thị giác mỗi ảnh giải thích ~60% sự biến thiên.
- Lựa chọn encoder giải thích ~20%.
- Kiến trúc connector giải thích ~5%.
- Mọi thứ khác (data mix, scheduler, LR) chiếm 15% còn lại.

Đây là một sự phân tách thô, nhưng là câu trả lời rõ ràng nhất cho câu hỏi "tôi nên ablate cái gì trước" trong tài liệu.

### Bộ chọn công thức cho năm 2026

Dựa trên bằng chứng, công thức open-VLM mặc định cho một dự án mới vào năm 2026:

- Encoder: SigLIP 2 SO400m/14 ở độ phân giải gốc với NaFlex, kết hợp với DINOv2 ViT-g/14 cho các đặc trưng dày đặc nếu bạn cần phân đoạn/định vị (segmentation/grounding).
- Connector: MLP 2 lớp trên các patch token. Bỏ qua Q-Former trừ khi bạn bị giới hạn về token.
- LLM: Qwen2.5 / Llama-3.1 / Gemma 2, 7B cho chi phí, 70B cho chất lượng, được chọn theo độ trễ mục tiêu.
- Data: PixMo + ShareGPT4V + Cauldron, bổ sung thêm dữ liệu instruction cụ thể cho tác vụ.
- Resolution: động (tối thiểu 256, tối đa 1280 pixel mỗi cạnh dài).
- Schedule: Giai đoạn 1 căn chỉnh (chỉ projector), Giai đoạn 2 tinh chỉnh toàn bộ, Giai đoạn 3 tinh chỉnh cụ thể cho tác vụ.

Mỗi mặc định đó đều bắt nguồn từ một thử nghiệm ablation được đo lường trong các bài báo được trích dẫn ở cuối bài học này.

```figure
l5-vlm-recipe-knobs
```

## Sử dụng nó

`code/main.py` là một trình phân tích bảng ablation và bộ chọn công thức. Nó mã hóa các bảng ablation của MM1 và Idefics2 (đã được tóm tắt) và cho phép bạn truy vấn:

- "Với ngân sách X và tác vụ Y, công thức nào thắng?"
- "Nếu tôi đổi SigLIP lấy CLIP trên Llama 7B, delta MMMU dự kiến là bao nhiêu?"
- "Trục nào tôi nên ablate trước để có câu trả lời với độ tin cậy 80%?"

Đầu ra là một danh sách công thức được xếp hạng với các delta benchmark dự kiến và khuyến nghị "ablate trước".

## Triển khai nó

Bài học này tạo ra `outputs/skill-vlm-recipe-picker.md`. Với một tập hợp tác vụ mục tiêu, ngân sách tính toán và mục tiêu độ trễ, nó phát ra một công thức đầy đủ (encoder, connector, LLM, data mix, resolution schedule) kèm theo các trích dẫn đến thử nghiệm ablation biện minh cho mỗi lựa chọn. Ngăn chặn các kỹ sư phải phát minh lại bảng ablation của Idefics2 mỗi khi một dự án VLM mới bắt đầu.

## Bài tập

1. Đọc MM1 Phần 3.2. Đối với LLM 2B cố định ở ngân sách 50M ảnh, encoder nào thắng? Câu trả lời có thay đổi ở LLM 13B không? Tại sao?

2. Cambrian-1 nhận thấy rằng việc kết hợp DINOv2 + SigLIP vượt trội hơn so với việc sử dụng riêng lẻ trên các benchmark tập trung vào thị giác nhưng không thêm tín hiệu nào trên MMMU. Dự đoán benchmark nào sẽ tăng và benchmark nào sẽ giữ nguyên.

3. Mục tiêu của bạn là một tác nhân UI di động trên LLM 2B. Chọn encoder, connector, độ phân giải và data mix. Biện minh cho mỗi lựa chọn bằng một bảng ablation cụ thể.

4. Molmo phát hành các mô hình 4B và 72B. Mô hình 4B cạnh tranh với các VLM 7B đóng; mô hình 72B đánh bại Llama-3.2-90B-Vision trên 11/11 benchmark. Điều đó cho bạn biết gì về giả thuyết bão hòa kích thước LLM?

5. Thiết kế một bảng ablation để tách biệt chất lượng data-mix khỏi chất lượng encoder trên một VLM 7B. Cần tối thiểu bao nhiêu lượt huấn luyện? Đề xuất bốn thiết lập trục.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|------------------------|
| Ablation | "Xoay một nút" | Huấn luyện nhiều lượt khác nhau chính xác ở một trục không gian thiết kế, giữ mọi thứ khác không đổi |
| Connector | "Cầu nối" / "projector" | Mô-đun có thể huấn luyện ánh xạ đầu ra của vision encoder vào không gian token của LLM (MLP, Q-Former, Perceiver) |
| Detailed human caption | "Chú thích dày đặc" | Mô tả do con người viết gồm nhiều câu (thường là 80-300 token) phong phú hơn văn bản alt trên web |
| Distillation | "Chú thích GPT-4V" | Dữ liệu huấn luyện được tạo bởi một VLM độc quyền mạnh hơn; tiện lợi nhưng dễ bị ảo giác kế thừa |
| AnyRes / dynamic res | "Đường dẫn độ phân giải cao" | Chiến lược cung cấp hình ảnh lớn hơn độ phân giải gốc của encoder thông qua tiling hoặc M-RoPE |
| Resolution ramp | "Chương trình giảng dạy" | Lịch trình huấn luyện bắt đầu với độ phân giải thấp và tăng dần, giúp tăng tốc học căn chỉnh |
| Vision-centric bench | "CV-Bench / BLINK" | Đánh giá nhấn mạnh vào nhận thức thị giác chi tiết thay vì lý luận nặng về ngôn ngữ |
| PixMo | "Dữ liệu của Molmo" | Tập dữ liệu hình ảnh được chú thích dày đặc 712K của Allen AI; giọng nói con người được chuyển thành chú thích dày đặc |

## Đọc thêm

- [McKinzie et al. — MM1 (arXiv:2403.09611)](https://arxiv.org/abs/2403.09611)
- [Laurençon et al. — Idefics2 / What matters building VLMs (arXiv:2405.02246)](https://arxiv.org/abs/2405.02246)
- [Deitke et al. — Molmo and PixMo (arXiv:2409.17146)](https://arxiv.org/abs/2409.17146)
- [Tong et al. — Cambrian-1 (arXiv:2406.16860)](https://arxiv.org/abs/2406.16860)
- [Karamcheti et al. — Prismatic VLMs (arXiv:2402.07865)](https://arxiv.org/abs/2402.07865)