# Janus-Pro: Decoupled Encoders cho các mô hình đa phương thức hợp nhất

> Các mô hình đa phương thức hợp nhất (unified multimodal models) luôn tồn tại một sự căng thẳng không thể tránh khỏi. Hiểu (understanding) cần các đặc trưng ngữ nghĩa — các vector đầu ra từ SigLIP hoặc DINOv2 chứa đựng thông tin ở cấp độ khái niệm. Tạo (generation) cần các mã (codes) thân thiện với việc tái tạo — các token VQ có thể ghép lại thành các pixel sắc nét. Hai mục tiêu này không tương thích trong cùng một encoder. Janus (DeepSeek, tháng 10 năm 2024) và Janus-Pro (DeepSeek, tháng 1 năm 2025) lập luận rằng giải pháp là ngừng cố gắng kết hợp chúng: hãy tách rời hai encoder. Chia sẻ phần thân transformer giữa các tác vụ, nhưng định tuyến (route) việc hiểu thông qua SigLIP và việc tạo thông qua một VQ tokenizer. Ở quy mô 7B, Janus-Pro vượt qua DALL-E 3 trên GenEval trong khi vẫn ngang bằng với LLaVA trên MMMU. Bài học này giải thích lý do tại sao hai encoder hoạt động hiệu quả trong khi một encoder lại thất bại.

**Type:** Build
**Languages:** Python (stdlib, dual-encoder routing + shared-body signal)
**Prerequisites:** Phase 12 · 13 (Transfusion), Phase 12 · 14 (Show-o)
**Time:** ~120 phút

## Mục tiêu học tập

- Giải thích lý do tại sao một encoder dùng chung làm giảm chất lượng của việc hiểu hoặc việc tạo.
- Mô tả cơ chế định tuyến của Janus-Pro: các đặc trưng SigLIP ở phía đầu vào cho việc hiểu, các token VQ ở cả đầu vào và đầu ra cho việc tạo.
- Theo dõi quá trình mở rộng (scaling) hỗn hợp dữ liệu giúp Janus-Pro thành công ở nơi Janus đã thất bại.
- So sánh các kiến trúc tách rời (Janus-Pro), liên tục-kết hợp (Transfusion) và rời rạc-kết hợp (Show-o).

## Vấn đề

Các mô hình hợp nhất chia sẻ một phần thân transformer cho cả việc hiểu và tạo. Các nỗ lực trước đây (Chameleon, Show-o, Transfusion) đều sử dụng một visual tokenizer cho cả hai hướng. Tokenizer này là một sự thỏa hiệp:

- Tối ưu hóa cho việc tái tạo (tạo): VQ-VAE nắm bắt chi tiết pixel tinh vi nhưng tạo ra các token có sự gắn kết ngữ nghĩa yếu.
- Tối ưu hóa cho ngữ nghĩa (hiểu): Các embedding SigLIP nhóm các hình ảnh "con mèo" gần với các token "con mèo" nhưng không cho phép tái tạo tốt.

Show-o và Transfusion phải trả giá bằng việc giảm chất lượng ở một trong hai hướng. Janus-Pro đặt câu hỏi: tại sao phải yêu cầu một tokenizer khi các tác vụ có nhu cầu khác nhau?

## Khái niệm

### Mã hóa hình ảnh tách rời (Decoupled visual encoding)

Kiến trúc của Janus-Pro tách biệt hai encoder:

- Đường dẫn hiểu (Understanding path). Hình ảnh đầu vào → SigLIP-SO400m → 2-layer MLP → phần thân transformer.
- Đường dẫn tạo (Generation path). Hình ảnh đầu vào (nếu điều kiện hóa trên một hình ảnh có sẵn) → VQ tokenizer → token IDs → phần thân transformer.
- Tạo đầu ra (Output generation). Các token hình ảnh được dự đoán bởi transformer → VQ decoder → pixel.

Phần thân transformer được chia sẻ. Mọi thứ phía trước và phía sau phần thân đều dành riêng cho từng tác vụ.

Các đầu vào được phân biệt bằng định dạng prompt: thẻ `<understand>` định tuyến qua SigLIP; `<generate>` định tuyến qua VQ. Hoặc việc định tuyến được thực hiện ngầm định từ tác vụ.

### Tại sao cách này hiệu quả

Loss của việc hiểu nhận được các đặc trưng SigLIP, vốn đã được tinh chỉnh thông qua tiền huấn luyện kiểu CLIP để đạt được sự tương đồng về ngữ nghĩa. Các benchmark về nhận thức của mô hình cải thiện so với Show-o / Transfusion vì các đặc trưng đầu vào phù hợp hơn với tác vụ.

Loss của việc tạo nhận được các token VQ, vốn đã được tinh chỉnh bởi tokenizer để tái tạo. Chất lượng hình ảnh cải thiện so với Show-o vì các mã VQ có thể ghép lại thành pixel một cách sạch sẽ.

Phần thân transformer dùng chung nhìn thấy hai phân phối đầu vào (SigLIP và VQ) và học cách làm việc với cả hai. Khẳng định ở đây là: đủ dữ liệu + đủ tham số, phần thân sẽ hấp thụ được sự chuyển đổi này.

### Mở rộng dữ liệu — Janus vs Janus-Pro

Janus (bản gốc, arXiv 2410.13848) đã giới thiệu sự tách rời nhưng ở quy mô nhỏ (1.3B tham số, dữ liệu hạn chế). Janus-Pro (arXiv 2501.17811) đã mở rộng:

- 7B tham số (so với 1.3B).
- 90 triệu cặp hình ảnh-văn bản cho giai đoạn 1 (căn chỉnh), tăng từ 72 triệu.
- 72 triệu cho giai đoạn 2 (hợp nhất), tăng từ 26 triệu.
- Thêm 200k mẫu hướng dẫn tạo hình ảnh cho giai đoạn 3.

Kết quả: Janus-Pro-7B ngang bằng với LLaVA trên MMMU (60.3 so với ~58) và vượt qua DALL-E 3 trên GenEval (0.80 so với 0.67). Một mô hình mã nguồn mở, cạnh tranh trên cả hai khía cạnh của phổ hợp nhất.

### JanusFlow — biến thể rectified flow

JanusFlow (arXiv 2411.07975) thay thế đường dẫn tạo VQ bằng đường dẫn tạo rectified-flow (liên tục). Sự phân tách trở thành SigLIP-cho-hiểu + rectified-flow-cho-tạo. Trần chất lượng được nâng cao hơn nữa. Kiến trúc vẫn giữ nguyên là các encoder tách rời với phần thân dùng chung.

### Công việc của phần thân dùng chung

Phần thân transformer xử lý một chuỗi hợp nhất nhưng với hai phân phối đầu vào. Công việc của nó là:

- Đối với hiểu: tiêu thụ các đặc trưng SigLIP + token văn bản → phát ra văn bản một cách tự hồi quy (autoregressively).
- Đối với tạo: tiêu thụ các token văn bản + (các token VQ hình ảnh tùy chọn) → phát ra các token VQ hình ảnh một cách tự hồi quy.

Phần thân không có trọng số đặc thù cho từng phương thức (modality-specific) trên mỗi block. Nó là transformer kiểu văn bản mà bạn mong đợi tìm thấy bên trong Qwen hoặc Llama, cộng với hai bộ chuyển đổi đầu vào (input adapters).

Thú vị là, điều này có nghĩa là phần thân của Janus-Pro có thể được khởi tạo từ một LLM đã tiền huấn luyện. Janus-Pro thực sự khởi tạo từ DeepSeek-MoE-7B. Lựa chọn đó rất quan trọng: LLM đóng góp khả năng suy luận mà các mô hình hợp nhất thuần túy từ đầu khó có thể đạt được.

### So sánh với InternVL-U

InternVL-U (Bài học 12.10) là phiên bản kế nhiệm năm 2026. Nó kết hợp:

- Tiền huấn luyện đa phương thức gốc (backbone InternVL3).
- Định tuyến encoder tách rời (SigLIP vào, VQ + diffusion heads ra).
- Hợp nhất hiểu + tạo + chỉnh sửa.

InternVL-U bao hàm lựa chọn kiến trúc của Janus-Pro vào một khung lớn hơn. Ý tưởng encoder tách rời hiện là mặc định cho các mô hình hợp nhất ở quy mô lớn.

### Hạn chế

Các encoder tách rời làm tăng độ phức tạp của kiến trúc. Hai tokenizer cần huấn luyện, hai đường dẫn đầu vào cần duy trì, hai bộ chế độ lỗi. Đối với các sản phẩm không cần tạo, Janus-Pro là quá mức cần thiết — hãy chọn một mô hình hiểu thuộc dòng LLaVA.

Đối với các sản phẩm không cần hiểu, Janus-Pro là quá dư thừa — hãy chọn mô hình Stable Diffusion 3 / Flux.

Đối với các sản phẩm cần cả hai, Janus-Pro hiện là kiến trúc mở tham chiếu.

```figure
l5-janus-decouple
```

## Sử dụng

`code/main.py` mô phỏng việc định tuyến của Janus-Pro:

- Hai mock encoder: kiểu SigLIP (tạo ra các vector ngữ nghĩa 256-dim) và kiểu VQ (tạo ra các mã số nguyên).
- Một bộ định tuyến prompt chọn encoder dựa trên thẻ tác vụ.
- Một phần thân dùng chung (đóng vai trò thay thế) xử lý các chuỗi token bất kể encoder nào đã tạo ra chúng.
- Một sự chuyển đổi từ giai đoạn 1 (căn chỉnh) sang giai đoạn 3 (tinh chỉnh hướng dẫn) với lịch trình lấy mẫu có trọng số.

In ra các đường dẫn đã định tuyến cho 3 ví dụ: QA hình ảnh, T2I, chỉnh sửa hình ảnh.

## Triển khai

Bài học này tạo ra `outputs/skill-decoupled-encoder-picker.md`. Với một sản phẩm cần tạo + hiểu hợp nhất ở chất lượng tiệm cận biên, nó chọn Janus-Pro, JanusFlow hoặc InternVL-U với khuyến nghị cụ thể về quy mô dữ liệu.

## Bài tập

1. Janus-Pro-7B vượt qua DALL-E 3 trên GenEval. Giải thích tại sao một mô hình mở 7B có thể ngang bằng với một mô hình độc quyền tiên tiến về tạo nhưng không phải về hiểu.

2. Triển khai một hàm định tuyến: với văn bản prompt, hãy phân loại là `understand` hoặc `generate`. Bạn xử lý các prompt mơ hồ như "mô tả rồi phác thảo" như thế nào?

3. JanusFlow thay thế đường dẫn VQ bằng rectified flow. Phần thân transformer bây giờ xuất ra cái gì, và điều gì thay đổi trong loss?

4. Đề xuất tác vụ thứ tư mà kiến trúc Janus-Pro có thể xử lý với thêm một encoder tách rời. Ví dụ: phân đoạn hình ảnh (kiểu DINO), độ sâu (kiểu MiDaS).

5. Đọc Janus-Pro Mục 4.2 về mở rộng dữ liệu. Giai đoạn dữ liệu nào đóng góp nhiều nhất vào mức tăng chất lượng T2I so với Janus?

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|------------------------|
| Decoupled encoding | "Hai encoder hình ảnh" | Tokenizer hoặc encoder riêng biệt cho mỗi hướng: ngữ nghĩa cho hiểu, tái tạo cho tạo |
| Shared body | "Một transformer" | Transformer duy nhất xử lý đầu ra của bất kỳ encoder nào; không có trọng số đặc thù phương thức |
| SigLIP for understanding | "Đặc trưng ngữ nghĩa" | Vision tower dòng CLIP cung cấp các đặc trưng khái niệm phong phú nhưng tái tạo kém |
| VQ for generation | "Mã tái tạo" | Các token được lượng tử hóa vector giải mã sạch sẽ trở lại thành pixel |
| JanusFlow | "Biến thể rectified-flow" | Janus-Pro với đầu tạo flow-matching liên tục thay vì VQ |
| Routing tag | "Thẻ tác vụ" | Đánh dấu prompt (`<understand>` / `<generate>`) chọn encoder đầu vào |

## Đọc thêm

- [Wu et al. — Janus (arXiv:2410.13848)](https://arxiv.org/abs/2410.13848)
- [Chen et al. — Janus-Pro (arXiv:2501.17811)](https://arxiv.org/abs/2501.17811)
- [Ma et al. — JanusFlow (arXiv:2411.07975)](https://arxiv.org/abs/2411.07975)
- [InternVL-U (arXiv:2603.09877)](https://arxiv.org/abs/2603.09877)
- [Dong et al. — DreamLLM (arXiv:2309.11499)](https://arxiv.org/abs/2309.11499)