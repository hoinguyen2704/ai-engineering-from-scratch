# Chameleon và các mô hình đa phương thức Early-Fusion chỉ dùng Token

> Mọi VLM mà chúng ta từng thấy cho đến nay đều tách biệt hình ảnh và văn bản. Các token hình ảnh đến từ một vision encoder, đi qua một projector, sau đó mới gặp văn bản bên trong LLM. Từ vựng của hình ảnh và văn bản không bao giờ chồng lấp. Chameleon (Meta, tháng 5 năm 2024) đã đặt câu hỏi: điều gì sẽ xảy ra nếu chúng chồng lấp? Hãy huấn luyện một VQ-VAE để biến hình ảnh thành một chuỗi các token rời rạc từ một từ vựng dùng chung. Giờ đây, mỗi tài liệu đa phương thức là một chuỗi duy nhất — các token văn bản và token hình ảnh xen kẽ nhau, với một hàm mất mát (loss) tự hồi quy duy nhất. Tác dụng phụ: mô hình có thể tạo ra các đầu ra hỗn hợp đa phương thức — xen kẽ các token văn bản và hình ảnh trong một lần gọi inference. Bài học này sẽ đọc luận điểm về early-fusion và xây dựng một phiên bản mô hình đồ chơi từ đầu đến cuối.

**Type:** Build
**Languages:** Python (stdlib, VQ-VAE tokenizer + interleaved decoder)
**Prerequisites:** Phase 12 · 05, Phase 8 (Generative AI)
**Time:** ~180 phút

## Mục tiêu học tập

- Giải thích lý do tại sao từ vựng dùng chung + một hàm loss duy nhất lại thay đổi khả năng của mô hình.
- Mô tả cách VQ-VAE mã hóa hình ảnh thành một chuỗi rời rạc tương thích với mục tiêu dự đoán token tiếp theo của transformer.
- Nêu tên các thủ thuật ổn định huấn luyện của Chameleon: QK-Norm, vị trí dropout, thứ tự LayerNorm.
- So sánh Chameleon với phương pháp Q-Former của BLIP-2 và mô tả khi nào nên chọn phương pháp nào.

## Vấn đề

Các VLM dựa trên adapter (LLaVA, BLIP-2, Qwen-VL) coi văn bản và hình ảnh là hai thực thể khác biệt. Một token văn bản đi qua `embed(text_token)`; một hình ảnh đi qua `visual_encoder(image) → projector → ... pseudo_tokens`. Mô hình có hai đường dẫn đầu vào hợp nhất ở giữa chặng đường.

Ba hệ quả:

1. LLM chỉ có thể tiêu thụ hình ảnh, không thể phát ra chúng. Đầu ra chỉ là văn bản.
2. Các tài liệu đa phương thức (xen kẽ đoạn văn và hình ảnh, như trong một bài báo) trở nên khó xử lý — bạn phải phân tích đầu vào đa phương thức bên ngoài mô hình hoặc xâu chuỗi các lần tạo.
3. Sự lệch pha về phân phối. Các token hình ảnh và token văn bản tồn tại trong các vùng khác nhau của không gian ẩn (hidden space), tạo ra các vấn đề căn chỉnh tinh vi.

Chameleon bác bỏ tiền đề đó: hình ảnh chỉ là các chuỗi token rời rạc từ một từ vựng dùng chung. Huấn luyện mô hình trên các tài liệu xen kẽ, một hàm loss, một bộ giải mã (decoder) tự hồi quy, và bạn sẽ mở khóa khả năng tạo đa phương thức hỗn hợp một cách miễn phí.

## Khái niệm

### VQ-VAE làm tokenizer hình ảnh

Tokenizer là một vector-quantized variational autoencoder. Kiến trúc bao gồm:

- Encoder: CNN + ViT ánh xạ hình ảnh thành bản đồ đặc trưng không gian, ví dụ 32x32 đặc trưng với số chiều 256.
- Codebook: một từ vựng đã học gồm K vector (Chameleon sử dụng 8192), cũng có số chiều 256.
- Quantization: với mỗi đặc trưng không gian, tìm kiếm mục codebook gần nhất theo khoảng cách L2. Thay thế đặc trưng liên tục bằng chỉ số nguyên.
- Decoder: CNN chuyển đổi các đặc trưng đã lượng tử hóa trở lại thành pixel.

Huấn luyện: VAE reconstruction loss + commitment loss + codebook loss. Các chỉ số codebook tạo thành một bảng chữ cái rời rạc cho hình ảnh.

Đối với Chameleon: một hình ảnh trở thành 32*32 = 1024 token được lấy từ từ vựng 8192. Nối với các token văn bản (từ từ vựng BPE của LLM, ví dụ 32000). Từ vựng cuối cùng: 40192. Transformer nhìn thấy một chuỗi duy nhất, một hàm loss duy nhất.

### Từ vựng dùng chung

Từ vựng của Chameleon kết hợp các token văn bản, token hình ảnh và các dấu phân cách phương thức. Mỗi token có một ID duy nhất. Lớp nhúng đầu vào (input embedding) ánh xạ mọi ID thành một vector ẩn D-chiều. Phép chiếu đầu ra (output projection) ánh xạ vector ẩn trở lại thành các logit từ vựng. Softmax chọn token tiếp theo, bất kể phương thức nào.

Các dấu phân cách rất quan trọng: các thẻ `<image>` và `</image>` bao quanh chuỗi token hình ảnh. Tại thời điểm tạo, nếu mô hình phát ra `<image>`, phần mềm hạ nguồn sẽ biết 1024 token tiếp theo là các chỉ số VQ cần gửi đến decoder để hiển thị pixel.

### Tạo đa phương thức hỗn hợp

Inference là dự đoán token tiếp theo trong từ vựng dùng chung. Ví dụ prompt: "Vẽ một con mèo và mô tả nó." Chameleon phát ra:

```
<image> 4821 1029 2891 ... (1024 image tokens) </image>
The cat is orange, sitting on a windowsill...
```

Mô hình tự chọn thứ tự — nó có thể tạo hình ảnh rồi đến văn bản, văn bản rồi đến hình ảnh, hoặc xen kẽ. Cùng một decoder, cùng một hàm loss.

So sánh với các VLM adapter nơi việc tạo chỉ là văn bản. Chameleon mở lại câu hỏi về các phương thức đầu ra của mô hình.

### Ổn định huấn luyện — QK-Norm, dropout, thứ tự LayerNorm

Huấn luyện early-fusion không ổn định ở quy mô lớn. Bài báo của Chameleon ghi lại ba thủ thuật:

- QK-Norm. Áp dụng LayerNorm cho các phép chiếu query và key bên trong attention, trước khi thực hiện tích vô hướng. Ngăn chặn sự bùng nổ độ lớn logit theo chiều sâu. Được sử dụng bởi nhiều mô hình lớn sau năm 2024.
- Vị trí Dropout. Dropout sau mỗi phép cộng residual, không chỉ sau attention và MLP. Cần nhiều chính quy hóa hơn khi gradient từ các token hình ảnh có thể chiếm ưu thế.
- Thứ tự LayerNorm. Pre-LN trên nhánh residual (tiêu chuẩn), cộng thêm một LN bổ sung trên kết nối skip của khối cuối cùng. Ổn định luồng gradient của lớp cuối.

Nếu không có các thủ thuật này, quá trình huấn luyện Chameleon 34B-param đã bị phân kỳ tại nhiều điểm kiểm tra. Với chúng, nó hội tụ. Công thức huấn luyện cũng là một đóng góp quan trọng như chính kiến trúc.

### Giới hạn tái tạo của tokenizer

VQ-VAE là loại mất dữ liệu (lossy). Với 8192 mục codebook và 1024 token cho mỗi hình ảnh 512x512, PSNR tái tạo đạt mức tối đa khoảng 26-28 dB. Điều này đủ để tạo hình ảnh dễ nhận biết nhưng rõ ràng kém hơn so với diffusion không gian liên tục (Stable Diffusion 3 đạt trên 32 dB).

Tokenizer là nút thắt cổ chai. Các tokenizer tốt hơn (MAGVIT-v2, IBQ, SBER-MoVQGAN) sẽ nâng cao giới hạn này. Emu3 (Bài 12.12) đạt được chất lượng tạo hình ảnh ngang ngửa SDXL chỉ nhờ một tokenizer tốt hơn.

### Chameleon vs BLIP-2 / LLaVA

Chameleon (early fusion, từ vựng dùng chung):
- Một hàm loss, một decoder.
- Tạo đầu ra đa phương thức hỗn hợp.
- Tokenizer là giới hạn chất lượng.
- Đắt đỏ: cần VQ-VAE decoder cho mỗi hình ảnh được tạo trên đường dẫn inference.

BLIP-2 / LLaVA (late fusion, các tháp riêng biệt):
- Đầu vào hình ảnh, chỉ đầu ra văn bản.
- Tái sử dụng LLM đã huấn luyện trước.
- Không có nút thắt cổ chai tokenizer cho việc hiểu.
- Rẻ: một lần forward pass duy nhất.

Chọn theo tác vụ. Nếu bạn cần tạo hình ảnh, hãy chọn dòng Chameleon. Nếu bạn chỉ cần hiểu, adapter-VLM đơn giản hơn và tái sử dụng nhiều tài nguyên tính toán đã huấn luyện trước hơn.

### Fuyu và AnyGPT

Fuyu (Adept, 2023) là một phương pháp liên quan: bỏ qua hoàn toàn vision encoder riêng biệt, đưa các patch hình ảnh thô qua phép chiếu đầu vào của LLM như thể chúng là các token, không cần tokenizer. Đơn giản hơn Chameleon, nhưng mất khả năng tạo đầu ra từ vựng dùng chung.

AnyGPT (Zhan và cộng sự, 2024) mở rộng Chameleon sang bốn phương thức: văn bản, hình ảnh, giọng nói, âm nhạc. Sử dụng cùng thủ thuật VQ-VAE cho mỗi phương thức, dùng chung transformer. Tạo bất kỳ-đến-bất kỳ. Được đề cập kỹ hơn trong Bài 12.16.

```figure
vq-codebook
```

## Sử dụng

`code/main.py` xây dựng một mô hình early-fusion đồ chơi từ đầu đến cuối:

- Một bộ lượng tử hóa kiểu VQ-VAE nhỏ ánh xạ các patch 8x8 thành các chỉ số codebook (K=16).
- Một từ vựng dùng chung gồm (id văn bản 0..31) + (id hình ảnh 32..47) + (dấu phân cách 48, 49).
- Một decoder tự hồi quy đồ chơi (bảng bigram) được huấn luyện trên các chú thích tổng hợp + chuỗi token hình ảnh.
- Vòng lặp lấy mẫu phát ra các token văn bản + hình ảnh xen kẽ dựa trên một prompt.

Mã nguồn cố tình giữ transformer ở mức nhỏ (bigram) để bạn có thể theo dõi luồng tín hiệu từ đầu đến cuối.

## Triển khai

Bài học này tạo ra `outputs/skill-tokenizer-vs-adapter-picker.md`. Dựa trên đặc tả sản phẩm (chỉ hiểu vs hiểu + tạo, chất lượng hình ảnh yêu cầu, ngân sách chi phí), nó chọn giữa dòng Chameleon (early fusion) và dòng LLaVA (late fusion) và biện minh bằng các quy tắc ngón tay cái định lượng.

## Bài tập

1. Chameleon sử dụng 8192 mục codebook và 1024 token cho mỗi hình ảnh 512x512. Hãy ước tính tỷ lệ nén so với hình ảnh RGB 24-bit. Nó có bị mất dữ liệu không? Mất bao nhiêu?

2. Một hình ảnh 4K (3840x2160) ở cùng mật độ VQ-VAE tạo ra bao nhiêu token hình ảnh? Một mô hình kiểu Chameleon có thể tạo hình ảnh 4K trong một lần gọi inference không? Điều gì sẽ hỏng trước tiên — ngữ cảnh, chất lượng tokenizer, hay KV cache?

3. Triển khai QK-Norm bằng Python thuần. Với query và key 64-chiều, hãy hiển thị tích vô hướng trước và sau LayerNorm. Tại sao kiểm soát độ lớn lại quan trọng ở chiều sâu?

4. Đọc Chameleon Mục 2.3 về ổn định huấn luyện. Mô tả chế độ lỗi chính xác mà bài báo quan sát được ở 34B mà không có QK-Norm. "Dấu hiệu bùng nổ chuẩn" (norm explosion) là gì?

5. Mở rộng decoder đồ chơi để phát ra phản hồi đa phương thức hỗn hợp dựa trên prompt chỉ có văn bản. Đo lường tần suất mô hình chọn hình ảnh trước so với văn bản trước dựa trên phân phối dữ liệu huấn luyện 60% văn bản trước / 40% hình ảnh trước.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|------------------------|
| Early fusion | "Token thống nhất" | Hình ảnh được chuyển đổi thành các token rời rạc chia sẻ từ vựng của transformer ngay từ bước đầu |
| VQ-VAE | "Tokenizer hình ảnh" | CNN + ViT + codebook ánh xạ hình ảnh thành các chỉ số nguyên mà transformer có thể dự đoán |
| Từ vựng dùng chung | "Một từ điển" | Một không gian ID token duy nhất bao gồm văn bản + hình ảnh + dấu phân cách phương thức |
| QK-Norm | "Bộ ổn định attention" | LayerNorm áp dụng cho query và key trước khi tích vô hướng, ngăn chặn bùng nổ chuẩn |
| Tạo đa phương thức hỗn hợp | "Đầu ra văn bản + hình ảnh" | Inference tự động tạo ra các token văn bản và hình ảnh xen kẽ trong một lần truyền |
| Kích thước codebook | "K mục" | Số lượng vector rời rạc mà VQ-VAE có thể lượng tử hóa; đánh đổi giữa nén và độ trung thực |
| Giới hạn tokenizer | "Giới hạn tái tạo" | PSNR tốt nhất có thể đạt được bằng cách giải mã các token VQ; giới hạn chất lượng hình ảnh của mô hình |

## Đọc thêm

- [Chameleon Team — Chameleon: Mixed-Modal Early-Fusion Foundation Models (arXiv:2405.09818)](https://arxiv.org/abs/2405.09818)
- [Aghajanyan et al. — CM3 (arXiv:2201.07520)](https://arxiv.org/abs/2201.07520)
- [Yu et al. — CM3Leon (arXiv:2309.02591)](https://arxiv.org/abs/2309.02591)
- [Zhan et al. — AnyGPT (arXiv:2402.12226)](https://arxiv.org/abs/2402.12226)
- [Adept — Fuyu-8B blog (adept.ai)](https://www.adept.ai/blog/fuyu-8b)