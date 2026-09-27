# Emu3: Dự đoán token tiếp theo cho tạo ảnh và video

> Emu3 của BAAI (Wang và cộng sự, tháng 9 năm 2024) là kết quả của năm 2024 đáng lẽ đã chấm dứt cuộc tranh luận giữa diffusion và autoregressive. Một Transformer decoder-only duy nhất theo phong cách Llama, chỉ được huấn luyện trên mục tiêu dự đoán token tiếp theo (next-token-prediction), trên một từ vựng thống nhất gồm văn bản + token ảnh VQ + token video 3D VQ, đã vượt qua SDXL trong việc tạo ảnh và LLaVA-1.6 trong việc nhận thức. Không có CLIP loss. Không có diffusion schedule. Classifier-free guidance được sử dụng khi suy luận để đảm bảo chất lượng, nhưng mục tiêu huấn luyện cốt lõi là dự đoán token tiếp theo với teacher forcing. Được xuất bản trên Nature. Bài học này đọc luận văn Emu3 — tại sao một tokenizer tốt hơn cộng với quy mô là tất cả những gì bạn cần — và đối chiếu với các phương pháp diffusion.

**Type:** Learn
**Languages:** Python (stdlib, toán học cho tokenizer video 3D + khung sampler autoregressive)
**Prerequisites:** Phase 12 · 11 (Chameleon)
**Time:** ~120 phút

## Mục tiêu học tập

- Giải thích tại sao mục tiêu dự đoán token tiếp theo với một loss duy nhất của Emu3 lại hiệu quả bất chấp giả định lâu nay rằng diffusion là bắt buộc để đạt chất lượng ảnh.
- Mô tả tokenizer video 3D: codebook VQ không gian-thời gian trông như thế nào, tại sao các patch lại trải dài theo thời gian.
- So sánh Emu3 với Stable Diffusion XL về (chi phí huấn luyện, chi phí suy luận, giới hạn chất lượng).
- Kể tên ba vai trò mà cùng một mô hình Emu3 đảm nhận: Emu3-Gen (tạo ảnh), Emu3-Chat (nhận thức), Emu3-Stage2 (tạo video).

## Vấn đề

Quan điểm thông thường trong suốt năm 2024: tạo ảnh cần diffusion. Lập luận: các token ảnh rời rạc làm mất quá nhiều thông tin để tái tạo chi tiết, và lấy mẫu autoregressive tích lũy sai số qua hàng nghìn token. Stable Diffusion, DALL-E 3, Imagen, Midjourney đều sử dụng một dạng diffusion nào đó. Chameleon (Bài học 12.11) đã bác bỏ một phần điều này ở quy mô nhỏ nhưng không đạt được chất lượng như SDXL.

Emu3 đã tấn công trực diện vào lập luận đó. Khẳng định: visual tokenizer tốt hơn + đủ quy mô + loss dự đoán token tiếp theo = tạo ảnh vượt trội hơn diffusion trong cùng một mô hình cũng thực hiện việc nhận thức.

Đặt cược này gây tranh cãi khi được công bố. Hai năm sau, dòng mô hình tạo nội dung thống nhất mã nguồn mở (Emu3, Show-o, Janus-Pro, Transfusion) là con đường mặc định cho nghiên cứu; các mô hình tiên phong trong sản xuất dường như đều sử dụng một biến thể nào đó.

## Khái niệm

### Tokenizer của Emu3

Thành phần chính là visual tokenizer. Emu3 huấn luyện một tokenizer tùy chỉnh thuộc lớp IBQ (Inverse Bottleneck Quantizer, họ SBER-MoVQGAN) với độ giảm phân giải 8x8 trên mỗi token. Một ảnh 512x512 trở thành 64x64 = 4096 token với kích thước codebook là 32768.

Con số này lớn hơn 1024 token trên mỗi 512x512 với K=8192 của Chameleon nhưng rẻ hơn trên mỗi token (tra cứu codebook nhỏ hơn, codec đơn giản hơn). Chỉ số chính: PSNR tái tạo ở mức 30.5 dB, cạnh tranh với không gian latent liên tục của Stable Diffusion ở mức 32 dB.

Đối với video: một tokenizer VQ 3D mã hóa một patch không gian-thời gian (4x4x4 pixel) thành một số nguyên. Một clip 4 giây ở 8 FPS có 32 khung hình; ở 256x256 với giảm không gian 4x và giảm thời gian 4x, số lượng token là (256/4) * (256/4) * (32/4) = 64 * 64 * 8 = 32,768 token.

Chất lượng tokenizer là giới hạn trần. Đóng góp của Emu3 một phần là "chúng tôi đã huấn luyện một tokenizer rất tốt".

### Huấn luyện với loss duy nhất

Emu3 sử dụng một mục tiêu duy nhất: dự đoán token tiếp theo trên một từ vựng chia sẻ giữa các token văn bản, token ảnh 2D và token video 3D. Trọng số được nhân với các hệ số đặc thù của từng phương thức trong quá trình huấn luyện để cân bằng đóng góp, nhưng hàm loss là giống hệt nhau.

Huấn luyện trên sự kết hợp của:
- Tạo ảnh: `<text caption> <image> image_tokens </image>`
- Nhận thức ảnh: `<image> image_tokens </image> <question> text_tokens`
- Tạo video: `<text caption> <video> video_tokens </video>`
- Nhận thức video: tương tự.
- Chỉ văn bản: NTP tiêu chuẩn.

Mô hình học cách khi nào cần phát ra token ảnh so với token văn bản từ phân phối dữ liệu. Việc tạo nội dung nảy sinh từ việc mô hình dự đoán các token ảnh sau thẻ `<image>`.

### Classifier-free guidance và temperature

Tạo ảnh autoregressive trở nên tốt hơn nhiều với classifier-free guidance (CFG) khi suy luận. Emu3 sử dụng nó: tạo hai lần, một lần với caption đầy đủ, một lần với caption trống, trộn các logit với trọng số guidance (thông thường 3.0-7.0). Đây là thủ thuật CFG tương tự mà diffusion sử dụng, được mượn sang bối cảnh autoregressive.

Temperature rất quan trọng: quá cao, gây nhiễu; quá thấp, gây mode collapse. Temperature khuyến nghị của Emu3 là 1.0 cho nhận thức, 0.8 cho tạo ảnh.

### Ba vai trò, một mô hình

Emu3 được phát hành dưới dạng ba API khác biệt về chức năng nhưng cùng một bộ trọng số cơ bản:

- Emu3-Gen. Tạo ảnh. Đầu vào là văn bản, đầu ra là các token ảnh.
- Emu3-Chat. VQA và chú thích ảnh. Đầu vào là ảnh (token), đầu ra là văn bản.
- Emu3-Stage2. Tạo video và video VQA. Đầu vào là văn bản hoặc video, đầu ra là văn bản hoặc video.

Không có các head chuyên biệt cho từng tác vụ. Chỉ là các prompt template khác nhau. Cùng một checkpoint.

### Benchmarks

Từ bài báo Emu3 (tháng 9 năm 2024):

- Tạo ảnh: vượt SDXL trên MJHQ-30K FID (5.4 so với 5.6), GenEval tổng thể (0.54 so với 0.55 — ngang bằng về mặt thống kê), và Deep-Eval composite ngang hàng.
- Nhận thức ảnh: vượt LLaVA-1.6 trên VQAv2 (75.1 so với 72.4) và xấp xỉ trên MMMU.
- Tạo video: chất lượng clip 4 giây có FVD cạnh tranh với các mô hình được benchmark công khai thời Sora.

Các con số không phải lúc nào cũng thắng — Emu3 đánh đổi một điểm ở chỗ này để lấy một điểm ở chỗ khác — nhưng khẳng định "dự đoán token tiếp theo là tất cả những gì bạn cần" là có thể bảo vệ được trên các phương thức.

### Chi phí tính toán

Emu3 được huấn luyện trên khoảng 300 tỷ token đa phương thức với mô hình 7B tham số. Số giờ GPU tương đương với việc huấn luyện trước Llama-2-7B (2k-4k GPU-năm trên phần cứng lớp A100). Các mô hình diffusion như Stable Diffusion 3 huấn luyện với ngân sách tương tự nhưng cần các bộ mã hóa văn bản riêng biệt và các pipeline phức tạp hơn.

Khi suy luận, Emu3 chậm hơn SDXL trên mỗi ảnh: 4096 token ảnh ở tốc độ 30 tok/s là khoảng 2 phút cho mỗi ảnh 512x512, so với 2-5 giây cho SDXL. Speculative decoding và tối ưu hóa KV-cache thu hẹp khoảng cách nhưng không xóa bỏ hoàn toàn. Tạo ảnh autoregressive rất nặng về tính toán; đây là sự đánh đổi hiện tại.

### Tại sao nó quan trọng

Đóng góp sâu sắc của Emu3 mang tính khái niệm. Nếu dự đoán token tiếp theo có thể mở rộng để sánh ngang với diffusion trong việc tạo ảnh, thì con đường mô hình thống nhất (một loss, một backbone, bất kỳ phương thức nào) là khả thi. Các mô hình tương lai không cần bộ mã hóa văn bản riêng, bộ lập lịch diffusion riêng, VAE riêng. Một Transformer, một tokenizer cho mỗi phương thức, quy mô.

Show-o, Janus-Pro và InternVL-U đều xây dựng dựa trên hoặc thách thức luận điểm này. Các phòng thí nghiệm Trung Quốc (BAAI, DeepSeek) xuất bản mạnh mẽ hơn theo hướng này so với các phòng thí nghiệm Mỹ trong suốt năm 2025.

```figure
l5-emu3-next-token
```

## Sử dụng

`code/main.py` xây dựng hai công cụ mô phỏng:

- Bộ tính toán số lượng token VQ 2D so với 3D: với (độ phân giải, patch, độ dài clip, FPS), tính toán số lượng token cho ảnh so với video.
- Bộ lấy mẫu token ảnh autoregressive với classifier-free guidance theo temperature.

Việc triển khai CFG khớp với công thức của Emu3 — trộn các logit có điều kiện và không điều kiện với trọng số guidance.

## Triển khai

Bài học này tạo ra `outputs/skill-token-gen-cost-analyzer.md`. Với một thông số kỹ thuật sản phẩm tạo nội dung (ảnh hoặc video, độ phân giải mục tiêu, cấp độ chất lượng, ngân sách độ trễ), nó tính toán số lượng token, chi phí suy luận và chọn giữa dòng Emu3 so với diffusion.

## Bài tập

1. Emu3 tạo ra 4096 token cho mỗi ảnh 512x512 ở mức giảm 8x8. Hãy tính toán con số tương đương cho 1024x1024 và 2048x2048. Điều gì xảy ra với độ trễ suy luận?

2. Đọc Emu3 Mục 3.3 về tokenizer video. Mô tả hình dạng patch VQ 3D và tại sao nó là 4x4x4 chứ không phải 8x8x1.

3. Trọng số classifier-free guidance 5.0 so với 3.0: hiệu ứng hình ảnh là gì? Hãy truy vết toán học trong `code/main.py`.

4. Tính toán FLOPs huấn luyện cho Emu3-7B ở 300B token và so sánh với Stable Diffusion 3. Mô hình nào đắt hơn để huấn luyện?

5. Emu3 vượt SDXL trên FID nhưng không vượt trên VQAv2 so với các VLM chuyên biệt. Giải thích tại sao phương pháp loss thống nhất lại cho thấy những điểm mạnh khác biệt so với các chuyên gia trên các benchmark khác nhau.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|------------------------|
| Next-token prediction | "NTP" | Loss autoregressive tiêu chuẩn: dự đoán token[i+1] dựa trên token[0..i]; hoạt động cho mọi phương thức khi đã được token hóa |
| IBQ tokenizer | "Inverse bottleneck quantizer" | Một lớp VQ-VAE với codebook lớn hơn (32768+) và khả năng tái tạo tốt hơn của Chameleon |
| 3D VQ | "Spatiotemporal quantizer" | Codebook được đánh chỉ mục bởi (thời gian, hàng, cột); một token bao phủ một khối pixel 4x4x4 |
| Classifier-free guidance | "CFG" | Trộn logit có điều kiện và không điều kiện với trọng số gamma; tăng chất lượng ảnh khi suy luận |
| Unified vocabulary | "Shared tokens" | Văn bản + ảnh + video đều lấy từ cùng một không gian số nguyên; mô hình dự đoán phương thức nào xuất hiện tiếp theo |
| MJHQ-30K | "Image gen benchmark" | Benchmark chất lượng Midjourney với 30k prompt; Emu3 báo cáo FID ở đây |

## Đọc thêm

- [Wang và cộng sự — Emu3: Next-Token Prediction is All You Need (arXiv:2409.18869)](https://arxiv.org/abs/2409.18869)
- [Sun và cộng sự — Emu: Generative Pretraining in Multimodality (arXiv:2307.05222)](https://arxiv.org/abs/2307.05222)
- [Liu và cộng sự — LWM (arXiv:2402.08268)](https://arxiv.org/abs/2402.08268)
- [Yu và cộng sự — MAGVIT-v2 (arXiv:2310.05737)](https://arxiv.org/abs/2310.05737)
- [Tian và cộng sự — VAR (arXiv:2404.02905)](https://arxiv.org/abs/2404.02905)