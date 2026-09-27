# LLaVA và Visual Instruction Tuning

> LLaVA (tháng 4 năm 2023) là kiến trúc đa phương thức (multimodal) được sao chép nhiều nhất trên thế giới. Nó thay thế Q-Former của BLIP-2 bằng một MLP 2 lớp, thay thế cơ chế gated cross-attention của Flamingo bằng việc nối token (token concatenation) đơn giản, và được huấn luyện trên 158 nghìn lượt hội thoại visual-instruction do GPT-4 tạo ra từ các chú thích văn bản. Bất kỳ kỹ sư nào xây dựng VLM trong giai đoạn 2023-2026 đều đã tạo ra một biến thể nào đó của LLaVA. LLaVA-1.5 bổ sung AnyRes. LLaVA-NeXT tăng độ phân giải. LLaVA-OneVision hợp nhất hình ảnh, đa hình ảnh và video vào một công thức duy nhất. Bài học này sẽ đọc công thức đó, triển khai projector và giải thích lý do tại sao "sự đơn giản đã chiến thắng".

**Type:** Build
**Languages:** Python (stdlib, projector + instruction-template builder)
**Prerequisites:** Phase 12 · 02 (CLIP), Phase 11 (LLM Engineering — instruction tuning)
**Time:** ~180 phút

## Mục tiêu học tập

- Xây dựng một projector MLP 2 lớp để ánh xạ các patch embedding của ViT (dim 1024) sang embedding dim của LLM (dim 4096).
- Đi qua công thức hai giai đoạn của LLaVA: (1) căn chỉnh projector trên 558 nghìn cặp chú thích, (2) visual instruction tuning trên 158 nghìn lượt hội thoại do GPT-4 tạo ra.
- Xây dựng prompt theo định dạng LLaVA với placeholder cho token hình ảnh, system prompt và các lượt hội thoại giữa người dùng/trợ lý.
- Giải thích lý do cộng đồng chuyển từ Q-Former sang MLP mặc dù Q-Former có lợi thế về ngân sách token.

## Vấn đề

Q-Former của BLIP-2 (Bài 12.03) nén một hình ảnh thành 32 token. Gọn gàng, hiệu quả, tốt cho các benchmark. Nhưng nó có hai vấn đề.

Thứ nhất, Q-Former có thể huấn luyện được nhưng hàm mất mát (loss) của nó không phải là tác vụ cuối cùng. Giai đoạn 1 huấn luyện ITC+ITM+ITG. Giai đoạn 2 huấn luyện LM loss. Các truy vấn (queries) học một biểu diễn trung gian mà sau đó LLM phải giải mã. Thông tin bị mất trong nút thắt cổ chai (bottleneck).

Thứ hai, Q-Former chiếm 188 triệu tham số, và ở quy mô năm 2023 của LLaVA, bạn phải đồng thiết kế nó với LLM mục tiêu. Thay đổi LLM, phải huấn luyện lại Q-Former. Thay đổi bộ mã hóa hình ảnh (vision encoder), phải huấn luyện lại. Mỗi sự kết hợp là một dự án R&D riêng biệt.

Câu trả lời của LLaVA gây ngạc nhiên vì sự đơn giản của nó: lấy 576 patch token của ViT, truyền từng token qua một MLP 2 lớp (`1024 → 4096 → 4096`), và đưa toàn bộ 576 token đó vào chuỗi đầu vào của LLM. Không có nút thắt cổ chai. Không có giai đoạn 1 tiền huấn luyện với các mục tiêu kỳ lạ. Chỉ cần huấn luyện MLP trên LM loss trực tiếp.

Dữ liệu lấy từ đâu? Thông tin chi tiết thứ hai của LLaVA: sử dụng GPT-4 (chỉ văn bản) để tạo dữ liệu hướng dẫn. Cung cấp cho GPT-4 dữ liệu chú thích COCO và bounding-box cho một hình ảnh, yêu cầu nó tạo ra các cuộc hội thoại, mô tả và câu hỏi suy luận phức tạp. 158 nghìn lượt hội thoại hướng dẫn-phản hồi miễn phí. Không cần gắn nhãn thủ công.

Kết quả: một VLM chạy trên 8 card A100 trong một ngày, đánh bại Flamingo trên MMMU và phát hành một checkpoint mở mà cộng đồng có thể mở rộng. Đến cuối năm 2023, nó đã tạo ra hơn 50 bản fork.

## Khái niệm

### Kiến trúc

LLaVA-1.5 ở quy mô 13B:
- Vision encoder: CLIP ViT-L/14 @ 336 (đóng băng trong giai đoạn 1, có thể mở đóng băng trong giai đoạn 2).
- Projector: MLP 2 lớp với kích hoạt GELU, `1024 → 4096 → 4096`.
- LLM: Vicuna-13B (sau này là Llama-3.1-8B).

Forward pass trên một hình ảnh + prompt văn bản:

```
img -> ViT -> 576 patches of dim 1024
patches -> MLP -> 576 tokens of dim 4096
prompt: system + "<image>" placeholder + user question
replace <image> token with the 576 projected tokens
feed the full sequence to the LLM
decode response
```

Hình ảnh chiếm 576 token trong ngữ cảnh của LLM. Với ngữ cảnh 2048, còn lại 1472 token cho văn bản. Với ngữ cảnh 32k, đây chỉ là sai số làm tròn.

### Giai đoạn 1: căn chỉnh projector

Đóng băng ViT. Đóng băng LLM. Chỉ huấn luyện MLP 2 lớp. Tập dữ liệu: 558 nghìn cặp hình ảnh-chú thích (LAION-CC-SBU). Loss: language modeling trên chú thích, có điều kiện là các token hình ảnh đã được chiếu (projected).

Trong một epoch với batch size 128, việc này hoàn thành trong vài giờ. Projector học cách ánh xạ không gian ViT sang không gian LLM. Không cần giám sát theo tác vụ cụ thể.

### Giai đoạn 2: visual instruction tuning

Mở đóng băng projector (vẫn có thể huấn luyện). Mở đóng băng LLM (thường là toàn bộ, đôi khi dùng LoRA). Huấn luyện trên 158 nghìn lượt hội thoại visual-instruction.

Dữ liệu hướng dẫn là chìa khóa. Liu và cộng sự đã tạo ra nó bằng cách:
1. Lấy một hình ảnh COCO.
2. Trích xuất mô tả văn bản (5 chú thích của con người + danh sách bounding-box).
3. Gửi tới GPT-4 với ba mẫu prompt:
   - Hội thoại: "Tạo một cuộc đối thoại qua lại giữa người dùng và trợ lý về hình ảnh này."
   - Mô tả chi tiết: "Đưa ra một mô tả phong phú, chi tiết về hình ảnh."
   - Suy luận phức tạp: "Đặt một câu hỏi đòi hỏi suy luận về hình ảnh, sau đó trả lời nó."
4. Phân tích đầu ra của GPT-4 thành các cặp (hướng dẫn, phản hồi).

Không có bước nào trong số này tác động trực tiếp đến hình ảnh — chỉ là mô tả văn bản. GPT-4 "ảo giác" ra nội dung hình ảnh hợp lý. Có một chút nhiễu, nhưng nó hiệu quả: 158 nghìn lượt hội thoại là đủ để mở khóa khả năng đối thoại.

### Tại sao cộng đồng sao chép cách này

- Không cần các hàm loss cụ thể cho giai đoạn 1 để tinh chỉnh. Chỉ dùng LM loss xuyên suốt.
- Projector huấn luyện trong vài giờ, không phải vài ngày.
- LLM có thể thay thế (LLaVA-Llama2, LLaVA-Mistral, LLaVA-Llama3) bằng cách chỉ huấn luyện lại projector.
- Pipeline dữ liệu visual-instruction sử dụng GPT-4 và chi phí thấp để tạo lại cho một miền mới.

### LLaVA-1.5 và LLaVA-NeXT

LLaVA-1.5 (tháng 10 năm 2023) bổ sung:
- Dữ liệu tác vụ học thuật (VQA, OKVQA, RefCOCO) được trộn vào quá trình instruction tuning.
- System prompt tốt hơn.
- Ngữ cảnh từ 2048 lên 32k.

LLaVA-NeXT (tháng 1 năm 2024) bổ sung:
- AnyRes: chia hình ảnh độ phân giải cao thành lưới 2x2 hoặc 1x3 các crop 336x336, cộng với một hình thu nhỏ (thumbnail) độ phân giải thấp toàn cục. Mỗi crop trở thành 576 token; tổng cộng khoảng 2880 token hình ảnh mỗi ảnh. Các tác vụ OCR và biểu đồ được cải thiện đáng kể.
- Hỗn hợp dữ liệu hướng dẫn tốt hơn với ShareGPT4V (các chú thích GPT-4V chất lượng cao).
- Các LLM nền tảng mạnh hơn (Mistral-7B, Yi-34B).

### LLaVA-OneVision

Bài 12.08 đề cập sâu về OneVision. Tóm tắt: cùng một projector, nhưng được huấn luyện với một chương trình bao gồm đơn hình ảnh, đa hình ảnh và video trong một mô hình với ngân sách token hình ảnh chia sẻ.

### So sánh với Q-Former

| | Q-Former (BLIP-2) | MLP (LLaVA) |
|---|---|---|
| Token hình ảnh mỗi ảnh | 32 | 576 (cơ bản) hoặc 2880 (AnyRes) |
| Tham số huấn luyện | 188M + LM | 40M + LM |
| Loss giai đoạn 1 | ITC+ITM+ITG | Chỉ LM |
| Thay thế LLM | Cần huấn luyện lại | Thay thế với huấn luyện lại tối thiểu |
| Đa hình ảnh | Khó khăn | Tự nhiên (nối chuỗi) |
| Video | Khó khăn | Tự nhiên (nối chuỗi theo khung hình) |
| Ngân sách token | Nhỏ | Lớn |

MLP thắng về sự đơn giản và tính linh hoạt của token. Q-Former thắng về ngân sách token. Đến cuối năm 2023, ngân sách token không còn là ràng buộc chính (ngữ cảnh LLM tăng lên 32k-128k+) và sự đơn giản chiếm ưu thế.

### Định dạng prompt

```
A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image> Describe this image in detail. ASSISTANT: The image shows ...
```

`<image>` là một token placeholder. Trước khi token hóa, nó được thay thế bằng 576 token hình ảnh (hoặc 2880 với AnyRes). Tokenizer thấy một chuỗi dài hơn một chút so với chuỗi mà nó được huấn luyện, nhưng LLM xử lý được đầu vào mới lạ này vì giai đoạn 1 đã dạy nó cách làm.

### Kinh tế tham số

Phân tích LLaVA-1.5-7B:
- CLIP ViT-L/14 @ 336: 303M (đóng băng giai đoạn 1, thường mở đóng băng giai đoạn 2).
- Projector (2x linear): ~22M có thể huấn luyện.
- Llama-7B: 7B.
- Tổng cộng: 7.3B tham số. Có thể huấn luyện trong giai đoạn 2: toàn bộ 7B + 22M projector.

Chi phí huấn luyện cho giai đoạn 2: ~20 giờ trên 8xA100. Đây là con số then chốt — một ngày, một node, có thể tái lập. Đó là lý do tại sao LLaVA lan rộng.

```figure
mm-llava-projector
```

## Sử dụng

`code/main.py` triển khai:

1. Projector MLP 2 lớp (dim 16 → 32 → 32 cho quy mô thử nghiệm) bằng Python thuần.
2. Pipeline xây dựng prompt: system prompt + `<image>` được thay thế bằng N token đã chiếu + lượt người dùng + placeholder cho phản hồi của trợ lý.
3. Trình trực quan hóa khối hình ảnh 576 token trông như thế nào trong ngữ cảnh LLM (phần trăm ngữ cảnh 2k / 32k / 128k bị tiêu thụ).

## Triển khai

Bài học này tạo ra `outputs/skill-llava-vibes-eval.md`. Với một checkpoint thuộc dòng LLaVA, nó chạy bộ đánh giá 10-prompt (3 chú thích, 3 VQA, 2 suy luận, 2 từ chối) và báo cáo bảng điểm dễ đọc. Không phải là một benchmark; mà là một bài kiểm tra nhanh (smoke test) để xác nhận projector và LLM đang kết nối tốt.

## Bài tập

1. Tính số lượng tham số có thể huấn luyện cho projector MLP 2 lớp tại `1024 → 4096 → 4096`. Với GELU và bias, nó chiếm bao nhiêu phần trăm của LLaVA-13B?

2. Xây dựng một prompt LLaVA cho trường hợp "từ chối" — hình ảnh chứa một cá nhân riêng tư. Viết phản hồi mong đợi của trợ lý. Tại sao LLaVA nên từ chối trường hợp zero-shot này và cần dữ liệu huấn luyện nào để củng cố việc từ chối?

3. Đọc phần AnyRes của blog LLaVA-NeXT. Tính số lượng token hình ảnh cho một hình ảnh 1344x672 tại AnyRes. So sánh với 576 token cơ bản tại 336x336.

4. Projector giai đoạn 1 của LLaVA được huấn luyện với LM loss trên các chú thích. Điều gì xảy ra nếu bạn bỏ qua giai đoạn 1 và đi thẳng đến giai đoạn 2 (visual instruction tuning)? Trích dẫn bài báo Prismatic VLMs (arXiv:2402.07865) để có câu trả lời.

5. LLaVA-Instruct-150k sử dụng GPT-4 với các chú thích COCO để tạo hướng dẫn. Đối với một miền mới (chụp X-quang y tế, hình ảnh vệ tinh), hãy mô tả pipeline dữ liệu bốn bước để tạo hướng dẫn cho miền đó. Điều gì có thể sai ở mỗi bước?

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|------------------------|
| Projector | "Cầu nối MLP" | MLP 2 lớp với GELU ánh xạ dim ViT sang dim LLM |
| Image token | "Placeholder <image>" | Đánh dấu trong prompt được thay thế bằng N token hình ảnh trước khi suy luận |
| Visual instruction tuning | "LLaVA giai đoạn 2" | Huấn luyện trên các bộ ba (hình ảnh, hướng dẫn, phản hồi) do GPT-4 tạo ra |
| Căn chỉnh giai đoạn 1 | "Tiền huấn luyện projector" | Đóng băng ViT và LLM, huấn luyện projector với LM loss trên chú thích |
| AnyRes | "Chia lưới đa crop" | Chia hình ảnh độ phân giải cao thành lưới các ô và nối token hình ảnh của mỗi ô |
| LLaVA-Instruct | "Do GPT-4 tạo ra" | 158 nghìn cặp hướng dẫn-phản hồi được tổng hợp từ chú thích COCO + GPT-4 |
| Đóng băng vision encoder | "Khóa backbone" | Trọng số CLIP không cập nhật trong giai đoạn 1, đôi khi không cập nhật cả giai đoạn 2 |
| ShareGPT4V | "Chú thích tốt hơn" | 1 triệu chú thích dày đặc do GPT-4V tạo ra, dùng cho căn chỉnh chất lượng cao hơn |
| VQA | "Visual question answering" | Tác vụ trả lời câu hỏi tự do về một hình ảnh |
| Prismatic VLMs | "Bài báo về không gian thiết kế" | Bài báo của Karamcheti 2024 thử nghiệm hệ thống các lựa chọn projector và dữ liệu |

## Đọc thêm

- [Liu et al. — Visual Instruction Tuning (arXiv:2304.08485)](https://arxiv.org/abs/2304.08485) — bài báo gốc về LLaVA.
- [Liu et al. — Improved Baselines with Visual Instruction Tuning (arXiv:2310.03744)](https://arxiv.org/abs/2310.03744) — LLaVA-1.5.
- [Chen et al. — ShareGPT4V (arXiv:2311.12793)](https://arxiv.org/abs/2311.12793) — tập dữ liệu chú thích dày đặc.
- [Karamcheti et al. — Prismatic VLMs (arXiv:2402.07865)](https://arxiv.org/abs/2402.07865) — các thử nghiệm về không gian thiết kế.
- [Li et al. — LLaVA-OneVision (arXiv:2408.03326)](https://arxiv.org/abs/2408.03326) — hợp nhất đơn hình ảnh, đa hình ảnh, video.