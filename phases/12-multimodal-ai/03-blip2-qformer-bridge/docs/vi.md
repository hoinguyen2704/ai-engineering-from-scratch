# Từ CLIP đến BLIP-2 — Q-Former với vai trò cầu nối đa phương thức (Modality Bridge)

> CLIP căn chỉnh hình ảnh và văn bản nhưng không thể tạo chú thích, trả lời câu hỏi hoặc duy trì hội thoại. BLIP-2 (Salesforce, 2023) đã giải quyết vấn đề đó bằng một cầu nối nhỏ có thể huấn luyện: 32 vector truy vấn (query) có khả năng học, thực hiện cross-attention trên các đặc trưng của một ViT đã đóng băng (frozen), sau đó đưa trực tiếp vào luồng đầu vào của một LLM đã đóng băng. 188 triệu tham số của cầu nối này đã kết nối một LLM 11B với một ViT-g/14. Mọi VLM dựa trên adapter cho đến năm 2026 — MiniGPT-4, InstructBLIP, các biến thể của LLaVA — đều là hậu duệ của nó. Bài học này tìm hiểu kiến trúc của Q-Former, giải thích quá trình huấn luyện hai giai đoạn và xây dựng một phiên bản mô phỏng (toy version) đưa các token hình ảnh vào một bộ giải mã văn bản (text decoder) đã đóng băng.

**Type:** Build
**Languages:** Python (stdlib, cross-attention + learnable-query demo)
**Prerequisites:** Phase 12 · 02 (CLIP), Phase 7 (Transformers)
**Time:** ~180 phút

## Mục tiêu học tập

- Giải thích lý do tại sao một nút thắt (bottleneck) có thể huấn luyện giữa bộ mã hóa hình ảnh (vision encoder) đã đóng băng và LLM đã đóng băng lại vượt trội hơn so với việc tinh chỉnh toàn bộ (end-to-end finetuning) về chi phí và độ ổn định.
- Triển khai khối cross-attention nơi một tập hợp cố định các truy vấn có thể học thực hiện attention trên các đặc trưng hình ảnh bên ngoài.
- Tìm hiểu quá trình tiền huấn luyện hai giai đoạn của BLIP-2: biểu diễn (ITC + ITM + ITG) sau đó là tạo sinh (LM loss với bộ giải mã đã đóng băng).
- So sánh Q-Former với bộ chiếu (projector) MLP đơn giản hơn được sử dụng trong LLaVA và lập luận khi nào mỗi lựa chọn sẽ chiếm ưu thế.

## Vấn đề

Bạn có một ViT đã đóng băng tạo ra 256 patch token với số chiều 1408 cho mỗi hình ảnh. Bạn có một LLM 7B đã đóng băng yêu cầu các embedding token với số chiều 4096. Cầu nối hiển nhiên nhất — một lớp tuyến tính từ 1408 đến 4096 — hoạt động tốt, nhưng việc đưa tất cả 256 patch token vào ngữ cảnh của LLM sẽ tốn thêm 256 token cho mỗi hình ảnh. Với một batch gồm 32 hình ảnh, đó là 8192 token chỉ dành riêng cho phương thức hình ảnh.

Câu hỏi của BLIP-2: liệu bạn có thể nén biểu diễn hình ảnh 256-token thành ít token hơn nhiều (ví dụ: 32) trong khi vẫn bảo toàn đủ thông tin để LLM có thể chú thích, trả lời câu hỏi và suy luận về hình ảnh không? Và liệu bạn có thể huấn luyện cầu nối này mà không cần chạm vào các backbone đã đóng băng, giữ chi phí huấn luyện chỉ ở mức các tham số của cầu nối?

Câu trả lời: Q-Former. 32 vector "truy vấn" có thể học thực hiện cross-attention với các patch token của ViT, tạo ra một bản tóm tắt hình ảnh 32-token mà LLM sẽ tiêu thụ. Tổng cộng 188 triệu tham số. Được huấn luyện với các mục tiêu tương phản (contrastive), khớp (matching) và tạo sinh (generative) trước khi chạm vào LLM.

## Khái niệm

### Các truy vấn có thể học (Learnable queries)

Thủ thuật cốt lõi của Q-Former: thay vì để các token văn bản của LLM thực hiện attention trên các patch hình ảnh, hãy giới thiệu một tập hợp mới gồm 32 vector truy vấn có thể học `Q` và để *chúng* thực hiện attention trên các patch hình ảnh. Các truy vấn này là các tham số của mô hình — chúng được học trong quá trình huấn luyện và cùng một tập 32 truy vấn được sử dụng cho mọi hình ảnh.

Sau khi thực hiện cross-attention, mỗi truy vấn nắm giữ một bản tóm tắt nén của hình ảnh — "mô tả vật thể chính", "mô tả nền", "đếm các vật thể", v.v. Các truy vấn không thực sự chuyên biệt hóa vào các nhãn ngữ nghĩa; chúng học bất kỳ mã hóa nào giúp giảm thiểu các hàm mất mát (loss) ở hạ nguồn.

### Kiến trúc

Q-Former là một transformer nhỏ (12 lớp, ~100 triệu tham số) với hai đường dẫn:

1. Đường dẫn truy vấn (Query path): 32 vector truy vấn đi qua self-attention (giữa chúng với nhau), sau đó là cross-attention trên các patch token của ViT đã đóng băng, rồi đến FFN.
2. Đường dẫn văn bản (Text path): một bộ mã hóa văn bản kiểu BERT chia sẻ các trọng số self-attention và FFN với đường dẫn truy vấn. Cross-attention bị vô hiệu hóa đối với đường dẫn văn bản.

Tại thời điểm huấn luyện, cả hai đường dẫn đều chạy. Các truy vấn và văn bản tương tác thông qua self-attention chia sẻ, nghĩa là các truy vấn có thể điều kiện hóa trên văn bản cho các tác vụ cần thiết (ITM, ITG). Tại thời điểm suy luận (inference) để chuyển giao cho VLM, chỉ có các truy vấn đi qua, tạo ra 32 token hình ảnh.

### Huấn luyện hai giai đoạn

BLIP-2 tiền huấn luyện theo hai giai đoạn:

Giai đoạn 1: học biểu diễn (không có LLM). Ba hàm mất mát:
- ITC (image-text contrastive): tương phản kiểu CLIP giữa các query token đã gộp (pooled) và token CLS của văn bản.
- ITM (image-text matching): bộ phân loại nhị phân — cặp hình ảnh-văn bản này có khớp không? Sử dụng kỹ thuật hard-negative-mined.
- ITG (image-grounded text generation): đầu LM nhân quả trên văn bản, có điều kiện trên các truy vấn. Buộc các truy vấn phải mã hóa nội dung có thể tạo sinh từ văn bản.

Chỉ Q-Former được huấn luyện. ViT bị đóng băng. Không có LLM tham gia.

Giai đoạn 2: học tạo sinh. Gắn một LLM đã đóng băng (OPT-2.7B hoặc Flan-T5-XL, v.v.). Chiếu 32 đầu ra của truy vấn sang số chiều embedding của LLM thông qua một lớp tuyến tính nhỏ. Đặt chúng trước prompt văn bản. Chỉ huấn luyện lớp chiếu tuyến tính và Q-Former dựa trên LM loss trên chuỗi kết hợp prompt + hình ảnh + chú thích.

Sau giai đoạn 2, Q-Former + lớp chiếu là bộ adapter hình ảnh hoàn chỉnh. Tại thời điểm suy luận: hình ảnh → ViT → Q-Former → lớp chiếu tuyến tính → đặt trước văn bản → LLM đã đóng băng phát ra đầu ra.

### Kinh tế tham số

BLIP-2 với ViT-g/14 (1.1B, đóng băng) + OPT-6.7B (6.7B, đóng băng) + Q-Former (188M, đã huấn luyện) = tổng 8B, 188M được huấn luyện. Riêng Q-Former chiếm ~2.4% tham số của toàn bộ hệ thống. Chi phí huấn luyện phản ánh điều này: vài ngày trên một số ít A100 so với hàng tuần cho huấn luyện end-to-end.

Chất lượng: BLIP-2 ngang bằng hoặc vượt qua Flamingo-80B về VQA zero-shot trong khi nhỏ hơn 50 lần. Cầu nối này thực sự hiệu quả.

### InstructBLIP và Q-Former nhận thức hướng dẫn (instruction-aware)

InstructBLIP (2023) mở rộng Q-Former với một đầu vào bổ sung: chính văn bản hướng dẫn. Tại thời điểm cross-attention, các truy vấn giờ đây có quyền truy cập vào cả các patch hình ảnh và hướng dẫn. Các truy vấn có thể chuyên biệt hóa theo từng hướng dẫn ("đếm số xe", "mô tả tâm trạng") thay vì học một bản tóm tắt cố định duy nhất. Đạt được mức tăng điểm trên các tác vụ kiểm thử (held-out tasks).

### MiniGPT-4 và cách tiếp cận chỉ dùng bộ chiếu (projector-only)

MiniGPT-4 giữ lại Q-Former nhưng chỉ huấn luyện lớp chiếu tuyến tính đầu ra trong khi đóng băng mọi thứ khác. Rẻ, nhưng cái giá phải trả là chất lượng — các truy vấn là của BLIP-2, không phải của bạn. Tốt cho việc lặp lại nhanh, không phải là kiến trúc tối ưu nhất.

### Tại sao LLaVA chọn cách đơn giản hơn

LLaVA (2023, Bài học 12.05) thay thế Q-Former bằng một MLP 2 lớp đơn giản chiếu mọi patch token của ViT vào không gian LLM — 576 token cho mỗi hình ảnh với lưới 24x24, tất cả được đưa vào LLM. Khả năng nén kém hơn nhưng cho phép LLM thực hiện attention trên các patch thô. Vào thời điểm đó, điều này gây tranh cãi; đến cuối năm 2023, nó trở nên thống trị vì dữ liệu hướng dẫn hình ảnh (LLaVA-Instruct-150k) chứng minh rằng MLP có thể được huấn luyện để bảo toàn đủ tín hiệu. Sự đánh đổi: ngữ cảnh của LLaVA đầy nhanh hơn, nhưng nó mở rộng tự nhiên sang đa hình ảnh và video.

Đến năm 2026, lĩnh vực này phân tách: Q-Former tồn tại ở những nơi ngân sách token quan trọng (video dài, nhiều hình ảnh); bộ chiếu MLP thống trị ở những nơi chất lượng thô trên mỗi token là ưu tiên hàng đầu.

### Gated cross-attention: Flamingo, tổ tiên của các mô hình

Flamingo (Bài học 12.04) ra đời trước BLIP-2 và sử dụng cùng ý tưởng cross-attention nhưng ở mọi lớp của LLM đã đóng băng, không phải là một cầu nối duy nhất. BLIP-2 cho thấy bạn có thể nén chỉ vào lớp đầu vào và vẫn hoạt động tốt. Gemini và Idefics kết hợp cả hai: các token đầu vào xen kẽ cộng với gated cross-attention tùy chọn cho in-context few-shot.

### Các hậu duệ năm 2026

- Q-Former: BLIP-2, InstructBLIP, MiniGPT-4, và hầu hết các mô hình video-ngôn ngữ vì lý do ngân sách token.
- Perceiver resampler: Biến thể của Flamingo (Bài học 12.04); họ Idefics, Eagle, OmniMAE.
- MLP projector: LLaVA, LLaVA-NeXT, LLaVA-OneVision, Cambrian-1.
- Attention pool: VILA, PaliGemma.

Cả bốn đều hợp lệ. Câu hỏi quyết định là bạn bị hạn chế về ngân sách token hay về chất lượng trên mỗi token.

```figure
modality-projection
```

## Sử dụng

`code/main.py` xây dựng một cross-attention kiểu Q-Former bằng thư viện chuẩn:

1. Mô phỏng 256 patch token hình ảnh (số chiều 128).
2. Khởi tạo 32 truy vấn có thể học (số chiều 128).
3. Chạy scaled-dot-product cross-attention (Q từ các truy vấn, K/V từ các patch).
4. Chiếu sang số chiều LLM (512) thông qua một lớp tuyến tính.
5. Xuất ra 32 token hình ảnh sẵn sàng cho LLM.

Tất cả toán học đều bằng Python thuần (các vòng lặp lồng nhau trên các vector). Dù là mô phỏng nhưng đúng về hình dạng. Ma trận trọng số attention được in ra để bạn có thể thấy mỗi truy vấn đã lấy thông tin từ những patch nào.

## Triển khai

Bài học này tạo ra `outputs/skill-modality-bridge-picker.md`. Với một cấu hình VLM mục tiêu (số lượng token của bộ mã hóa hình ảnh, ngân sách ngữ cảnh LLM, các ràng buộc triển khai, mục tiêu chất lượng), nó đưa ra khuyến nghị chọn Q-Former, MLP hay Perceiver resampler kèm theo lý do ngắn gọn và ước tính số lượng tham số cho mỗi cầu nối.

## Bài tập

1. Triển khai khối cross-attention trong PyTorch. Xác minh rằng với 32 truy vấn và 256 keys/values, ma trận trọng số attention là 32 x 256 và mỗi hàng có tổng bằng 1 sau khi qua softmax.

2. Trong BLIP-2 giai đoạn 1, Q-Former chạy ba hàm mất mát đồng thời: ITC, ITM, ITG. Viết chữ ký hàm forward cho mỗi hàm dưới dạng mã giả. Hàm nào yêu cầu đường dẫn bộ mã hóa văn bản phải hoạt động?

3. So sánh số lượng tham số: Q-Former (12 lớp, 768 hidden) so với bộ chiếu MLP 2 lớp (1408 → 4096, hai lớp). Ở quy mô LLM nào thì chi phí 188 triệu tham số của Q-Former mang lại hiệu quả huấn luyện?

4. Đọc Mục 3.2 của bài báo BLIP-2 (arXiv:2301.12597) về cách Q-Former được khởi tạo. Giải thích tại sao việc khởi tạo từ BERT-base (thay vì ngẫu nhiên) lại giúp tăng tốc độ hội tụ.

5. Đối với một video 10 phút ở tốc độ 1 FPS được lấy mẫu thành 60 khung hình, hãy tính chi phí token mỗi khung hình với (Q-Former → 32 token/khung hình) so với (MLP projector → 576 token/khung hình). Cái nào phù hợp với cửa sổ ngữ cảnh LLM 128k-token?

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|------------------------|
| Q-Former | "Querying transformer" | Transformer nhỏ với 32 vector truy vấn có thể học thực hiện cross-attention trên các đặc trưng ViT đã đóng băng |
| Learnable queries | "Soft prompt for vision" | Một tập hợp tham số cố định đóng vai trò là phía truy vấn của cross-attention; được học theo từng mô hình, chia sẻ trên mọi đầu vào |
| Cross-attention | "Q from here, K/V from there" | Attention nơi query, key và value đến từ các nguồn khác nhau; cách các truy vấn lấy thông tin từ các patch ViT |
| ITC | "Image-text contrastive" | Hàm mất mát kiểu CLIP áp dụng cho các truy vấn đã gộp của Q-Former so với văn bản CLS |
| ITM | "Image-text matching" | Bộ phân loại nhị phân trên các cặp hard-negative-mined; buộc các truy vấn phân biệt các sự không khớp tinh vi |
| ITG | "Image-grounded text generation" | Hàm mất mát LM nhân quả nơi văn bản được tạo ra có điều kiện trên các truy vấn; buộc các truy vấn mã hóa nội dung có thể giải mã thành văn bản |
| Two-stage pretraining | "Representation then generative" | Giai đoạn 1 huấn luyện riêng Q-Former (ITC/ITM/ITG); Giai đoạn 2 gắn LLM đã đóng băng và chỉ huấn luyện lớp chiếu + Q-Former |
| Frozen backbone | "Do not finetune" | Trọng số của bộ mã hóa hình ảnh và LLM được cố định; chỉ cầu nối được huấn luyện |
| Projection head | "Linear to LLM dim" | Lớp tuyến tính cuối cùng ánh xạ đầu ra của Q-Former sang số chiều embedding của LLM |
| Perceiver resampler | "Flamingo's version" | Cross-attention với truy vấn có thể học tương tự, được Flamingo sử dụng ở mọi lớp thay vì chỉ là một cầu nối duy nhất |

## Đọc thêm

- [Li et al. — BLIP-2 (arXiv:2301.12597)](https://arxiv.org/abs/2301.12597) — bài báo cốt lõi.
- [Li et al. — BLIP (arXiv:2201.12086)](https://arxiv.org/abs/2201.12086) — tiền thân với bộ ba ITC/ITM/ITG.
- [Li et al. — ALBEF (arXiv:2107.07651)](https://arxiv.org/abs/2107.07651) — "align before fuse" — tổ tiên khái niệm của huấn luyện giai đoạn 1.
- [Dai et al. — InstructBLIP (arXiv:2305.06500)](https://arxiv.org/abs/2305.06500) — Q-Former nhận thức hướng dẫn.
- [Zhu et al. — MiniGPT-4 (arXiv:2304.10592)](https://arxiv.org/abs/2304.10592) — cách tiếp cận chỉ dùng bộ chiếu.
- [Jaegle et al. — Perceiver IO (arXiv:2107.14795)](https://arxiv.org/abs/2107.14795) — kiến trúc tổng quát cho cross-attention với truy vấn có thể học.