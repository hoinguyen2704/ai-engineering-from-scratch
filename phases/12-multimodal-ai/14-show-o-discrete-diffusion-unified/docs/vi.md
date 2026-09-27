# Show-o và các mô hình hợp nhất Discrete-Diffusion

> Transfusion kết hợp các biểu diễn liên tục và rời rạc. Show-o (Xie và cộng sự, tháng 8 năm 2024) đi theo hướng ngược lại: các token văn bản sử dụng dự đoán token tiếp theo (causal next-token prediction), còn các token hình ảnh sử dụng masked discrete diffusion theo tinh thần của MaskGIT. Cả hai đều nằm trong một transformer với mặt nạ chú ý (attention mask) lai. Kết quả là sự hợp nhất của VQA, text-to-image, inpainting và tạo nội dung đa phương thức trên một backbone duy nhất, một tokenizer cho mỗi phương thức và một công thức hàm mất mát (next-token được mở rộng sang dự đoán có che mask). Bài học này sẽ đi sâu vào thiết kế của Show-o — tại sao masked discrete diffusion lại là một trình tạo hình ảnh song song, ít bước — và so sánh nó với Transfusion và Emu3.

**Type:** Learn
**Languages:** Python (stdlib, masked-discrete-diffusion sampler)
**Prerequisites:** Phase 12 · 13 (Transfusion)
**Time:** ~120 phút

## Mục tiêu học tập

- Giải thích masked discrete diffusion: lịch trình che mask các token một cách đồng nhất, sau đó yêu cầu transformer khôi phục chúng.
- So sánh giải mã hình ảnh song song (Show-o, MaskGIT) với giải mã hình ảnh tự hồi quy (Chameleon, Emu3) về tốc độ và chất lượng.
- Liệt kê ba tác vụ mà Show-o xử lý trong một checkpoint: T2I, VQA, inpainting hình ảnh.
- Chọn một lịch trình che mask (cosine, linear, truncated) và lập luận về ảnh hưởng của nó đến chất lượng mẫu.

## Vấn đề

Việc huấn luyện với hai hàm mất mát của Transfusion có hiệu quả nhưng động lực học lại phức tạp hơn — hàm mất mát diffusion liên tục nằm trên một thang đo số học khác với hàm mất mát NTP rời rạc. Việc cân bằng trọng số hàm mất mát là một quá trình tìm kiếm siêu tham số. Kiến trúc này hiệu quả nhưng phức tạp.

Câu trả lời của Show-o: giữ cả hai phương thức ở dạng rời rạc (giống như Chameleon), nhưng tạo hình ảnh song song thông qua masked discrete diffusion thay vì tuần tự. Mục tiêu huấn luyện trở thành một dự đoán token bị che duy nhất, khái quát hóa một cách tự nhiên cho dự đoán token tiếp theo.

## Khái niệm

### Masked discrete diffusion (MaskGIT)

Thủ thuật MaskGIT của Chang và cộng sự (2022) rất tinh tế. Bắt đầu từ một hình ảnh bị che hoàn toàn (mỗi token là id đặc biệt `<MASK>`). Tại mỗi bước, dự đoán tất cả các token bị che một cách song song, sau đó giữ lại top-K dự đoán tự tin nhất và che lại phần còn lại. Sau khoảng 8-16 lần lặp, tất cả các token sẽ được lấp đầy. Lịch trình về số lượng token cần bỏ che mỗi bước được tinh chỉnh — các lịch trình cosine hoạt động rất tốt.

Việc huấn luyện rất đơn giản: lấy mẫu tỷ lệ che mask đồng nhất từ [0, 1], áp dụng nó vào các token VQ của hình ảnh, huấn luyện transformer để khôi phục các token bị che. Chính xác là những gì BERT đã làm cho văn bản, được mở rộng sang tạo hình ảnh.

### Show-o: một transformer, mặt nạ lai

Show-o đặt MaskGIT vào bên trong một transformer mô hình ngôn ngữ nhân quả (causal-language-model). Mặt nạ chú ý (attention mask) là:

- Token văn bản: nhân quả (LLM tiêu chuẩn).
- Token hình ảnh: hai chiều đầy đủ trong khối hình ảnh (để các token bị che có thể nhìn thấy mọi token hình ảnh khác trong quá trình dự đoán).
- Text-to-image: văn bản chú ý đến các hình ảnh trước đó, hình ảnh chú ý đến văn bản trước đó.

Huấn luyện luân phiên giữa:
1. NTP tiêu chuẩn trên các chuỗi văn bản.
2. Các mẫu T2I: văn bản → hình ảnh với các token hình ảnh bị che, hàm mất mát dự đoán token bị che.
3. Các mẫu VQA: hình ảnh → văn bản với các token văn bản bị che (thực chất chỉ là NTP).

Hàm mất mát hợp nhất là cross-entropy trên các token `<MASK>`, bao gồm cả NTP văn bản (chỉ token cuối cùng là "bị che") và diffusion hình ảnh bị che (một tập hợp con ngẫu nhiên bị che).

### Lấy mẫu song song

Show-o tạo ra một hình ảnh trong khoảng 16 bước thay vì khoảng 1000 (tự hồi quy trên mỗi token) hoặc khoảng 20 (diffusion). Tại mỗi bước, dự đoán tất cả các token bị che một cách song song; cam kết top-K tự tin nhất; lặp lại.

So sánh:
- Chameleon / Emu3 (tự hồi quy trên các token): N_tokens forward passes, thường là 1024-4096 cho mỗi hình ảnh.
- Transfusion (diffusion liên tục): ~20 bước, mỗi bước là một lần chạy transformer đầy đủ.
- Show-o (masked discrete diffusion): ~16 bước, mỗi bước là một lần chạy transformer đầy đủ.

Show-o nhanh hơn Chameleon ở các mô hình có quy mô tương đương, số bước xấp xỉ Transfusion với chi phí mỗi bước thấp hơn (logits từ vựng rời rạc so với hàm mất mát MSE liên tục).

### Các tác vụ trong một checkpoint

Show-o hỗ trợ bốn tác vụ tại thời điểm suy luận, được chọn theo định dạng prompt:

- Tạo văn bản: đầu ra văn bản tự hồi quy tiêu chuẩn.
- VQA: đầu vào hình ảnh, đầu ra văn bản.
- T2I: đầu vào văn bản, đầu ra hình ảnh thông qua masked discrete diffusion.
- Inpainting: hình ảnh với một số token bị che, lấp đầy phần còn thiếu.

Khả năng inpainting có được miễn phí từ quá trình huấn luyện dự đoán bị che. Che một vùng của lưới token VQ, đưa phần còn lại cùng với một prompt văn bản vào, dự đoán các token bị che.

### Lịch trình che mask

Lịch trình về số lượng token cần bỏ che mỗi bước quyết định chất lượng. Show-o khuyến nghị sử dụng cosine:

```
mask_ratio(t) = cos(pi * t / (2 * T))   # t = 0..T
```

Tại bước 0, tất cả các token bị che (tỷ lệ 1.0). Tại bước T, không có token nào bị che. Cosine tập trung khối lượng vào các tỷ lệ tầm trung nơi dự đoán mang lại nhiều thông tin nhất. Các lịch trình tuyến tính cũng hoạt động nhưng đạt trạng thái bão hòa nhanh hơn.

### Show-o2

Show-o2 (bản cập nhật năm 2025, arXiv 2506.15564) mở rộng Show-o: nền tảng LLM lớn hơn, tokenizer tốt hơn, lịch trình che mask được cải thiện. Cùng một mô hình kiến trúc.

### Vị trí của Show-o

Trong phân loại năm 2026:

- Token rời rạc + NTP: Chameleon, Emu3. Đơn giản nhưng suy luận chậm.
- Token rời rạc + masked diffusion: Show-o, MaskGIT, LlamaGen, Muse. Lấy mẫu song song, vẫn bị mất mát do tokenizer.
- Liên tục + diffusion: Transfusion, MMDiT, DiT. Chất lượng cao nhất, huấn luyện phức tạp hơn.
- Liên tục + flow matching trong VLM: JanusFlow, InternVL-U. Mới nhất.

Chọn theo tác vụ: Show-o khi bạn muốn T2I + inpainting + VQA trong một mô hình mở với tốc độ hợp lý; Transfusion khi chất lượng là ưu tiên hàng đầu và bạn có thể chấp nhận sự phức tạp của hai hàm mất mát.

```figure
masked-diffusion-unmask
```

## Sử dụng

`code/main.py` mô phỏng quá trình lấy mẫu của Show-o:

- Một lưới đồ chơi gồm 16 token VQ.
- Một "transformer" giả lập dự đoán logits dựa trên prompt và các token hiện chưa bị che.
- Lấy mẫu che mask song song qua 8 bước với lịch trình cosine.
- In ra các trạng thái trung gian (sự tiến hóa của mẫu mask) và các token cuối cùng.

Hãy chạy thử và quan sát lớp mask tan biến từng bước một.

## Triển khai

Bài học này tạo ra `outputs/skill-unified-gen-model-picker.md`. Với một sản phẩm cần cả khả năng hiểu (VQA, chú thích) và tạo (T2I, inpainting) với ràng buộc trọng số mở, hãy lựa chọn giữa gia đình Show-o, gia đình Transfusion/MMDiT và gia đình Emu3 / Chameleon với những đánh đổi cụ thể.

## Bài tập

1. Các mẫu masked discrete diffusion trong khoảng 16 bước. Tại sao không phải là 1? Điều gì sẽ xảy ra nếu bạn bỏ che tất cả tại bước 0?

2. Inpainting có sẵn với masked diffusion. Hãy đề xuất một trường hợp sử dụng sản phẩm (thực tế hoặc giả định) mà tính năng inpainting của Show-o vượt trội hơn một mô hình chuyên biệt.

3. Lịch trình Cosine so với lịch trình tuyến tính: hãy theo dõi số lượng token được bỏ che mỗi bước cho T=8. Cái nào cân bằng hơn?

4. Một hình ảnh Show-o 512x512 là 1024 token. Với từ vựng K=16384, mô hình phát ra 1024 * log2(16384) = 14,336 bit (~1.75 KiB) dữ liệu. Stable Diffusion xuất ra 512*512*24 bit = 6,291,456 bit (~768 KiB) pixel thô. Tỷ lệ nén là bao nhiêu và nó mang lại chất lượng như thế nào?

5. Đọc LlamaGen (arXiv:2406.06525). Mô hình hình ảnh tự hồi quy có điều kiện theo lớp của LlamaGen khác với phương pháp masked của Show-o như thế nào?

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|------------------------|
| Masked discrete diffusion | "Kiểu MaskGIT" | Huấn luyện để dự đoán các token bị che; khi suy luận, lặp lại việc bỏ che các dự đoán tự tin nhất |
| Cosine schedule | "Lịch trình bỏ che" | Sự suy giảm của tỷ lệ che mask qua các bước suy luận; tập trung sự tăng trưởng độ tự tin ở mức trung bình |
| Parallel decoding | "Tất cả token cùng lúc" | Mỗi bước dự đoán toàn bộ chuỗi token bị che trong một lần chạy, sau đó cam kết top-K |
| Hybrid attention | "Nhân quả + hai chiều" | Mặt nạ có tính nhân quả trên các token văn bản và hai chiều trong các khối hình ảnh |
| Inpainting | "Tạo nội dung lấp đầy" | Điều kiện hóa trên một hình ảnh với một số token bị che, dự đoán các phần còn thiếu; có sẵn từ mục tiêu huấn luyện |
| Commitment rate | "Top-K mỗi bước" | Số lượng token được tuyên bố "xong" mỗi lần lặp; kiểm soát sự đánh đổi giữa suy luận và chất lượng |

## Đọc thêm

- [Xie và cộng sự — Show-o (arXiv:2408.12528)](https://arxiv.org/abs/2408.12528)
- [Show-o2 (arXiv:2506.15564)](https://arxiv.org/abs/2506.15564)
- [Chang và cộng sự — MaskGIT (arXiv:2202.04200)](https://arxiv.org/abs/2202.04200)
- [Sun và cộng sự — LlamaGen (arXiv:2406.06525)](https://arxiv.org/abs/2406.06525)
- [Chang và cộng sự — Muse (arXiv:2301.00704)](https://arxiv.org/abs/2301.00704)