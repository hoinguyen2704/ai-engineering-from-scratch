# Embodied VLAs: RT-2, OpenVLA, π0, GR00T

> Lần đầu tiên một mô hình đọc công thức nấu ăn từ một trang web và thực hiện nó trên một robot nhà bếp là RT-2 (Google DeepMind, tháng 7 năm 2023). RT-2 rời rạc hóa các hành động thành các token văn bản, đồng tinh chỉnh (co-fine-tune) một VLM trên dữ liệu web kết hợp với dữ liệu hành động của robot, và chứng minh rằng kiến thức thị giác-ngôn ngữ ở quy mô web có thể chuyển đổi sang điều khiển robot. OpenVLA (tháng 6 năm 2024) đã phát hành bản tham chiếu 7B mã nguồn mở. Dòng π0 của Physical Intelligence (2024-2025) đã bổ sung các chuyên gia hành động flow-matching. GR00T N1 của NVIDIA (tháng 3 năm 2025) đã mang đến khả năng điều khiển hệ thống kép (System 1 / System 2) cho robot hình người ở quy mô lớn. Nguyên mẫu VLA — vision-language-action, một mô hình duy nhất có khả năng nhìn, đọc và hành động — chính là cầu nối giữa các mô hình hiểu biết của giai đoạn này và các hệ thống tự hành trong Giai đoạn 15.

**Type:** Learn
**Languages:** Python (stdlib, action tokenizer + VLA inference skeleton)
**Prerequisites:** Phase 12 · 05 (LLaVA), Phase 15 (Autonomous Systems, referenced)
**Time:** ~180 phút

## Mục tiêu học tập

- Mô tả quá trình token hóa hành động: mã hóa bin rời rạc (RT-2), token hành động hiệu quả FAST, hành động flow-matching liên tục (π0).
- Giải thích lý do tại sao việc đồng tinh chỉnh trên dữ liệu web + robot giúp bảo toàn khả năng chuyển đổi kiến thức tổng quát sang các tác vụ mới.
- So sánh OpenVLA (Llama 7B + VLM mã nguồn mở), π0 (flow-matching), và GR00T N1 (hệ thống kép) trên cùng một tác vụ robot.
- Nêu tên tập dữ liệu Open X-Embodiment và vai trò của nó như là tập dữ liệu huấn luyện RT-X.

## Vấn đề

Một robot thực hiện các công việc nhà từ các hướng dẫn ngôn ngữ tự nhiên đã là mục tiêu nghiên cứu từ những năm 1970. Câu trả lời của những năm 2020: mô hình vision-language-action (VLA). Sử dụng cùng kiến trúc VLM cho VQA, nhưng đầu ra là các hành động (mô-men xoắn khớp, tư thế của bộ phận cuối, các lệnh rời rạc) thay vì văn bản.

Các thách thức đặc thù đối với VLA:

1. Không gian hành động là liên tục (góc khớp, lực) và có số chiều cao (cánh tay 7-DOF + kẹp 3-DOF = 10 chiều ở tần số 30 Hz).
2. Dữ liệu huấn luyện đặc thù cho robot rất khan hiếm. Open X-Embodiment có khoảng 1 triệu quỹ đạo; trong khi dữ liệu văn bản-hình ảnh trên web là hơn 5 tỷ.
3. Tần số điều khiển rất quan trọng. Vòng lặp điều khiển 30 Hz nghĩa là ngân sách 33ms cho mỗi hành động.
4. An toàn. Một hành động sai có thể làm hỏng phần cứng, gây hại cho con người hoặc tài sản.

## Khái niệm

### Token hóa hành động (RT-2)

Thủ thuật của RT-2: biểu diễn mỗi mục tiêu khớp dưới dạng một token văn bản được lượng tử hóa. Rời rạc hóa phạm vi chuẩn hóa [-1, 1] thành 256 bin, ánh xạ mỗi bin tới một ID từ vựng. Một hành động 10-DOF trở thành 10 token tại mỗi bước điều khiển.

Đồng tinh chỉnh một VLM PaLM-X trên hỗn hợp:

- Các cặp hình ảnh-văn bản trên web (captioning, VQA).
- Các bản trình diễn của robot, với hành động được biểu diễn dưới dạng token.

Mô hình nhìn thấy "nhặt khối màu đỏ" (ngôn ngữ) → hình ảnh (thị giác) → chuỗi hành động 10-token (các mục tiêu khớp đã rời rạc hóa). Việc tiền huấn luyện trên web bảo toàn khả năng chuyển đổi kiến thức tổng quát: RT-2 có thể thực hiện "di chuyển về phía vật thể đang di chuyển nhanh" ngay cả khi cụm từ "di chuyển nhanh" không có trong dữ liệu huấn luyện.

Suy luận ở mức 3-5 Hz trong bài báo RT-2, bị giới hạn bởi quá trình giải mã tự hồi quy (autoregressive decode) của VLM.

### OpenVLA — bản tham chiếu 7B mã nguồn mở

OpenVLA (Kim và cộng sự, tháng 6 năm 2024) là phiên bản tương đương RT-2 với trọng số mở. Sử dụng backbone Llama 7B, bộ mã hóa thị giác kép DINOv2 + SigLIP, và token hóa hành động trên 256 bin.

Được huấn luyện trên Open X-Embodiment (970 nghìn quỹ đạo trên 22 loại robot). Đi kèm với hỗ trợ tinh chỉnh LoRA để thích nghi với các robot mới.

Suy luận: 4-5 Hz trên A100 với lượng tử hóa. Đủ nhanh cho các thao tác chậm, nhưng chưa đủ cho điều khiển tần số cao.

### Tokenizer FAST — giải mã hành động nhanh hơn

Pertsch và cộng sự (2024) đã chỉ ra rằng token hóa theo bin rời rạc là không hiệu quả — hầu hết các hành động tập trung trong một vùng nhỏ của không gian bin. FAST (Frequency-domain Action Sequence Tokenizer) nén các chuỗi hành động thông qua DCT và lượng tử hóa các hệ số.

Một quỹ đạo hành động 30 bước trở thành khoảng 10 token FAST thay vì 300 token bin rời rạc. Tốc độ suy luận tăng 3-5 lần mà không làm giảm chất lượng.

### π0 và hành động flow-matching

π0 của Physical Intelligence (Black và cộng sự, tháng 10 năm 2024) thay thế các token hành động rời rạc bằng một chuyên gia hành động flow-matching:

- Một transformer hành động nhỏ đọc các trạng thái ẩn của VLM và xuất ra một chuỗi hành động liên tục 50 bước thông qua rectified flow.
- Head hành động được huấn luyện với hàm mất mát flow-matching; quá trình tiền huấn luyện VLM vẫn giữ nguyên.
- Suy luận: toàn bộ chuỗi hành động được phát ra trong khoảng 5 bước khử nhiễu, tương đương với điều khiển 50 Hz.

Tuyên bố của π0: vượt trội hơn OpenVLA và Octo trên một loạt các tác vụ thao tác. Công thức hành động liên tục bảo toàn độ mượt mà mà việc rời rạc hóa làm mất đi.

π0.5 và π0-FAST là các bản nâng cấp gia tăng. π0-FAST kết hợp token hóa FAST với flow matching.

### GR00T N1 — hệ thống kép cho robot hình người

GR00T N1 của NVIDIA (tháng 3 năm 2025) được xây dựng cho robot hình người (>30 DOF, toàn thân):

- System 2: một VLM lớn đọc cảnh + hướng dẫn, tạo ra các mục tiêu phụ cấp cao ở mức ~1 Hz.
- System 1: một transformer head hành động nhỏ tạo ra các lệnh khớp cấp thấp 50-100 Hz dựa trên các mục tiêu phụ.

Sự phân chia này ánh xạ tới tư duy nhanh và chậm của Kahneman: System 2 lập kế hoạch, System 1 hành động. Lợi ích: việc lập kế hoạch chậm của VLM không chặn quá trình điều khiển nhanh; System 1 giữ được kích thước nhỏ để giảm độ trễ.

GR00T N1.7 (cuối năm 2025) cải thiện khả năng mở rộng dữ liệu. GR00T tinh chỉnh với dữ liệu sim-to-real từ Omniverse.

### Open X-Embodiment

Dữ liệu huấn luyện. RT-X (tháng 10 năm 2023) đã tập hợp 22 tập dữ liệu bao gồm 1 triệu quỹ đạo trên 22 robot. Open X-Embodiment là kho dữ liệu mà mọi người đều sử dụng:

- ALOHA / Bridge V2 / Droid / RT-2 Kitchen / Language Table.
- Mỗi mẫu: (trạng thái robot, góc nhìn camera, hướng dẫn, chuỗi hành động).
- Vệ sinh dữ liệu huấn luyện: thống nhất không gian hành động, chuẩn hóa phạm vi khớp, thay đổi kích thước camera.

OpenVLA và π0 huấn luyện trên Open X-Embodiment. Khoảng cách miền (domain gap) đối với bất kỳ robot cụ thể nào được thu hẹp bằng cách tinh chỉnh LoRA trên 100-1000 bản trình diễn đặc thù cho tác vụ.

### Đồng tinh chỉnh vs chỉ dùng dữ liệu robot

Đồng tinh chỉnh trộn dữ liệu VQA web với các quỹ đạo robot. Tỷ lệ rất quan trọng: quá nhiều VQA thì mô hình quên hành động; quá nhiều dữ liệu robot thì mô hình mất kiến thức tổng quát.

Tỷ lệ của RT-2: ~1:1. OpenVLA: ~0.5:1 web-trên-robot. π0: tương tự. Tỷ lệ chính xác là một siêu tham số cần điều chỉnh theo quy mô tập dữ liệu.

Huấn luyện chỉ với dữ liệu robot tạo ra các mô hình đặc thù cho tác vụ nhưng thất bại với các hướng dẫn nằm ngoài phân phối (out-of-distribution). Đồng tinh chỉnh tạo ra sự khác biệt giữa "nhặt khối màu đỏ (trong bản demo)" và "nhặt vật thể lớn thứ ba từ bên trái (cách diễn đạt mới)."

### An toàn và giới hạn hành động

Mọi VLA thương mại đều đi kèm với:

- Giới hạn khớp cứng (không thể tạo mô-men xoắn vượt quá thông số kỹ thuật).
- Giới hạn vận tốc (soft clipping).
- Giới hạn không gian làm việc (bộ phận cuối không thể rời khỏi bàn).
- Phê duyệt của con người (human-in-the-loop) cho các tác vụ mới.

Những cơ chế này nằm ngoài VLA như các kiểm tra lớp điều khiển. Đầu ra của VLA chỉ là một gợi ý, không phải là một mệnh lệnh.

```figure
mm-action-tokens
```

## Sử dụng

`code/main.py`:

- Triển khai token hóa và giải mã hành động 256-bin.
- Phác thảo một tokenizer FAST dựa trên DCT + lượng tử hóa.
- So sánh số lượng token trên mỗi bước hành động giữa (bin rời rạc, FAST, flow liên tục).
- In tóm tắt dòng dõi từ RT-2 → OpenVLA → π0 → GR00T.

## Triển khai

Bài học này tạo ra `outputs/skill-vla-action-format-picker.md`. Với một tác vụ robot (thao tác, điều hướng, toàn thân hình người), lựa chọn giữa bin rời rạc + RT-2, FAST + OpenVLA, flow-matching + π0, hoặc hệ thống kép + GR00T.

## Bài tập

1. Một cánh tay 10-DOF ở tốc độ điều khiển 30 Hz. Token hóa bin rời rạc tại 256 bin phát ra bao nhiêu token mỗi giây? Một VLM 7B có thể theo kịp không?

2. Token hóa FAST nén các quỹ đạo 30 bước thành khoảng 10 token. Người dùng mất gì nếu quỹ đạo có chuyển động tần số cao (ví dụ: đánh trống)?

3. Head flow-matching của π0 khử nhiễu trong khoảng 5 bước. So sánh thông lượng với giải mã tự hồi quy của OpenVLA ở mức 4-5 Hz.

4. Sự phân chia System 1 / System 2 của GR00T ánh xạ tới Kahneman. Hãy đề xuất một sự phân chia khác (System 3?) có thể giúp ích cho việc đi bộ bằng hai chân.

5. Đọc Open X-Embodiment Phần 4 về quản lý tập dữ liệu. Nêu tên ba quy tắc quản lý giúp ngăn chặn rò rỉ miền (domain leakage).

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|------------------------|
| VLA | "Vision-language-action" | Mô hình nhận hình ảnh + hướng dẫn và xuất ra các lệnh hành động |
| Action tokenization | "Discrete bins" | Lượng tử hóa các mục tiêu khớp liên tục thành 256 bin mỗi chiều, mỗi bin là một ID từ vựng |
| FAST tokenizer | "Frequency action tokens" | DCT + lượng tử hóa để nén các quỹ đạo 30 bước thành ~10 token |
| Co-fine-tune | "Mix web + robot" | Huấn luyện trên dữ liệu VQA web cùng với các bản demo robot để bảo toàn kiến thức tổng quát |
| Flow-matching action head | "π0 continuous output" | Transformer nhỏ xuất ra chuỗi hành động 50 bước thông qua rectified flow |
| System 1 / System 2 | "Dual-system control" | VLM lớn lập kế hoạch chậm, head hành động nhỏ thực hiện nhanh; mô hình GR00T |
| Open X-Embodiment | "RT-X dataset" | Tập dữ liệu đa robot với 1 triệu quỹ đạo; kho dữ liệu huấn luyện |

## Đọc thêm

- [Brohan và cộng sự — RT-2 (arXiv:2307.15818)](https://arxiv.org/abs/2307.15818)
- [Kim và cộng sự — OpenVLA (arXiv:2406.09246)](https://arxiv.org/abs/2406.09246)
- [Black và cộng sự — π0 (arXiv:2410.24164)](https://arxiv.org/abs/2410.24164)
- [NVIDIA — GR00T N1 (arXiv:2503.14734)](https://arxiv.org/abs/2503.14734)
- [Open X-Embodiment Collab — RT-X (arXiv:2310.08864)](https://arxiv.org/abs/2310.08864)