# Họ Qwen-VL và Video Dynamic-FPS

> Họ Qwen-VL — Qwen-VL (2023), Qwen2-VL (2024), Qwen2.5-VL (2025), Qwen3-VL (2025) — là dòng mô hình ngôn ngữ-thị giác (vision-language model) mã nguồn mở có ảnh hưởng nhất vào năm 2026. Mỗi thế hệ đều thực hiện một bước đi kiến trúc mang tính quyết định mà phần còn lại của hệ sinh thái mã nguồn mở đã sao chép trong vòng mười hai tháng: độ phân giải động gốc thông qua M-RoPE, lấy mẫu dynamic-FPS với căn chỉnh thời gian tuyệt đối, window attention trong ViT, và các định dạng đầu ra tác nhân (agent) có cấu trúc. Đến Qwen3-VL, công thức đã trở nên ổn định: bộ mã hóa 2D-RoPE-ViT với đầu vào có tỷ lệ khung hình gốc, bộ chiếu MLP vào một nền tảng ngôn ngữ Qwen3 lớn, và các giai đoạn huấn luyện nhấn mạnh vào OCR, grounding (định vị), và hành vi tác nhân như các mục tiêu ưu tiên hàng đầu. Bài học này xem xét dòng họ này theo trình tự thời gian để bạn hiểu tại sao mọi núm điều khiển lại nằm ở vị trí của chúng.

**Type:** Learn
**Languages:** Python (stdlib, M-RoPE encoder + dynamic-FPS sampler)
**Prerequisites:** Phase 12 · 06 (patch-n'-pack)
**Time:** ~120 phút

## Mục tiêu học tập

- Tính toán các phép quay ba trục của M-RoPE (thời gian, chiều cao, chiều rộng) và giải thích tại sao cần cả ba.
- Chọn chiến lược lấy mẫu dynamic-FPS cho video và lập luận về sự đánh đổi giữa số token trên mỗi giây và độ chính xác của việc phát hiện sự kiện.
- Kể tên bốn nâng cấp thế hệ của Qwen-VL theo thứ tự và những gì mỗi nâng cấp đã kích hoạt.
- Xây dựng định dạng đầu ra tác nhân JSON theo kiểu Qwen2.5-VL và phân tích cú pháp các lệnh gọi công cụ có cấu trúc từ phản hồi của VLM.

## Vấn đề

Qwen-VL ra mắt vào tháng 8 năm 2023 như một phản ứng trực tiếp đối với LLaVA-1.5 và BLIP-2. Khoảng cách mà nhóm Qwen nhắm đến có ba khía cạnh: độ phân giải, video và đầu ra có cấu trúc.

Độ phân giải: LLaVA-1.5 chạy ở 336x336. Tốt cho ảnh, nhưng vô dụng đối với hóa đơn tiếng Trung hoặc ảnh chụp màn hình bảng tính dày đặc. Đổi mới đầu tiên của Qwen-VL là 448x448 và đầu ra bounding-box (hộp bao) được định vị, cho phép mô hình chỉ vào các đối tượng.

Video: Video-LLaMA xếp chồng các bộ mã hóa theo từng khung hình và đưa chúng vào LLM. Nó hoạt động với các clip ngắn, không phải cho các video dài nhiều phút nơi trục thời gian là tín hiệu chính. Nhóm Qwen muốn một bộ mã hóa duy nhất hiểu được thời gian.

Đầu ra có cấu trúc: LLaVA phát ra văn bản tự do. Một tác nhân cần JSON. Qwen-VL được huấn luyện trên các định dạng đầu ra JSON rõ ràng bao gồm tọa độ bounding-box dưới dạng văn bản.

Mỗi thế hệ Qwen-VL đều mở rộng một trong ba trục này.

## Khái niệm

### Qwen-VL (Tháng 8 năm 2023)

Thế hệ đầu tiên: OpenCLIP ViT-bigG/14 làm bộ mã hóa (2.5B tham số), Q-Former tương thích với LLama (1 bước với 256 truy vấn), nền tảng Qwen-7B. Đóng góp:

- Độ phân giải 448x448 (lúc đó là SOTA cho VLM mã nguồn mở).
- Grounding: huấn luyện trên các cặp ảnh-văn bản với đầu ra token tọa độ rõ ràng. "Con mèo ở tại <box>(112, 204), (280, 344)</box>".
- Huấn luyện đa ngôn ngữ Trung + Anh ngay từ đầu.

Các benchmark tại thời điểm đó: cạnh tranh với GPT-4V trên tiếng Anh, chiếm ưu thế trên tiếng Trung. Sự giám sát grounding là tiêu đề thực sự.

### Qwen2-VL (Tháng 9 năm 2024) — M-RoPE và độ phân giải gốc

Qwen2-VL đã thay thế ngăn xếp độ phân giải cố định + Q-Former bằng bộ mã hóa ViT độ phân giải động gốc. Các thay đổi chính:

- Độ phân giải động gốc. ViT chấp nhận bất kỳ HxW nào chia hết cho 28 (patch 14 với 2x gộp không gian). Một hình ảnh ở 1120x672 (40x24 patch gộp) tạo ra 960 token hình ảnh. Không thay đổi kích thước, không tiling, không thumbnail.
- M-RoPE (Multimodal RoPE). Mỗi token mang một vị trí 3D (t, h, w) thay vì 1D. Đối với ảnh t=0, đối với video t = chỉ số khung hình. RoPE xoay các vector truy vấn/khóa theo tần số trên mỗi trục. Không có bảng nhúng vị trí (positional embedding table).
- Bộ chiếu MLP. Loại bỏ Q-Former; sử dụng MLP 2 lớp trên các token patch đã gộp.
- Video với dynamic FPS. Video được lấy mẫu ở 1-2 FPS theo mặc định, nhưng mô hình chấp nhận số lượng khung hình tùy ý.

Kết quả: Qwen2-VL-7B ngang bằng với GPT-4o trên một số benchmark đa phương thức và đánh bại nó trên DocVQA (94.5 so với 88.4). Thay đổi kiến trúc là bước đi quyết định.

### Qwen2.5-VL (Tháng 2 năm 2025) — dynamic FPS + thời gian tuyệt đối

Bước chuyển lớn của Qwen2.5-VL là video. Dynamic FPS không chỉ là "lấy mẫu nhiều khung hình hơn khi cần". Bài báo đã chính thức hóa:

- Token thời gian tuyệt đối. Thay vì chỉ số vị trí (khung hình 0, 1, 2...), hãy sử dụng dấu thời gian thực tế. "Tại 0:04, con mèo nhảy." Mô hình nhìn thấy các token `<time>0.04</time>` xen kẽ với các token khung hình.
- Dynamic FPS. Lấy mẫu ở 1 FPS cho cảnh quay chậm, 4+ FPS cho hành động. Người dùng hoặc người huấn luyện chọn; M-RoPE thích ứng.
- Window attention trong ViT. Attention không gian được chia cửa sổ (cục bộ trong các khối) để đạt thông lượng; attention toàn cục mỗi vài lớp.
- Định dạng đầu ra JSON rõ ràng. Được huấn luyện trên dữ liệu gọi công cụ: "{\"tool\": \"click\", \"coords\": [380, 220]}". Sẵn sàng cho tác nhân ngay khi xuất xưởng.
- MRoPE-v2 scaling. Các vị trí được mở rộng theo kích thước đầu vào tối đa để video 10 phút không bị vượt quá phạm vi tần số.

Benchmarks: Qwen2.5-VL-72B đánh bại GPT-4o trên hầu hết các benchmark video, ngang bằng với Gemini 2.0 trên tài liệu, và thiết lập SOTA mô hình mở cho GUI grounding (ScreenSpot: độ chính xác 84% so với 38% của GPT-4o).

### Qwen3-VL (Tháng 11 năm 2025)

Qwen3-VL là một bản nâng cấp gia tăng nhằm củng cố thay vì tái tạo: xương sống LLM lớn hơn (Qwen3-72B), dữ liệu huấn luyện mở rộng, OCR cải tiến, suy luận mạnh mẽ hơn thông qua "chế độ suy nghĩ" của Qwen3. ViT và M-RoPE vẫn được giữ nguyên. Bài báo tập trung vào các cải tiến về dữ liệu và huấn luyện hơn là kiến trúc.

Bài học từ dòng họ này: đến năm 2025, kiến trúc Qwen-VL đã ổn định. Các thế hệ bổ sung mở rộng quy mô tính toán và dữ liệu, không phải các nguyên hàm.

### M-RoPE về mặt toán học

RoPE cổ điển xoay một truy vấn `q` có chiều `d` theo vị trí `m` sử dụng các tọa độ ghép cặp:

```
q_rot[2i]   = q[2i]   * cos(m * theta_i) - q[2i+1] * sin(m * theta_i)
q_rot[2i+1] = q[2i]   * sin(m * theta_i) + q[2i+1] * cos(m * theta_i)
theta_i     = 10000^(-2i/d)
```

M-RoPE chia chiều ẩn thành ba dải. Giả sử `d = 96`. Gán 32 chiều cho thời gian, 32 cho chiều cao, 32 cho chiều rộng. Mỗi dải xoay theo vị trí trục riêng của nó. Một patch tại (t=5, h=10, w=20) nhận các phép quay `R_t(5)`, `R_h(10)`, `R_w(20)` áp dụng cho ba dải của nó.

Các token văn bản sử dụng `t = text_index, h = 0, w = 0` (hoặc một lựa chọn chuẩn hóa), giữ tính tương thích. Các khung hình video sử dụng `t = frame_time, h = row, w = col`. Ảnh đơn sử dụng `t = 0`.

Lợi ích: một mã hóa vị trí xử lý văn bản, hình ảnh và video mà không cần mã phân nhánh hoặc các bảng vị trí khác nhau.

### Logic lấy mẫu Dynamic-FPS

Với một video có thời lượng `T` giây và ngân sách token mục tiêu `B`:

1. Tính toán FPS tối đa bạn có thể chi trả: `fps_max = B / (T * tokens_per_frame)`.
2. Chọn một FPS mục tiêu từ `{1, 2, 4, 8}` thỏa mãn `fps <= fps_max`.
3. Nếu chuyển động cao (heuristic dòng quang học hoặc yêu cầu rõ ràng từ người dùng), hãy chọn FPS cao hơn. Nếu chuyển động thấp, hãy chọn thấp hơn.
4. Lấy mẫu đồng nhất tại FPS đã chọn; chèn các token `<time>t</time>` giữa các khung hình.

Qwen2.5-VL huấn luyện logic này một cách ngầm định; tại thời điểm suy luận, người dùng kiểm soát thông qua tham số `fps`. Một chuỗi hành động 60 giây ở 4 FPS với 81 token mỗi khung hình = 19440 token, có thể quản lý được trong ngữ cảnh 32k.

### Đầu ra tác nhân có cấu trúc

Huấn luyện tác nhân của Qwen2.5-VL nhắm mục tiêu rõ ràng vào các lệnh gọi công cụ có cấu trúc:

```
{
  "tool": "mouse_click",
  "coords": [1024, 512],
  "button": "left",
  "modifier": null
}
```

Phân tích cú pháp là xác định: JSON.parse trên đầu ra của mô hình. So sánh với "click at (1024, 512)" dạng tự do vốn yêu cầu regex và xử lý sự mơ hồ. Sự thay đổi này là lý do tại sao điểm số ScreenSpot của Qwen2.5-VL tăng từ 55% của Qwen2-VL lên 84%.

```figure
mm-mrope-axes
```

## Sử dụng

`code/main.py` triển khai:

- Tính toán vị trí M-RoPE cho một chuỗi đóng gói trộn lẫn văn bản, patch hình ảnh và khung hình video.
- Bộ lấy mẫu Dynamic-FPS: với (thời lượng, ngân sách, mức độ chuyển động), chọn FPS và phát ra dấu thời gian khung hình.
- Một trình phân tích cú pháp JSON-output Qwen2.5-VL mẫu xử lý các phản hồi gọi công cụ với các trường tọa độ.

Hãy chạy nó, sau đó cảm nhận sự khác biệt khi bạn hoán đổi FPS cố định sang dynamic-FPS trên một video dài 5 phút.

## Triển khai

Bài học này tạo ra `outputs/skill-qwen-vl-pipeline-designer.md`. Với một tác vụ video (giám sát, tác nhân, nhận dạng hành động, khả năng truy cập), nó phát ra cấu hình Qwen2.5-VL (ngân sách khung hình, chiến lược FPS, cờ window-attention, chế độ đầu ra tác nhân) và ước tính độ trễ. Sử dụng điều này bất cứ khi nào bạn triển khai mô hình họ Qwen-VL cho một sản phẩm video.

## Bài tập

1. Tính toán các phép quay M-RoPE cho một patch tại (t=3, h=5, w=7) với chiều ẩn 48 (16 mỗi dải, theta cơ sở 10000). Hiển thị các góc quay cho ba cặp đầu tiên trong mỗi dải.

2. Một bản ghi camera an ninh 10 phút ở 1 FPS tạo ra bao nhiêu khung hình? Ở độ phân giải 384 với pool 3x, tổng cộng bao nhiêu token? Ngữ cảnh 32k mặc định của Qwen2.5-VL có xử lý được không?

3. Chọn FPS cho một trận tennis 30 giây so với một video demo công thức nấu ăn 30 giây so với một bản ghi tác nhân UI 30 giây. Biện minh cho từng cái bằng logic dynamic-FPS.

4. Qwen2.5-VL loại bỏ hoàn toàn Q-Former. Tại sao một MLP đơn giản lại hoạt động vào năm 2025 nhưng không phải năm 2023? (Gợi ý: quy mô dữ liệu và chất lượng bộ mã hóa.)

5. Phân tích ba đầu ra gọi công cụ JSON của Qwen2.5-VL thành các dict Python. Điều gì xảy ra với JSON bị lỗi và chiến lược phục hồi nào mà sách hướng dẫn Qwen khuyến nghị?

## Thuật ngữ chính

| Thuật ngữ | Mọi người nói gì | Ý nghĩa thực sự |
|------|-----------------|------------------------|
| M-RoPE | "Multimodal RoPE" | Nhúng vị trí xoay 3D với các dải thời gian, chiều cao và chiều rộng trong chiều ẩn |
| Dynamic FPS | "Lấy mẫu thông minh" | Tốc độ lấy mẫu khung hình được chọn cho mỗi video dựa trên chuyển động, thời lượng và ngân sách token |
| Absolute time token | "Token dấu thời gian" | `<time>t</time>` xen kẽ trong chuỗi để mô hình nhìn thấy giây thực tế thay vì chỉ số khung hình |
| Window attention | "Attention cục bộ" | Self-attention không gian bị giới hạn trong các cửa sổ nhỏ để tăng tốc; attention toàn cục được thêm định kỳ |
| Structured agent output | "Chế độ JSON" | Giám sát dữ liệu huấn luyện dạy VLM phát ra JSON có thể phân tích cú pháp với tọa độ và tên công cụ |
| min_pixels / max_pixels | "Giới hạn độ phân giải" | Các điều khiển Qwen2.5-VL theo yêu cầu giới hạn tổng số pixel và do đó là số lượng token |
| Grounding | "Chỉ vào nó" | Xuất tọa độ bounding-box dưới dạng token văn bản; được sử dụng từ Qwen-VL v1 |

## Đọc thêm

- [Bai et al. — Qwen-VL (arXiv:2308.12966)](https://arxiv.org/abs/2308.12966)
- [Wang et al. — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)
- [Qwen Team — Qwen2.5-VL Technical Report (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
- [Qwen Team — Qwen3-VL (arXiv:2511.21631)](https://arxiv.org/abs/2511.21631)
- [Zhu et al. — InternVL3 (arXiv:2504.10479)](https://arxiv.org/abs/2504.10479)