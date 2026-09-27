# Hiểu video dài với ngữ cảnh triệu token

> Một video 4K dài 1 giờ ở tốc độ 24 FPS, sau khi được chia patch và nhúng (embed), tạo ra khoảng 60 triệu token. Một tập podcast dài 2 giờ sau khi chuyển thành văn bản (transcribe) là 30.000 token. Một bộ phim Blu-ray đầy đủ, ngay cả khi được nén bằng pooling mạnh, cũng lên tới hàng trăm nghìn token. Gemini 1.5 của Google (tháng 3 năm 2024) đã mở ra kỷ nguyên này với ngữ cảnh 10 triệu token, thực hiện việc truy xuất "kim đáy bể" (needle-in-a-haystack) đáng tin cậy trên các video dài hàng giờ. LWM (Liu và cộng sự, tháng 2 năm 2024) đã chỉ ra con đường mở rộng của ring attention. LongVILA và Video-XL đã mở rộng khả năng nạp dữ liệu hơn nữa. VideoAgent thay thế ngữ cảnh thô bằng truy xuất tác tử (agentic retrieval). Mỗi phương pháp là một sự đánh đổi khác nhau về tính toán, khả năng truy xuất và độ phức tạp kỹ thuật. Bài học này sẽ phân tích chúng song song với nhau.

**Type:** Build
**Languages:** Python (stdlib, trình mô phỏng needle-in-haystack + bộ định tuyến truy xuất tác tử)
**Prerequisites:** Phase 12 · 17 (token thời gian video)
**Time:** ~180 phút

## Mục tiêu học tập

- Tính tổng số lượng visual token cho video dài ở các mức FPS và pooling khác nhau.
- Giải thích ba con đường mở rộng: ngữ cảnh thô (Gemini 1.5), ring attention (LWM), nén token (LongVILA / Video-XL).
- So sánh các VLM video ngữ cảnh thô với VLM video truy xuất tác tử (VideoAgent) về độ chính xác và độ trễ.
- Thiết kế bài kiểm tra needle-in-a-haystack cho video 30 phút và đo lường khả năng truy xuất tại một phút cụ thể.

## Vấn đề

Một khung hình đơn lẻ với các patch kích thước Qwen2.5-VL ở độ phân giải gốc 384 là khoảng 729 token. Với pooling 3x3, con số này là 81 token mỗi khung hình. Một clip 30 phút ở 1 FPS = 1800 khung hình = 145.800 token. Các VLM mã nguồn mở năm 2025 có thể xử lý được, nhưng khá sát giới hạn. Ở 2 FPS, con số là 291.600 token — chỉ những ngữ cảnh lớn nhất mới chứa được.

Một bộ phim 2 giờ ở 1 FPS là 583k token. Vượt quá hầu hết các mô hình mở năm 2026; đòi hỏi Gemini 2.5 Pro hoặc pooling mạnh hơn nữa.

Ba con đường mở rộng đã xuất hiện.

## Khái niệm

### Con đường 1: Ngữ cảnh thô (Gemini 1.5, Claude Opus)

Dùng phần cứng để giải quyết vấn đề. Mở rộng ngữ cảnh lên hàng triệu token, xử lý mọi thứ trong một lần forward pass.

Gemini 1.5 Pro ra mắt với 1M token; Gemini 1.5 Ultra lên tới 10M; Gemini 2.5 Pro vào năm 2026 xử lý video dài hàng giờ một cách đáng tin cậy. Bài báo (arXiv:2403.05530) ghi lại khả năng truy xuất needle-in-a-haystack đạt 99,7% lên tới khoảng 9,5M token.

Kỹ thuật: triển khai attention tùy chỉnh với phân cấp bộ nhớ (cục bộ + toàn cục + thưa thớt) cộng với định tuyến chuyên gia MoE để đạt hiệu quả ngữ cảnh dài. Không được công bố chi tiết đầy đủ. Không phải mã nguồn mở.

### Con đường 2: Ring attention (LWM, LongVILA)

Ring attention phân phối các chuỗi dài trên các thiết bị trong một "vòng" (ring), nơi mỗi thiết bị giữ một phần dữ liệu. Attention trên toàn bộ chuỗi xảy ra bằng cách mỗi thiết bị gửi phần dữ liệu của mình cho thiết bị tiếp theo theo mô hình vòng tròn, tính toán attention từng phần và tổng hợp lại.

LWM (Liu và cộng sự, 2024) đã huấn luyện mô hình ngữ cảnh 1M-token theo cách này. Tính toán huấn luyện mở rộng tuyến tính theo ngữ cảnh, không phải bậc hai — tác động bậc hai của attention được phân bổ trên các thiết bị trong vòng.

LongVILA (arXiv:2408.10188) đã điều chỉnh mô hình này cho VLM. Video 1400 khung hình ở 192 token mỗi khung hình = 268k ngữ cảnh, được huấn luyện với ring attention trên 8-way parallelism.

### Con đường 3: Nén token (Video-XL, LongVA)

Rẻ hơn ngữ cảnh thô: nén mạnh trước khi LLM nhìn thấy chuỗi.

Video-XL (arXiv:2409.14485) sử dụng một token tóm tắt hình ảnh: mỗi clip gồm N khung hình tạo ra một token "tóm tắt" duy nhất chú ý (attend) trên N khung hình đó. Khi suy luận (inference), LLM chỉ thấy một token tóm tắt mỗi clip, làm giảm đáng kể ngữ cảnh.

LongVA mở rộng ngữ cảnh LLM từ 200k lên 2M với kỹ thuật "chuyển đổi ngữ cảnh dài" (long context transfer). Huấn luyện trên văn bản ngữ cảnh dài, chuyển sang video ngữ cảnh dài thông qua biểu diễn chia sẻ.

Nén token đánh đổi khả năng truy xuất tại các mốc thời gian cụ thể để lấy khả năng mở rộng. Mô hình biết chung về những gì đã xảy ra nhưng đôi khi bỏ lỡ các khung hình chính xác.

### Con đường 4: Truy xuất tác tử (VideoAgent)

Không nạp toàn bộ video vào LLM. Thay vào đó, coi video như một cơ sở dữ liệu và sử dụng LLM để truy vấn nó.

VideoAgent (arXiv:2403.10517):

1. LLM đọc câu hỏi.
2. LLM yêu cầu một công cụ truy xuất các clip liên quan ("cho tôi xem các phân đoạn có con mèo").
3. Công cụ trả về các mốc thời gian clip khớp.
4. LLM đọc các clip đó thông qua một VLM.
5. LLM soạn câu trả lời hoặc đặt các truy vấn tiếp theo.

Đây là mô hình LLM-as-agent áp dụng cho video dài. Suy luận rẻ hơn (chỉ các clip liên quan được mã hóa), kỹ thuật khó hơn (chất lượng truy xuất trở thành nút thắt cổ chai).

### Các benchmark Needle-in-a-haystack

Bài kiểm tra ngữ cảnh dài tiêu chuẩn: chèn một dấu hiệu hình ảnh hoặc văn bản độc nhất tại một điểm ngẫu nhiên trong video, sau đó đặt một truy vấn yêu cầu truy xuất nó.

Chỉ số: Recall@k theo độ dài video và vị trí dấu hiệu.

Gemini 2.5 Pro đạt điểm truy xuất >99% ở các video dài tới 90 phút. Các mô hình 72B mở (Qwen2.5-VL-72B, InternVL3-78B) đạt khoảng 85-90% ở 30 phút và giảm dần sau 60 phút.

VideoAgent có thể sánh ngang hoặc vượt qua các mô hình ngữ cảnh thô ở video dài 2 giờ trở lên vì việc truy xuất sẽ tìm thấy "cây kim" nếu công cụ đủ tốt.

### Chọn con đường nào

Đối với clip 15 phút cần độ chính xác cao nhất: ngữ cảnh gốc của mô hình 72B thường hoạt động tốt. Hãy chọn Qwen2.5-VL-72B.

Đối với nội dung từ 30 phút đến 1 giờ: LongVILA hoặc Video-XL cho mô hình mở; Gemini 2.5 Pro cho mô hình đóng. Thanh chất lượng rất quan trọng — các mô hình tiên phong (frontier) thường là mô hình đóng.

Đối với nội dung dài 2 giờ trở lên: VideoAgent hoặc các mô hình truy xuất tương tự. Ngoài ra, hãy tóm tắt thành các đoạn nhỏ hơn và nạp các bản tóm tắt phân cấp.

### Mô hình sản xuất năm 2026

Trong thực tế, các pipeline video dài trong sản xuất là mô hình lai:

1. Chạy lấy mẫu dynamic-FPS + pooling mạnh trên toàn bộ video (để có biểu diễn toàn cục 100k-token).
2. Chuyển cho VLM 72B để có bản tóm tắt toàn cục.
3. Nếu người dùng đặt câu hỏi chi tiết, hãy chạy truy xuất tác tử bằng cách sử dụng bản tóm tắt làm chỉ mục.

Cách này kết hợp ngữ cảnh thô để hiểu toàn cục và truy xuất để lấy chi tiết cục bộ.

```figure
mm-video-token-budget
```

## Sử dụng

`code/main.py`:

- Tính toán ngân sách token cho video từ 1 phút đến 3 giờ ở các mức FPS + pooling khác nhau.
- Mô phỏng một lần chạy needle-in-a-haystack: chèn dấu hiệu tại một mốc thời gian ngẫu nhiên, đặt câu hỏi, chấm điểm truy xuất.
- Bao gồm một trình mô phỏng bộ định tuyến truy xuất tác tử chọn các clip cụ thể để nạp vào VLM hạ nguồn.

Hãy chạy bảng ngân sách và cảm nhận khoảng cách về quy mô.

## Triển khai

Bài học này tạo ra `outputs/skill-long-video-strategy-planner.md`. Với thời lượng video và độ phức tạp của truy vấn, nó chọn giữa ngữ cảnh thô, nén và truy xuất tác tử, đồng thời tính toán độ trễ + kỳ vọng về chất lượng.

## Bài tập

1. Một bài giảng 45 phút ở 1 FPS, 81 token mỗi khung hình. Tổng số token là bao nhiêu? Phù hợp với ngữ cảnh của những mô hình nào?

2. Thiết kế bài kiểm tra needle-in-a-haystack: bạn chèn dấu hiệu vào phút thứ mấy và định dạng truy vấn chính xác là gì?

3. So sánh Qwen2.5-VL-72B ngữ cảnh thô (80k ngữ cảnh) với VideoAgent (Claude 3.5 + truy xuất) trên video 1 giờ. Cái nào thắng về khả năng truy xuất? Cái nào thắng về độ trễ?

4. Chi phí bộ nhớ của ring attention mở rộng tuyến tính theo độ dài chuỗi và tuyến tính theo số lượng thiết bị. Giải thích tại sao và điều gì sẽ xảy ra nếu bạn bỏ qua giai đoạn xoay vòng (ring-rotation).

5. Đọc Gemini 1.5 Phần 5 về needle-in-a-haystack. Bài báo đã phát hiện điều gì về khả năng truy xuất ở ranh giới 1M so với 10M token?

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|------------------------|
| Brute context | "Chỉ là thêm token" | Mở rộng ngữ cảnh LLM lên hàng triệu token; xử lý mọi thứ trong một lần |
| Ring attention | "Song song kiểu LWM" | Mô hình attention phân tán nơi mỗi thiết bị giữ một phần và xoay vòng |
| Token compression | "Token tóm tắt" | Giảm token mỗi clip thông qua bộ nén đã học trước khi vào LLM |
| Needle-in-haystack | "Kiểm tra NIH" | Chèn dấu hiệu độc nhất tại điểm ngẫu nhiên, yêu cầu mô hình truy xuất khi kiểm tra |
| Agentic retrieval | "LLM làm lập kế hoạch truy vấn" | LLM yêu cầu công cụ truy xuất các clip liên quan, đọc qua VLM, soạn câu trả lời |
| VideoAgent | "Mô hình truy xuất cho video" | Thiết kế truy xuất tác tử chuẩn: câu hỏi -> công cụ -> clip -> câu trả lời |

## Đọc thêm

- [Gemini Team — Gemini 1.5 (arXiv:2403.05530)](https://arxiv.org/abs/2403.05530)
- [Liu và cộng sự — LWM / RingAttention (arXiv:2402.08268)](https://arxiv.org/abs/2402.08268)
- [Xue và cộng sự — LongVILA (arXiv:2408.10188)](https://arxiv.org/abs/2408.10188)
- [Shu và cộng sự — Video-XL (arXiv:2409.14485)](https://arxiv.org/abs/2409.14485)
- [Wang và cộng sự — VideoAgent (arXiv:2403.10517)](https://arxiv.org/abs/2403.10517)