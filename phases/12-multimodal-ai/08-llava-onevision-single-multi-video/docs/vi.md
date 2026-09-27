# LLaVA-OneVision: Single-Image, Multi-Image, Video trong một mô hình duy nhất

> Trước khi LLaVA-OneVision (Li và cộng sự, tháng 8 năm 2024) ra đời, thế giới open-VLM tồn tại các dòng mô hình riêng biệt: LLaVA-1.5 cho ảnh đơn, các mô hình đa ảnh như Mantis và VILA, và các mô hình video như Video-LLaVA và Video-LLaMA. Mỗi mô hình đều dẫn đầu trong các benchmark của riêng mình nhưng lại thất bại ở các lĩnh vực khác. LLaVA-OneVision lập luận rằng một chương trình đào tạo (curriculum) duy nhất có thể huấn luyện một mô hình thống trị cả ba kịch bản, và các hiệu ứng chuyển giao kỹ năng (task-transfer) mới nổi (kỹ năng ảnh đơn được chuyển sang video, khả năng suy luận đa ảnh được chuyển sang ảnh đơn) mang lại hiệu quả vượt trội so với tổng các mô hình chuyên biệt. Công thức này đơn giản đến bất ngờ: một ngân sách visual-token cố định trên mọi kịch bản, cộng với một chương trình đào tạo rõ ràng đi từ ảnh đơn sang OneVision (đa ảnh) rồi đến video. Bài học này sẽ phân tích về ngân sách, chương trình đào tạo và các hành vi mới nổi này.

**Type:** Build
**Languages:** Python (stdlib, token budget solver + curriculum planner)
**Prerequisites:** Phase 12 · 05 (LLaVA), Phase 12 · 06 (any-resolution)
**Time:** ~180 phút

## Mục tiêu học tập

- Thiết kế ngân sách visual-token giữ nguyên giá trị trên các đầu vào ảnh đơn, đa ảnh và video.
- Sắp xếp chương trình đào tạo để chuyển giao kỹ năng từ ảnh đơn sang video mà không gây ra hiện tượng quên lãng thảm họa (catastrophic forgetting).
- Giải thích lý do tại sao một mô hình duy nhất lại vượt trội hơn các mô hình chuyên biệt ở cùng số lượng tham số khi chương trình đào tạo được thực hiện đúng cách.
- Kể tên ba khả năng mới nổi được báo cáo bởi LLaVA-OneVision: suy luận đa camera, prompting bằng set-of-mark, và tác nhân (agent) đọc ảnh chụp màn hình iPhone.

## Vấn đề

Ảnh đơn, đa ảnh và video đều gây áp lực lên mô hình theo những cách khác nhau.

Ảnh đơn đòi hỏi các token độ phân giải cao (AnyRes, ~2880 visual tokens) để nắm bắt OCR và chi tiết tinh vi. Ngân sách mỗi mẫu: một ảnh, 2880 tokens.

Đa ảnh đòi hỏi nhiều ảnh ở độ phân giải trung bình (~576 tokens mỗi ảnh) để việc suy luận giữa các ảnh nằm gọn trong ngữ cảnh. Ngân sách mỗi mẫu: 4-8 ảnh, 576 tokens mỗi ảnh, tổng 2300-4600 tokens.

Video đòi hỏi nhiều khung hình ở độ phân giải thấp (~196 tokens mỗi khung hình sau khi pooling) để nắm bắt động lực học theo thời gian. Ngân sách mỗi mẫu: 8-32 khung hình, 196 tokens mỗi khung hình, tổng 1600-6200 tokens.

Nếu huấn luyện các mô hình riêng biệt, bạn phải chọn một ngân sách. Nếu huấn luyện một mô hình duy nhất, bạn cần ngân sách đó phải mở rộng hợp lý qua các kịch bản mà không làm tràn ngữ cảnh.

Trước OneVision, câu trả lời mặc định là "huấn luyện một kịch bản, bỏ qua các kịch bản khác". Video-LLaVA đã tích hợp video vào mô hình ảnh với các giai đoạn huấn luyện bổ sung. LLaVA-NeXT thêm hỗ trợ đa ảnh bằng cách chia ô (tiling). Không có mô hình nào xử lý cả ba một cách gọn gàng.

## Khái niệm

### Ngân sách token OneVision

LLaVA-OneVision chọn ngân sách visual-token thống nhất khoảng 3000-4000 tokens mỗi mẫu, được phân bổ khác nhau tùy theo kịch bản:

- Ảnh đơn: AnyRes-9 (3x3 ô + ảnh thu nhỏ), mỗi ô ở độ phân giải 384 với 729 patches, sử dụng bilinear pooling mạnh 2x2 → 182 tokens mỗi ô. Tổng: 9 * 182 + 182 = 1820 tokens. Hoặc AnyRes-4 với 729 tokens mỗi ô = 2916 + 729.
- Đa ảnh: mỗi ảnh ở độ phân giải trung bình (384, không chia ô), 729 tokens không pooling. Ngân sách 6 ảnh → 4374 tokens.
- Video: 32 khung hình ở độ phân giải 384 với bilinear pool 3x3 mạnh → 81 tokens mỗi khung hình. Tổng: 32 * 81 = 2592 tokens.

Việc phân bổ này duy trì tổng số token gần như không đổi. LLM không bao giờ gặp phải batch làm tràn ngữ cảnh. Bộ mã hóa (encoder) tạo ra hình học khác nhau tùy theo kịch bản, nhưng LLM tiêu thụ cùng một ngân sách.

### Chương trình đào tạo ba giai đoạn

LLaVA-OneVision huấn luyện qua ba giai đoạn:

1. SFT ảnh đơn (giai đoạn SI). Tất cả dữ liệu là ảnh đơn kèm văn bản. Huấn luyện trên đầu vào AnyRes độ phân giải cao. Điều này dạy cho mô hình về nhận thức, OCR và hiểu biết chi tiết. Sử dụng dữ liệu LLaVA-NeXT cộng với dữ liệu ảnh đơn đặc thù của OneVision.
2. SFT OneVision (giai đoạn OV). Trộn ảnh đơn + đa ảnh + video (lấy mẫu khung hình đồng nhất). Huấn luyện trên ngân sách token thống nhất. Điều này dạy mô hình xử lý các hình dạng batch không đồng nhất. Không reset trọng số — tiếp tục từ giai đoạn SI.
3. Chuyển giao tác vụ (giai đoạn TT). Tiếp tục với hỗn hợp tác vụ mục tiêu, thường tập trung nhiều hơn vào đa ảnh hoặc video tùy thuộc vào sản phẩm. Tinh chỉnh tùy chọn để triển khai.

Quan trọng: thứ tự chương trình đào tạo rất quan trọng. Huấn luyện video trước hoặc đa ảnh trước sẽ tạo ra hiệu suất ảnh kém hơn so với huấn luyện ảnh đơn trước, ngay cả với cùng một dữ liệu. Bài báo đã thực hiện ablation study về điều này một cách rõ ràng.

### Tại sao chương trình đào tạo lại hiệu quả

Huấn luyện ảnh đơn xây dựng nền tảng nhận thức. Các patch token mang các đặc trưng thị giác chi tiết; LLM học cách tích hợp chúng với văn bản. Đa ảnh và video giới thiệu các thách thức về cấu trúc (ảnh nào là ảnh nào, điều gì xảy ra trước) vốn khó học nếu không có nền tảng nhận thức vững chắc.

Nếu bạn huấn luyện tất cả các kịch bản từ đầu cùng nhau, mô hình sẽ thiếu hụt về nhận thức (dữ liệu ảnh đơn hạn chế mỗi batch) và quá khớp (overfit) về cấu trúc (nhiều dữ liệu đa ảnh/video). Kết quả: một mô hình tuân theo các mẫu suy luận giữa các ảnh nhưng lại nông cạn về thị giác.

Thứ tự chương trình đào tạo mang lại sức mạnh nhận thức từ giai đoạn SI, sau đó là suy luận cấu trúc/thời gian từ giai đoạn OV, mà không làm mất đi bất kỳ kỹ năng nào.

### Các kỹ năng mới nổi giữa các kịch bản

Bài báo LLaVA-OneVision báo cáo ba khả năng mới nổi:

1. Suy luận đa camera. Được huấn luyện trên đa ảnh + video riêng biệt; khi suy luận, mô hình được yêu cầu suy luận về một cảnh lái xe đa camera. Mô hình tích hợp chính xác các góc nhìn mặc dù chưa bao giờ thấy định dạng chính xác đó trong quá trình huấn luyện.
2. Prompting bằng set-of-mark. Người dùng chú thích các đối tượng trong ảnh bằng các dấu số; mô hình suy luận về "dấu số 3 đang làm gì so với dấu số 7". Không được huấn luyện trên các dấu số hay chú thích; học được từ sự kết hợp giữa định vị không gian + tham chiếu đa ảnh.
3. Tác nhân (agent) đọc ảnh chụp màn hình iPhone. Người dùng cung cấp ảnh chụp màn hình iPhone và yêu cầu lập kế hoạch cho lần nhấp tiếp theo. Được huấn luyện trên ảnh chụp màn hình UI, video quy trình làm việc của người dùng và các cặp ảnh trước/sau. Khả năng tổng quát hóa sang trường hợp sử dụng tác nhân.

Đây không phải là các tác vụ được huấn luyện; chúng nảy sinh từ cấu trúc thành phần của chương trình đào tạo.

### Visual-token pooling

Ngân sách token đòi hỏi phải có pooling. OneVision sử dụng nội suy song tuyến (bilinear interpolation) trên lưới patch 2D: 24x24 = 576 patches trở thành 12x12 = 144 (hệ số 2x) hoặc 8x8 = 64 (hệ số 3x). Pooling được thực hiện trong không gian lưới patch, không phải không gian token, để bảo toàn tính cục bộ.

Việc chọn hệ số pooling cho mỗi kịch bản bản thân nó là một siêu tham số. Pooling ít hơn = nhiều token hơn = biểu diễn phong phú hơn. Pooling nhiều hơn = ít token hơn = chứa được nhiều khung hình/ảnh hơn.

### LLaVA-OneVision-1.5

Phiên bản tiếp theo năm 2025 (LLaVA-OneVision-1.5, arXiv 2509.23661) là "hoàn toàn mở" về dữ liệu huấn luyện, trọng số mô hình và mã nguồn. Nó thu hẹp khoảng cách với các mô hình độc quyền trên một số benchmark và dân chủ hóa công thức này. Cùng chương trình đào tạo, nhiều dữ liệu hơn, LLM nền tảng tốt hơn. Không thay đổi kiến trúc.

### Đối chiếu với Qwen2.5-VL

Qwen2.5-VL (Bài học 12.09) đưa ra các lựa chọn khác. Nó sử dụng M-RoPE và FPS động thay vì pooling cố định. Ngân sách của nó mở rộng theo đầu vào — một video 1 phút sử dụng nhiều token hơn video 5 giây. LLaVA-OneVision cố định ngân sách và thay đổi hệ số pooling. Cả hai đều hiệu quả; chúng đánh đổi khả năng cấu hình lấy khả năng dự đoán.

```figure
l5-onevision-budget
```

## Sử dụng

`code/main.py` là một công cụ lập kế hoạch chương trình đào tạo và ngân sách cho VLM kiểu OneVision. Với ngân sách token mỗi mẫu và hỗn hợp kịch bản mục tiêu (ví dụ: 40% ảnh đơn, 30% đa ảnh, 30% video), nó:

- Phân bổ độ phân giải, hệ số pooling và số khung hình mỗi kịch bản.
- Kiểm tra xem mọi kịch bản có nằm trong ngân sách chia sẻ hay không.
- Báo cáo số lượng token dự kiến, LLM FLOPs và kịch bản nào đang bị thiếu token.
- In ra lịch trình huấn luyện từng giai đoạn.

Sử dụng nó để lập kế hoạch tinh chỉnh OneVision hoặc kiểm tra chi phí mỗi yêu cầu của việc triển khai VLM.

## Triển khai

Bài học này tạo ra `outputs/skill-onevision-budget-planner.md`. Với phân phối tác vụ mục tiêu và ngân sách mỗi mẫu, nó xuất ra hệ số AnyRes, pooling mỗi khung hình, số khung hình video và trọng số giai đoạn chương trình đào tạo. Sử dụng công cụ này bất cứ khi nào bạn huấn luyện hoặc tinh chỉnh một VLM kịch bản thống nhất.

## Bài tập

1. Sản phẩm của bạn hỗ trợ 80% ảnh đơn, 10% đa ảnh (2-4 ảnh), 10% video (8-16 khung hình). Hãy thiết kế ngân sách token. Bạn sẽ sử dụng ngân sách dư thừa từ việc không thực hiện đa ảnh nặng ở đâu?

2. Đọc LLaVA-OneVision Phần 4.3 (các khả năng mới nổi). Đề xuất một kỹ năng mới nổi thứ tư mà chương trình đào tạo có khả năng mở khóa nhưng bài báo chưa báo cáo.

3. Đảo ngược thứ tự chương trình đào tạo — huấn luyện đa ảnh trước, sau đó là ảnh đơn, rồi đến video. Dự đoán benchmark nào sẽ suy giảm và tại sao.

4. Bài báo báo cáo các benchmark video được huấn luyện chỉ với 8 khung hình mỗi mẫu. Điều đó có tổng quát hóa cho các video 30 giây khi suy luận không? Điều gì sẽ hỏng trước — ngân sách token hay suy luận thời gian?

5. Bilinear pooling từ 24x24 patches xuống 12x12 là giảm 4x mỗi chiều. Hãy triển khai pooling bằng Python stdlib và xác minh rằng giá trị trung bình trên mỗi khối 2x2 khớp với đầu ra bilinear.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|------------------------|
| OneVision scenario | "Ảnh đơn, đa ảnh hoặc video" | Một trong ba hình dạng đầu vào mà VLM thống nhất xử lý; ngân sách giữ nguyên |
| Token budget | "Bao nhiêu token mỗi mẫu" | Tổng số visual token mà LLM thấy mỗi mẫu huấn luyện/suy luận, thường là 3000-4000 |
| Curriculum | "Thứ tự huấn luyện" | Thứ tự giai đoạn (ảnh đơn → đa ảnh → video) được chọn để chuyển giao kỹ năng |
| Bilinear pooling | "Thu nhỏ token" | Áp dụng nội suy song tuyến lên lưới patch (2D) để giảm số lượng token trong khi bảo toàn tính cục bộ |
| Emergent skill | "Không được huấn luyện, vẫn hoạt động" | Khả năng xuất hiện khi suy luận mà không cần dữ liệu huấn luyện tương ứng, nhờ cấu trúc chương trình đào tạo |
| AnyRes-k | "Thiết lập k-tile" | k ô con có độ phân giải cố định cộng với một ảnh thu nhỏ, thường k ∈ {4, 9} |
| Task transfer | "Tổng quát hóa giữa các kịch bản" | Các kỹ năng học được trên ảnh đơn áp dụng cho video (và ngược lại) thông qua backbone chia sẻ |

## Đọc thêm

- [Li và cộng sự — LLaVA-OneVision (arXiv:2408.03326)](https://arxiv.org/abs/2408.03326)
- [LLaVA-OneVision-1.5: Fully Open Framework (arXiv:2509.23661)](https://arxiv.org/abs/2509.23661)
- [Lin và cộng sự — Video-LLaVA (arXiv:2311.10122)](https://arxiv.org/abs/2311.10122)
- [Lin và cộng sự — VILA (arXiv:2312.07533)](https://arxiv.org/abs/2312.07533)
- [Wang và cộng sự — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)