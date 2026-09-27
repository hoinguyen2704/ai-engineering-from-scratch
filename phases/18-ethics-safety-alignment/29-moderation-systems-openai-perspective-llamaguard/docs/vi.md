# Hệ thống kiểm duyệt — OpenAI, Perspective, Llama Guard

> Các hệ thống kiểm duyệt trong môi trường production giúp hiện thực hóa các chính sách an toàn đã được định nghĩa trong Bài 12-16. OpenAI Moderation API: `omni-moderation-latest` (2024) được xây dựng trên GPT-4o, phân loại văn bản + hình ảnh trong một lần gọi; hiệu quả hơn 42% trên tập kiểm thử đa ngôn ngữ so với phiên bản trước; schema phản hồi trả về 13 boolean danh mục — harassment, harassment/threatening, hate, hate/threatening, illicit, illicit/violent, self-harm, self-harm/intent, self-harm/instructions, sexual, sexual/minors, violence, violence/graphic; miễn phí cho hầu hết các nhà phát triển. Các mô hình phân lớp: Kiểm duyệt đầu vào (trước khi tạo), Kiểm duyệt đầu ra (sau khi tạo), Kiểm duyệt tùy chỉnh (quy tắc miền). Các lệnh gọi bất đồng bộ song song giúp ẩn độ trễ; phản hồi giữ chỗ (placeholder) khi bị gắn cờ. Llama Guard 3/4 (Bài 16): 14 mối nguy hại theo MLCommons, Code Interpreter Abuse, 8 ngôn ngữ (v3), đa hình ảnh (v4). Perspective API (Google Jigsaw): hệ thống chấm điểm độc hại có từ trước làn sóng LLM-as-moderator; chủ yếu là độc hại đơn chiều với các biến thể severe-toxicity/insult/profanity; là baseline cho nghiên cứu kiểm duyệt nội dung. Các tính năng bị loại bỏ: Azure Content Moderator bị ngừng hỗ trợ từ tháng 2/2024, nghỉ hưu vào tháng 2/2027, được thay thế bởi Azure AI Content Safety.

**Type:** Build
**Languages:** Python (stdlib, three-layer moderation harness)
**Prerequisites:** Phase 18 · 16 (Llama Guard / Garak / PyRIT)
**Time:** ~60 phút

## Mục tiêu học tập

- Mô tả hệ thống phân loại danh mục của OpenAI Moderation API và sự khác biệt so với tập MLCommons của Llama Guard 3.
- Mô tả mô hình kiểm duyệt ba lớp (đầu vào, đầu ra, tùy chỉnh) và nêu tên một chế độ lỗi (failure mode) của mỗi lớp.
- Mô tả vị thế của Perspective API như một baseline từ thời kỳ tiền LLM và lý do tại sao nó vẫn được sử dụng trong nghiên cứu.
- Nêu mốc thời gian ngừng hỗ trợ của Azure.

## Vấn đề

Bài 12-16 mô tả các cuộc tấn công và công cụ phòng thủ. Bài 29 đề cập đến các hệ thống kiểm duyệt đã triển khai giúp hiện thực hóa các biện pháp phòng thủ tại điểm tiếp xúc giữa người dùng và sản phẩm. Mô hình ba lớp là cấu hình mặc định cho năm 2026.

## Khái niệm

### OpenAI Moderation API

`omni-moderation-latest` (2024). Được xây dựng trên GPT-4o. Phân loại văn bản + hình ảnh trong một lần gọi. Miễn phí cho hầu hết các nhà phát triển.

Các danh mục (13 boolean trong schema phản hồi):
- harassment, harassment/threatening
- hate, hate/threatening
- self-harm, self-harm/intent, self-harm/instructions
- sexual, sexual/minors
- violence, violence/graphic
- illicit, illicit/violent

Hỗ trợ đa phương thức áp dụng cho `violence`, `self-harm`, và `sexual` nhưng không áp dụng cho `sexual/minors`; các danh mục còn lại chỉ dành cho văn bản.

Đối với bộ công cụ mã nguồn trong `code/main.py`, chúng ta gộp các danh mục con `/threatening`, `/intent`, `/instructions`, và `/graphic` vào các danh mục cha cấp cao nhất để đơn giản hóa việc giảng dạy. Mã nguồn production nên sử dụng schema đầy đủ 13 danh mục.

Hiệu quả hơn 42% trên tập kiểm thử đa ngôn ngữ so với endpoint kiểm duyệt thế hệ trước. Điểm số theo từng danh mục; các ứng dụng sẽ thiết lập ngưỡng (threshold).

### Llama Guard 3/4

Được đề cập trong Bài 16. 14 danh mục mối nguy hại MLCommons (được tổ chức khác với 13 boolean trong schema phản hồi của OpenAI). Hỗ trợ 8 ngôn ngữ (v3). Llama Guard 4 (tháng 4/2025) là mô hình đa phương thức gốc, 12B.

Hệ thống phân loại của OpenAI và Llama Guard có sự chồng lấp nhưng cũng có sự khác biệt. OpenAI có "illicit" là một danh mục rộng; Llama Guard tách biệt "violent crimes" và "non-violent crimes". Các triển khai sẽ lựa chọn dựa trên sự phù hợp với hệ thống chính sách của họ.

### Perspective API (Google Jigsaw)

Hệ thống chấm điểm độc hại có từ trước làn sóng LLM-as-moderator (trước 2020). Các danh mục: TOXICITY, SEVERE_TOXICITY, INSULT, PROFANITY, THREAT, IDENTITY_ATTACK. Điểm số chính đơn chiều (TOXICITY) với các biến thể danh mục con.

Được sử dụng rộng rãi như một baseline nghiên cứu kiểm duyệt nội dung vì API ổn định, có tài liệu đầy đủ và có dữ liệu hiệu chuẩn qua nhiều năm. Đối với các trường hợp sử dụng hiện đại liên quan đến LLM, Llama Guard hoặc OpenAI Moderation thường là lựa chọn phù hợp hơn.

### Mô hình ba lớp

1. **Kiểm duyệt đầu vào (Input moderation).** Phân loại prompt của người dùng trước khi tạo. Từ chối nếu bị gắn cờ. Độ trễ: một lần gọi bộ phân loại.
2. **Kiểm duyệt đầu ra (Output moderation).** Phân loại đầu ra của mô hình trước khi gửi đến người dùng. Thay thế bằng phản hồi từ chối nếu bị gắn cờ. Độ trễ: một lần gọi bộ phân loại sau khi tạo.
3. **Kiểm duyệt tùy chỉnh (Custom moderation).** Các quy tắc đặc thù cho miền (regex, allowlist, chính sách kinh doanh). Chạy ở đầu vào hoặc đầu ra.

Ba lớp được thiết kế theo trình tự: kiểm duyệt đầu vào phải hoàn tất trước khi tạo, và kiểm duyệt đầu ra chạy sau khi tạo. Tính song song áp dụng trong cùng một lớp — chạy nhiều bộ phân loại (ví dụ: OpenAI Moderation + Llama Guard + Perspective) đồng thời trên cùng một văn bản giúp ẩn độ trễ của từng bộ phân loại. Như một tối ưu hóa tùy chọn, phản hồi giữ chỗ ("vui lòng chờ trong giây lát...") có thể được hiển thị trong khi kiểm duyệt đầu vào hoàn tất và việc streaming token-1 bị trì hoãn. Hành vi khi bị gắn cờ có thể cấu hình: từ chối, làm sạch (sanitize), hoặc chuyển cho con người xem xét.

### Các chế độ lỗi (Failure modes)

- **Chỉ kiểm duyệt đầu vào.** Không bắt được các trường hợp hallucination ở đầu ra (các cuộc tấn công mã hóa Bài 12-14 có thể vượt qua bộ phân loại đầu vào).
- **Chỉ kiểm duyệt đầu ra.** Cho phép mọi đầu vào tiếp cận mô hình; tăng chi phí; làm lộ suy luận nội bộ cho kẻ tấn công.
- **Chỉ kiểm duyệt tùy chỉnh.** Không mạnh mẽ trên các danh mục; regex thường rất dễ bị phá vỡ.

Phân lớp là cấu hình mặc định. "Thắt lưng buộc bụng" (Belt-and-suspenders).

### Ngừng hỗ trợ Azure

Azure Content Moderator: ngừng hỗ trợ từ tháng 2/2024, nghỉ hưu vào tháng 2/2027. Được thay thế bởi Azure AI Content Safety, dựa trên LLM và tích hợp với Azure OpenAI. Việc di chuyển là một dự án cấp thực địa từ 2024-2027 cho các triển khai Azure.

### Vị trí trong Giai đoạn 18

Bài 16 bao gồm các công cụ kiểm duyệt trong bối cảnh red-team. Bài 29 bao gồm kiểm duyệt đã triển khai. Bài 30 kết thúc với bằng chứng về khả năng sử dụng kép (dual-use) hiện tại.

```figure
an-moderation-layers
```

## Sử dụng

`code/main.py` xây dựng một bộ công cụ kiểm duyệt ba lớp: bộ kiểm duyệt đầu vào (từ khóa + điểm danh mục), bộ kiểm duyệt đầu ra (cùng bộ phân loại trên đầu ra), bộ kiểm duyệt tùy chỉnh (quy tắc miền). Bạn có thể chạy các đầu vào qua và quan sát lớp nào bắt được nội dung gì.

## Triển khai

Bài học này tạo ra `outputs/skill-moderation-stack.md`. Với một triển khai cụ thể, nó đề xuất cấu hình ngăn xếp kiểm duyệt: bộ phân loại nào ở đầu vào, bộ nào ở đầu ra, quy tắc tùy chỉnh nào, và bộ đánh giá (judge) nào cho các trường hợp biên.

## Bài tập

1. Chạy `code/main.py`. Chạy một đầu vào lành tính, biên (borderline), và độc hại qua cả ba lớp. Báo cáo lớp nào kích hoạt cho từng trường hợp.

2. Mở rộng bộ công cụ với chấm điểm độc hại theo phong cách Perspective-API trên một danh mục cụ thể. So sánh hành vi ngưỡng của nó với điểm danh mục.

3. Đọc tài liệu OpenAI Moderation API và danh sách danh mục Llama Guard 3. Ánh xạ từng danh mục OpenAI sang các danh mục Llama Guard gần nhất. Xác định ba danh mục không ánh xạ rõ ràng.

4. Thiết kế một ngăn xếp kiểm duyệt cho triển khai trợ lý lập trình (ví dụ: GitHub Copilot). Xác định các danh mục liên quan nhất và ít liên quan nhất, đồng thời đề xuất các quy tắc tùy chỉnh.

5. Azure Content Moderator nghỉ hưu vào tháng 2/2027. Lập kế hoạch di chuyển sang Azure AI Content Safety. Xác định yếu tố rủi ro cao nhất của quá trình di chuyển.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|------------------------|
| OpenAI Moderation | "omni-moderation-latest" | Bộ phân loại 13 danh mục (văn bản) dựa trên GPT-4o với hỗ trợ đa phương thức một phần |
| Perspective API | "Google Jigsaw toxicity" | Baseline chấm điểm độc hại thời kỳ tiền LLM |
| Llama Guard | "MLCommons 14-category" | Bộ phân loại mối nguy hại của Meta (v3: 8B văn bản, 8 ngôn ngữ; v4: 12B đa phương thức) |
| Input moderation | "pre-generation filter" | Bộ phân loại trên prompt người dùng trước khi gọi mô hình |
| Output moderation | "post-generation filter" | Bộ phân loại trên đầu ra của mô hình trước khi gửi đến người dùng |
| Custom moderation | "domain rules" | Các quy tắc đặc thù cho triển khai (regex, allowlist, chính sách) |
| Layered moderation | "all three layers" | Mô hình triển khai production tiêu chuẩn |

## Đọc thêm

- [Tài liệu OpenAI Moderation API](https://platform.openai.com/docs/api-reference/moderations) — omni-moderation endpoint
- [Meta PurpleLlama + Llama Guard](https://github.com/meta-llama/PurpleLlama) — Llama Guard repo
- [Google Jigsaw Perspective API](https://perspectiveapi.com/) — chấm điểm độc hại
- [Azure AI Content Safety](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/) — thay thế cho Azure