# Red-Team Tooling — Garak, Llama Guard, PyRIT

> Ba công cụ sản xuất định hình stack red-team năm 2026. Llama Guard (Meta) — một bộ phân loại Llama-3.1-8B được tinh chỉnh trên 14 danh mục nguy cơ của MLCommons; Llama Guard 4 năm 2025 là bộ phân loại đa phương thức (multimodal) 12B được tỉa từ Llama 4 Scout. Garak (NVIDIA) — trình quét lỗ hổng LLM mã nguồn mở với các probe tĩnh, động và thích ứng cho các vấn đề như hallucination, rò rỉ dữ liệu, prompt injection, độc hại và jailbreak. PyRIT (Microsoft) — các chiến dịch red-team đa lượt (multi-turn) với Crescendo, TAP và các chuỗi converter tùy chỉnh để khai thác sâu. Llama Guard 3 được ghi lại trong tài liệu "Llama 3 Herd of Models" của Meta (arXiv:2407.21783); Llama Guard 3-1B-INT4 trong arXiv:2411.17713; kiến trúc probe của Garak tại github.com/NVIDIA/garak. Những công cụ này là giao diện sản xuất năm 2026 giữa nghiên cứu red-team (Bài 12-15) và triển khai (Bài 17+).

**Type:** Build
**Languages:** Python (stdlib, trình mô phỏng kiến trúc công cụ và mock bộ phân loại kiểu Llama Guard)
**Prerequisites:** Giai đoạn 18 · 12-15 (jailbreaks và IPI)
**Time:** ~75 phút

## Mục tiêu học tập

- Mô tả vị trí của Llama Guard 3/4 trong stack an toàn: bộ phân loại đầu vào, bộ phân loại đầu ra, hoặc cả hai.
- Kể tên 14 danh mục nguy cơ của MLCommons và nêu một danh mục không hiển nhiên (Code Interpreter Abuse).
- Mô tả kiến trúc probe của Garak: probes, detectors, harnesses.
- Mô tả cấu trúc chiến dịch đa lượt của PyRIT và cách nó kết hợp với các probe của Garak.

## Vấn đề

Các bài 12-15 trình bày bề mặt tấn công. Các triển khai sản xuất cần đánh giá có thể lặp lại và mở rộng. Ba công cụ thống trị năm 2026: Llama Guard (bộ phân loại phòng thủ), Garak (trình quét), PyRIT (trình điều phối chiến dịch). Mỗi công cụ nhắm vào một lớp khác nhau của vòng đời red-team.

## Khái niệm

### Llama Guard (Meta)

Llama Guard 3 là mô hình Llama-3.1-8B được tinh chỉnh để phân loại đầu vào/đầu ra dựa trên 14 danh mục AILuminate của MLCommons:
- Tội phạm bạo lực, tội phạm phi bạo lực, liên quan đến tình dục, CSAM, phỉ báng
- Lời khuyên chuyên môn, quyền riêng tư, IP, vũ khí bừa bãi, thù ghét
- Tự tử/tự hại, nội dung tình dục, bầu cử, lạm dụng trình thông dịch mã (code-interpreter abuse)

Hỗ trợ 8 ngôn ngữ. Cách sử dụng: đặt trước LLM (kiểm duyệt đầu vào), sau LLM (kiểm duyệt đầu ra), hoặc cả hai. Hai cách sử dụng này tạo ra các phân phối huấn luyện khác nhau — Llama Guard 3 được phát hành dưới dạng một mô hình duy nhất xử lý cả hai.

Llama Guard 3-1B-INT4 (arXiv:2411.17713, 440MB, ~30 tokens/s trên CPU di động) là biến thể edge đã được lượng tử hóa.

Llama Guard 4 (tháng 4 năm 2025) là mô hình 12B, đa phương thức tự nhiên, được tỉa từ Llama 4 Scout. Nó thay thế cả hai phiên bản tiền nhiệm 8B văn bản và 11B hình ảnh bằng một bộ phân loại duy nhất tiếp nhận cả văn bản + hình ảnh.

### Garak (NVIDIA)

Trình quét lỗ hổng mã nguồn mở. Kiến trúc:
- **Probes.** Các trình tạo tấn công cho hallucination, rò rỉ dữ liệu, prompt injection, độc hại, jailbreaks. Tĩnh (các prompt cố định), động (các prompt được tạo ra), thích ứng (phản hồi lại đầu ra của mục tiêu).
- **Detectors.** Chấm điểm đầu ra dựa trên các chế độ lỗi dự kiến — độc hại, bị rò rỉ, bị jailbreak.
- **Harnesses.** Quản lý các cặp probe-detector, chạy chiến dịch, tạo báo cáo.

TrustyAI tích hợp Garak với các lá chắn Llama-Stack (bộ phân loại đầu vào Prompt-Guard-86M, bộ phân loại đầu ra Llama-Guard-3-8B) để đánh giá mục tiêu được bảo vệ từ đầu đến cuối. Chấm điểm theo cấp độ (TBSA) thay thế cho pass/fail nhị phân — một mô hình có thể vượt qua ở cấp độ nghiêm trọng 3 nhưng thất bại ở cấp độ nghiêm trọng 5 trên cùng một probe.

### PyRIT (Microsoft)

Python Risk Identification Toolkit. Các chiến dịch red-team đa lượt. Được xây dựng xung quanh:
- **Converters.** Chuyển đổi một seed prompt — diễn giải lại, mã hóa, dịch thuật, đóng vai.
- **Orchestrators.** Chạy chiến dịch: Crescendo (leo thang), TAP (phân nhánh), RedTeaming (vòng lặp tùy chỉnh).
- **Scoring.** LLM-as-judge hoặc classifier-as-judge.

PyRIT là người anh em "nặng ký" hơn của Garak. Garak chạy hàng ngàn probe đơn lượt; PyRIT chạy các chiến dịch đa lượt sâu được thiết kế để phá vỡ các chế độ lỗi cụ thể.

### Stack

Đặt Llama Guard ở cả hai phía của mô hình. Chạy Garak hàng đêm để kiểm tra hồi quy. Chạy PyRIT cho các chiến dịch trước khi phát hành. Đây là cấu hình mặc định năm 2026 cho hầu hết các triển khai sản xuất.

### Các cạm bẫy khi đánh giá

- **Danh tính của Judge.** Cả ba công cụ đều có thể sử dụng một LLM judge; việc hiệu chuẩn judge sẽ ảnh hưởng đến ASR được báo cáo (Bài 12). Hãy chỉ định judge cùng với công cụ.
- **Sự lỗi thời của Probe.** Các probe của Garak sẽ cũ đi khi các mô hình được vá lỗi chống lại chúng. Các probe thích ứng (dạng PAIR) sẽ cũ chậm hơn so với các probe tĩnh.
- **FPR của Llama Guard trên nội dung lành tính.** Các phiên bản Llama Guard đời đầu thường gắn cờ nhầm nội dung chính trị và LGBTQ+; các hiệu chuẩn của Llama Guard 3/4 đã được cải thiện nhưng chưa được hiệu chuẩn cho từng triển khai cụ thể.

### Vị trí trong Giai đoạn 18

Các bài 12-15 là các họ tấn công. Bài 16 là công cụ sản xuất. Bài 17 (WMDP) là đánh giá cho khả năng sử dụng kép. Bài 18 là các khung an toàn tiên phong bao bọc các công cụ này trong một cấu trúc chính sách.

```figure
al-guard-stack
```

## Sử dụng

`code/main.py` xây dựng một bộ phân loại kiểu Llama Guard (từ khóa + đặc trưng ngữ nghĩa trên 14 danh mục), một harness Garak mô phỏng (vòng lặp probe-detector), và một chuỗi converter đa lượt kiểu PyRIT. Bạn có thể chạy ba công cụ này chống lại một mục tiêu giả lập và quan sát các dấu hiệu bao phủ khác nhau.

## Triển khai

Bài học này tạo ra `outputs/skill-red-team-stack.md`. Với một mô tả triển khai, nó chỉ ra công cụ nào trong ba công cụ trên là phù hợp, cần cấu hình gì trong mỗi công cụ và tần suất hồi quy cần chạy.

## Bài tập

1. Chạy `code/main.py`. So sánh tỷ lệ phát hiện của bộ phân loại kiểu Llama-Guard trên các cuộc tấn công đơn lượt so với đa lượt.

2. Triển khai một probe Garak mới: một yêu cầu độc hại được mã hóa base64. Đo lường khả năng phát hiện của nó bởi bộ phân loại kiểu Llama-Guard.

3. Mở rộng chuỗi converter kiểu PyRIT với một converter "dịch sang tiếng Pháp, sau đó diễn giải lại". Đo lường lại tỷ lệ thành công của cuộc tấn công.

4. Đọc danh sách danh mục nguy cơ của Llama Guard 3. Xác định hai danh mục mà dữ liệu huấn luyện có khả năng tạo ra tỷ lệ dương tính giả cao trên nội dung hợp lệ của nhà phát triển.

5. So sánh các nguyên tắc thiết kế của Garak và PyRIT. Lập luận cho một triển khai mà mỗi công cụ là lựa chọn phù hợp.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|------------------------|
| Llama Guard | "bộ phân loại" | Bộ phân loại an toàn Llama-3.1-8B/4-12B đã tinh chỉnh với 14 danh mục nguy cơ |
| Garak | "trình quét" | Trình quét lỗ hổng mã nguồn mở của NVIDIA; probes, detectors, harnesses |
| PyRIT | "công cụ chiến dịch" | Trình điều phối red-team đa lượt của Microsoft; converters, orchestrators, scoring |
| Prompt-Guard | "bộ phân loại nhỏ" | Bộ phân loại prompt-injection 86M của Meta, đi kèm với Llama Guard |
| TBSA | "chấm điểm theo cấp độ" | Cách pass/fail theo cấp độ của Garak thay thế cho kết quả nhị phân |
| Converter chain | "diễn giải + mã hóa + ..." | Nguyên thủy kết hợp của PyRIT để xây dựng các cuộc tấn công đa bước |
| MLCommons hazard categories | "14 phân loại" | Phân loại tiêu chuẩn ngành mà Llama Guard nhắm tới |

## Đọc thêm

- [Meta — Llama Guard 3 (trong bài báo Llama 3 Herd, arXiv:2407.21783)](https://arxiv.org/abs/2407.21783) — bộ phân loại 8B
- [Meta — Llama Guard 3-1B-INT4 (arXiv:2411.17713)](https://arxiv.org/abs/2411.17713) — bộ phân loại di động đã lượng tử hóa
- [NVIDIA Garak — GitHub](https://github.com/NVIDIA/garak) — repo trình quét và tài liệu
- [Microsoft PyRIT — GitHub](https://github.com/Azure/PyRIT) — bộ công cụ chiến dịch