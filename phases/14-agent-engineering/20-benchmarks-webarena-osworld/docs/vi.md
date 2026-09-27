# Benchmarks: WebArena và OSWorld

> WebArena kiểm tra khả năng của web-agent trên bốn ứng dụng tự lưu trữ (self-hosted). OSWorld kiểm tra khả năng của desktop-agent trên Ubuntu, Windows và macOS. Tại thời điểm ra mắt (2023–2024), cả hai đều cho thấy khoảng cách lớn giữa các agent tốt nhất và con người. Khoảng cách này đang dần thu hẹp; tuy nhiên, các dạng lỗi (failure modes) vẫn không thay đổi.

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 19 (SWE-bench, GAIA)
**Time:** ~60 minutes

## Mục tiêu học tập

- Mô tả bốn ứng dụng tự lưu trữ của WebArena và lý do tại sao đánh giá dựa trên thực thi (execution-based evaluation) lại quan trọng.
- Giải thích lý do tại sao OSWorld sử dụng ảnh chụp màn hình OS thực tế thay vì các API hỗ trợ tiếp cận (accessibility APIs).
- Nêu tên hai dạng lỗi chính của OSWorld: GUI grounding và kiến thức vận hành (operational knowledge).
- Tóm tắt những gì OSWorld-G và OSWorld-Human bổ sung vào benchmark cơ sở.

## Vấn đề

Các agent tổng quát có thể gọi công cụ. Liệu chúng có thể điều khiển trình duyệt qua 20 cú nhấp chuột để hoàn tất thanh toán mua sắm? Liệu chúng có thể cấu hình một máy Linux chỉ bằng bàn phím và chuột? Đây là những câu hỏi mà WebArena và OSWorld giải đáp.

## Khái niệm

### WebArena (Zhou et al., ICLR 2024)

- 812 tác vụ dài hạn trên bốn ứng dụng web tự lưu trữ: một trang mua sắm, một diễn đàn, một công cụ phát triển giống GitLab, một CMS doanh nghiệp.
- Cộng thêm các tiện ích: bản đồ, máy tính, nháp.
- Đánh giá dựa trên thực thi thông qua gym APIs — đơn hàng đã được đặt chưa, vấn đề đã được đóng chưa, trang CMS đã được cập nhật chưa?
- Tại thời điểm ra mắt: agent GPT-4 tốt nhất đạt tỷ lệ thành công 14,41% so với 78,24% của con người.

Việc thiết lập tự lưu trữ rất quan trọng — benchmark không bị nhiễu vì các ứng dụng mục tiêu được ghim phiên bản và có khả năng tái lập.

### Các phần mở rộng

- **VisualWebArena** — các tác vụ dựa trên hình ảnh, nơi thành công phụ thuộc vào việc diễn giải hình ảnh (ảnh chụp màn hình là quan sát hạng nhất).
- **TheAgentCompany** (Tháng 12/2024) — bổ sung terminal + lập trình; giống môi trường làm việc từ xa thực tế hơn.

### OSWorld (Xie et al., NeurIPS 2024)

- 369 tác vụ máy tính thực tế trên Ubuntu, Windows, macOS.
- Điều khiển bàn phím và chuột tự do trên các ứng dụng thực.
- Ảnh chụp màn hình 1920×1080 làm quan sát.
- Tại thời điểm ra mắt: mô hình tốt nhất đạt 12,24% so với 72,36% của con người.

### Các dạng lỗi chính

1. **GUI grounding.** Ánh xạ từ Pixel → phần tử. Các mô hình gặp khó khăn trong việc xác định vị trí các phần tử UI một cách đáng tin cậy ở độ phân giải 1920×1080.
2. **Operational knowledge.** Menu nào chứa cài đặt, phím tắt nào, bảng tùy chọn nào. Đây là phần kiến thức chuyên sâu mà con người tích lũy qua nhiều năm.

### Các nghiên cứu tiếp theo

- **OSWorld-G** — bộ dữ liệu grounding gồm 564 mẫu + tập huấn luyện Jedi. Phân tách grounding khỏi lập kế hoạch để bạn có thể đo lường chúng riêng biệt.
- **OSWorld-Human** — các quỹ đạo hành động chuẩn (gold trajectories) được tuyển chọn thủ công. Cho thấy các agent hàng đầu sử dụng số bước nhiều gấp 1,4-2,7 lần mức cần thiết (khoảng cách về hiệu suất quỹ đạo).

### Tại sao điều này quan trọng

Claude computer use, OpenAI CUA, Gemini 2.5 Computer Use (Bài 21) đều huấn luyện trên các khối lượng công việc được định hình bởi WebArena và OSWorld. Các benchmark là mục tiêu; các mô hình sản phẩm là câu trả lời được triển khai.

### Những sai lầm trong việc làm benchmark

- **Chỉ đánh giá bằng ảnh chụp màn hình.** OSWorld dựa trên ảnh chụp màn hình; việc đánh giá một agent sử dụng DOM hoặc accessibility APIs trên OSWorld sẽ bỏ lỡ thách thức về grounding.
- **Bỏ qua độ dài quỹ đạo.** Chỉ chấm điểm tỷ lệ thành công sẽ bỏ lỡ sự kém hiệu quả về số bước (1,4-2,7 lần) mà OSWorld-Human chỉ ra.
- **Các ứng dụng tự lưu trữ lỗi thời.** Các ứng dụng của WebArena ghim các phiên bản cụ thể; cập nhật mà không quản lý lại sẽ phá vỡ khả năng so sánh.

```figure
ae-agent-human-gap
```

## Xây dựng

`code/main.py` triển khai một bộ khung web-agent mô phỏng:

- Một máy trạng thái "ứng dụng mua sắm" tối giản: list_items, add_to_cart, checkout.
- Các quỹ đạo chuẩn cho 3 tác vụ.
- Một agent dạng kịch bản cố gắng thực hiện từng tác vụ.
- Bộ đánh giá dựa trên thực thi (kiểm tra trạng thái) và chỉ số hiệu suất quỹ đạo (số bước so với quỹ đạo chuẩn).

Chạy nó:

```
python3 code/main.py
```

Đầu ra: tỷ lệ thành công trên mỗi tác vụ và hiệu suất quỹ đạo, phản ánh phương pháp luận của OSWorld-Human.

## Sử dụng

- **WebArena Verified** tự lưu trữ trên cụm nội bộ để đánh giá liên tục.
- **OSWorld** trong đội máy ảo (VM fleet) cho các desktop agent.
- **Computer-use agents** (Bài 21) — Claude, OpenAI CUA, Gemini — tất cả đều được huấn luyện trên các khối lượng công việc như thế này.
- **Quy trình sản phẩm của riêng bạn** — ghi lại các quỹ đạo chuẩn cho 20 tác vụ hàng đầu của bạn; chạy các agent đối chiếu với chúng hàng tuần.

## Triển khai

`outputs/skill-web-desktop-harness.md` xây dựng một bộ khung web/desktop agent với đánh giá dựa trên thực thi và chỉ số hiệu suất quỹ đạo.

## Bài tập

1. Mở rộng bộ khung mô phỏng với ứng dụng thứ hai (một diễn đàn). Viết 3 tác vụ cộng với các quỹ đạo chuẩn.
2. Thêm báo cáo hiệu suất quỹ đạo cho mỗi tác vụ. Trên mô hình của bạn, agent đang thực hiện gấp 1x, 2x hay 3x so với quỹ đạo chuẩn?
3. Triển khai một công cụ "gây nhiễu" — công cụ mà quỹ đạo chuẩn không bao giờ sử dụng. Agent dạng kịch bản có bị mắc bẫy không?
4. Đọc OSWorld-G. Bạn sẽ tách biệt các lỗi grounding khỏi lỗi lập kế hoạch trong các đánh giá của riêng mình như thế nào?
5. Đọc README của các ứng dụng WebArena. Điều gì sẽ xảy ra khi bạn nâng cấp một trong các phiên bản ứng dụng đã được ghim?

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|------------------------|
| WebArena | "Web agent benchmark" | 812 tác vụ trên 4 ứng dụng tự lưu trữ; đánh giá kiểu gym |
| VisualWebArena | "Visual WebArena" | WebArena dựa trên hình ảnh; ảnh chụp màn hình là quan sát |
| OSWorld | "Desktop agent benchmark" | 369 tác vụ trên Ubuntu/Windows/macOS thực |
| GUI grounding | "Pixel-to-element mapping" | Mô hình xác định vị trí các phần tử UI trong 1920x1080 |
| Operational knowledge | "OS know-how" | Menu nào, phím tắt nào, bảng tùy chọn nào |
| OSWorld-G | "Grounding suite" | 564 mẫu chỉ dành cho grounding + tập huấn luyện |
| OSWorld-Human | "Gold trajectories" | Chuỗi hành động chuyên gia thủ công để đo hiệu suất |
| Trajectory efficiency | "Steps over gold" | Số bước của agent chia cho số bước tối thiểu của con người |

## Đọc thêm

- [Zhou et al., WebArena (arXiv:2307.13854)](https://arxiv.org/abs/2307.13854) — benchmark web bốn ứng dụng
- [Xie et al., OSWorld (arXiv:2404.07972)](https://arxiv.org/abs/2404.07972) — benchmark desktop đa OS
- [Anthropic, Introducing computer use](https://www.anthropic.com/news/3-5-models-and-computer-use) — khả năng của Claude được định hình bởi benchmark
- [OpenAI, Computer-Using Agent](https://openai.com/index/computer-using-agent/) — các con số từ OSWorld và WebArena