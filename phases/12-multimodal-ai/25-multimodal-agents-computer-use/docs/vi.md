# Multimodal Agents and Computer-Use (Capstone)

> Sản phẩm tiên phong năm 2026 là một multimodal agent có khả năng đọc ảnh chụp màn hình, nhấp chuột, điều hướng giao diện web, điền biểu mẫu và hoàn thành các quy trình làm việc từ đầu đến cuối. SeeClick và CogAgent (2024) đã chứng minh được nguyên lý GUI-grounding. Ferret-UI bổ sung khả năng cho thiết bị di động. ChartAgent giới thiệu việc sử dụng công cụ trực quan cho biểu đồ. VisualWebArena và AgentVista (2026) là các bộ benchmark mà các sản phẩm tiên phong đang theo đuổi — và ngay cả Gemini 3 Pro hay Claude Opus 4.7 cũng chỉ đạt khoảng 30% trong các tác vụ khó của AgentVista. Đồ án này tổng hợp mọi khía cạnh của Giai đoạn 12: nhận thức (VLM độ phân giải cao), suy luận (LLM với khả năng sử dụng công cụ), grounding (xuất tọa độ), bộ nhớ dài hạn và đánh giá.

**Type:** Capstone
**Languages:** Python (stdlib, action schema + agent loop skeleton)
**Prerequisites:** Phase 12 · 05 (LLaVA), Phase 12 · 09 (Qwen-VL JSON), Phase 14 (Agent Engineering)
**Time:** ~240 phút

## Mục tiêu học tập

- Thiết kế một vòng lặp multimodal agent: nhận thức → suy luận → hành động → quan sát → lặp lại.
- Xây dựng schema đầu ra cho GUI grounding (tọa độ nhấp chuột, nhập văn bản, cuộn, kéo thả) mà VLM có thể xuất ra dưới dạng JSON.
- So sánh các agent chỉ dùng ảnh chụp màn hình vs agent dùng cây truy cập (accessibility-tree) vs agent lai.
- Thiết lập đánh giá benchmark cho multimodal agent trên một phần nhỏ của VisualWebArena.

## Vấn đề

Một quy trình trên trang web đặt vé: "tìm cho tôi chuyến bay đến Tokyo vào ngày 15 tháng 4, ghế gần lối đi với giá dưới 800 đô la, và đặt vé."

Một multimodal agent cần:

1. Chụp ảnh màn hình trình duyệt.
2. Phân tích ảnh chụp màn hình + URL + mục tiêu thành một kế hoạch.
3. Xuất ra một hành động có cấu trúc: nhấp (tại x,y), nhập "Tokyo" (tại phần tử E), cuộn xuống, chọn (nút radio).
4. Áp dụng hành động vào trình duyệt.
5. Quan sát trạng thái mới (ảnh chụp màn hình tiếp theo).
6. Lặp lại cho đến khi hoàn thành tác vụ.

Mỗi bước là một lần gọi multimodal VLM. Đầu ra của VLM phải là JSON có thể phân tích cú pháp. Các lỗi sẽ tích tụ qua từng bước, vì vậy khả năng phục hồi là rất quan trọng.

## Khái niệm

### GUI grounding — nguyên lý cơ bản

GUI grounding là: với một ảnh chụp màn hình và một hướng dẫn bằng ngôn ngữ tự nhiên, hãy xuất ra tọa độ (x, y) để nhấp (hoặc hành động khác).

SeeClick (arXiv:2401.10935) là kết quả mở đầu tiên ở quy mô lớn: tinh chỉnh (fine-tune) VLM trên dữ liệu GUI tổng hợp + thực tế, xuất tọa độ dưới dạng các token văn bản thuần túy. Hoạt động hiệu quả.

CogAgent (arXiv:2312.08914) đã bổ sung mã hóa độ phân giải cao 1120x1120 cho các giao diện dày đặc. Điểm số: ~84% trong điều hướng web.

Ferret-UI (arXiv:2404.05719) tập trung vào giao diện di động, tích hợp với dữ liệu accessibility của iOS.

Định dạng đầu ra thường là JSON:

```json
{"action": "click", "x": 384, "y": 220, "element_desc": "Search button"}
```

`element_desc` giúp ích cho việc phục hồi: nếu tọa độ bị lệch giữa các ảnh chụp màn hình, gợi ý ngữ nghĩa cho phép hệ thống thực hiện grounding lại.

### Action schemas

Một action schema điển hình có 6-10 loại hành động:

- `click`: (x, y)
- `type`: (text, x?, y?)
- `scroll`: (direction, amount)
- `drag`: (x0, y0, x1, y1)
- `select`: (option_index)
- `hover`: (x, y)
- `navigate`: (url)
- `wait`: (ms)
- `done`: (success, explanation)

Agent xuất ra một hành động mỗi bước. Trình bao bọc trình duyệt thực thi và trả về trạng thái mới.

### Chỉ dùng ảnh chụp màn hình vs cây truy cập (accessibility-tree)

Hai chế độ đầu vào:

- Chỉ dùng ảnh chụp màn hình: hình ảnh đầy đủ, không có thông tin cấu trúc. Tổng quát nhất; hoạt động trên mọi ứng dụng.
- Cây truy cập: DOM có cấu trúc / thông tin accessibility của iOS. Đáng tin cậy hơn nhiều cho việc grounding; hoạt động ở nơi có cây truy cập.
- Lai: cả hai, với cây truy cập đóng vai trò là bộ grounding đáng tin cậy cho các hành động nguyên tử và ảnh chụp màn hình cho ngữ cảnh ngữ nghĩa.

Các agent trong sản xuất sử dụng mô hình lai khi có thể. Tự động hóa trình duyệt (Selenium + accessibility) luôn có cây truy cập; các ứng dụng máy tính để bàn đôi khi cũng có.

### Bộ nhớ dài hạn

Một quy trình 20 bước tạo ra 20 ảnh chụp màn hình. Ngữ cảnh của VLM sẽ nhanh chóng bị đầy. Ba chiến lược nén:

- Summary-chain: sau mỗi 5 bước, tóm tắt những gì đã xảy ra, loại bỏ các ảnh chụp màn hình cũ.
- Skip-frame: giữ lại ảnh đầu tiên, ảnh cuối cùng và mỗi ảnh thứ 3.
- Nhật ký ghi lại công cụ: thực thi hành động, giữ nhật ký văn bản về những gì đã làm; không xem lại các ảnh chụp màn hình cũ.

API computer-use của Claude sử dụng mô hình nhật ký. Đơn giản hơn, đáng tin cậy hơn.

### Sử dụng công cụ trực quan

ChartAgent (arXiv:2510.04514) giới thiệu việc sử dụng công cụ trực quan để hiểu biểu đồ: cắt, phóng to, OCR, gọi phát hiện bên ngoài. Agent có thể xuất ra "cắt vùng (100, 200, 300, 400) sau đó gọi OCR" dưới dạng một lệnh gọi công cụ. Công cụ trả về văn bản; VLM tiếp tục suy luận.

Mô hình này được khái quát hóa: set-of-mark prompting, chú thích vùng và các công cụ phát hiện bên ngoài đều phù hợp với cùng một schema "xuất lệnh gọi công cụ, nhận phản hồi có cấu trúc".

### Các bộ benchmark năm 2026

- ScreenSpot-Pro. GUI grounding trên ~1k ảnh chụp màn hình web. Open SOTA Qwen2.5-VL-72B ~85%. Tiên phong ~90%.
- VisualWebArena. Các tác vụ web từ đầu đến cuối (mua sắm, diễn đàn, rao vặt). Open SOTA ~20%. Gemini 3 Pro ~27%.
- AgentVista (arXiv:2602.23166). Bộ benchmark khó nhất năm 2026. Các quy trình làm việc thực tế trên 12 lĩnh vực. Các mô hình tiên phong đạt 27-40%; các mô hình mở đạt 10-20%.
- WebArena / WebShop. Các bộ benchmark cũ hơn; đã bị các mô hình tiên phong vượt qua.

### Tại sao nó vẫn khó

Các nút thắt cổ chai về hiệu suất của agent:

1. Visual grounding ở quy mô chi tiết. "Nhấp vào chữ X nhỏ" thường thất bại ở độ phân giải di động.
2. Lập kế hoạch dài hạn. Sau 10 hành động, agent bị lệch khỏi mục tiêu.
3. Phục hồi lỗi. Khi một cú nhấp chuột thất bại (sai nút), việc phát hiện + phục hồi hiếm khi có trong dữ liệu huấn luyện.
4. Ngữ cảnh giữa các trang. Việc nhảy giữa các tab hoặc các biểu mẫu dài làm mất trạng thái.

Hướng nghiên cứu: kiến trúc bộ nhớ, lập kế hoạch lại rõ ràng, xác minh đa phương thức (khớp ảnh chụp màn hình để kiểm tra thành công của hành động).

### Xây dựng đồ án

Tác vụ đồ án: xây dựng một computer-use agent có khả năng:

1. Đọc HTML + ảnh chụp màn hình của một trang web đặt vé giả lập.
2. Lập kế hoạch cho một chuỗi nhiều bước: tìm kiếm → chọn → điền biểu mẫu → gửi.
3. Xuất các hành động JSON khớp với action schema.
4. Đánh giá trên một tập hợp cố định gồm 10 tác vụ.

Bài học cung cấp mã khung (scaffold code) dễ dàng mở rộng thành một trình duyệt thực tế.

```figure
mm-agent-loop
```

## Sử dụng

`code/main.py` là khung đồ án:

- Định nghĩa JSON cho action schema (10 hành động).
- Trạng thái trình duyệt giả lập dưới dạng dict.
- Khung vòng lặp agent: nhận trạng thái, xuất hành động, áp dụng, lặp lại.
- Mini-benchmark 10 tác vụ (các trang tổng hợp) để đo tỷ lệ thành công từ đầu đến cuối.
- Hook phục hồi lỗi khi một hành động thất bại.

## Triển khai

Bài học này tạo ra `outputs/skill-multimodal-agent-designer.md`. Với một sản phẩm computer-use (lĩnh vực, tập hành động, mục tiêu đánh giá), thiết kế toàn bộ vòng lặp agent, chiến lược bộ nhớ, chế độ grounding và điểm số benchmark kỳ vọng.

## Bài tập

1. Mở rộng action schema với công cụ `screenshot_region` (cắt + phóng to). Những tác vụ nào được hưởng lợi?

2. Đọc AgentVista (arXiv:2602.23166). Mô tả danh mục tác vụ khó nhất và tại sao các mô hình tiên phong vẫn thất bại.

3. Nén bộ nhớ dài hạn: thiết kế một summary-chain với tối đa 4 ảnh chụp màn hình được giữ trực tiếp, bất kỳ số lượng nào được ghi nhật ký.

4. Xây dựng hook phục hồi lỗi: khi hành động thất bại (không tìm thấy nút), agent sẽ làm gì tiếp theo?

5. So sánh Claude 4.7 chỉ dùng ảnh chụp màn hình với Qwen2.5-VL lai ảnh chụp màn hình + cây truy cập trên 10 tác vụ web. Cái nào thắng ở tác vụ nào?

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|------------------------|
| GUI grounding | "Tọa độ nhấp chuột" | Mô hình xuất ra (x,y) cho mục tiêu của hướng dẫn trên ảnh chụp màn hình |
| Action schema | "Định nghĩa công cụ" | Mô tả JSON của các hành động hợp lệ (nhấp, nhập, cuộn, kéo) |
| Accessibility tree | "DOM có cấu trúc" | Hệ thống phân cấp giao diện máy đọc được từ API trình duyệt/iOS |
| Hybrid agent | "Ảnh chụp + cây" | Sử dụng cả hình ảnh và thông tin cấu trúc; đáng tin cậy hơn khi dùng riêng lẻ |
| Visual tool use | "Phóng to/cắt/phát hiện" | Agent gọi các công cụ thị giác bên ngoài (OCR, phát hiện) giữa kế hoạch |
| Summary-chain | "Nén bộ nhớ" | Các bản tóm tắt văn bản định kỳ thay thế lịch sử ảnh chụp màn hình dài |
| VisualWebArena | "Benchmark web E2E" | Benchmark năm 2024 cho các tác vụ web từ đầu đến cuối |
| AgentVista | "Benchmark khó 2026" | Quy trình thực tế 12 lĩnh vực; ngay cả Gemini 3 Pro cũng chỉ đạt ~30% |

## Đọc thêm

- [Cheng et al. — SeeClick (arXiv:2401.10935)](https://arxiv.org/abs/2401.10935)
- [Hong et al. — CogAgent (arXiv:2312.08914)](https://arxiv.org/abs/2312.08914)
- [You et al. — Ferret-UI (arXiv:2404.05719)](https://arxiv.org/abs/2404.05719)
- [ChartAgent (arXiv:2510.04514)](https://arxiv.org/abs/2510.04514)
- [Koh et al. — VisualWebArena (arXiv:2401.13649)](https://arxiv.org/abs/2401.13649)
- [AgentVista (arXiv:2602.23166)](https://arxiv.org/abs/2602.23166)