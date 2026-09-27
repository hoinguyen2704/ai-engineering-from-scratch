# Memory Blocks và Sleep-Time Compute

> Các khối bộ nhớ chức năng rời rạc mà model có thể chỉnh sửa trực tiếp, và một agent chạy trong thời gian nghỉ (sleep-time agent) thực hiện hợp nhất bộ nhớ một cách bất đồng bộ trong khi agent chính đang rảnh rỗi. Hai ý tưởng này là cách bạn mở rộng bộ nhớ vượt ra ngoài một phiên hội thoại đơn lẻ.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 07 (MemGPT)
**Time:** ~75 phút

## Mục tiêu học tập

- Nêu tên ba tầng bộ nhớ mà Letta sử dụng (core, recall, archival) và vai trò của từng tầng.
- Giải thích mô hình khối bộ nhớ (memory-block pattern): Khối Human, khối Persona và các khối do người dùng định nghĩa như là các đối tượng có kiểu dữ liệu hạng nhất (first-class typed objects).
- Mô tả sleep-time compute là gì, tại sao nó nằm ngoài đường dẫn quan trọng (critical path) và tại sao nó có thể chạy một model mạnh hơn so với agent chính.
- Triển khai một vòng lặp hai agent theo kịch bản, trong đó một agent chính phục vụ phản hồi và một sleep-time agent thực hiện hợp nhất các khối giữa các lượt hội thoại.

## Vấn đề

MemGPT (Bài 07) đã giải quyết luồng điều khiển bộ nhớ ảo. Ba vấn đề thực tế đã nảy sinh:

1. **Độ trễ (Latency).** Mọi thao tác bộ nhớ đều nằm trên đường dẫn quan trọng. Nếu agent phải cắt tỉa, tóm tắt hoặc đối chiếu trong khi người dùng đang chờ, độ trễ đuôi (tail latency) sẽ tăng vọt.
2. **Sự suy giảm bộ nhớ (Memory rot).** Các bản ghi tích tụ. Các dữ kiện mâu thuẫn vẫn tồn tại. Việc truy xuất bị nhấn chìm trong nội dung cũ.
3. **Mất cấu trúc (Structure loss).** Một kho lưu trữ archival phẳng không thể diễn đạt "khối Human luôn nằm trong prompt; khối Persona luôn nằm trong prompt; khối Task thay đổi theo từng phiên."

Letta (letta.com) là tên nền tảng mà dự án MemGPT ban đầu đã áp dụng vào năm 2024 — mô hình trong bài báo vẫn giữ tên MemGPT — và bản viết lại Letta V1 năm 2026 là một bước đi sau đó, tách biệt. Các khối bộ nhớ làm cho cấu trúc trở nên rõ ràng; sleep-time compute di chuyển việc hợp nhất ra khỏi đường dẫn quan trọng.

## Khái niệm

### Ba tầng bộ nhớ

| Tầng | Phạm vi | Nơi lưu trữ | Được ghi bởi |
|------|-------|----------------|------------|
| Core | Luôn hiển thị | Bên trong prompt chính | Tool call của agent + sleep-time rewrite |
| Recall | Lịch sử hội thoại | Có thể truy xuất | Tự động ghi log lượt hội thoại |
| Archival | Dữ kiện tùy ý | Vector + KV + đồ thị | Tool call của agent + sleep-time ingest |

Core là lõi của MemGPT. Recall là bộ đệm hội thoại với phần đuôi đã bị loại bỏ. Archival là kho lưu trữ bên ngoài. Sự phân chia này giúp làm sạch việc quá tải hai tầng của MemGPT.

### Các khối bộ nhớ (Memory blocks)

Một khối là một phần có kiểu dữ liệu, bền vững và có thể chỉnh sửa trong tầng core. Bài báo MemGPT gốc định nghĩa hai khối:

- **Khối Human** — các dữ kiện về người dùng (tên, vai trò, sở thích, mục tiêu).
- **Khối Persona** — khái niệm về bản thân của agent (danh tính, giọng điệu, các ràng buộc).

Letta tổng quát hóa thành các khối tùy ý do người dùng định nghĩa: một khối `Task` cho mục tiêu hiện tại, một khối `Project` cho các dữ kiện về codebase, một khối `Safety` cho các ràng buộc cứng. Mỗi khối có một `id`, `label`, `value`, `limit` (giới hạn ký tự), `description` (để model biết khi nào cần chỉnh sửa).

Các khối có thể chỉnh sửa thông qua giao diện tool:

- `block_append(label, text)`
- `block_replace(label, old, new)`
- `block_read(label)`
- `block_summarize(label)` — nén một khối đang gần đạt giới hạn.

### Sleep-time compute

Bổ sung của Letta năm 2025: chạy một agent thứ hai ở chế độ nền, nằm ngoài đường dẫn quan trọng. Các sleep-time agent xử lý các bản ghi hội thoại và ngữ cảnh codebase, ghi `learned_context` vào các khối dùng chung, và hợp nhất hoặc vô hiệu hóa các bản ghi archival.

Các đặc tính đạt được:

- **Không tốn độ trễ.** Các phản hồi chính không phải chờ đợi các thao tác bộ nhớ.
- **Cho phép sử dụng model mạnh hơn.** Sleep-time agent có thể là một model đắt tiền hơn, chậm hơn vì nó không bị ràng buộc bởi độ trễ.
- **Cửa sổ hợp nhất tự nhiên.** Khử trùng lặp, tóm tắt, vô hiệu hóa các dữ kiện mâu thuẫn khi người dùng không chờ đợi.

Hình thái này khớp với cách con người làm việc: bạn thực hiện nhiệm vụ, bạn ngủ để suy ngẫm về nó, và bộ nhớ dài hạn sẽ ổn định qua đêm.

### Native reasoning

Letta V1 (`letta_v1_agent`, 2026) loại bỏ `send_message`/heartbeat và các token `Thought:` nội dòng để chuyển sang native reasoning. Responses API (OpenAI) và Messages API với extended thinking (Anthropic) phát ra suy luận trên một kênh riêng biệt, được truyền qua các lượt hội thoại (được mã hóa giữa các nhà cung cấp trong môi trường production). Vòng lặp điều khiển vẫn là ReAct. Dấu vết suy nghĩ (thought trace) mang tính cấu trúc, không phải dạng prompt.

### Nơi mô hình này gặp lỗi

- **Block bloat.** `block_append` vô hạn sẽ nhanh chóng đạt giới hạn. Hãy thiết lập một bộ tóm tắt khối trước khi thực hiện ghi nếu nó vượt quá giới hạn.
- **Silent drift.** Sleep-time agent viết lại một khối và agent chính không bao giờ nhận ra. Hãy đánh phiên bản các khối và hiển thị sự khác biệt (diffs) trong dấu vết.
- **Poisoned consolidation.** Sleep-time agent xử lý nội dung mà kẻ tấn công có thể tiếp cận vào core. Bài 27 cũng áp dụng cho bề mặt sleep-time.

```figure
memory-blocks
```

## Xây dựng

`code/main.py` triển khai:

- `Block` — id, nhãn, giá trị, giới hạn, mô tả.
- `BlockStore` — CRUD + helper `near_limit(label)`.
- Hai agent theo kịch bản — `PrimaryAgent` phục vụ một lượt hội thoại, `SleepTimeAgent` hợp nhất giữa các lượt.
- Một dấu vết hiển thị hội thoại ba lượt với các lần ghi khối, cộng với một lượt sleep-time để tóm tắt một khối và vô hiệu hóa một dữ kiện cũ.

Chạy nó:

```
python3 code/main.py
```

Bản ghi cho thấy sự phân tách: các lượt chính diễn ra nhanh và tạo ra các bản ghi thô; lượt sleep-time thực hiện nén và làm sạch.

## Sử dụng

- **Letta** (letta.com) cho bản triển khai tham chiếu. Tự lưu trữ hoặc quản lý trên cloud.
- **Claude Agent SDK skills** như là kiến thức dạng khối — một skill là một khối hướng dẫn được đặt tên, đánh phiên bản, có thể truy xuất mà agent tải theo yêu cầu.
- **Custom builds** cho các đội ngũ muốn kiểm soát backend lưu trữ. Hãy sử dụng hợp đồng API của Letta để bạn có thể di chuyển sau này.

## Triển khai

`outputs/skill-memory-blocks.md` tạo ra một hệ thống khối theo kiểu Letta với các hook sleep-time cho bất kỳ runtime nào, bao gồm các quy tắc an toàn và liên kết trích dẫn.

## Bài tập

1. Thêm một tool `block_summarize` thay thế giá trị khối bằng một bản tóm tắt do model tạo ra khi `near_limit` trả về true. Ngưỡng kích hoạt nào giúp giảm thiểu cả số lần gọi tóm tắt và tình trạng tràn khối?
2. Triển khai khử trùng lặp sleep-time trên archival: hai bản ghi có văn bản trùng lặp token >90% sẽ gộp thành một. Chỉ thực hiện trong lượt sleep-time, không bao giờ thực hiện trên đường dẫn quan trọng.
3. Đánh phiên bản các khối. Trong mỗi lần ghi, hãy lưu lại giá trị cũ và một bản diff. Hiển thị `block_history(label)` để người vận hành có thể gỡ lỗi "tại sao agent lại quên X."
4. Đối xử với sleep-time agent như những người viết không đáng tin cậy. Khi chúng chạm vào khối Persona hoặc Safety, hãy yêu cầu một sự xem xét từ agent thứ hai trước khi commit.
5. Chuyển ví dụ sang sử dụng Letta API (`letta_v1_agent`). Những gì thay đổi trong schema khối, và native reasoning làm thay đổi hình thái dấu vết như thế nào?

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|------------------------|
| Memory block | "Phần prompt có thể chỉnh sửa" | Phân đoạn bộ nhớ core có kiểu, bền vững, có thể chỉnh sửa bởi LLM |
| Human block | "Bộ nhớ người dùng" | Các dữ kiện về người dùng, được ghim trong core |
| Persona block | "Danh tính agent" | Khái niệm bản thân, giọng điệu, ràng buộc, được ghim trong core |
| Sleep-time compute | "Công việc bộ nhớ bất đồng bộ" | Agent thứ hai thực hiện hợp nhất ngoài đường dẫn quan trọng |
| Core / Recall / Archival | "Các tầng" | Phân chia bộ nhớ ba lớp: luôn hiển thị / hội thoại / bên ngoài |
| Block limit | "Giới hạn" | Giới hạn ký tự mỗi khối; buộc phải tóm tắt |
| Native reasoning | "Kênh suy nghĩ" | Đầu ra suy luận ở cấp độ nhà cung cấp, không phải `Thought:` cấp prompt |
| Learned context | "Đầu ra sleep-time" | Các dữ kiện mà sleep-time agent ghi vào các khối dùng chung |

## Đọc thêm

- [Letta, Memory Blocks blog](https://www.letta.com/blog/memory-blocks) — mô hình khối
- [Letta, Sleep-time Compute blog](https://www.letta.com/blog/sleep-time-compute) — hợp nhất bất đồng bộ
- [Letta, Rearchitecting the Agent Loop](https://www.letta.com/blog/letta-v1-agent) — viết lại native reasoning
- [Packer et al., MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560) — nguồn gốc