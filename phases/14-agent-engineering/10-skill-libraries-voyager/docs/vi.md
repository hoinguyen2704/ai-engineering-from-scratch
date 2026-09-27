# Skill Libraries and Lifelong Learning (Voyager)

> Voyager (Wang et al., TMLR 2024) coi mã thực thi là một kỹ năng (skill). Các kỹ năng được đặt tên, có thể truy xuất, có thể kết hợp và được tinh chỉnh thông qua phản hồi từ môi trường. Đây là kiến trúc tham chiếu cho các kỹ năng trong Claude Agent SDK, skillkit và mô hình thư viện kỹ năng năm 2026.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 07 (MemGPT), Phase 14 · 08 (Letta Blocks)
**Time:** ~75 phút

## Mục tiêu học tập

- Nêu tên ba thành phần của Voyager — chương trình giảng dạy tự động (automatic curriculum), thư viện kỹ năng (skill library), cơ chế nhắc lệnh lặp (iterative prompting) — và vai trò của từng thành phần.
- Giải thích lý do tại sao Voyager biến không gian hành động thành mã (code) thay vì các lệnh nguyên thủy (primitive commands).
- Triển khai một thư viện kỹ năng stdlib với các tính năng đăng ký, truy xuất, kết hợp và tinh chỉnh dựa trên lỗi (failure-driven refinement).
- Ánh xạ mô hình của Voyager vào các kỹ năng trong Claude Agent SDK năm 2026 và hệ sinh thái skillkit.

## Vấn đề

Các tác nhân (agent) xây dựng lại mọi khả năng từ đầu trong mỗi phiên làm việc thường mắc ba sai lầm:

1. **Lãng phí token.** Mỗi tác vụ đều yêu cầu suy luận lại từ đầu.
2. **Mất tiến trình.** Một chỉnh sửa đã học được trong phiên A không được chuyển sang phiên B.
3. **Thất bại trong việc kết hợp dài hạn.** Các tác vụ phức tạp cần phân cấp khả năng; các prompt one-shot không thể diễn đạt được điều này.

Câu trả lời của Voyager: coi mỗi khả năng có thể tái sử dụng là một đoạn mã được đặt tên, lưu trữ trong thư viện, có thể truy xuất theo độ tương đồng, kết hợp với các kỹ năng khác và được tinh chỉnh bởi phản hồi thực thi.

## Khái niệm

### Ba thành phần

Voyager (arXiv:2305.16291) cấu trúc một tác nhân xung quanh:

1. **Chương trình giảng dạy tự động.** Một bộ đề xuất dựa trên sự tò mò sẽ chọn tác vụ tiếp theo dựa trên bộ kỹ năng hiện tại của tác nhân và trạng thái môi trường. Việc khám phá diễn ra theo hướng từ dưới lên (bottom-up).
2. **Thư viện kỹ năng.** Mỗi kỹ năng là một đoạn mã thực thi. Các kỹ năng mới được thêm vào khi một tác vụ thành công. Kỹ năng được truy xuất bằng độ tương đồng giữa truy vấn và mô tả.
3. **Cơ chế nhắc lệnh lặp.** Khi thất bại, tác nhân nhận được các lỗi thực thi, phản hồi từ môi trường và kết quả tự xác minh, sau đó tinh chỉnh kỹ năng.

Đánh giá trên Minecraft (Wang et al., 2024): tạo ra nhiều vật phẩm độc đáo hơn 3.3 lần, công cụ đá nhanh hơn 8.5 lần, công cụ sắt nhanh hơn 6.4 lần, di chuyển trên bản đồ xa hơn 2.3 lần so với các mô hình cơ sở. Các con số này là đặc thù của Minecraft, nhưng mô hình này có thể áp dụng rộng rãi.

### Không gian hành động = mã

Hầu hết các tác nhân phát ra các lệnh nguyên thủy. Voyager phát ra các hàm JavaScript. Một kỹ năng là:

```
async function craftIronPickaxe(bot) {
  await mineIron(bot, 3);
  await mineStick(bot, 2);
  await placeCraftingTable(bot);
  await craft(bot, 'iron_pickaxe');
}
```

Được kết hợp từ các kỹ năng con. Được lưu trữ dựa trên mô tả và embedding. Được truy xuất dưới dạng một chương trình, không phải một prompt.

Đây chính là kỹ năng của Claude Agent SDK năm 2026: một đoạn mã được đặt tên, có thể truy xuất cùng với các hướng dẫn mà tác nhân tải theo yêu cầu.

### Truy xuất kỹ năng

Tác vụ mới "chế tạo một chiếc cúp kim cương". Tác nhân:

1. Nhúng (embed) mô tả tác vụ.
2. Truy vấn thư viện kỹ năng để tìm top-k kỹ năng tương tự.
3. Truy xuất `craftIronPickaxe`, `mineDiamond`, `placeCraftingTable`, v.v.
4. Kết hợp kỹ năng mới từ các kỹ năng nguyên thủy đã truy xuất + logic mới.

Đây là mô hình mà các tài nguyên MCP (Phase 13) và kỹ năng của Agent SDK triển khai: truy xuất trên bề mặt tri thức/mã, giới hạn trong tác vụ hiện tại.

### Tinh chỉnh lặp

Vòng lặp phản hồi của Voyager:

1. Tác nhân viết một kỹ năng.
2. Kỹ năng chạy trong môi trường.
3. Một trong ba tín hiệu trả về: `success`, `error` (kèm stack trace), `self-verification failure`.
4. Tác nhân viết lại kỹ năng bằng cách sử dụng tín hiệu làm ngữ cảnh.
5. Lặp lại cho đến khi thành công hoặc đạt số vòng tối đa.

Đây là Self-Refine (Bài 05) được áp dụng cho việc tạo mã với xác minh dựa trên môi trường. CRITIC (Bài 05) là cùng một mô hình với các công cụ bên ngoài đóng vai trò là bộ xác minh.

### Chương trình giảng dạy và khám phá

Mô-đun chương trình giảng dạy của Voyager đề xuất các tác vụ như "xây dựng nơi trú ẩn gần hồ" dựa trên những gì tác nhân có và những gì chưa làm. Bộ đề xuất sử dụng trạng thái môi trường + danh mục kỹ năng để chọn một tác vụ nằm ngay trên khả năng hiện tại — điểm ngọt của sự khám phá.

Đối với các tác nhân trong môi trường sản xuất, điều này chuyển thành toán tử "cái gì còn thiếu": dựa trên thư viện kỹ năng hiện tại và một lĩnh vực, những kỹ năng nào chúng ta chưa bao phủ? Các nhóm thường triển khai thủ công việc này dưới dạng đánh giá chương trình giảng dạy.

### Nơi mô hình này đi chệch hướng

- **Thoái hóa thư viện kỹ năng.** Cùng một kỹ năng được thêm 10 lần với các mô tả hơi khác nhau. Hãy thêm tính năng khử trùng lặp khi ghi; truy xuất chỉ trả về một kết quả.
- **Trôi dạt kỹ năng kết hợp.** Kỹ năng cha phụ thuộc vào kỹ năng con đã được tinh chỉnh. Hãy đánh phiên bản cho kỹ năng; kỹ năng cha được ghim ở v1 sẽ không tự động cập nhật lên v3.
- **Chất lượng truy xuất.** Truy xuất vector trên mô tả kỹ năng sẽ giảm chất lượng khi thư viện vượt quá vài trăm mục. Hãy bổ sung bằng các bộ lọc thẻ và các ràng buộc cứng ("chỉ các kỹ năng có `category=tooling`").

```figure
voyager-skills
```

## Xây dựng

`code/main.py` triển khai một thư viện kỹ năng stdlib:

- `Skill` — tên, mô tả, mã (dưới dạng chuỗi), phiên bản, thẻ và các phụ thuộc.
- `SkillLibrary` — đăng ký, tìm kiếm (trùng lặp token), kết hợp (sắp xếp topo các phụ thuộc) và tinh chỉnh (tăng phiên bản khi cập nhật).
- Một tác nhân kịch bản đăng ký ba kỹ năng nguyên thủy, kết hợp kỹ năng thứ tư, gặp lỗi và tinh chỉnh.

Chạy nó:

```
python3 code/main.py
```

Dấu vết (trace) cho thấy các thao tác ghi thư viện, truy xuất, kết hợp, thực thi thất bại và tinh chỉnh v2 — vòng lặp Voyager từ đầu đến cuối.

## Sử dụng

- **Kỹ năng Claude Agent SDK** (Anthropic) — tham chiếu năm 2026: mỗi kỹ năng có mô tả, mã và hướng dẫn; được tải theo yêu cầu trong phiên làm việc của tác nhân.
- **skillkit** (npm: skillkit) — quản lý kỹ năng đa tác nhân cho hơn 32 tác nhân lập trình AI.
- **Thư viện kỹ năng tùy chỉnh** — đặc thù theo lĩnh vực (kỹ năng SQL cho tác nhân dữ liệu, kỹ năng Terraform cho tác nhân hạ tầng). Mô hình Voyager có thể mở rộng xuống quy mô nhỏ.
- **OpenAI Agents SDK `tools`** — ở mức độ cơ bản; mỗi công cụ là một kỹ năng nhẹ.

## Triển khai

`outputs/skill-skill-library.md` tạo ra một thư viện kỹ năng theo hình mẫu Voyager với các tính năng đăng ký, truy xuất, đánh phiên bản và tinh chỉnh được kết nối cho bất kỳ runtime mục tiêu nào.

## Bài tập

1. Thêm bộ phát hiện chu kỳ phụ thuộc vào `compose()`. Điều gì xảy ra khi kỹ năng A phụ thuộc vào B, B lại phụ thuộc vào A? Lỗi hay cảnh báo?
2. Triển khai ghim phiên bản cho từng kỹ năng. Khi một kỹ năng cha kết hợp kỹ năng con `crafting@1`, việc tinh chỉnh lên `crafting@2` không được tự động nâng cấp kỹ năng cha.
3. Thay thế truy xuất trùng lặp token bằng embedding sentence-transformers (hoặc triển khai stdlib BM25). Đo lường retrieval@5 trên thư viện mẫu 50 kỹ năng.
4. Thêm một tác nhân "chương trình giảng dạy": dựa trên thư viện hiện tại và mô tả lĩnh vực, đề xuất 5 kỹ năng còn thiếu. Gọi nó hàng tuần.
5. Đọc tài liệu về kỹ năng của Claude Agent SDK từ Anthropic. Chuyển thư viện mẫu sang lược đồ kỹ năng của SDK. Điều gì thay đổi về khả năng khám phá?

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|------------------------|
| Skill | "Khả năng tái sử dụng" | Đoạn mã được đặt tên + mô tả, có thể truy xuất theo độ tương đồng |
| Skill library | "Bộ nhớ cách làm của tác nhân" | Kho lưu trữ kỹ năng bền vững, có thể tìm kiếm và kết hợp |
| Curriculum | "Bộ đề xuất tác vụ" | Trình tạo mục tiêu từ dưới lên dựa trên khoảng cách khả năng hiện tại |
| Composition | "Đồ thị kỹ năng (DAG)" | Các kỹ năng gọi kỹ năng khác; được sắp xếp topo khi thực thi |
| Iterative refinement | "Vòng lặp tự sửa lỗi" | Phản hồi môi trường + lỗi + tự xác minh quay ngược lại phiên bản tiếp theo |
| Action-space-as-code | "Hành động lập trình" | Phát ra các hàm, không phải lệnh nguyên thủy, cho hành vi mở rộng theo thời gian |
| Dedup on write | "Hợp nhất kỹ năng" | Các mô tả gần giống nhau được hợp nhất thành một kỹ năng chuẩn |

## Đọc thêm

- [Wang et al., Voyager (arXiv:2305.16291)](https://arxiv.org/abs/2305.16291) — bài báo gốc về thư viện kỹ năng
- [Tổng quan về Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) — kỹ năng như một sản phẩm năm 2026
- [Anthropic, Xây dựng tác nhân với Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) — kỹ năng và tác nhân con trong thực tế
- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651) — vòng lặp tinh chỉnh bên dưới Voyager