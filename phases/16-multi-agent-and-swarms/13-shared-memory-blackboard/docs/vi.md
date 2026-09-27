# Shared Memory và Blackboard Patterns

> Hai phương pháp tiếp cận cùng tồn tại trong các hệ thống đa tác nhân (multi-agent) năm 2026: **message pool** (mọi người đều thấy tin nhắn của nhau, như trong AutoGen GroupChat hoặc MetaGPT) và **blackboard với cơ chế đăng ký (subscription)** (các tác nhân đăng ký các sự kiện liên quan, như trong Context-Aware MCP hoặc framework Matrix). Cả hai đều là phần duy nhất có trạng thái (stateful) của một hệ thống đa tác nhân — nghĩa là đây là nơi chứa đựng những lỗi thú vị nhất. Chế độ lỗi tham chiếu là **memory poisoning** (nhiễm độc bộ nhớ): một tác nhân tạo ra "sự thật" ảo tưởng, các tác nhân khác coi đó là đã được xác minh, và độ chính xác suy giảm dần theo cách khó gỡ lỗi hơn nhiều so với một sự cố sập hệ thống ngay lập tức. Bài học này xây dựng cả hai cấu trúc từ stdlib, thực hiện một cuộc tấn công nhiễm độc, và chỉ ra ba biện pháp giảm thiểu thực sự hiệu quả trong môi trường production.

**Type:** Learn + Build
**Languages:** Python (stdlib, `threading`)
**Prerequisites:** Phase 16 · 04 (Primitive Model), Phase 16 · 09 (Parallel Swarm Networks)
**Time:** ~75 phút

## Vấn đề

Các hệ thống đa tác nhân cần một nơi để các tác nhân chia sẻ dữ liệu thực tế. Một lựa chọn theo nghĩa đen là "chuyển mọi thứ trong tin nhắn" — nhưng điều đó tạo lại trạng thái chia sẻ với việc sao chép dư thừa. Một lựa chọn khác là "cung cấp cho mọi người một nhật ký toàn cục (global log)" — nhưng nhật ký toàn cục phát triển không giới hạn và dễ bị nhiễm độc. Lựa chọn thứ ba là "chiếu một view cho mỗi tác nhân" — có khả năng mở rộng nhưng đòi hỏi schema phức tạp.

Khi một trong các tác nhân bị ảo tưởng và ghi sự ảo tưởng đó vào trạng thái chia sẻ, mọi tác nhân hạ nguồn đọc trạng thái đó sẽ chấp nhận sự ảo tưởng như một sự thật. Đến khi con người nhận ra, chuỗi suy luận đã sâu năm bước và nguyên nhân gốc rễ nằm ở tin nhắn thứ ba từng được viết. Việc gỡ lỗi sự suy giảm độ chính xác trong hệ thống đa tác nhân khó hơn nhiều so với gỡ lỗi một sự cố sập hệ thống.

Đây chính là memory poisoning. Đây là họ lỗi được ghi chép nhiều thứ hai trong phân loại MAST (Cemri và cộng sự, arXiv:2503.13657) và nó mang tính cấu trúc: bất kỳ thiết kế bộ nhớ chia sẻ nào không có nguồn gốc (provenance) và bộ xác minh không thể ghi (unwritable verifier) cuối cùng đều sẽ gặp phải lỗi này.

## Khái niệm

### Hai cấu trúc liên kết chính

**Full message pool.** Mọi tác nhân đều đọc mọi tin nhắn. AutoGen GroupChat và MetaGPT sử dụng cách này. Đơn giản, minh bạch, dễ kiểm tra, nhưng không thể mở rộng quá ~10 tác nhân vì ngữ cảnh của mỗi tác nhân sẽ bị lấp đầy bởi công việc của các tác nhân khác.

```
agent-A ──write──▶ ┌────────────────┐ ◀──read── agent-D
                   │ message pool   │
agent-B ──write──▶ │                │ ◀──read── agent-E
                   │ (global log)   │
agent-C ──write──▶ └────────────────┘ ◀──read── agent-F
```

**Blackboard với subscription.** Các tác nhân khai báo sự quan tâm đến các chủ đề; nền tảng chỉ định tuyến các tin nhắn liên quan. CA-MCP (arXiv:2601.11595) và framework phi tập trung Matrix (arXiv:2511.21686) sử dụng cách này. Khả năng mở rộng tốt hơn, nhưng đòi hỏi thiết kế schema từ trước để việc đăng ký trở nên có ý nghĩa.

```
                   ┌─ topic: prices ──┐
agent-A ──pub────▶ │                  │ ──▶ agent-D (subscribed)
                   ├─ topic: orders ──┤
agent-B ──pub────▶ │                  │ ──▶ agent-E (subscribed)
                   ├─ topic: alerts ──┤
agent-C ──pub────▶ │                  │ ──▶ agent-F (subscribed)
                   └──────────────────┘
```

### Khi nào mỗi loại chiếm ưu thế

- **Full pool** thắng khi số lượng tác nhân ít (< 10), không đồng nhất và cuộc hội thoại có tầm nhìn ngắn hạn. Việc suy luận ai đã nói gì là tầm thường khi mọi người đều thấy mọi thứ.
- **Blackboard** thắng khi số lượng tác nhân nhiều, đồng nhất về vai trò nhưng đông đảo về số lượng (swarms), và cuộc hội thoại kéo dài. Việc định tuyến giúp tiết kiệm chi phí token và tránh ô nhiễm ngữ cảnh.

Các hệ thống production thường kết hợp cả hai: một full pool nhỏ ở trên cùng (lớp lập kế hoạch), các blackboard ở bên dưới (lớp worker).

### Memory poisoning, trong một kịch bản

Ba tác nhân làm việc trong một nhiệm vụ nghiên cứu. Tác nhân A là tác nhân truy xuất. Tác nhân B là tác nhân tóm tắt. Tác nhân C là tác nhân phân tích.

1. A lấy một trang và viết một tin nhắn vào trạng thái chia sẻ: "Nghiên cứu báo cáo mức cải thiện độ chính xác 42%."
2. Trang được lấy thực tế nói "cải thiện 4.2%." A đã ảo tưởng về dấu thập phân.
3. B, đọc trạng thái chia sẻ, viết: "Báo cáo mức tăng độ chính xác 42% (nguồn: A)."
4. C, đọc trạng thái chia sẻ, viết: "Khuyến nghị áp dụng — mức tăng 42% là mang tính chuyển đổi."
5. Báo cáo cuối cùng trích dẫn con số 42% chưa từng tồn tại.

Không tác nhân nào bị sập. Không bài kiểm tra nào thất bại. Hệ thống "đã hoạt động." Sự ảo tưởng đã vượt qua từ ngữ cảnh của một tác nhân vào suy luận của mọi tác nhân hạ nguồn thông qua trạng thái chia sẻ.

### Tại sao đây là vấn đề cấu trúc

Nếu không có trạng thái chia sẻ, sự ảo tưởng của tác nhân A sẽ chỉ nằm trong ngữ cảnh của A. Các tác nhân hạ nguồn sẽ truy xuất lại hoặc suy luận lại và có thể phát hiện ra lỗi. Với trạng thái chia sẻ ngây thơ, ngữ cảnh của A trở thành ngữ cảnh của mọi người, và sự ảo tưởng được "rửa sạch" thành sự thật.

Vấn đề không phải là trạng thái chia sẻ tự thân nó — mà là trạng thái chia sẻ **không có nguồn gốc và không có bộ xác minh độc lập**. Ba biện pháp giảm thiểu giải quyết vấn đề này:

1. **Gán nguồn gốc cho mọi lần ghi.** Mỗi mục trong trạng thái chia sẻ ghi lại ai đã viết, khi nào, dưới prompt nào, và (nếu có) nguồn nào mà tác nhân đã trích dẫn. Các tác nhân hạ nguồn đọc với sự hoài nghi dựa trên nguồn gốc.
2. **Đánh phiên bản các lần ghi; coi chúng là chỉ-thêm (append-only).** Một sự điều chỉnh là một mục mới thay thế mục cũ, không phải là cập nhật tại chỗ. Dấu vết kiểm toán được bảo tồn.
3. **Giữ ít nhất một tác nhân không thể ghi vào trạng thái chia sẻ.** Một tác nhân xác minh chỉ-đọc (read-only) lấy mẫu các mục, truy xuất lại các nguồn và gắn cờ các điểm không nhất quán. Vì nó không thể ghi vào pool, nó không thể bị nhiễm độc bởi pool.

### Tiền lệ Blackboard (Hayes-Roth, 1985)

Mô hình blackboard có trước các tác nhân LLM bốn thập kỷ. Hayes-Roth (1985, "A Blackboard Architecture for Control") đã mô tả các Nguồn Kiến thức (Knowledge Sources) chuyên biệt quan sát một blackboard toàn cục, đóng góp các giải pháp một phần và kích hoạt các nguồn khác. Blackboard năm 2026 (CA-MCP, Matrix) là cùng một mô hình với các tác nhân LLM đóng vai trò là Nguồn Kiến thức và các JSON blob là các giải pháp một phần. Tài liệu cũ đã ghi lại các giải pháp cho tranh chấp ghi, kiểm soát cơ hội và tính nhất quán mà các hệ thống hiện đại đang khám phá lại.

### Projection so với full view

Một blackboard thuần túy cung cấp cho mọi người đăng ký cùng một projection (theo chủ đề). Một thiết kế mạnh mẽ hơn là **projection theo từng tác nhân**: mỗi tác nhân nhận được một view tùy chỉnh theo vai trò của nó. Các state reducer của LangGraph là triển khai chuẩn mực năm 2026 — hàm reducer gộp trạng thái toàn cục thành một lát cắt cụ thể cho vai trò.

Projection theo từng tác nhân có khả năng mở rộng tốt hơn nhưng cần một schema. Nếu không có nó, bạn sẽ phải xây dựng lại projection ad-hoc trong prompt của mỗi tác nhân.

### Các mô hình tranh chấp ghi (write-contention)

Nhiều tác nhân ghi đồng thời là một vấn đề về đồng thời (concurrency), không chỉ là vấn đề của LLM. Ba mô hình hoạt động hiệu quả:

- **Sequential writer (single producer).** Tất cả các lần ghi đi qua một tác nhân điều phối duy nhất để tuần tự hóa. Đơn giản, nhưng là một nút thắt cổ chai.
- **Optimistic concurrency với versioning.** Mỗi mục có một phiên bản; các tác nhân ghi sẽ thất bại khi không khớp phiên bản và thử lại. Kỹ thuật cơ sở dữ liệu cổ điển.
- **Phân vùng chủ đề (Topic partitioning).** Các tác nhân khác nhau sở hữu các chủ đề khác nhau. Không có tranh chấp chéo chủ đề. Đòi hỏi thiết kế ranh giới phân vùng.

Hầu hết các framework năm 2026 mặc định sử dụng sequential writer vì các cuộc gọi LLM đủ chậm để tranh chấp hiếm khi xảy ra và nút thắt cổ chai không gây hại.

### Bộ xác minh không thể ghi (unwritable verifier)

Biện pháp giảm thiểu quan trọng nhất là bộ xác minh chỉ-đọc. Các quy tắc triển khai:

- Bộ xác minh chia sẻ trạng thái với nhóm (đọc blackboard hoặc pool).
- Bộ xác minh không có quyền ghi vào trạng thái chia sẻ — chỉ có quyền ghi vào một kênh xác minh riêng biệt.
- Bộ xác minh truy xuất độc lập các nguồn được trích dẫn trong các lần ghi. Gắn cờ các điểm không nhất quán.
- Đầu ra của bộ xác minh được định tuyến đến con người hoặc một tác nhân ra quyết định riêng biệt, không bao giờ được đưa ngược lại vào pool.

Nếu không có sự tách biệt này, đầu ra của bộ xác minh sẽ trở thành các mục mới trong pool, nghĩa là một pool bị nhiễm độc sẽ làm nhiễm độc bộ xác minh, từ đó làm nhiễm độc các xác minh của nó.

```figure
swarm-blackboard
```

## Xây dựng

`code/main.py` triển khai cả hai cấu trúc liên kết trong Python stdlib cùng với một cuộc tấn công nhiễm độc đồ chơi và ba biện pháp giảm thiểu.

- `MessagePool` — nhật ký chỉ-thêm an toàn cho luồng (thread-safe) với khả năng đọc toàn bộ.
- `Blackboard` — pub/sub theo chủ đề với các đăng ký theo từng tác nhân.
- `ProvenanceEntry` — mỗi lần ghi đều ghi lại (writer, timestamp, prompt_hash, source_uri).
- `PoisoningScenario` — chạy một nhiệm vụ nghiên cứu ba tác nhân nơi tác nhân A ảo tưởng về một dấu thập phân. In báo cáo cuối cùng.
- `Verifier` — một tác nhân chỉ-đọc truy xuất lại các nguồn và gắn cờ các điểm không nhất quán. Chạy cùng kịch bản với bộ xác minh hiện diện.

Chạy:

```
python3 code/main.py
```

Kết quả mong đợi:
- Chạy 1 (không có bộ xác minh): con số 42% ảo tưởng lan truyền vào báo cáo cuối cùng.
- Chạy 2 (có bộ xác minh): bộ xác minh gắn cờ sự không nhất quán, pool được dán nhãn "flagged", báo cáo cuối cùng bao gồm một phần đính chính.

## Sử dụng

`outputs/skill-memory-auditor.md` là một kỹ năng kiểm toán thiết kế bộ nhớ chia sẻ của bất kỳ hệ thống đa tác nhân nào về nguồn gốc, phiên bản và sự tách biệt bộ xác minh. Hãy chạy nó trên các kiến trúc đa tác nhân mới trước khi đưa vào production.

## Triển khai

Đối với bất kỳ thiết kế bộ nhớ chia sẻ nào:

- Ghi lại nguồn gốc trên mỗi lần ghi: `(writer, timestamp, prompt_hash, tool_calls_cited, source_uri)`.
- Làm cho nhật ký chỉ-thêm (append-only). Các chỉnh sửa là các mục mới tham chiếu đến mục bị thay thế.
- Triển khai ít nhất một tác nhân xác minh chỉ-đọc với quyền truy cập nguồn độc lập.
- Định tuyến đầu ra của bộ xác minh đến một kênh riêng biệt, không đưa ngược lại vào pool chia sẻ.
- Ghi nhật ký tỷ lệ các lần ghi là sự thay thế — tỷ lệ tăng dần là bằng chứng sớm của các mô hình ảo tưởng.

## Bài tập

1. Chạy `code/main.py`. Xác nhận lần chạy 1 lan truyền sự ảo tưởng và lần chạy 2 bắt được nó.
2. Thêm một sự ảo tưởng thứ hai: tác nhân B bịa đặt kích thước tập dữ liệu. Bộ xác minh sẽ bắt được cả hai mà không cần điều chỉnh thủ công cho từng cái.
3. Chuyển full pool sang blackboard với các phân vùng chủ đề (`prices`, `summaries`, `analyses`). Phân vùng chủ đề làm cho các kịch bản nhiễm độc nào khó thực hiện hơn, và nó không giúp ích cho kịch bản nào?
4. Đọc Hayes-Roth (1985, "A Blackboard Architecture for Control"). Xác định hai mô hình kiểm soát từ bài báo không được thảo luận trong bài học này mà các hệ thống năm 2026 sẽ được hưởng lợi.
5. Đọc CA-MCP (arXiv:2601.11595). Ánh xạ Shared Context Store của nó vào lớp MessagePool hoặc Blackboard trong `code/main.py`. CA-MCP thêm các primitive nào lên trên?

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|------------------------|
| Message pool | "Lịch sử chat chia sẻ" | Nhật ký chỉ-thêm mà mọi tác nhân đều đọc. Minh bạch hoàn toàn, khả năng mở rộng kém. |
| Blackboard | "Không gian làm việc chia sẻ" | Pub/sub theo chủ đề. Các tác nhân đăng ký các chủ đề liên quan. Khả năng mở rộng tốt hơn. |
| Provenance | "Ai đã viết cái gì" | Metadata trên mỗi lần ghi: tác nhân, thời gian, prompt, nguồn. |
| Memory poisoning | "Sự lan truyền ảo tưởng" | Lỗi của một tác nhân đi vào trạng thái chia sẻ, các tác nhân hạ nguồn chấp nhận nó như sự thật. |
| Append-only | "Không cập nhật tại chỗ" | Các chỉnh sửa là các mục mới thay thế mục cũ. Bảo tồn dấu vết kiểm toán. |
| Unwritable verifier | "Kiểm toán viên độc lập" | Tác nhân chỉ-đọc truy xuất lại nguồn và gắn cờ các điểm không nhất quán. |
| Projection | "View có phạm vi" | View theo từng tác nhân được tính toán từ trạng thái toàn cục. LangGraph reducer là trường hợp chuẩn mực. |
| Knowledge Source | "Tác nhân chuyên gia" | Thuật ngữ năm 1985 của Hayes-Roth cho một người tham gia blackboard. |

## Đọc thêm

- [Cemri và cộng sự — Tại sao các hệ thống LLM đa tác nhân thất bại?](https://arxiv.org/abs/2503.13657) — Phân loại MAST; memory poisoning là một phân họ của lỗi phối hợp.
- [CA-MCP — Context-Aware Multi-Server MCP](https://arxiv.org/abs/2601.11595) — Shared Context Store cho các máy chủ MCP phối hợp.
- [Matrix — framework đa tác nhân phi tập trung](https://arxiv.org/abs/2511.21686) — blackboard dựa trên hàng đợi tin nhắn không có bộ điều phối trung tâm.
- [Trạng thái và reducer của LangGraph](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — mô hình projection theo từng tác nhân trong production.
- [Anthropic — Cách chúng tôi xây dựng hệ thống nghiên cứu đa tác nhân](https://www.anthropic.com/engineering/multi-agent-research-system) — ghi chú về nguồn gốc và xác minh từ một triển khai production.