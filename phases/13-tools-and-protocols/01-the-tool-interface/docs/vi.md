# Giao diện Công cụ — Tại sao các Agent cần I/O có cấu trúc

> Một mô hình ngôn ngữ tạo ra các token. Một chương trình thực hiện các hành động. Khoảng cách giữa hai yếu tố đó chính là giao diện công cụ: một hợp đồng cho phép mô hình yêu cầu một hành động và máy chủ (host) thực thi nó. Mọi stack năm 2026 — function calling trên OpenAI, Anthropic và Gemini; `tools/call` của MCP; các task parts của A2A — đều là những cách mã hóa khác nhau của cùng một vòng lặp bốn bước. Bài học này đặt tên cho vòng lặp đó và chỉ ra các cơ chế tối thiểu để vận hành nó.

**Type:** Learn
**Languages:** Python (stdlib, no LLM)
**Prerequisites:** Phase 11 (LLM completion APIs)
**Time:** ~45 phút

## Mục tiêu học tập

- Giải thích lý do tại sao một LLM chỉ có khả năng tạo văn bản lại không thể tự mình thực hiện các hành động trong thế giới thực.
- Vẽ vòng lặp gọi công cụ bốn bước (mô tả → quyết định → thực thi → quan sát) và xác định chủ thể chịu trách nhiệm cho từng bước.
- Viết mô tả công cụ gồm ba phần: tên, input JSON Schema và hàm thực thi (executor) tất định.
- Phân biệt công cụ thuần túy (pure) và công cụ có tác dụng phụ (side-effecting), đồng thời nêu lý do tại sao sự phân tách này quan trọng đối với tính an toàn.

## Vấn đề

Một LLM phát ra một phân phối xác suất trên token tiếp theo. Đó là toàn bộ bề mặt đầu ra của nó. Nếu bạn hỏi một chat model "thời tiết ở Bengaluru hiện tại như thế nào", nó có thể viết một câu trả lời hợp lý, nhưng nó không thể kết nối vào một API thời tiết. Câu trả lời đó có thể đúng do trùng hợp hoặc đã cũ từ ba ngày trước.

Thu hẹp khoảng cách đó là mục đích của giao diện công cụ. Chương trình máy chủ — runtime của agent, Claude Desktop, ChatGPT, Cursor hoặc một script tùy chỉnh — quảng bá một danh sách các công cụ có thể gọi cho mô hình. Khi mô hình quyết định cần một hành động, nó sẽ phát ra một payload có cấu trúc nêu tên công cụ và các đối số của nó. Máy chủ phân tích payload đó, chạy công cụ thực tế và phản hồi kết quả lại. Vòng lặp tiếp tục cho đến khi mô hình quyết định không cần gọi thêm nữa.

Phiên bản đầu tiên của hợp đồng này được ra mắt vào tháng 6 năm 2023 dưới dạng tham số "functions" của OpenAI. Anthropic theo sau với các khối `tool_use` trong Claude 2.1. Gemini đã thêm `functionDeclarations` vài tháng sau đó. Mọi nhà cung cấp hiện nay đều hiển thị cùng một hình thái: danh sách công cụ theo định dạng JSON-Schema ở đầu vào, và lời gọi công cụ theo payload JSON ở đầu ra. Model Context Protocol (tháng 11 năm 2024) đã tổng quát hóa hợp đồng này để một registry công cụ có thể phục vụ mọi mô hình. A2A (tháng 4 năm 2026, v1.0) đã xếp chồng cùng một primitive này cho việc ủy quyền giữa các agent (agent-to-agent).

Vòng lặp bốn bước là bất biến nằm bên dưới tất cả những điều này. Mọi thứ khác trong Phase 13 đều là sự mở rộng.

## Khái niệm

### Bước một: mô tả (describe)

Máy chủ khai báo mỗi công cụ với ba trường.

- **Tên (Name).** Một định danh ổn định, máy có thể đọc được. `get_weather`, không phải "weather thing".
- **Mô tả (Description).** Một đoạn tóm tắt bằng ngôn ngữ tự nhiên. "Sử dụng khi người dùng hỏi về điều kiện hiện tại của một thành phố cụ thể. Không sử dụng cho dữ liệu lịch sử."
- **Input schema.** Một đối tượng JSON Schema (draft 2020-12) mô tả các đối số của công cụ.

Mô hình nhận danh sách này. Các nhà cung cấp hiện đại serialize các khai báo này vào system prompt bằng cách sử dụng template riêng của nhà cung cấp, vì vậy bạn với tư cách là người gọi chỉ cần xử lý dạng có cấu trúc.

### Bước hai: quyết định (decide)

Với tin nhắn của người dùng và các công cụ khả dụng, mô hình chọn một trong ba hành vi.

1. **Trả lời trực tiếp** bằng văn bản. Không gọi công cụ.
2. **Gọi một hoặc nhiều công cụ.** Phát ra các đối tượng gọi có cấu trúc. Dưới `parallel_tool_calls: true` (mặc định trên OpenAI và Gemini, tùy chọn trên Anthropic), mô hình có thể phát ra nhiều lời gọi trong một lượt.
3. **Từ chối.** Các structured output ở chế độ nghiêm ngặt (strict-mode) có thể tạo ra một khối `refusal` có kiểu thay vì một lời gọi.

Payload của một lời gọi công cụ có ba trường ổn định: một `id` lời gọi, một `name` công cụ và một đối tượng `arguments` JSON. Id tồn tại để máy chủ có thể tương quan kết quả sau đó với lời gọi cụ thể, điều này quan trọng khi các lời gọi song song trả về không theo thứ tự.

### Bước ba: thực thi (execute)

Máy chủ nhận lời gọi, xác thực các đối số dựa trên schema đã khai báo và chạy executor. Các đối số không hợp lệ có nghĩa là mô hình đã hallucinate (ảo tưởng) một trường hoặc sử dụng sai kiểu dữ liệu — một dạng lỗi rất phổ biến trên các mô hình yếu. Các máy chủ sản xuất (production) thực hiện một trong ba việc khi đối số không hợp lệ: dừng ngay lập tức và hiển thị lỗi cho mô hình, sửa JSON bằng trình phân tích cú pháp ràng buộc, hoặc thử lại với mô hình kèm theo lỗi xác thực trong prompt.

Bản thân executor là mã nguồn thông thường. Python, TypeScript, lệnh shell, truy vấn cơ sở dữ liệu. Nó tạo ra một kết quả, thường là một chuỗi nhưng có thể là bất kỳ giá trị JSON nào hoặc một khối nội dung có cấu trúc (văn bản, hình ảnh hoặc tham chiếu tài nguyên trong MCP). Kết quả phải có khả năng serialize.

### Bước bốn: quan sát (observe)

Máy chủ thêm kết quả công cụ vào cuộc hội thoại (dưới dạng tin nhắn có vai trò `tool` với `id` tương ứng) và gọi lại mô hình. Mô hình bây giờ đã có đầu ra của công cụ trong ngữ cảnh và có thể đưa ra câu trả lời cuối cùng hoặc yêu cầu thêm các lời gọi khác. Quá trình này tiếp tục cho đến khi mô hình ngừng phát ra các lời gọi hoặc máy chủ đạt đến giới hạn an toàn về số lần lặp.

### Sự phân tách tin cậy (trust split)

Các công cụ có hai loại quan trọng đối với tính an toàn.

- **Thuần túy (Pure).** Chỉ đọc, tất định, không có tác dụng phụ. `get_weather`, `search_docs`, `get_current_time`. An toàn để gọi thử nghiệm.
- **Hệ quả (Consequential).** Thay đổi trạng thái, tiêu tốn tiền, truy cập dữ liệu người dùng. `send_email`, `delete_file`, `execute_trade`. Phải được kiểm soát (gated).

"Quy tắc hai" (Rule of Two) năm 2026 của Meta về bảo mật agent cho biết một lượt (turn) có thể kết hợp tối đa hai trong số: đầu vào không đáng tin cậy, dữ liệu nhạy cảm, hành động có hệ quả. Giao diện công cụ là nơi bạn thực thi quy tắc đó — bằng cách từ chối các lời gọi, yêu cầu xác nhận của người dùng hoặc nâng cấp phạm vi quyền hạn. Xem Phase 13 · 15 để biết chương bảo mật đầy đủ và Phase 14 · 09 cho các chính sách cấp phép ở cấp độ agent.

### Vòng lặp nằm ở đâu

| Ngữ cảnh | Ai mô tả | Ai quyết định | Ai thực thi |
|---------|---------------|-------------|--------------|
| Function calling đơn lượt (OpenAI/Anthropic/Gemini) | Lập trình viên ứng dụng | LLM | Lập trình viên ứng dụng |
| MCP | MCP server | LLM qua MCP client | MCP server |
| A2A | Nhà xuất bản Agent Card | Agent gọi | Agent được gọi |
| Trình duyệt web (agent gọi hàm) | Tiện ích mở rộng / WebMCP | LLM | Runtime trình duyệt |

Ở mọi nơi, đều là bốn bước giống nhau. Tên cột thay đổi; cấu trúc thì không.

### Tại sao không chỉ prompt mô hình phát ra JSON?

"Yêu cầu mô hình trả lời bằng JSON" là mô hình trước khi có function calling. Nó thất bại khoảng 5 đến 15 phần trăm thời gian trên các mô hình tiên phong và nhiều hơn nữa trên các mô hình nhỏ hơn. Các dạng lỗi bao gồm thiếu dấu ngoặc nhọn, dấu phẩy thừa, các trường ảo tưởng và sai kiểu dữ liệu. Sau đó, bạn cần một bước sửa lỗi JSON, thử lại hoặc bộ giải mã có ràng buộc.

Native function calling tốt hơn vì ba lý do. Thứ nhất, nhà cung cấp huấn luyện mô hình end-to-end trên hình thái lời gọi chính xác, vì vậy tỷ lệ JSON hợp lệ tăng lên 98 đến 99 phần trăm ở chế độ nghiêm ngặt. Thứ hai, payload lời gọi nằm trong khe giao thức riêng của nó, không nằm trong văn bản tự do — vì vậy một lời gọi công cụ không bao giờ bị rò rỉ vào câu trả lời hiển thị cho người dùng. Thứ ba, các nhà cung cấp thực thi tuân thủ schema bằng cách giải mã có ràng buộc (chế độ nghiêm ngặt của OpenAI, `tool_use` của Anthropic, `responseSchema` của Gemini). Đầu ra được đảm bảo xác thực.

Phase 13 · 02 so sánh ba API của nhà cung cấp cạnh nhau. Phase 13 · 04 đi sâu vào các đầu ra có cấu trúc.

### Bộ ngắt mạch (Circuit breakers)

Vòng lặp kết thúc khi mô hình ngừng phát ra các lời gọi hoặc máy chủ đạt đến số lượt tối đa. Các máy chủ sản xuất đặt con số này từ 5 đến 20 lượt. Vượt quá con số đó, gần như chắc chắn bạn đang ở trong một vòng lặp mà mô hình không thể thoát ra. Claude Code mặc định là 20; OpenAI Assistants là 10; chế độ agent của Cursor là 25.

Giải pháp thay thế — các vòng lặp không giới hạn — xuất hiện mỗi sáu tháng dưới dạng các bài báo cáo "agent tiêu tốn 400 đô la tiền gọi API qua đêm". Đừng phát hành mà không có giới hạn.

Phase 14 · 12 đề cập sâu về phục hồi lỗi và tự chữa lành; Phase 17 đề cập đến giới hạn tốc độ sản xuất.

### Phase 13 sẽ đi đến đâu từ đây

- Các bài học từ 02 đến 05 hoàn thiện bề mặt gọi công cụ ở cấp độ nhà cung cấp.
- Các bài học từ 06 đến 14 tổng quát hóa vòng lặp thành MCP.
- Các bài học từ 15 đến 18 bảo vệ vòng lặp trước các máy chủ thù địch, người dùng đối nghịch và các bề mặt xác thực từ xa không được kiểm chứng.
- Các bài học từ 19 đến 22 mở rộng mô hình sang cộng tác giữa các agent, khả năng quan sát, định tuyến và đóng gói.
- Bài học 23 phát hành một hệ sinh thái hoàn chỉnh sử dụng mọi primitive.

Mọi bài học còn lại đều là sự mở rộng của vòng lặp bốn bước này. Hãy ghi nhớ nó như một bất biến.

```figure
tp-tool-loop
```

## Sử dụng nó

`code/main.py` chạy vòng lặp bốn bước mà không cần LLM. Một hàm "decider" giả lập mô hình bằng cách khớp mẫu trên tin nhắn người dùng; executor, trình xác thực schema và bộ khung bước quan sát là thật. Hãy chạy nó để xem toàn bộ vũ đạo yêu cầu/phản hồi với trạng thái trung gian có thể in ra, sau đó thay thế decider giả bằng bất kỳ nhà cung cấp thực tế nào trong bài học sau.

Những điều cần xem xét:

- Registry công cụ giữ ba trường cho mỗi công cụ: tên, mô tả, schema và tham chiếu executor.
- Trình xác thực là một tập con JSON Schema tối thiểu (types, required, enum, min/max) chỉ viết bằng stdlib. Phase 13 · 04 sẽ cung cấp một bản đầy đủ hơn.
- Vòng lặp giới hạn số lần lặp ở mức năm. Các agent sản xuất cần chính xác loại bộ ngắt mạch này.

## Phát hành nó

Bài học này tạo ra `outputs/skill-tool-interface-reviewer.md`. Với một bản nháp định nghĩa công cụ (tên + mô tả + schema + dàn ý executor), kỹ năng này kiểm tra tính phù hợp với vòng lặp: tên có ổn định với máy không, mô tả có phải là bản tóm tắt sử dụng đầy đủ không, schema có sử dụng đúng JSON Schema 2020-12 không và phân loại thuần túy-so-với-hệ quả có rõ ràng không.

## Bài tập

1. Thêm công cụ thứ tư vào `code/main.py` có tên là `get_stock_price(ticker)`. Viết mô tả của nó là "Sử dụng khi người dùng hỏi giá cổ phiếu hiện tại theo mã ticker. Không sử dụng cho giá lịch sử hoặc tóm tắt thị trường." Chạy bộ khung và xác nhận decider giả định tuyến các truy vấn đề cập đến ticker đến công cụ mới.

2. Phá vỡ trình xác thực schema. Truyền một lời gọi mà đối tượng `arguments` của nó thiếu một trường bắt buộc và xác nhận máy chủ từ chối nó trước khi thực thi. Sau đó truyền một lời gọi với một trường lạ không xác định. Quyết định: máy chủ nên từ chối hay bỏ qua? Biện minh cho lựa chọn của bạn bằng một lập luận về an toàn.

3. Phân loại từng công cụ trong bộ khung là thuần túy hay có hệ quả. Thêm cờ `consequential: true` vào các mục registry cần thiết và thay đổi vòng lặp để in dòng "sẽ xác nhận với người dùng" bất cứ khi nào một công cụ có hệ quả được chọn. Đây là hình thái của cổng xác nhận mà mọi máy chủ sản xuất cần.

4. Vẽ vòng lặp bốn bước trên giấy với bảng cột nhà cung cấp ở trên được điền cho client yêu thích của bạn (Claude Desktop, Cursor, ChatGPT hoặc một stack tùy chỉnh). Đối chiếu với biến thể dành riêng cho MCP trong Phase 13 · 06.

5. Đọc hướng dẫn function calling của OpenAI từ đầu đến cuối. Xác định một trường nằm trong yêu cầu nhưng không nằm trong vòng lặp bốn bước như đã trình bày ở đây. Giải thích nó thêm vào điều gì và tại sao nó thuận tiện thay vì thiết yếu.

## Thuật ngữ chính

| Thuật ngữ | Mọi người nói | Ý nghĩa thực sự |
|------|----------------|------------------------|
| Tool | "Một thứ mô hình có thể gọi" | Một bộ ba gồm tên + input kiểu JSON-Schema + hàm executor |
| Function calling | "Sử dụng công cụ gốc" | Hỗ trợ API cấp nhà cung cấp để phát ra các lời gọi công cụ có cấu trúc thay vì văn xuôi |
| Tool call | "Yêu cầu hành động của mô hình" | Một payload JSON với `id`, `name`, `arguments` do mô hình phát ra |
| Tool result | "Những gì công cụ trả về" | Đầu ra của executor, được bọc trong tin nhắn vai trò `tool` với id tương ứng |
| Parallel tool calls | "Nhiều lời gọi cùng lúc" | Nhiều đối tượng lời gọi trong một lượt mô hình, độc lập và có thể sắp xếp theo id |
| Strict mode | "JSON được đảm bảo" | Giải mã có ràng buộc buộc đầu ra của mô hình phải xác thực theo schema đã khai báo |
| Pure tool | "Công cụ chỉ đọc" | Không có tác dụng phụ; an toàn để chạy lại |
| Consequential tool | "Công cụ hành động" | Thay đổi trạng thái bên ngoài; yêu cầu cổng, kiểm toán hoặc xác nhận của người dùng |
| Four-step loop | "Chu kỳ gọi công cụ" | mô tả → quyết định → thực thi → quan sát |
| Host | "Agent runtime" | Chương trình giữ registry công cụ, gọi mô hình và chạy executor |

## Đọc thêm

- [OpenAI — Hướng dẫn function calling](https://platform.openai.com/docs/guides/function-calling) — tài liệu tham khảo chính thức cho các khai báo công cụ và hình thái lời gọi kiểu OpenAI
- [Anthropic — Tổng quan về sử dụng công cụ](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) — định dạng khối `tool_use` / `tool_result` của Claude
- [Google — Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling) — `functionDeclarations` và ngữ nghĩa lời gọi song song trong Gemini
- [Model Context Protocol — Đặc tả 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28) — sự tổng quát hóa giao diện công cụ không trạng thái, không phụ thuộc nhà cung cấp hiện nay
- [JSON Schema — Ghi chú phát hành 2020-12](https://json-schema.org/draft/2020-12/release-notes) — phương ngữ schema mà mọi API công cụ hiện đại đều sử dụng