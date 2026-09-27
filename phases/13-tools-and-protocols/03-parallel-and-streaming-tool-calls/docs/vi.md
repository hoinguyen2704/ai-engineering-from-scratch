# Parallel Tool Calls and Streaming with Tools

> Ba yêu cầu tra cứu thời tiết độc lập nếu thực hiện tuần tự sẽ mất ba vòng lặp (round trips). Hãy chạy chúng song song và tổng thời gian sẽ giảm xuống bằng thời gian của yêu cầu chậm nhất. Mọi nhà cung cấp mô hình tiên phong hiện nay đều phát ra nhiều tool call trong một lượt phản hồi. Lợi ích là rất thực tế; nhưng kỹ thuật triển khai lại khá tinh tế. Bài học này sẽ đi qua hai phần: fan-out song song và tái cấu trúc các đối số (argument) được stream, với trọng tâm là cái bẫy tương quan id.

**Type:** Build
**Languages:** Python (stdlib, thread pool + streaming harness)
**Prerequisites:** Phase 13 · 02 (function calling deep dive)
**Time:** ~75 minutes

## Learning Objectives

- Giải thích lý do tại sao `parallel_tool_calls: true` tồn tại và khi nào nên vô hiệu hóa nó.
- Tương quan các đoạn (chunk) đối số được stream với đúng tool-call id trong quá trình fan-out song song.
- Tái cấu trúc các chuỗi `arguments` một phần thành JSON hoàn chỉnh mà không cần phân tích cú pháp (parse) quá sớm.
- Chạy benchmark thời tiết cho ba thành phố để chứng minh sự khác biệt về độ trễ giữa tuần tự và song song.

## The Problem

Nếu không có parallel calls, một agent trả lời câu hỏi "thời tiết ở Bengaluru, Tokyo và Zurich như thế nào" sẽ thực hiện như sau:

```
user -> LLM
LLM -> call get_weather(Bengaluru)
host -> run executor, reply with result
LLM -> call get_weather(Tokyo)
host -> run executor, reply with result
LLM -> call get_weather(Zurich)
host -> run executor, reply with result
LLM -> final text answer
```

Ba vòng lặp LLM, mỗi vòng lặp đều phải chịu thêm độ trễ của executor. Tổng thời gian thực tế (wall-clock time) gấp khoảng 4 lần so với lý tưởng.

Với parallel calls:

```
user -> LLM
LLM -> call get_weather(Bengaluru); call get_weather(Tokyo); call get_weather(Zurich)
host -> run all three executors concurrently, reply with three results
LLM -> final text answer
```

Chỉ một vòng lặp LLM. Thời gian thực thi là giá trị lớn nhất trong ba yêu cầu, không phải tổng của chúng. Các benchmark thực tế trên OpenAI, Anthropic và Gemini cho thấy mức giảm 60 đến 70 phần trăm thời gian thực tế trên các khối lượng công việc fan-out.

Cái giá phải trả là sự phức tạp trong việc tương quan. Khi ba yêu cầu hoàn thành không theo thứ tự, kết quả của bạn phải mang theo `tool_call_id` tương ứng để mô hình có thể khớp chúng lại. Khi kết quả được stream, bạn phải lắp ghép các đoạn đối số một phần thành JSON hoàn chỉnh trước khi thực thi. Gemini 3 đã thêm các id duy nhất một phần để giải quyết vấn đề thực tế là hai parallel call đến cùng một công cụ không thể phân biệt được với nhau.

## The Concept

### Enabling parallel

- **OpenAI.** `parallel_tool_calls: true` được bật mặc định. Thiết lập `false` để buộc thực hiện tuần tự.
- **Anthropic.** Parallel thông qua `disable_parallel_tool_use: false` (mặc định trên Claude 3.5 trở lên). Thiết lập `true` để thực hiện tuần tự.
- **Gemini.** Luôn có khả năng chạy song song; `tool_config.function_calling_config.mode = "AUTO"` cho phép mô hình tự quyết định.

Hãy vô hiệu hóa parallel khi các công cụ có phụ thuộc về thứ tự (`create_file` rồi đến `write_file`), khi đầu ra của một yêu cầu là đầu vào của yêu cầu khác, hoặc khi bộ giới hạn tốc độ (rate limiter) không thể xử lý fan-out.

### Id correlation

Mỗi yêu cầu mà mô hình phát ra đều có một `id`. Mọi kết quả mà host trả về phải bao gồm cùng id đó. Nếu không, các kết quả sẽ trở nên mơ hồ.

- **OpenAI.** `tool_call_id` trên mỗi tin nhắn có role là tool.
- **Anthropic.** `tool_use_id` trên mỗi khối `tool_result`.
- **Gemini.** `id` trên mỗi `functionResponse` (Gemini 3 trở lên; Gemini 2 khớp theo tên, điều này gây lỗi khi có các parallel call cùng tên).

### Running calls concurrently

Host chạy executor của mỗi yêu cầu trên thread, coroutine hoặc remote worker riêng. Harness đơn giản nhất sử dụng thread pool; trong môi trường production, người ta sử dụng asyncio với `asyncio.gather` hoặc structured concurrency. Thứ tự hoàn thành là không thể đoán trước — id chính là định danh.

Một lỗi phổ biến: trả lời kết quả theo thứ tự danh sách yêu cầu thay vì thứ tự hoàn thành. Điều này thường vẫn hoạt động vì mô hình chỉ quan tâm đến `tool_call_id`, nhưng nếu một kết quả bị mất hoặc trùng lặp, việc gửi không theo thứ tự sẽ gây khó khăn cho việc debug. Hãy ưu tiên trả lời theo thứ tự hoàn thành với các id rõ ràng.

### Streaming tool calls

Khi mô hình stream, `arguments` sẽ đến từng phần. Ba luồng chunk riêng biệt cho ba parallel call sẽ xen kẽ trên đường truyền. Bạn cần một bộ tích lũy (accumulator) cho mỗi id.

Hình thái theo nhà cung cấp:

- **OpenAI.** Mỗi chunk là `choices[0].delta.tool_calls[i].function.arguments` (chuỗi một phần). Chunk mang theo `index` (vị trí trong danh sách yêu cầu). Bạn tích lũy theo chỉ mục, đọc `id` khi nó xuất hiện lần đầu và phân tích JSON khi `finish_reason = "tool_calls"`.
- **Anthropic.** Các sự kiện stream là `message_start`, sau đó là một `content_block_start` cho mỗi khối với type là `tool_use` (chứa id, tên, input trống). Các sự kiện `content_block_delta` mang theo các chunk `input_json_delta`. `content_block_stop` đóng mỗi khối.
- **Gemini.** `streamFunctionCallArguments` (Gemini 3 trở lên) phát ra các chunk với `functionCallId` để các yêu cầu xen kẽ một cách sạch sẽ. Trước Gemini 3, streaming trả về từng yêu cầu hoàn chỉnh một.

### Partial JSON and the parse-early trap

Bạn không thể phân tích `arguments` cho đến khi nó hoàn tất. JSON một phần như `{"city": "Beng` không hợp lệ và sẽ gây lỗi. Cổng kiểm soát chính xác là tín hiệu kết thúc yêu cầu của nhà cung cấp: `finish_reason = "tool_calls"` của OpenAI, `content_block_stop` của Anthropic, hoặc sự kiện kết thúc stream của Gemini. Chỉ khi đó mới thử `json.loads`. Một cách tiếp cận mạnh mẽ hơn là sử dụng trình phân tích JSON tăng dần (incremental JSON parser) để tạo ra các sự kiện khi cấu trúc hoàn tất; hướng dẫn streaming của OpenAI khuyến nghị cách này để có UX hiển thị chỉ báo "đang suy nghĩ" trực tiếp. Việc đếm dấu ngoặc nhọn không đáng tin cậy để kiểm tra tính hoàn chỉnh (dấu ngoặc bên trong chuỗi trích dẫn hoặc nội dung đã escape sẽ gây ra kết quả dương tính giả) và chỉ nên được sử dụng như một heuristic debug không chính thức.

### Out-of-order completion

```
call_A: fast API, returns first
call_B: slow API, returns second
call_C: median API, returns third
```

Phản hồi của host vẫn phải trích dẫn các id:

```
[{role: "tool", tool_call_id: "call_A", content: ...},
 {role: "tool", tool_call_id: "call_B", content: ...},
 {role: "tool", tool_call_id: "call_C", content: ...}]
```

Thứ tự trong phản hồi không quan trọng đối với tính đúng đắn trên OpenAI hoặc Anthropic. Gemini chấp nhận bất kỳ thứ tự nào miễn là các id khớp nhau.

### Benchmark: sequential vs parallel

Harness trong `code/main.py` mô phỏng ba executor với độ trễ 400, 600 và 800 ms. Chạy tuần tự mất tổng cộng 1800 ms. Chạy song song mất max(400, 600, 800) = 800 ms. Sự khác biệt là hằng số, không phải tỷ lệ, vì vậy mức tiết kiệm tăng lên cùng với số lượng công cụ.

Lưu ý thực tế: parallel call gây áp lực lên các API hạ nguồn. Một fan-out 10 yêu cầu đến một dịch vụ bị giới hạn tốc độ sẽ thất bại. Phase 13 · 17 đề cập đến backpressure ở cấp gateway; ngữ nghĩa thử lại (retry) được lên kế hoạch cho một phase tương lai.

### Streaming fan-out wall-clock

Nếu bản thân mô hình stream, bạn có thể bắt đầu thực thi ngay khi đối số của một yêu cầu hoàn tất, thay vì đợi tất cả các yêu cầu hoàn tất. Đây là một tối ưu hóa mà OpenAI ghi lại nhưng không phải SDK nào cũng hỗ trợ. Harness trong bài học này thực hiện điều đó: ngay khi luồng mô phỏng tạo ra một đối tượng đối số hoàn chỉnh, host sẽ kích hoạt yêu cầu đó.

```figure
tp-parallel-fanout
```

## Use It

`code/main.py` có hai phần. Phần đầu chạy ba yêu cầu thời tiết mô phỏng tuần tự và song song bằng cách sử dụng `concurrent.futures.ThreadPoolExecutor` và in ra thời gian thực tế. Phần thứ hai phát lại một phản hồi streaming giả — các chunk của `arguments` cho ba parallel call xen kẽ trên một luồng — và lắp ghép chúng theo từng id với `StreamAccumulator`. Không LLM, không mạng, chỉ là logic lắp ghép.

Những điều cần quan sát:

- Bộ đếm thời gian tuần tự đạt 1.8 giây. Bộ đếm thời gian song song đạt 0.8 giây trên cùng các độ trễ giả lập.
- Bộ tích lũy xử lý các chunk đến không theo thứ tự bằng cách đệm theo từng id và chỉ phân tích khi JSON của mỗi yêu cầu hoàn tất.
- Executor kích hoạt ngay khi đối số của một id hoàn tất, không phải sau khi tất cả các luồng kết thúc.

## Ship It

Bài học này tạo ra `outputs/skill-parallel-call-safety-check.md`. Với một registry công cụ, kỹ năng này kiểm tra xem công cụ nào an toàn để chạy song song, công cụ nào có phụ thuộc về thứ tự và công cụ nào sẽ làm quá tải giới hạn tốc độ hạ nguồn — trả về một registry đã sửa đổi với các cờ `parallel_safe` cho từng công cụ.

## Exercises

1. Chạy `code/main.py` và thay đổi các độ trễ mô phỏng. Xác nhận rằng tỷ lệ song song trên tuần tự xấp xỉ `max/sum` (các lần chạy thực tế sai lệch một chút so với lý tưởng do lập lịch luồng, tuần tự hóa và overhead của harness). Tại phân phối độ trễ nào thì song song không còn quan trọng?

2. Mở rộng bộ tích lũy để xử lý trường hợp "yêu cầu bị hủy giữa chừng" bằng cách xóa bộ đệm của nó và phát ra sự kiện `cancelled`. Nhà cung cấp nào ghi lại trường hợp này một cách rõ ràng? Kiểm tra ngữ nghĩa `content_block_stop` của Anthropic và hành vi `finish_reason: "length"` của OpenAI.

3. Thay thế thread pool bằng `asyncio.gather`. Benchmark cả hai. Bạn sẽ thấy những lợi ích nhỏ trên async do chi phí chuyển đổi ngữ cảnh thấp hơn, nhưng chỉ khi các executor thực hiện I/O thực sự.

4. Chọn hai công cụ KHÔNG NÊN chạy song song (ví dụ: `create_file` rồi đến `write_file`). Thêm đồ thị `ordering_dependency` vào registry và chặn fan-out song song dựa trên đồ thị đó. Đây là cơ chế tối thiểu cho việc lập lịch nhận biết phụ thuộc, điều mà một phase kỹ thuật agent tương lai sẽ chính thức hóa.

5. Đọc phần parallel-function-calling của OpenAI và tài liệu `disable_parallel_tool_use` của Anthropic. Xác định một loại công cụ thực tế mà Anthropic khuyến nghị vô hiệu hóa tính song song. (Gợi ý: các thay đổi có hệ quả trên cùng một tài nguyên.)

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| Parallel tool calls | "Fan-out in one turn" | Mô hình phát ra nhiều tool call trong một tin nhắn assistant duy nhất |
| `parallel_tool_calls` | "OpenAI's flag" | Bật hoặc tắt việc phát ra nhiều yêu cầu |
| `disable_parallel_tool_use` | "Anthropic's inverse" | Cờ opt-out; mặc định là bật song song |
| Tool call id | "Correlation handle" | Định danh cho mỗi yêu cầu mà tin nhắn kết quả phải lặp lại |
| Accumulator | "Stream buffer" | Bộ đệm chuỗi theo từng id cho các chunk `arguments` một phần |
| Out-of-order completion | "Fastest first" | Các parallel call hoàn thành theo thứ tự không thể đoán trước; id là chất keo kết nối |
| Dependency graph | "Ordering constraints" | Các công cụ có đầu ra cung cấp cho đầu vào của công cụ khác; không thể chạy song song |
| Parse-early trap | "JSON.parse exploded" | Cố gắng phân tích một chuỗi `arguments` chưa hoàn chỉnh |
| `streamFunctionCallArguments` | "Gemini 3 feature" | Các chunk đối số được stream với id duy nhất cho mỗi yêu cầu |
| Completion-order reply | "Don't wait for all" | Trả lời kết quả ngay khi chúng đến, được khóa bởi id |

## Further Reading

- [OpenAI — Parallel function calling](https://platform.openai.com/docs/guides/function-calling#parallel-function-calling) — hành vi mặc định và cờ opt-out
- [Anthropic — Tool use: implementing tool use](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/implementing-tool-use) — `disable_parallel_tool_use` và batching kết quả
- [Google — Gemini function calling parallel section](https://ai.google.dev/gemini-api/docs/function-calling) — parallel call tương quan id từ Gemini 3
- [OpenAI — Streaming responses with tools](https://platform.openai.com/docs/api-reference/responses-streaming) — lắp ghép đối số dạng chunk cho các stream của OpenAI
- [Anthropic — Streaming messages](https://docs.anthropic.com/en/api/messages-streaming) — `content_block_delta` với `input_json_delta`