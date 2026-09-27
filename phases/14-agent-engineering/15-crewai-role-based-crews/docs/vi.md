# Role-Based Agent Teams — Roles, Tasks, Processes

> Bốn thành phần cơ bản: Agent, Task, Crew, Process. Hai hình thái cấp cao: Crews (cộng tác tự chủ, dựa trên vai trò) và Flows (hướng sự kiện, có tính xác định). CrewAI là bản triển khai tham chiếu cho năm 2026, và tài liệu của nó rất thẳng thắn: "đối với bất kỳ ứng dụng nào sẵn sàng cho môi trường production, hãy bắt đầu với một Flow."

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 12 (Workflow Patterns), Phase 14 · 14 (Actor Model)
**Time:** ~75 phút

## Mục tiêu học tập

- Gọi tên bốn thành phần cơ bản của CrewAI (Agent, Task, Crew, Process) và những gì mỗi thành phần quản lý.
- Phân biệt các quy trình Sequential, Hierarchical và Consensus (đang được lên kế hoạch); chọn một quy trình cho mỗi khối lượng công việc.
- Phân biệt Crews (dựa trên vai trò tự chủ) với Flows (hướng sự kiện, có tính xác định), và giải thích khuyến nghị về production trong tài liệu.
- Kết nối các công cụ với decorator `@tool` và lớp con `BaseTool`; lập luận về structured outputs so với văn bản tự do.
- Gọi tên bốn loại bộ nhớ của CrewAI và thời điểm mỗi loại phát huy tác dụng.
- Triển khai một crew gồm ba agent (nghiên cứu, viết, biên tập) bằng stdlib để tạo ra một bản tóm tắt.
- Phát hiện ba chế độ thất bại của CrewAI: prompt-bloat, thuế manager-LLM, và các điểm chuyển giao mong manh.

## Vấn đề

Các nhóm áp dụng framework đa agent đều gặp chung một bức tường. "Cộng tác tự chủ" nghe rất tuyệt trong bản demo. Sau đó, khách hàng báo lỗi và bạn cần khả năng phát lại (replay) có tính xác định. Hoặc bộ phận tài chính hỏi chi phí cho mỗi lần chạy của một crew được điều phối bởi LLM là bao nhiêu. Hoặc bộ phận trực ca cần biết agent nào bị treo lúc 3 giờ sáng.

Các crew được điều phối bởi LLM dạng tự do không trả lời rõ ràng cho bất kỳ câu hỏi nào trong số đó. Các DAG thuần túy trả lời được tất cả nhưng lại mất đi hình thái khám phá mà một agent động não cần.

Sự phân chia của CrewAI rất trung thực về sự đánh đổi này. Crews dành cho công việc cộng tác, dựa trên vai trò, mang tính khám phá. Flows dành cho công việc production hướng sự kiện, sở hữu bởi mã nguồn, có thể kiểm toán. Cùng một framework, hai hình thái, hãy chọn tùy theo bề mặt ứng dụng.

## Khái niệm

### Bốn thành phần cơ bản

Bề mặt của CrewAI rất nhỏ. Hãy ghi nhớ điều này và phần còn lại chỉ là cấu hình.

- **Agent.** `role + goal + backstory + tools + (optional) llm`. Backstory đóng vai trò chịu tải. Nó định hình giọng điệu, khả năng phán đoán, và thời điểm agent dừng lại. Tools là các hàm mà agent có thể gọi (xem thêm bên dưới).
- **Task.** `description + expected_output + agent + (optional) context + (optional) output_pydantic`. Một đơn vị công việc có thể tái sử dụng. `expected_output` là hợp đồng. `context` liệt kê các task thượng nguồn mà kết quả của chúng được truyền vào. `output_pydantic` ép buộc một hình thái có cấu trúc.
- **Crew.** Container. Sở hữu danh sách `agents`, danh sách `tasks`, `process`, và các cài đặt tùy chọn `memory` + `verbose` + `manager_llm`.
- **Process.** Chiến lược thực thi. Sequential, Hierarchical, Consensus (đã lên kế hoạch). Chọn hình thái cho lần chạy.

Các agent không nhìn thấy nhau trực tiếp. Các task tham chiếu đến các agent. Crew sắp xếp thứ tự các task. Process quyết định ai chọn task tiếp theo. Đó là toàn bộ mô hình tư duy.

> **Được xác thực với** CrewAI 0.86 (2026-05). Các phiên bản mới hơn có thể đổi tên hoặc hợp nhất các loại process; hãy kiểm tra [tài liệu CrewAI Processes](https://docs.crewai.com/concepts/processes) trước khi dựa vào một hình thái cụ thể.

### Sequential vs Hierarchical vs Consensus

- **Sequential.** Các task chạy theo thứ tự khai báo. Kết quả của task N có sẵn dưới dạng `context` cho task N+1. Chi phí thấp nhất. Dễ dự đoán nhất. Sử dụng khi thứ tự là cố định.
- **Hierarchical.** Một Agent quản lý (gọi LLM riêng) điều phối giữa các chuyên gia. CrewAI tạo ra người quản lý từ cấu hình `manager_llm` của bạn hoặc mặc định. Người quản lý chọn task tiếp theo mỗi vòng và có thể từ chối hoặc điều phối lại. Sử dụng khi bạn có từ bốn chuyên gia trở lên và thứ tự thực sự phụ thuộc vào kết quả trước đó.
- **Consensus.** Đã lên kế hoạch, hiện chưa được triển khai trong API công khai. Tài liệu dành tên này cho một quy trình dựa trên bỏ phiếu trong tương lai. Đừng dựa vào nó ở thời điểm hiện tại.

Hierarchical thêm một lệnh gọi LLM mỗi vòng (người quản lý) trên mỗi lệnh gọi chuyên gia. Chi phí token có thể tăng gấp ba trong một lần chạy năm bước. Chỉ trả phí cho nó khi bạn cần sự điều phối.

### Crews vs Flows

Đây là khung tư duy mà tài liệu dẫn dắt vào năm 2026.

- **Crew.** Sự tự chủ do LLM điều khiển. Framework chọn hình thái tại thời điểm chạy. Tốt cho: nghiên cứu, động não, bản nháp đầu tiên, bất cứ nơi nào lộ trình là một phần của câu trả lời. Khó phát lại. Khó kiểm thử. Chi phí tạo mẫu thấp.
- **Flow.** Đồ thị hướng sự kiện do bạn sở hữu. `@start` đánh dấu điểm vào. `@listen(topic)` đánh dấu một bước kích hoạt khi một bước khác phát ra chủ đề đó. Mỗi bước là Python thuần (có thể gọi một Crew bên trong). Tốt cho: production. Có thể quan sát. Có thể kiểm thử. Có tính xác định.

Khuyến nghị production năm 2026 của tài liệu: hãy bắt đầu với một Flow. Lồng ghép các Crew dưới dạng các lệnh gọi `Crew.kickoff()` từ bên trong các bước của Flow khi sự tự chủ xứng đáng với chi phí của nó. Flow cung cấp cho bạn dấu vết kiểm toán, Crew cung cấp cho bạn sự khám phá. Hãy kết hợp, đừng chỉ chọn một.

### Tích hợp công cụ

Ba cách để cung cấp công cụ cho một Agent. Hãy chọn cách đơn giản nhất phù hợp.

1. **Decorator `@tool`.** Các hàm thuần túy trở thành công cụ. Chữ ký là schema; docstring là mô tả mà LLM nhìn thấy. Tốt nhất cho các trình trợ giúp một lần.

   ```python
   from crewai.tools import tool

   @tool("Search the web")
   def search(query: str) -> str:
       """Return top results for the query."""
       return run_search(query)
   ```

2. **Lớp con `BaseTool`.** Công cụ dựa trên lớp với schema đối số rõ ràng, hỗ trợ async, thử lại. Sử dụng khi công cụ có trạng thái (client, cache) hoặc cần các đối số có cấu trúc.

   ```python
   from crewai.tools import BaseTool
   from pydantic import BaseModel

   class SearchArgs(BaseModel):
       query: str
       limit: int = 10

   class SearchTool(BaseTool):
       name = "web_search"
       description = "Search the web and return top results."
       args_schema = SearchArgs

       def _run(self, query: str, limit: int = 10) -> str:
           return self.client.search(query, limit=limit)
   ```

3. **Bộ công cụ tích hợp.** CrewAI cung cấp các bộ điều hợp chính chủ: `SerperDevTool`, `FileReadTool`, `DirectoryReadTool`, `CodeInterpreterTool`, `RagTool`, `WebsiteSearchTool`. Kết nối với một lệnh import.

Các kết quả có cấu trúc sử dụng Pydantic. Truyền `output_pydantic=MyModel` vào Task. CrewAI xác thực phản hồi của LLM dựa trên model và thực hiện ép kiểu hoặc thử lại. Kết hợp điều này với một chuỗi `expected_output` chặt chẽ. Kết quả văn bản tự do phù hợp cho bản nháp; kết quả có cấu trúc là thứ mà các Flow hạ nguồn có thể tiêu thụ.

### Các hook bộ nhớ

CrewAI cung cấp bốn loại bộ nhớ ngay khi cài đặt. Chúng kết hợp với nhau: một Crew có thể kích hoạt cả bốn cùng lúc.

> **Được xác thực với** CrewAI 0.86 (2026-05). Các bản phát hành gần đây định tuyến mọi thứ thông qua hệ thống `Memory` thống nhất bao bọc bốn kho lưu trữ này. Mô hình khái niệm bên dưới vẫn giữ nguyên, nhưng bề mặt lớp công khai có thể thu gọn thành một điểm truy cập `Memory` duy nhất trong các phiên bản mới hơn; hãy kiểm tra [tài liệu bộ nhớ CrewAI](https://docs.crewai.com/concepts/memory) để biết API hiện tại.

- **Short-term.** Bộ đệm hội thoại trong một lần chạy duy nhất. Bị xóa khi kết thúc.
- **Long-term.** Được lưu trữ bền vững qua các lần chạy. Lưu trong vector DB (mặc định là Chroma, có thể thay thế). Được truy xuất theo độ tương đồng với task hiện tại.
- **Entity.** Các sự kiện theo thực thể. "Khách hàng X đang sử dụng gói doanh nghiệp." Được khóa theo thực thể, không phải theo độ tương đồng. Tồn tại qua các lần chạy.
- **Contextual.** Truy xuất tại thời điểm lắp ráp. Kéo bộ nhớ liên quan tại thời điểm Agent cần, không tải trước.

Kích hoạt trên Crew với `memory=True` hoặc cấu hình theo từng loại. Được hỗ trợ bởi nhà cung cấp embedding mà bạn cấu hình (mặc định là OpenAI, có thể thay thế bằng local). Bộ nhớ là một trong những nơi CrewAI chứng minh giá trị so với các framework mỏng hơn; LangGraph thuần túy yêu cầu bạn phải tự kết nối từng loại này.

### Khi nào các nhóm dựa trên vai trò phù hợp

- Ba đến sáu agent với các vai trò được đặt tên và quy trình làm việc cộng tác. Soạn thảo, đánh giá, lập kế hoạch, động não.
- Điều phối nơi khả năng phán đoán của LLM về bước tiếp theo là một phần của giá trị (Hierarchical).
- Bất cứ nơi nào nhóm cảm thấy thoải mái hơn khi đọc `role + goal + backstory` thay vì đọc định nghĩa đồ thị.

### Khi nào chúng không phù hợp

- Các DAG có tính xác định với thứ tự nghiêm ngặt. Hãy sử dụng LangGraph (Bài 13). Hình thái đồ thị là sự trừu tượng hóa đúng đắn; khung vai trò của CrewAI gây ra ma sát.
- Ngân sách độ trễ dưới một giây. Hierarchical thêm các vòng lặp. Ngay cả Sequential cũng tuần tự hóa các prompt bao gồm backstory và kết quả trước đó.
- Các vòng lặp đơn agent. Hãy bỏ qua framework; một vòng lặp agent (Bài 1) cộng với một registry công cụ sẽ ngắn gọn hơn.

Bài 17 (Đánh đổi Framework Agent) trình bày điều này trong một ma trận. Phiên bản ngắn gọn: CrewAI nằm ở góc "cộng tác dựa trên vai trò".

### Hình thái phụ thuộc

Độc lập với LangChain. Python 3.10 đến 3.13. Sử dụng `uv`. Số sao: xem [crewAIInc/crewAI](https://github.com/crewAIInc/crewAI) (ảnh chụp nhanh tính đến 2026-05). Tích hợp AWS Bedrock đã được ghi lại; các điểm chuẩn của nhà cung cấp báo cáo tốc độ tăng đáng kể so với LangGraph trên các khối lượng công việc QA, nhưng phương pháp luận (tập dữ liệu, phần cứng, chỉ số đánh giá) không được công bố, vì vậy hãy coi các con số của nhà cung cấp framework chỉ mang tính định hướng.

### Nơi mô hình này đi chệch hướng

- **Prompt-bloat từ backstory.** Một backstory 2000 từ mỗi agent và một crew năm agent sẽ đốt cháy ngân sách ngữ cảnh trước khi lệnh gọi công cụ đầu tiên diễn ra. Giữ backstory dưới 200 từ. Tái sử dụng các cụm từ giữa các agent; đừng lặp lại phong cách công ty năm lần.
- **Thuế token manager-LLM.** Quy trình Hierarchical thêm một lệnh gọi LLM quản lý trước mỗi lệnh gọi chuyên gia. Trên một crew năm task, đó là sáu lệnh gọi LLM thay vì năm, và lệnh gọi quản lý mang theo toàn bộ danh sách task cộng với kết quả trước đó. Chuyển sang Sequential trừ khi việc điều phối phụ thuộc vào kết quả.
- **Chuyển giao mong manh.** `expected_output` của Task N là "một dàn ý". Task N+1 đọc nó dưới dạng `context` và cố gắng phân tích ba phần. LLM đã tạo ra bốn phần. Agent hạ nguồn phải ứng biến. Khắc phục bằng `output_pydantic` trên Task N để Task N+1 đọc một đối tượng có kiểu, không phải văn bản tự do.
- **Crew-as-prod.** Crew dạng tự do được đưa vào production mà không có wrapper Flow. Độ biến thiên kết quả cao; không thể phát lại; bộ phận trực ca không thể so sánh một lần chạy xấu với một lần chạy tốt. Hãy bao bọc bằng một Flow.

```figure
ae-crew-vs-flow
```

## Xây dựng

`code/main.py` triển khai các phiên bản stdlib của cả hai hình thái cộng với một crew ba agent.

Hình thái:

- `Agent`, `Task` các dataclass khớp với bề mặt của CrewAI.
- `SequentialCrew.kickoff(inputs)` chạy các task theo thứ tự khai báo, luồng kết quả dưới dạng `context`.
- `HierarchicalCrew.kickoff(topic)` thêm một Agent quản lý chọn chuyên gia tiếp theo mỗi vòng, dừng lại khi "xong".
- `Flow` với các decorator `@start` và `@listen(topic)`, một vòng lặp sự kiện nhỏ, và một dấu vết (trace).
- Decorator `tool(name)` phản chiếu hình thái `@tool` của CrewAI.
- `Memory` với các kho lưu trữ `short_term`, `long_term`, `entity`; độ tương đồng giả lập sử dụng numpy.
- Các phản hồi LLM giả lập là các chuỗi được mã hóa cứng dựa trên vai trò cộng với tiền tố đầu vào. Không có mạng. Có tính xác định.

Demo cụ thể: crew nghiên cứu, viết, biên tập tạo ra một bản tóm tắt về "agent engineering 2026". Người nghiên cứu lấy các nguồn (giả lập). Người viết soạn thảo. Người biên tập thắt chặt. Cùng một crew chạy qua một Flow để hiển thị hình thái có tính xác định.

Chạy nó:

```bash
python3 code/main.py
```

Dấu vết bao gồm: crew tuần tự luồng kết quả qua `context`, crew phân cấp với các lựa chọn của người quản lý (nghiên cứu, viết, biên tập, sau đó là "xong"), flow chạy ba bước tương tự với các chủ đề rõ ràng (`researched`, `drafted`, `edited`), các lệnh gọi công cụ được định tuyến qua `@tool`, và bộ nhớ dài hạn tồn tại qua hai lần khởi chạy.

Dấu vết Crew rất linh hoạt; người quản lý về nguyên tắc có thể sắp xếp lại. Dấu vết Flow là cố định. Sự lựa chọn đó chính là bài học.

## Sử dụng

- **CrewAI Flow** cho production. Ngay cả khi Flow chỉ là một bước gọi `Crew.kickoff()`. Flow cung cấp ranh giới kiểm toán.
- **CrewAI Crew (Sequential)** cho công việc cộng tác có thứ tự rõ ràng, đặc biệt là các bản nháp đầu tiên và vòng lặp đánh giá.
- **CrewAI Crew (Hierarchical)** khi việc điều phối phụ thuộc vào kết quả và bạn có từ bốn chuyên gia trở lên.
- **LangGraph** (Bài 13) cho các máy trạng thái rõ ràng, khả năng tiếp tục bền vững, thứ tự nghiêm ngặt.
- **AutoGen v0.4** (Bài 14) cho tính đồng thời của mô hình actor và cách ly lỗi.
- **OpenAI Agents SDK** (Bài 16) cho các sản phẩm ưu tiên OpenAI với các điểm chuyển giao và rào chắn.
- **Claude Agent SDK** (Bài 17) cho các sản phẩm ưu tiên Claude với các subagent và lưu trữ phiên.

## Triển khai

`outputs/skill-crew-or-flow.md` chọn Crew vs Flow cho một task và tạo khung triển khai tối thiểu. Từ chối cứng đối với Crew-không-có-backstory, Flow-không-có-chủ-đề-rõ-ràng, Hierarchical với dưới ba chuyên gia.

## Cạm bẫy

- **Backstory như hương vị.** Nó định hình kết quả. Kiểm tra ba biến thể mỗi agent; sự biến thiên là có thật. Chọn một, đóng băng nó.
- **Bỏ qua `expected_output`.** Nếu không có hợp đồng mỗi task, các task hạ nguồn sẽ lấy bất cứ thứ gì LLM tạo ra. Crew chạy; kiểm toán thất bại.
- **Bộ nhớ luôn bật.** Ghi dài hạn mỗi lần chạy. Vector DB phát triển. Truy xuất trở nên ồn ào. Giới hạn ghi vào các task nơi sự kiện đó là bền vững.
- **Manager prompt drift.** Prompt của người quản lý trong Hierarchical là ngầm định. Nếu việc điều phối trở nên kỳ lạ, hãy đổ nó ra chế độ verbose và đọc.
- **Tác dụng phụ của công cụ trong Crews.** Một Crew có thể gọi một công cụ nhiều lần hơn dự kiến. POST, DELETE, thanh toán thuộc về một bước Flow, không bao giờ là một công cụ Crew.

## Bài tập

1. Chuyển đổi crew Sequential thành một Flow. Đếm các điểm tiếp xúc nơi độ biến thiên giảm xuống. Lưu ý nơi khả năng đọc giảm xuống.
2. Thêm bộ nhớ thực thể vào crew: các sự kiện về khách hàng tồn tại qua các lần khởi chạy. Xác minh việc truy xuất kéo đúng thực thể.
3. Triển khai quy trình Hierarchical nơi người quản lý từ chối điều phối đến người biên tập cho đến khi kết quả của người viết có ít nhất ba đoạn văn. Theo dõi việc thử lại.
4. Kết nối một lớp con `BaseTool` cho một tìm kiếm web (giả lập). So sánh hình thái dấu vết với phiên bản decorator `@tool`.
5. Thêm `output_pydantic=Brief` vào task biên tập, nơi `Brief` có `title`, `summary`, `sections`. Làm cho task viết tạo ra JSON sai định dạng một lần; xác minh hành vi thử lại của CrewAI trong dấu vết.
6. Đọc phần giới thiệu tài liệu của CrewAI. Chuyển đổi toy sang API `crewai` thực tế. Phiên bản stdlib đã bỏ qua những đảm bảo nào?
7. Kết nối AgentOps hoặc Langfuse (Bài 24) với một lần chạy thực tế. Bạn đã bỏ lỡ những dấu vết nào trong phiên bản stdlib?

## Thuật ngữ chính

| Thuật ngữ | Mọi người nói | Ý nghĩa thực sự |
|------|----------------|------------------------|
| Agent | "Persona" | Vai trò + mục tiêu + backstory + công cụ |
| Task | "Đơn vị công việc" | Mô tả + kết quả mong đợi + người thực hiện + kết quả có cấu trúc tùy chọn |
| Crew | "Nhóm agent" | Container cho Agents + Tasks + Process |
| Process | "Chiến lược thực thi" | Sequential / Hierarchical / Consensus (đã lên kế hoạch) |
| Flow | "Quy trình xác định" | Hướng sự kiện, sở hữu bởi mã nguồn, có thể kiểm thử |
| Backstory | "Prompt persona" | Định hình giọng điệu và khả năng phán đoán cho Agent |
| `@tool` | "Công cụ hàm" | Decorator biến hàm thành công cụ mà Agent có thể gọi |
| `BaseTool` | "Công cụ lớp" | Công cụ dựa trên lớp với schema đối số, thử lại, hỗ trợ async |
| Entity memory | "Sự kiện theo thực thể" | Bộ nhớ giới hạn cho một khách hàng / tài khoản / vấn đề |
| Long-term memory | "Bộ nhớ xuyên lần chạy" | Bộ nhớ hỗ trợ vector tồn tại giữa các lần khởi chạy |
| Contextual memory | "Truy xuất tức thời" | Bộ nhớ được kéo tại thời điểm Agent cần |
| Manager LLM | "Agent điều phối" | LLM bổ sung trong quy trình Hierarchical chọn task tiếp theo |
| `expected_output` | "Hợp đồng task" | Chuỗi cho Agent (và kiểm toán) biết hình thái cần trả về |

## Đọc thêm

- [Giới thiệu tài liệu CrewAI](https://docs.crewai.com/en/introduction): các khái niệm và lộ trình production được khuyến nghị
- [Hướng dẫn CrewAI Flows](https://docs.crewai.com/en/concepts/flows): hình thái hướng sự kiện, `@start`, `@listen`
- [Tham chiếu công cụ CrewAI](https://docs.crewai.com/en/concepts/tools): `@tool`, `BaseTool`, bộ công cụ tích hợp
- [Bộ nhớ CrewAI](https://docs.crewai.com/en/concepts/memory): ngắn hạn, dài hạn, thực thể, ngữ cảnh
- [Anthropic, Xây dựng các Agent hiệu quả](https://www.anthropic.com/research/building-effective-agents): khi nào đa agent giúp ích và khi nào không
- [Tổng quan LangGraph](https://docs.langchain.com/oss/python/langgraph/overview): giải pháp thay thế máy trạng thái