# Agent State Machines — Graphs, Nodes, Checkpoints

> Một vòng lặp ReAct được viết thủ công là một `while True`. Cùng vòng lặp đó nếu được viết dưới dạng một đồ thị tường minh (explicit graph) sẽ là thứ mà bạn có thể checkpoint, ngắt quãng (interrupt), phân nhánh và du hành thời gian (time-travel). Agent không hề thay đổi. Chỉ có bộ khung bao quanh nó là thay đổi.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 11 · 09 (Function Calling), Phase 11 · 14 (Model Context Protocol)
**Time:** ~75 minutes

## Vấn đề

Bạn triển khai một agent có khả năng gọi hàm (function-calling agent). Nó hoạt động tốt trong ba lượt, rồi có sự cố xảy ra: model thử dùng một tool trả về lỗi 500, người dùng đổi ý giữa chừng, hoặc agent quyết định hoàn tiền một đơn hàng mà không có sự phê duyệt của con người. Vòng lặp `while True:` không có các điểm móc (hooks). Bạn không thể tạm dừng, không thể tua lại và không thể phân nhánh theo kiểu "điều gì sẽ xảy ra nếu model chọn tool kia". Ngay khi bạn đưa nó ra khỏi môi trường demo, agent trở thành một hộp đen mà kết quả chỉ có thể là "chạy được" hoặc "không chạy được".

Bước tiếp theo rất rõ ràng khi bạn nhận ra điều này. Agent vốn dĩ đã là một máy trạng thái (state machine) — system prompt cộng với lịch sử tin nhắn cộng với các tool call đang chờ xử lý cộng với hành động tiếp theo. Hãy làm cho máy trạng thái đó trở nên tường minh: các node cho "model đang suy nghĩ", "tool đang chạy", "con người phê duyệt", và các cạnh (edges) cho các chuyển đổi có điều kiện giữa chúng. Một khi đồ thị đã tường minh, bộ khung sẽ nhận được bốn thứ miễn phí: checkpointing (lưu trạng thái giữa các bước), interrupts (tạm dừng cho con người), streaming (truyền luồng token và các sự kiện trung gian), và time-travel (quay lại trạng thái trước đó và thử một nhánh khác).

Triển khai tham chiếu của khái niệm này là LangGraph. Nó không phải là một framework agent theo kiểu LangChain ("đây là AgentExecutor, chúc may mắn"). Nó là một runtime đồ thị với trạng thái hạng nhất (first-class state), tính bền vững hạng nhất (first-class persistence) và khả năng ngắt quãng hạng nhất (first-class interrupts). Vòng lặp agent là thứ bạn vẽ ra, không phải thứ bạn viết tay.

## Khái niệm

![LangGraph StateGraph: nodes, edges, and the checkpointer](../assets/langgraph-stategraph.svg)

Một `StateGraph` có ba thành phần.

1. **State.** Một dict có kiểu (TypedDict hoặc Pydantic model) chảy xuyên suốt đồ thị. Mỗi node nhận toàn bộ trạng thái và trả về một bản cập nhật một phần, LangGraph sẽ hợp nhất chúng bằng một *reducer* cho mỗi trường — `operator.add` cho các danh sách cần tích lũy, mặc định là ghi đè.
2. **Nodes.** Các hàm Python `state -> partial_state`. Mỗi node là một bước rời rạc: "gọi model", "chạy tool", "tóm tắt".
3. **Edges.** Các chuyển đổi giữa các node. Các cạnh tĩnh (static edges) đi đến một nơi cố định. Các cạnh có điều kiện (conditional edges) sử dụng một hàm định tuyến `state -> next_node_name` để đồ thị có thể phân nhánh dựa trên đầu ra của model.

Bạn biên dịch (compile) đồ thị. Compile sẽ liên kết cấu trúc liên kết (topology), đính kèm một checkpointer (tùy chọn nhưng cần thiết cho production) và trả về một runnable. Bạn gọi nó với một trạng thái ban đầu và một `thread_id`. Mỗi bước thực thi sẽ lưu một checkpoint được khóa bởi `(thread_id, checkpoint_id)`.

### Bốn siêu năng lực

**Checkpointing.** Mỗi lần chuyển đổi node sẽ ghi trạng thái mới vào một kho lưu trữ (in-memory cho kiểm thử, Postgres/Redis/SQLite cho production). Tiếp tục bằng cách gọi lại đồ thị với cùng `thread_id`. Đồ thị sẽ bắt đầu từ nơi nó đã tạm dừng.

**Interrupts.** Đánh dấu một node với `interrupt_before=["human_review"]` và quá trình thực thi sẽ dừng lại trước khi node đó chạy. Trạng thái được lưu lại. API của bạn phản hồi người dùng với thông báo "đang chờ phê duyệt". Một yêu cầu sau đó đến cùng `thread_id` với `Command(resume=...)` sẽ tiếp tục thực thi.

**Streaming.** `graph.stream(state, mode="updates")` trả về các thay đổi trạng thái (deltas) ngay khi chúng xảy ra. `mode="messages"` truyền luồng các token LLM bên trong các node model. `mode="values"` trả về các bản chụp (snapshots) đầy đủ. Bạn chọn những gì cần hiển thị trên UI của mình.

**Time-travel.** `graph.get_state_history(thread_id)` trả về nhật ký checkpoint đầy đủ. Truyền bất kỳ `checkpoint_id` nào trước đó vào `graph.invoke` và bạn sẽ phân nhánh từ điểm đó. Rất hữu ích cho việc gỡ lỗi ("điều gì sẽ xảy ra nếu model chọn tool B thay vì tool A?") và cho các bài kiểm thử hồi quy (regression tests) chạy lại các dấu vết từ production.

### Reducer là điểm mấu chốt

Mỗi trường trạng thái đều có một reducer. Hầu hết các giá trị mặc định đều ổn — giá trị mới ghi đè giá trị cũ. Nhưng các danh sách tin nhắn cần `operator.add` để các tin nhắn mới được thêm vào thay vì thay thế. Các cạnh song song hợp nhất các bản cập nhật của chúng thông qua reducer. Nếu hai node cùng cập nhật `messages` và bạn quên `Annotated[list, add_messages]`, node thứ hai sẽ thắng một cách âm thầm và bạn mất một nửa lượt thực thi. Reducer là thứ tinh tế duy nhất trong thư viện; hãy làm đúng nó và phần còn lại sẽ tự kết hợp.

### Đồ thị ReAct trong bốn node

Một agent ReAct trong production gồm bốn node và hai cạnh:

1. `agent` — gọi LLM với lịch sử tin nhắn hiện tại. Trả về tin nhắn của trợ lý (có thể chứa các tool_calls).
2. `tools` — thực thi bất kỳ tool_calls nào trong tin nhắn trợ lý gần nhất, thêm kết quả của tool dưới dạng các tin nhắn tool.
3. Một cạnh có điều kiện từ `agent` định tuyến đến `tools` nếu tin nhắn cuối cùng có tool_calls, ngược lại đến `END`.
4. Một cạnh tĩnh từ `tools` quay lại `agent`.

Chỉ vậy thôi. Bạn có được vòng lặp ReAct đầy đủ (Thought → Action → Observation → Thought → …) với checkpointing, interrupts và streaming, chỉ trong khoảng 40 dòng code.

### StateGraph vs Send (fanout)

`Send(node_name, state)` cho phép một node điều phối các đồ thị con song song. Ví dụ: agent quyết định truy vấn ba trình truy xuất (retrievers) cùng một lúc. Mỗi `Send` sẽ tạo ra một quá trình thực thi song song của node mục tiêu; đầu ra của chúng được hợp nhất thông qua reducer trạng thái. Đây là cách LangGraph thể hiện mô hình orchestrator-workers mà không cần các nguyên hàm luồng (threading primitives).

### Subgraphs

Một đồ thị đã biên dịch có thể là một node trong một đồ thị khác. Đồ thị bên ngoài nhìn thấy một node duy nhất; đồ thị bên trong có trạng thái riêng và checkpoint riêng. Đây là cách các đội ngũ xây dựng các agent supervisor-worker: đồ thị supervisor định tuyến ý định của người dùng đến một subgraph worker theo từng lĩnh vực.

```figure
l5-state-graph-ledger
```

## Xây dựng

### Bước 1: trạng thái và các node

```python
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def agent_node(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: State) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END

tool_node = ToolNode(tools=[search_web, read_file])

graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

app = graph.compile(checkpointer=MemorySaver())
```

`add_messages` là reducer giúp danh sách tin nhắn tích lũy thay vì ghi đè. Quên nó là lỗi phổ biến nhất trong LangGraph.

### Bước 2: chạy với một thread

```python
config = {"configurable": {"thread_id": "user-42"}}
for event in app.stream(
    {"messages": [HumanMessage("find the Anthropic headquarters address")]},
    config,
    stream_mode="updates",
):
    print(event)
```

Mỗi bản cập nhật là một dict `{node_name: state_delta}`. Frontend của bạn có thể truyền luồng các bản cập nhật này đến UI để người dùng thấy "agent đang suy nghĩ… đang gọi search_web… đã có kết quả… đang trả lời."

### Bước 3: thêm interrupt (human-in-the-loop)

Đánh dấu một node để quá trình thực thi tạm dừng trước khi nó chạy.

```python
app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["tools"],  # pause before every tool call
)

state = app.invoke({"messages": [HumanMessage("delete the production database")]}, config)
# state["__interrupt__"] is set. Inspect proposed tool calls.
# If approved:
from langgraph.types import Command
app.invoke(Command(resume=True), config)
# If denied: write a rejection message and resume
app.update_state(config, {"messages": [AIMessage("Blocked by human reviewer.")]})
```

Trạng thái, checkpoint và thread đều tồn tại xuyên suốt quá trình ngắt quãng. Không có gì nằm trong bộ nhớ ngoại trừ trong quá trình thực thi.

### Bước 4: time-travel để gỡ lỗi

```python
history = list(app.get_state_history(config))
for snapshot in history:
    print(snapshot.values["messages"][-1].content[:80], snapshot.config)

# Fork from a prior checkpoint
target = history[3].config  # three steps back
for event in app.stream(None, target, stream_mode="values"):
    pass  # replay from that point forward
```

Truyền `None` làm đầu vào sẽ chạy lại từ checkpoint đã cho; truyền một giá trị sẽ thêm nó như một bản cập nhật vào trạng thái của checkpoint đó trước khi tiếp tục. Đây là cách bạn tái tạo một lần chạy agent lỗi mà không cần chạy lại toàn bộ cuộc hội thoại.

### Bước 5: thay đổi checkpointer cho production

```python
from langgraph.checkpoint.postgres import PostgresSaver

with PostgresSaver.from_conn_string("postgresql://...") as checkpointer:
    checkpointer.setup()
    app = graph.compile(checkpointer=checkpointer)
```

SQLite, Redis và Postgres đã được hỗ trợ sẵn. `MemorySaver` dành cho kiểm thử. Bất cứ thứ gì cần tồn tại sau khi khởi động lại đều cần một kho lưu trữ thực sự.

## Kỹ năng

> Bạn xây dựng các agent dưới dạng đồ thị, không phải dưới dạng các vòng lặp `while True`.

Trước khi sử dụng LangGraph, hãy thiết kế trong 60 giây:

1. **Đặt tên các node.** Mỗi quyết định rời rạc hoặc hành động có tác dụng phụ đều là một node. "Agent suy nghĩ", "tool chạy", "người đánh giá phê duyệt", "phản hồi truyền luồng". Nếu bạn không thể liệt kê chúng, tác vụ đó chưa có hình dạng của một agent.
2. **Khai báo trạng thái.** TypedDict tối giản với một reducer cho mỗi trường danh sách. Đừng nhồi nhét mọi thứ vào `messages`; hãy đưa các trường cụ thể của tác vụ (một `plan` đang làm việc, một bộ đếm `budget`, một danh sách `retrieved_docs`) lên cấp cao nhất.
3. **Vẽ các cạnh.** Mặc định là tĩnh trừ khi bước tiếp theo phụ thuộc vào đầu ra của model. Mỗi cạnh có điều kiện cần một hàm định tuyến với các nhánh được đặt tên.
4. **Chọn checkpointer ngay từ đầu.** `MemorySaver` cho kiểm thử, Postgres/Redis/SQLite cho mọi thứ khác. Đừng triển khai mà không có nó — không có checkpointer nghĩa là không thể tiếp tục, không thể ngắt quãng, không thể du hành thời gian.
5. **Quyết định các interrupt trước khi tool chạy, không phải sau đó.** Các phê duyệt nằm trên cạnh dẫn vào một node có tác dụng phụ để bạn có thể hủy trước khi gây hại; xác thực nằm trên cạnh dẫn ra khỏi model để bạn có thể từ chối các lệnh gọi sai với chi phí thấp.
6. **Mặc định streaming.** `mode="updates"` cho UI, `mode="messages"` cho streaming ở cấp độ token bên trong các node model, `mode="values"` cho các bản chụp đầy đủ trong quá trình đánh giá.

Từ chối triển khai một LangGraph agent không có checkpointer. Từ chối triển khai một agent ngắt quãng *sau khi* tác dụng phụ đã xảy ra. Từ chối triển khai một trường `messages` mà không có `add_messages` làm reducer.

## Bài tập

1. **Dễ.** Triển khai đồ thị ReAct bốn node ở trên với một tool máy tính và một tool tìm kiếm web. Xác minh rằng `list(app.get_state_history(config))` trả về ít nhất bốn checkpoint cho một cuộc hội thoại hai lượt.
2. **Trung bình.** Thêm một node `planner` chạy trước `agent` và ghi một `plan: list[str]` có cấu trúc vào trạng thái. Yêu cầu `agent` đánh dấu các bước kế hoạch là đã hoàn thành. Làm thất bại bài kiểm thử nếu `plan` bị mất sau khi khôi phục checkpoint (sai reducer).
3. **Khó.** Xây dựng một đồ thị supervisor định tuyến giữa ba subgraph (`researcher`, `writer`, `reviewer`) sử dụng `Send`. Mỗi subgraph có trạng thái và checkpointer riêng. Thêm một `interrupt_before=["writer"]` trên đồ thị bên ngoài để con người có thể phê duyệt bản tóm tắt nghiên cứu. Xác nhận rằng time-travel từ một checkpoint trước đó chỉ chạy lại nhánh đã phân nhánh.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| StateGraph | "Đồ thị LangGraph" | Đối tượng builder mà bạn thêm các node và cạnh vào trước khi biên dịch. |
| Reducer | "Cách trường hợp nhất" | Một hàm `(old, new) -> merged` được áp dụng khi một node trả về bản cập nhật cho trường đó; mặc định là ghi đè, `add_messages` là thêm vào. |
| Thread | "ID cuộc hội thoại" | Một chuỗi `thread_id` xác định phạm vi tất cả các checkpoint cho một phiên làm việc. |
| Checkpoint | "Trạng thái tạm dừng" | Một bản chụp bền vững của toàn bộ trạng thái đồ thị sau khi chuyển đổi node, được khóa bởi `(thread_id, checkpoint_id)`. |
| Interrupt | "Tạm dừng cho con người" | `interrupt_before` / `interrupt_after` dừng thực thi tại ranh giới node; tiếp tục với `Command(resume=...)`. |
| Time-travel | "Phân nhánh từ bước trước" | `graph.invoke(None, config_with_old_checkpoint_id)` chạy lại từ checkpoint đó trở đi. |
| Send | "Điều phối subgraph song song" | Một constructor mà một node có thể trả về để tạo ra N quá trình thực thi song song của một node mục tiêu. |
| Subgraph | "Đồ thị đã biên dịch làm node" | Một StateGraph đã biên dịch được sử dụng như một node trong đồ thị khác; bảo toàn phạm vi trạng thái riêng của nó. |

## Đọc thêm

- [Tài liệu LangGraph](https://langchain-ai.github.io/langgraph/) — tài liệu tham khảo chính thức cho StateGraph, reducers, checkpointers và interrupts.
- [Khái niệm LangGraph: state, reducers, checkpointers](https://langchain-ai.github.io/langgraph/concepts/low_level/) — mô hình tư duy mà bài học này sử dụng, trực tiếp từ nguồn.
- [LangGraph Persistence và Checkpoints](https://langchain-ai.github.io/langgraph/concepts/persistence/) — chi tiết về các kho lưu trữ Postgres/SQLite/Redis, không gian tên checkpoint và thread IDs.
- [LangGraph Human-in-the-loop](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/) — `interrupt_before`, `interrupt_after`, `Command(resume=...)` và mô hình edit-state.
- [Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (ICLR 2023)](https://arxiv.org/abs/2210.03629) — mô hình mà mọi LangGraph agent đều triển khai; hãy đọc để hiểu lý do đằng sau dấu vết suy luận.
- [Anthropic — Building effective agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) — các hình dạng đồ thị (chain, router, orchestrator-workers, evaluator-optimizer) nên ưu tiên và khi nào.
- Phase 11 · 09 (Function Calling) — nguyên hàm tool-call mà mọi node agent LangGraph đều tái sử dụng.
- Phase 11 · 14 (Model Context Protocol) — khám phá tool bên ngoài kết nối vào một `ToolNode` của LangGraph thông qua bộ chuyển đổi MCP.
- Phase 11 · 17 (Đánh đổi giữa các framework agent) — khi nào nên chọn LangGraph thay vì CrewAI, AutoGen hoặc Agno.