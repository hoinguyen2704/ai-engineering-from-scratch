# Tại sao cần Multi-Agent?

> Một agent đâm đầu vào tường. Giải pháp thông minh không phải là tạo ra một agent lớn hơn - mà là thêm nhiều agent hơn.

**Type:** Learn
**Languages:** TypeScript
**Prerequisites:** Phase 14 (Agent Engineering)
**Time:** ~60 phút

## Mục tiêu học tập

- Xác định giới hạn của single-agent (tràn ngữ cảnh, chuyên môn hỗn tạp, nút thắt tuần tự) và giải thích khi nào việc chia nhỏ thành nhiều agent là bước đi đúng đắn.
- So sánh các mô hình điều phối (pipeline, parallel fan-out, supervisor, hierarchical) và chọn mô hình phù hợp cho cấu trúc tác vụ cụ thể.
- Thiết kế một hệ thống multi-agent với ranh giới vai trò rõ ràng, trạng thái chia sẻ và hợp đồng giao tiếp.
- Phân tích sự đánh đổi giữa độ phức tạp của multi-agent (độ trễ, chi phí, khó khăn khi debug) so với sự đơn giản của single-agent.

## Vấn đề

Bạn đã xây dựng một single-agent ở Phase 14. Nó hoạt động tốt. Nó có thể đọc tệp, chạy lệnh, gọi API và suy luận về kết quả. Sau đó, bạn áp dụng nó vào một codebase thực tế: 200 tệp, ba ngôn ngữ, các bài kiểm thử phụ thuộc vào hạ tầng và yêu cầu nghiên cứu các API bên ngoài trước khi viết mã.

Agent bị nghẽn. Không phải vì LLM kém, mà vì tác vụ vượt quá khả năng xử lý của một vòng lặp agent. Cửa sổ ngữ cảnh (context window) bị lấp đầy bởi nội dung tệp. Agent quên mất những gì nó đã đọc từ 40 lần gọi công cụ trước đó. Nó cố gắng trở thành một nhà nghiên cứu, một lập trình viên và một người đánh giá cùng một lúc, và làm cả ba việc đều không tốt.

Đây là giới hạn của single-agent. Bạn sẽ gặp phải nó bất cứ khi nào tác vụ yêu cầu:

- **Nhiều ngữ cảnh hơn mức có thể chứa trong một cửa sổ** - đọc 50 tệp sẽ vượt quá 200k token.
- **Chuyên môn khác nhau ở các giai đoạn khác nhau** - nghiên cứu yêu cầu kỹ thuật prompting khác với tạo mã.
- **Công việc có thể diễn ra song song** - tại sao phải đọc ba tệp tuần tự khi bạn có thể đọc chúng cùng lúc?

## Khái niệm

### Giới hạn của Single-Agent

Một single-agent là một vòng lặp, một cửa sổ ngữ cảnh, một system prompt. Hãy hình dung:

```
┌─────────────────────────────────────────┐
│            SINGLE AGENT                 │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │         Context Window            │  │
│  │                                   │  │
│  │  research notes                   │  │
│  │  + code files                     │  │
│  │  + test output                    │  │
│  │  + review feedback                │  │
│  │  + API docs                       │  │
│  │  + ...                            │  │
│  │                                   │  │
│  │  ██████████████████████ FULL ███  │  │
│  └───────────────────────────────────┘  │
│                                         │
│  One system prompt tries to cover       │
│  research + coding + review + testing   │
│                                         │
│  Result: mediocre at everything         │
└─────────────────────────────────────────┘
```

Ba thứ sẽ đổ vỡ:

1. **Bão hòa ngữ cảnh** - kết quả công cụ chồng chất. Đến lượt thứ 30, agent đã tiêu thụ 150k token nội dung tệp, đầu ra lệnh và suy luận trước đó. Các chi tiết quan trọng từ lượt thứ 5 bị mất.

2. **Nhầm lẫn vai trò** - một system prompt nói rằng "bạn là nhà nghiên cứu, lập trình viên, người đánh giá và người kiểm thử" sẽ tạo ra một agent làm nửa vời việc nghiên cứu, nửa vời việc viết mã và không bao giờ hoàn thành việc đánh giá.

3. **Nút thắt tuần tự** - agent đọc tệp A, sau đó tệp B, rồi tệp C. Ba lệnh gọi LLM tuần tự. Ba lần thực thi công cụ tuần tự. Không có sự song song.

### Giải pháp Multi-Agent

Chia nhỏ công việc. Giao cho mỗi agent một công việc, một cửa sổ ngữ cảnh và một system prompt được tinh chỉnh cho công việc đó:

```
┌──────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                          │
│                                                          │
│  "Build a REST API for user management"                  │
│                                                          │
│         ┌──────────┬──────────┬──────────┐               │
│         │          │          │          │               │
│         ▼          ▼          ▼          ▼               │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│   │RESEARCHER│ │  CODER   │ │ REVIEWER │ │  TESTER  │  │
│   │          │ │          │ │          │ │          │  │
│   │ Reads    │ │ Writes   │ │ Checks   │ │ Runs     │  │
│   │ docs,    │ │ code     │ │ code     │ │ tests,   │  │
│   │ finds    │ │ based on │ │ quality, │ │ reports  │  │
│   │ patterns │ │ research │ │ finds    │ │ results  │  │
│   │          │ │ + spec   │ │ bugs     │ │          │  │
│   └─────┬────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │
│         │           │            │             │         │
│         └───────────┴────────────┴─────────────┘         │
│                          │                               │
│                     Merge results                        │
└──────────────────────────────────────────────────────────┘
```

Mỗi agent có:
- Một system prompt tập trung ("Bạn là người đánh giá mã. Công việc duy nhất của bạn là tìm lỗi.")
- Cửa sổ ngữ cảnh riêng (không bị ô nhiễm bởi công việc của các agent khác)
- Một hợp đồng đầu vào/đầu ra rõ ràng (nhận ghi chú nghiên cứu, xuất ra mã)

### Các hệ thống thực tế áp dụng điều này

**Claude Code subagents** - khi Claude Code tạo ra một subagent với `Task`, nó tạo ra một agent con với tác vụ được giới hạn phạm vi. Agent cha giữ cho ngữ cảnh của nó sạch sẽ. Agent con thực hiện công việc tập trung và trả về bản tóm tắt.

**Devin** - chạy một agent lập kế hoạch, một agent viết mã và một agent trình duyệt. Agent lập kế hoạch chia nhỏ công việc thành các bước. Agent viết mã thực hiện viết. Agent trình duyệt nghiên cứu tài liệu. Mỗi agent có ngữ cảnh riêng biệt.

**Multi-agent coding teams (SWE-bench)** - các hệ thống đạt hiệu suất cao nhất trên SWE-bench sử dụng một nhà nghiên cứu để đọc codebase, một người lập kế hoạch để thiết kế bản sửa lỗi và một lập trình viên để triển khai nó. Các hệ thống single-agent đạt điểm thấp hơn.

**ChatGPT Deep Research** - tạo ra nhiều agent tìm kiếm song song, mỗi agent khám phá một góc độ khác nhau, sau đó tổng hợp kết quả.

### Phổ Multi-Agent

Multi-agent không phải là nhị phân. Nó là một phổ:

```
SIMPLE ──────────────────────────────────────────── COMPLEX

 Single        Sub-         Pipeline      Team         Swarm
 Agent         agents

 ┌───┐       ┌───┐        ┌───┐───┐    ┌───┐───┐    ┌─┐┌─┐┌─┐
 │ A │       │ A │        │ A │ B │    │ A │ B │    │ ││ ││ │
 └───┘       └─┬─┘        └───┘─┬─┘    └─┬─┘─┬─┘    └┬┘└┬┘└┬┘
               │                │        │   │       ┌┴──┴──┴┐
             ┌─┴─┐          ┌───┘───┐    │   │       │shared │
             │ a │          │ C │ D │  ┌─┴───┴─┐    │ state │
             └───┘          └───┘───┘  │  msg   │    └───────┘
                                       │  bus   │
 1 loop      Parent +      Stage by    │       │    N peers,
 1 context   child tasks   stage       └───────┘    emergent
                                       Explicit      behavior
                                       roles
```

**Single agent** - một vòng lặp, một prompt. Tốt cho các tác vụ đơn giản.

**Subagents** - một agent cha tạo ra các agent con cho các tác vụ phụ tập trung. Agent cha duy trì kế hoạch. Các agent con báo cáo lại. Đây là cách Claude Code hoạt động.

**Pipeline** - các agent chạy tuần tự. Đầu ra của Agent A trở thành đầu vào của Agent B. Tốt cho các quy trình theo giai đoạn: nghiên cứu -> viết mã -> đánh giá -> kiểm thử.

**Team** - các agent chạy song song với một bus tin nhắn chia sẻ. Mỗi agent có một vai trò. Một người điều phối (orchestrator) sẽ điều phối công việc. Tốt khi cần nhiều kỹ năng cùng lúc.

**Swarm** - nhiều agent giống hệt hoặc gần như giống hệt nhau với trạng thái chia sẻ. Không có người điều phối cố định. Các agent lấy công việc từ hàng đợi. Tốt cho các tác vụ song song có lưu lượng cao.

### Bốn mô hình Multi-Agent

#### Mô hình 1: Pipeline

```
Input ──▶ Agent A ──▶ Agent B ──▶ Agent C ──▶ Output
          (research)  (code)      (review)
```

Mỗi agent chuyển đổi dữ liệu và chuyển tiếp nó. Dễ hiểu. Lỗi ở một giai đoạn sẽ chặn các giai đoạn còn lại.

#### Mô hình 2: Fan-out / Fan-in

```
                ┌──▶ Agent A ──┐
                │              │
Input ──▶ Split ├──▶ Agent B ──├──▶ Merge ──▶ Output
                │              │
                └──▶ Agent C ──┘
```

Chia nhỏ công việc cho các agent song song, sau đó hợp nhất kết quả. Tốt cho các tác vụ có thể phân tách thành các tác vụ phụ độc lập.

#### Mô hình 3: Orchestrator-Worker

```
                    ┌──────────┐
                    │  Orch.   │
                    └──┬───┬───┘
                  task │   │ task
                 ┌─────┘   └─────┐
                 ▼               ▼
           ┌──────────┐   ┌──────────┐
           │ Worker A │   │ Worker B │
           └──────────┘   └──────────┘
```

Một người điều phối thông minh quyết định việc cần làm, ủy quyền cho các worker và tổng hợp kết quả. Bản thân người điều phối cũng là một agent với các công cụ để tạo ra các worker.

#### Mô hình 4: Peer Swarm

```
         ┌───┐ ◄──── msg ────▶ ┌───┐
         │ A │                  │ B │
         └─┬─┘                  └─┬─┘
           │                      │
      msg  │    ┌───────────┐     │ msg
           └───▶│  Shared   │◄────┘
                │  State    │
           ┌───▶│  / Queue  │◄────┐
           │    └───────────┘     │
      msg  │                      │ msg
         ┌─┴─┐                  ┌─┴─┐
         │ C │ ◄──── msg ────▶ │ D │
         └───┘                  └───┘
```

Không có người điều phối trung tâm. Các agent giao tiếp ngang hàng. Các quyết định nảy sinh từ sự tương tác. Khó debug hơn, nhưng có thể mở rộng cho nhiều agent.

### Khi nào KHÔNG nên sử dụng Multi-Agent

Multi-agent làm tăng độ phức tạp. Mỗi tin nhắn giữa các agent là một điểm lỗi tiềm ẩn. Việc debug chuyển từ "đọc một cuộc hội thoại" sang "truy vết tin nhắn qua năm agent".

**Hãy giữ ở dạng single-agent khi:**
- Tác vụ vừa với một cửa sổ ngữ cảnh (dưới ~100k token dữ liệu làm việc)
- Bạn không cần các system prompt khác nhau cho các giai đoạn khác nhau
- Thực thi tuần tự đủ nhanh
- Tác vụ đủ đơn giản để việc chia nhỏ nó tạo ra nhiều chi phí quản lý hơn là giá trị

**Chi phí của độ phức tạp:**
- Mỗi ranh giới agent là một bước nén mất dữ liệu: toàn bộ ngữ cảnh của agent A được tóm tắt thành một tin nhắn cho agent B
- Logic điều phối (ai làm gì, khi nào, theo thứ tự nào) là một nguồn gây lỗi riêng
- Độ trễ tăng lên: N agent nghĩa là tối thiểu N lần gọi LLM tuần tự, thậm chí nhiều hơn nếu chúng cần trao đổi qua lại
- Chi phí nhân lên: mỗi agent tiêu tốn token độc lập

Quy tắc ngón tay cái: nếu một tác vụ mất ít hơn 20 lần gọi công cụ và vừa với 100k token, hãy giữ nó ở dạng single-agent.

```figure
swarm-messages
```

## Xây dựng

### Bước 1: Single Agent bị quá tải

Đây là một single-agent cố gắng làm mọi thứ. Nó có một system prompt khổng lồ và một cửa sổ ngữ cảnh chứa nghiên cứu, mã và đánh giá:

```typescript
type AgentResult = {
  content: string;
  tokensUsed: number;
  toolCalls: number;
};

async function singleAgentApproach(task: string): Promise<AgentResult> {
  const systemPrompt = `You are a full-stack developer. You must:
1. Research the requirements
2. Write the code
3. Review the code for bugs
4. Write tests
Do ALL of these in a single conversation.`;

  const contextWindow: string[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const research = await fakeLLMCall(systemPrompt, `Research: ${task}`);
  contextWindow.push(research.output);
  totalTokens += research.tokens;
  totalToolCalls += research.calls;

  const code = await fakeLLMCall(
    systemPrompt,
    `Given this research:\n${contextWindow.join("\n")}\n\nNow write code for: ${task}`
  );
  contextWindow.push(code.output);
  totalTokens += code.tokens;
  totalToolCalls += code.calls;

  const review = await fakeLLMCall(
    systemPrompt,
    `Given all previous context:\n${contextWindow.join("\n")}\n\nReview the code.`
  );
  contextWindow.push(review.output);
  totalTokens += review.tokens;
  totalToolCalls += review.calls;

  return {
    content: contextWindow.join("\n---\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

Các vấn đề với cách tiếp cận này:
- Cửa sổ ngữ cảnh tăng lên theo từng giai đoạn. Đến bước đánh giá, nó chứa ghi chú nghiên cứu VÀ mã VÀ suy luận trước đó.
- System prompt mang tính chung chung. Nó không thể được tinh chỉnh cho từng giai đoạn.
- Không có gì chạy song song.

### Bước 2: Specialist Agents

Bây giờ hãy chia nhỏ nó ra. Mỗi agent nhận một công việc:

```typescript
type SpecialistAgent = {
  name: string;
  systemPrompt: string;
  run: (input: string) => Promise<AgentResult>;
};

function createSpecialist(name: string, systemPrompt: string): SpecialistAgent {
  return {
    name,
    systemPrompt,
    run: async (input: string) => {
      const result = await fakeLLMCall(systemPrompt, input);
      return {
        content: result.output,
        tokensUsed: result.tokens,
        toolCalls: result.calls,
      };
    },
  };
}

const researcher = createSpecialist(
  "researcher",
  "You are a technical researcher. Read documentation, find patterns, and summarize findings. Output only the facts needed for implementation."
);

const coder = createSpecialist(
  "coder",
  "You are a senior TypeScript developer. Given requirements and research notes, write clean, tested code. Nothing else."
);

const reviewer = createSpecialist(
  "reviewer",
  "You are a code reviewer. Find bugs, security issues, and logic errors. Be specific. Cite line numbers."
);
```

Mỗi chuyên gia có một prompt tập trung. Mỗi chuyên gia nhận một cửa sổ ngữ cảnh sạch sẽ chỉ với đầu vào mà nó cần.

### Bước 3: Điều phối thông qua tin nhắn

Kết nối các chuyên gia lại với nhau bằng cách truyền tin nhắn rõ ràng:

```typescript
type AgentMessage = {
  from: string;
  to: string;
  content: string;
  timestamp: number;
};

async function multiAgentApproach(task: string): Promise<AgentResult> {
  const messages: AgentMessage[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const researchResult = await researcher.run(task);
  messages.push({
    from: "researcher",
    to: "coder",
    content: researchResult.content,
    timestamp: Date.now(),
  });
  totalTokens += researchResult.tokensUsed;
  totalToolCalls += researchResult.toolCalls;

  const coderInput = messages
    .filter((m) => m.to === "coder")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const codeResult = await coder.run(coderInput);
  messages.push({
    from: "coder",
    to: "reviewer",
    content: codeResult.content,
    timestamp: Date.now(),
  });
  totalTokens += codeResult.tokensUsed;
  totalToolCalls += codeResult.toolCalls;

  const reviewerInput = messages
    .filter((m) => m.to === "reviewer")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const reviewResult = await reviewer.run(reviewerInput);
  messages.push({
    from: "reviewer",
    to: "orchestrator",
    content: reviewResult.content,
    timestamp: Date.now(),
  });
  totalTokens += reviewResult.tokensUsed;
  totalToolCalls += reviewResult.toolCalls;

  return {
    content: messages.map((m) => `[${m.from} -> ${m.to}]: ${m.content}`).join("\n\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

Mỗi agent chỉ nhận các tin nhắn gửi cho nó. Không có sự ô nhiễm ngữ cảnh. 50k token đọc tài liệu của nhà nghiên cứu không bao giờ đi vào ngữ cảnh của người đánh giá.

### Bước 4: So sánh

```typescript
async function compare() {
  const task = "Build a rate limiter middleware for an Express.js API";

  console.log("=== Single Agent ===");
  const single = await singleAgentApproach(task);
  console.log(`Tokens: ${single.tokensUsed}`);
  console.log(`Tool calls: ${single.toolCalls}`);

  console.log("\n=== Multi-Agent ===");
  const multi = await multiAgentApproach(task);
  console.log(`Tokens: ${multi.tokensUsed}`);
  console.log(`Tool calls: ${multi.toolCalls}`);
}
```

Phiên bản multi-agent sử dụng nhiều token hơn (ba agent, ba lần gọi LLM riêng biệt) nhưng ngữ cảnh của mỗi agent vẫn sạch sẽ. Chất lượng của từng giai đoạn được cải thiện vì system prompt được chuyên môn hóa.

## Sử dụng

Bài học này tạo ra một prompt có thể tái sử dụng để quyết định khi nào nên chuyển sang multi-agent. Xem `outputs/prompt-multi-agent-decision.md`.

## Bài tập

1. Thêm chuyên gia thứ tư: một agent "kiểm thử" nhận mã từ lập trình viên và phản hồi đánh giá từ người đánh giá, sau đó viết các bài kiểm thử.
2. Sửa đổi pipeline để người đánh giá có thể gửi phản hồi lại cho lập trình viên để thực hiện vòng lặp sửa đổi (tối đa 2 vòng).
3. Chuyển đổi pipeline tuần tự thành fan-out: chạy nhà nghiên cứu và một agent "phân tích yêu cầu" song song, sau đó hợp nhất đầu ra của chúng trước khi chuyển cho lập trình viên.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|----------------------|
| Swarm | "Một trí tuệ bầy đàn của các AI agent" | Một tập hợp các agent ngang hàng với trạng thái chia sẻ và không có lãnh đạo cố định. Hành vi nảy sinh từ các tương tác cục bộ. |
| Orchestrator | "Agent sếp" | Một agent có các công cụ bao gồm việc tạo và quản lý các agent khác. Nó lập kế hoạch và ủy quyền nhưng có thể không trực tiếp làm công việc. |
| Coordinator | "Cảnh sát giao thông" | Một thành phần không phải agent (thường chỉ là mã, không phải LLM) định tuyến tin nhắn giữa các agent dựa trên các quy tắc. |
| Consensus | "Các agent đồng ý" | Một giao thức trong đó nhiều agent phải đạt được thỏa thuận trước khi tiếp tục. Được sử dụng khi các đầu ra xung đột cần giải quyết. |
| Emergent behavior | "Các agent tự tìm ra cách" | Các mô hình ở cấp độ hệ thống nảy sinh từ tương tác của các agent nhưng không được lập trình rõ ràng. Có thể hữu ích hoặc có hại. |
| Fan-out / fan-in | "Map-reduce cho các agent" | Chia nhỏ một tác vụ cho các agent song song (fan-out), sau đó kết hợp kết quả của chúng (fan-in). |
| Message passing | "Các agent nói chuyện với nhau" | Cơ chế giao tiếp giữa các agent: dữ liệu có cấu trúc được gửi từ agent này sang agent khác, thay thế cho các cửa sổ ngữ cảnh chia sẻ. |

## Đọc thêm

- [The Landscape of Emerging AI Agent Architectures](https://arxiv.org/abs/2409.02977) - khảo sát các mô hình multi-agent
- [AutoGen: Enabling Next-Gen LLM Applications](https://arxiv.org/abs/2308.08155) - khung hội thoại multi-agent của Microsoft
- [Claude Code subagents documentation](https://docs.anthropic.com/en/docs/claude-code) - cách Claude Code ủy quyền với Task
- [CrewAI documentation](https://docs.crewai.com/) - khung multi-agent dựa trên vai trò