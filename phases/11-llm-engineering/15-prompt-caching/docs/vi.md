# Prompt Caching và Context Caching

> System prompt của bạn là 4.000 tokens. Ngữ cảnh RAG của bạn là 20.000 tokens. Bạn gửi cả hai trong mỗi yêu cầu. Bạn cũng phải trả phí cho cả hai — mỗi lần như vậy. Prompt caching cho phép nhà cung cấp giữ phần tiền tố (prefix) đó ở trạng thái "nóng" trên hệ thống của họ và tính phí bạn 10% mức giá thông thường khi tái sử dụng. Nếu được sử dụng đúng cách, nó giúp cắt giảm chi phí suy luận (inference) từ 50–90% và độ trễ token đầu tiên (first-token latency) từ 40–85%.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 11 · 01 (Prompt Engineering), Phase 11 · 05 (Context Engineering), Phase 11 · 11 (Caching and Cost)
**Time:** ~60 phút

## Vấn đề

Một coding agent gửi cùng một system prompt 15.000 tokens tới Claude trong mỗi lượt hội thoại. Hai mươi lượt với $3/M input tokens is $0.90 chỉ tính riêng chi phí đầu vào — trước khi có bất kỳ tin nhắn thực tế nào của người dùng. Nhân với 10.000 cuộc hội thoại mỗi ngày, hóa đơn sẽ lên tới 9.000 USD/ngày cho một đoạn văn bản không bao giờ thay đổi.

Bạn không thể rút ngắn prompt mà không làm giảm chất lượng. Bạn không thể tránh việc gửi nó — mô hình cần nó trong mỗi lượt. Cách duy nhất là ngừng trả giá đầy đủ cho một tiền tố mà nhà cung cấp đã thấy trước đó.

Đó chính là prompt caching. Anthropic đã ra mắt tính năng này vào tháng 8 năm 2024 (với biến thể TTL mở rộng 1 giờ vào năm 2025), OpenAI tự động hóa nó vào cuối năm đó, Google ra mắt context caching rõ ràng cùng với Gemini 1.5, và cả ba hiện đều cung cấp nó như một tính năng hạng nhất trên các mô hình tiên phong của họ.

## Khái niệm

![Prompt caching: write once, read cheap](../assets/prompt-caching.svg)

**Cơ chế.** Khi tiền tố của một yêu cầu khớp với tiền tố từ một yêu cầu gần đây, nhà cung cấp sẽ phục vụ KV-cache từ lần chạy trước thay vì mã hóa lại các tokens. Bạn trả một khoản phí ghi (write premium) nhỏ trong lần đầu tiên và nhận mức chiết khấu đọc (read discount) lớn cho mọi lần sau đó.

**Ba phong cách nhà cung cấp trong năm 2026.**

| Nhà cung cấp | Kiểu API | Chiết khấu khi khớp | Phí ghi | TTL mặc định | Tối thiểu để cache |
|---------|-----------|--------------|---------------|-------------|---------------|
| Anthropic | Đánh dấu `cache_control` rõ ràng trên các khối nội dung | Giảm 90% input | Phụ phí 25% | 5 phút (có thể mở rộng lên 1 giờ) | 1.024 tokens (Sonnet/Opus), 2.048 (Haiku) |
| OpenAI | Tự động phát hiện tiền tố | Giảm 50% input | không có | Lên đến 1 giờ (nỗ lực tốt nhất) | 1.024 tokens |
| Google (Gemini) | API `CachedContent` rõ ràng | Tính phí lưu trữ; đọc ở mức ~25% bình thường | Phí lưu trữ mỗi token·giờ | Người dùng thiết lập (mặc định 1 giờ) | 4.096 tokens (Flash), 32.768 (Pro) |

**Bất biến.** Cả ba chỉ cache phần tiền tố. Nếu bất kỳ token nào khác biệt giữa các yêu cầu, mọi thứ sau token khác biệt đầu tiên sẽ bị coi là miss (không khớp). Hãy đặt các phần *ổn định* ở trên cùng, các phần *biến đổi* ở dưới cùng.

### Bố cục thân thiện với cache

```
[system prompt]          <-- cache this
[tool definitions]       <-- cache this
[few-shot examples]      <-- cache this
[retrieved documents]    <-- cache if reused, else don't
[conversation history]   <-- cache up to last turn
[current user message]   <-- never cache (different every time)
```

Vi phạm thứ tự — đặt tin nhắn người dùng phía trên system prompt, chèn các truy vấn động giữa các few-shots — và cache sẽ không bao giờ khớp.

### Tính toán điểm hòa vốn

Phí ghi 25% của Anthropic có nghĩa là một khối được cache phải được đọc ít nhất hai lần để thu hồi vốn. 1 lần ghi + 1 lần đọc trung bình tốn 0,675x chi phí mỗi yêu cầu (tiết kiệm 32%); 1 lần ghi + 10 lần đọc trung bình tốn 0,205x (tiết kiệm 80%). Quy tắc ngón tay cái: hãy cache bất cứ thứ gì bạn dự định tái sử dụng ít nhất 3 lần trong thời gian TTL.

```figure
prompt-cache-hit
```

## Xây dựng

### Bước 1: Anthropic prompt caching với các đánh dấu rõ ràng

```python
import anthropic

client = anthropic.Anthropic()

SYSTEM = [
    {
        "type": "text",
        "text": "You are a senior Python reviewer. Follow the rubric exactly.\n\n" + RUBRIC_15K_TOKENS,
        "cache_control": {"type": "ephemeral"},
    }
]

def review(code: str):
    return client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        system=SYSTEM,
        messages=[{"role": "user", "content": code}],
    )
```

Đánh dấu `cache_control` yêu cầu Anthropic lưu trữ khối này trong 5 phút. Tái sử dụng trong khoảng thời gian đó sẽ khớp; tái sử dụng sau khi hết hạn sẽ ghi lại.

**Các trường sử dụng phản hồi:**

```python
response = review(code_a)
response.usage
# InputTokensUsage(
#     input_tokens=120,
#     cache_creation_input_tokens=15023,   # paid at 1.25x
#     cache_read_input_tokens=0,
#     output_tokens=340,
# )

response_b = review(code_b)
response_b.usage
# cache_creation_input_tokens=0
# cache_read_input_tokens=15023           # paid at 0.1x
```

Kiểm tra cả hai trường trong CI — nếu `cache_read_input_tokens` vẫn bằng 0 qua các yêu cầu, các khóa cache của bạn đang bị lệch.

### Bước 2: TTL mở rộng một giờ

Đối với các công việc hàng loạt (batch jobs) chạy dài, mặc định 5 phút sẽ hết hạn giữa các công việc. Thiết lập `ttl`:

```python
{"type": "text", "text": RUBRIC, "cache_control": {"type": "ephemeral", "ttl": "1h"}}
```

TTL 1 giờ tốn gấp 2 lần phí ghi (50% so với mức cơ bản thay vì 25%) nhưng hoàn vốn nhanh chóng cho bất kỳ batch nào tái sử dụng tiền tố hơn 5 lần.

### Bước 3: OpenAI tự động caching

OpenAI không yêu cầu bạn cấu hình gì cả. Bất kỳ tiền tố nào trên 1.024 tokens khớp với một yêu cầu gần đây sẽ tự động được giảm giá 50%.

```python
from openai import OpenAI
client = OpenAI()

resp = client.chat.completions.create(
    model="gpt-5",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},   # long and stable
        {"role": "user", "content": user_msg},
    ],
)
resp.usage.prompt_tokens_details.cached_tokens  # the discounted portion
```

Quy tắc bố cục thân thiện với cache tương tự vẫn áp dụng. Hai điều làm hỏng cache của OpenAI mà không làm hỏng của Anthropic: thay đổi trường `user` (được sử dụng như một thành phần khóa cache) và sắp xếp lại các tools.

### Bước 4: Gemini context caching rõ ràng

Gemini coi cache là một đối tượng hạng nhất mà bạn tạo và đặt tên:

```python
from google import genai
from google.genai import types

client = genai.Client()

cache = client.caches.create(
    model="gemini-3-pro",
    config=types.CreateCachedContentConfig(
        display_name="rubric-v3",
        system_instruction=RUBRIC,
        contents=[FEW_SHOT_EXAMPLES],
        ttl="3600s",
    ),
)

resp = client.models.generate_content(
    model="gemini-3-pro",
    contents=["Review this code:\n" + code],
    config=types.GenerateContentConfig(cached_content=cache.name),
)
```

Gemini tính phí lưu trữ theo token·giờ miễn là cache còn tồn tại, và đọc ở mức ~25% tốc độ input bình thường. Đây là hình thức phù hợp khi bạn tái sử dụng cùng một prompt khổng lồ qua nhiều phiên trong nhiều ngày.

### Bước 5: đo lường tỷ lệ khớp trong sản xuất

Xem `code/main.py` để biết bộ đếm ba nhà cung cấp mô phỏng theo dõi số lần ghi/đọc/miss và tính toán chi phí hỗn hợp trên 1.000 yêu cầu. Kiểm soát việc triển khai dựa trên tỷ lệ khớp mục tiêu — hầu hết các thiết lập Anthropic trong sản xuất nên đạt tỷ lệ đọc >80% sau khi khởi động.

## Các cạm bẫy vẫn tồn tại vào năm 2026

- **Dấu thời gian động ở trên cùng.** `"Current time: 2026-04-22 15:30:02"` ở đầu system prompt. Mọi yêu cầu đều bị miss. Hãy di chuyển dấu thời gian xuống dưới điểm ngắt cache.
- **Sắp xếp lại công cụ.** Tuần tự hóa các công cụ theo thứ tự ổn định — việc xáo trộn dict giữa các lần triển khai sẽ làm hỏng mọi lần khớp.
- **Các văn bản gần giống nhau.** "You are helpful." so với "You are a helpful assistant." — chênh lệch một byte = miss hoàn toàn.
- **Các khối quá nhỏ.** Anthropic áp đặt mức sàn 1.024 tokens (2.048 cho Haiku). Các khối nhỏ hơn sẽ không được cache một cách âm thầm.
- **Bảng điều khiển chi phí mù quáng.** Hãy tách "input tokens" thành loại được cache và không được cache. Nếu không, sự sụt giảm lưu lượng truy cập trông giống như một chiến thắng về cache.

## Sử dụng

Stack caching năm 2026:

| Tình huống | Lựa chọn |
|-----------|------|
| Agent với system prompt ổn định 10k+ tokens, nhiều lượt | Anthropic `cache_control` với TTL 5 phút |
| Batch job tái sử dụng tiền tố trong 30+ phút | Anthropic với `ttl: "1h"` |
| Serverless endpoints trên GPT-5, không có hạ tầng tùy chỉnh | OpenAI tự động (chỉ cần làm cho tiền tố của bạn ổn định và dài) |
| Tái sử dụng nhiều ngày một kho tài liệu/mã nguồn khổng lồ | Gemini `CachedContent` rõ ràng |
| Dự phòng đa nhà cung cấp | Giữ bố cục tiền tố có thể cache giống hệt nhau giữa các nhà cung cấp để bất kỳ lần khớp nào cũng hoạt động |

Kết hợp với semantic caching (Phase 11 · 11) cho lớp tin nhắn người dùng: prompt caching xử lý việc tái sử dụng *giống hệt token*, semantic caching xử lý việc tái sử dụng *giống hệt ý nghĩa*.

## Triển khai

Tiết kiệm `outputs/skill-prompt-caching-planner.md`:

```markdown
---
name: prompt-caching-planner
description: Design a cache-friendly prompt layout and pick the right provider caching mode.
version: 1.0.0
phase: 11
lesson: 15
tags: [llm-engineering, caching, cost]
---

Given a prompt (system + tools + few-shot + retrieval + history + user) and a usage profile (requests per hour, TTL needed, provider), output:

1. Layout. Reordered sections with a single cache breakpoint marked; explain which sections are stable, which are volatile.
2. Provider mode. Anthropic cache_control, OpenAI automatic, or Gemini CachedContent. Justify from TTL and reuse pattern.
3. Break-even. Expected reads per write within TTL; net cost vs no-cache with math.
4. Verification plan. CI assertion that cache_read_input_tokens > 0 on the second identical request; dashboard split by cached vs uncached tokens.
5. Failure modes. List the three most likely reasons the cache will miss in this setup (dynamic timestamp, tool reorder, near-duplicate text) and how you will prevent each.

Refuse to ship a cache plan that places a dynamic field above the breakpoint. Refuse to enable 1h TTL without a reuse count that makes the 2x write premium pay back.
```

## Bài tập

1. **Dễ.** Thực hiện một cuộc hội thoại 10 lượt với system prompt 5.000 tokens trên Claude. Chạy nó mà không có `cache_control` và sau đó là có. Báo cáo hóa đơn input-token cho mỗi trường hợp.
2. **Trung bình.** Viết một bộ kiểm thử (test harness) mà khi có một mẫu prompt và nhật ký yêu cầu, nó sẽ tính toán tỷ lệ khớp dự kiến và số tiền tiết kiệm được cho mỗi nhà cung cấp (Anthropic 5m, Anthropic 1h, OpenAI tự động, Gemini rõ ràng).
3. **Khó.** Xây dựng một trình tối ưu hóa bố cục: với một prompt và danh sách các trường được đánh dấu `stable=True/False`, hãy viết lại prompt để đặt một điểm ngắt cache duy nhất tại vị trí thân thiện với cache tối đa mà không làm mất thông tin. Xác minh trên một endpoint Anthropic thực tế.

## Thuật ngữ chính

| Thuật ngữ | Mọi người nói | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| Prompt caching | "Làm cho các prompt dài trở nên rẻ" | Tái sử dụng KV-cache phía nhà cung cấp cho các tiền tố khớp; chiết khấu 50-90% trên các input tokens lặp lại. |
| `cache_control` | "Đánh dấu của Anthropic" | Thuộc tính khối nội dung tuyên bố "mọi thứ đến đây đều có thể cache"; `{"type": "ephemeral"}`. |
| Cache write | "Trả phí phụ" | Yêu cầu đầu tiên điền vào cache; tính phí ~1,25x tốc độ input trên Anthropic, miễn phí trên OpenAI. |
| Cache read | "Chiết khấu" | Các yêu cầu tiếp theo khớp với tiền tố; tính phí 10% (Anthropic), 50% (OpenAI), ~25% (Gemini). |
| TTL | "Nó sống bao lâu" | Số giây cache duy trì trạng thái nóng; mặc định 5 phút trên Anthropic (có thể mở rộng 1 giờ), nỗ lực tốt nhất 1 giờ trên OpenAI, người dùng thiết lập trên Gemini. |
| Extended TTL | "Cache Anthropic 1 giờ" | `{"type": "ephemeral", "ttl": "1h"}`; phí ghi gấp 2 lần nhưng xứng đáng cho việc tái sử dụng hàng loạt. |
| Prefix match | "Tại sao cache của tôi bị miss" | Cache chỉ khớp khi mọi token từ đầu đến điểm ngắt giống hệt nhau về byte. |
| Context caching (Gemini) | "Loại rõ ràng" | Đối tượng cache được đặt tên, tính phí lưu trữ của Google; tốt nhất cho việc tái sử dụng kho dữ liệu lớn trong nhiều ngày. |

## Đọc thêm

- [Anthropic — Prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — `cache_control`, TTL 1 giờ, bảng hòa vốn.
- [OpenAI — Prompt caching](https://platform.openai.com/docs/guides/prompt-caching) — khớp tiền tố tự động.
- [Google — Context caching](https://ai.google.dev/gemini-api/docs/caching) — API `CachedContent` và giá lưu trữ.
- [Anthropic engineering — Prompt caching for long-context workloads](https://www.anthropic.com/news/prompt-caching) — bài đăng ra mắt gốc với các con số về độ trễ.
- Phase 11 · 05 (Context Engineering) — nơi cắt prompt để cache có thể hoạt động.
- Phase 11 · 11 (Caching and Cost) — kết hợp prompt caching với semantic cache trên tin nhắn người dùng.
- [Pope et al., "Efficiently Scaling Transformer Inference" (2022)](https://arxiv.org/abs/2211.05102) — mô hình bộ nhớ KV-cache mà prompt caching phơi bày cho người dùng; giải thích tại sao tiền tố được cache rẻ hơn ~10 lần để đọc lại so với tính toán lại.
- [Agrawal et al., "SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills" (2023)](https://arxiv.org/abs/2308.16369) — prefill là giai đoạn mà prompt caching rút ngắn; bài báo này giải thích tại sao TTFT giảm đáng kể khi khớp cache trong khi TPOT không bị ảnh hưởng.
- [Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (2023)](https://arxiv.org/abs/2211.17192) — prompt caching nằm cùng với speculative decoding, Flash Attention, và MQA/GQA như các đòn bẩy làm cong đường cong chi phí suy luận; hãy đọc bài này để biết ba kỹ thuật còn lại.