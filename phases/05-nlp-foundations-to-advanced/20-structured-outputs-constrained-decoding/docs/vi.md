# Structured Outputs & Constrained Decoding

> Yêu cầu LLM trả về JSON. Hầu hết thời gian bạn sẽ nhận được JSON. Trong môi trường production, "hầu hết" chính là vấn đề. Constrained decoding biến "hầu hết" thành "luôn luôn" bằng cách chỉnh sửa các logit trước khi lấy mẫu (sampling).

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 17 (Chatbots), Phase 5 · 19 (Subword Tokenization)
**Time:** ~60 phút

## Vấn đề

Một bộ phân loại (classifier) đưa ra prompt cho LLM: "Trả về một trong các giá trị {positive, negative, neutral}." Mô hình trả về "The sentiment is positive — this review is overwhelmingly favorable because the customer explicitly states that they ...". Trình phân tích cú pháp (parser) của bạn bị lỗi. F1 score của bộ phân loại là 0.0.

Tạo văn bản tự do (free-form generation) không phải là một hợp đồng. Nó chỉ là một gợi ý. Một hệ thống production cần một hợp đồng.

Có ba lớp tồn tại vào năm 2026.

1. **Prompting.** Yêu cầu lịch sự. "Chỉ trả về đối tượng JSON." Hoạt động ~80% trên các mô hình tiên tiến, thấp hơn trên các mô hình nhỏ.
2. **Native structured output APIs.** OpenAI `response_format`, Anthropic tool use, Gemini JSON mode. Đáng tin cậy trên các schema được hỗ trợ. Bị khóa bởi nhà cung cấp (vendor-locked).
3. **Constrained decoding.** Sửa đổi các logit tại mỗi bước tạo văn bản để mô hình *không thể* xuất ra các token không hợp lệ. Đảm bảo 100% hợp lệ theo cấu trúc. Hoạt động trên mọi mô hình cục bộ (local model).

Bài học này xây dựng tư duy cho cả ba phương pháp và xác định khi nào nên sử dụng phương pháp nào.

## Khái niệm

![Constrained decoding masking invalid tokens at each step](../assets/constrained-decoding.svg)

**Cách thức hoạt động của constrained decoding.** Tại mỗi bước tạo văn bản, LLM tạo ra một vector logit trên toàn bộ từ vựng (~100k token). Một *logit processor* nằm giữa mô hình và bộ lấy mẫu (sampler). Nó tính toán những token nào hợp lệ dựa trên vị trí hiện tại trong ngữ pháp mục tiêu — JSON Schema, regex, ngữ pháp phi ngữ cảnh (context-free grammar) — và đặt logit của tất cả các token không hợp lệ thành âm vô cùng. Softmax trên các logit còn lại chỉ đặt trọng số xác suất vào các phần tiếp nối hợp lệ.

Các triển khai vào năm 2026:

- **Outlines.** Biên dịch JSON Schema hoặc regex thành một máy trạng thái hữu hạn (finite-state machine). Mỗi token có một tra cứu O(1) cho token hợp lệ tiếp theo. Dựa trên FSM, vì vậy các schema đệ quy cần được làm phẳng.
- **XGrammar / llguidance.** Các công cụ ngữ pháp phi ngữ cảnh (CFG). Xử lý JSON Schema đệ quy. Độ trễ giải mã gần bằng 0. OpenAI đã ghi nhận llguidance trong triển khai structured output của họ vào năm 2025.
- **vLLM guided decoding.** Tích hợp sẵn `guided_json`, `guided_regex`, `guided_choice`, `guided_grammar` thông qua các backend Outlines, XGrammar hoặc lm-format-enforcer.
- **Instructor.** Wrapper dựa trên Pydantic cho bất kỳ LLM nào. Thử lại (retry) khi xác thực thất bại. Đa nhà cung cấp, nhưng không sửa đổi logit — nó dựa vào việc thử lại + các prompt nhận biết structured-output.

### Kết quả phản trực giác

Constrained decoding thường *nhanh hơn* so với tạo văn bản không bị ràng buộc. Có hai lý do. Thứ nhất, nó thu hẹp không gian tìm kiếm token tiếp theo. Thứ hai, các triển khai thông minh bỏ qua hoàn toàn việc tạo token cho các token bắt buộc (scaffolding như `{"name": "` — mọi byte đều đã được xác định).

### Cái bẫy gây tốn kém

Thứ tự trường (field order) rất quan trọng. Đặt `answer` trước `reasoning`, và mô hình sẽ cam kết với một câu trả lời trước khi nó kịp suy nghĩ. JSON thì hợp lệ. Câu trả lời thì sai. Không có xác thực nào bắt được lỗi này.

```json
// BAD
{"answer": "yes", "reasoning": "because ..."}

// GOOD
{"reasoning": "... therefore ...", "answer": "yes"}
```

Thứ tự trường trong schema là logic, không phải định dạng.

```figure
constrained-decoder
```

## Xây dựng

### Bước 1: Tạo văn bản bị ràng buộc bởi regex từ đầu

Xem `code/main.py` để biết triển khai FSM độc lập. Ý tưởng cốt lõi trong 30 dòng:

```python
def mask_logits(logits, valid_token_ids):
    mask = [float("-inf")] * len(logits)
    for tid in valid_token_ids:
        mask[tid] = logits[tid]
    return mask


def generate_constrained(model, tokenizer, prompt, fsm):
    ids = tokenizer.encode(prompt)
    state = fsm.initial_state
    while not fsm.is_accept(state):
        logits = model.next_token_logits(ids)
        valid = fsm.valid_tokens(state, tokenizer)
        logits = mask_logits(logits, valid)
        tok = sample(logits)
        ids.append(tok)
        state = fsm.transition(state, tok)
    return tokenizer.decode(ids)
```

FSM theo dõi những phần nào của ngữ pháp mà chúng ta đã thỏa mãn cho đến nay. `valid_tokens(state, tokenizer)` tính toán những token từ vựng nào có thể thúc đẩy FSM mà không rời khỏi đường dẫn chấp nhận.

### Bước 2: Outlines cho JSON Schema

```python
from pydantic import BaseModel
from typing import Literal
import outlines


class Review(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    evidence_span: str


model = outlines.models.transformers("meta-llama/Llama-3.2-3B-Instruct")
generator = outlines.generate.json(model, Review)

result = generator("Classify: 'The wait staff was attentive and the food arrived hot.'")
print(result)
# Review(sentiment='positive', confidence=0.93, evidence_span='attentive ... hot')
```

Không có lỗi xác thực. Bao giờ cũng vậy. FSM làm cho đầu ra không hợp lệ trở nên không thể đạt tới.

### Bước 3: Instructor cho Pydantic đa nhà cung cấp

```python
import instructor
from anthropic import Anthropic
from pydantic import BaseModel, Field


class Invoice(BaseModel):
    vendor: str
    total_usd: float = Field(ge=0)
    line_items: list[str]


client = instructor.from_anthropic(Anthropic())
invoice = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=1024,
    response_model=Invoice,
    messages=[{"role": "user", "content": "Extract from: 'Acme Corp $420. Widget, Gizmo.'"}],
)
```

Cơ chế khác biệt. Instructor không chạm vào logit. Nó định dạng schema vào prompt, phân tích đầu ra và thử lại khi xác thực thất bại (mặc định 3 lần). Hoạt động với bất kỳ nhà cung cấp nào. Việc thử lại làm tăng độ trễ và chi phí. Khả năng di động giữa các nhà cung cấp là điểm bán hàng chính.

### Bước 4: Native vendor APIs

```python
from openai import OpenAI

client = OpenAI()
response = client.responses.create(
    model="gpt-5",
    input=[{"role": "user", "content": "Classify: 'The food was cold.'"}],
    text={"format": {"type": "json_schema", "name": "sentiment",
          "schema": {"type": "object", "required": ["sentiment"],
                     "properties": {"sentiment": {"type": "string",
                                                  "enum": ["positive", "negative", "neutral"]}}}}},
)
print(response.output_parsed)
```

Constrained decoding phía máy chủ. Độ tin cậy ngang bằng với Outlines cho các schema được hỗ trợ. Không cần quản lý mô hình cục bộ. Khóa bạn vào nhà cung cấp.

## Các cạm bẫy

- **Recursive schemas.** Outlines làm phẳng đệ quy đến một độ sâu cố định. Các đầu ra có cấu trúc cây (nested comments, AST) cần XGrammar hoặc llguidance (dựa trên CFG).
- **Huge enums.** Enum với 10,000 tùy chọn biên dịch chậm hoặc bị timeout. Hãy chuyển sang bộ truy xuất (retriever): dự đoán top-k ứng viên trước, sau đó ràng buộc vào các ứng viên đó.
- **Grammar quá nghiêm ngặt.** Ép buộc regex `date: "YYYY-MM-DD"` và mô hình không thể xuất ra `"unknown"` cho các ngày bị thiếu. Mô hình bù đắp bằng cách tự bịa ra một ngày. Hãy cho phép `null` hoặc một sentinel.
- **Cam kết sớm (Premature commitment).** Xem cạm bẫy về thứ tự trường ở trên. Luôn đặt phần suy luận (reasoning) lên trước.
- **Vendor JSON mode không có schema.** Chế độ JSON thuần túy chỉ đảm bảo cú pháp JSON hợp lệ, không đảm bảo hợp lệ *cho trường hợp sử dụng của bạn*. Luôn cung cấp một schema đầy đủ.

## Sử dụng

Stack năm 2026:

| Tình huống | Lựa chọn |
|-----------|------|
| Mô hình OpenAI/Anthropic/Google, schema đơn giản | Native vendor structured output |
| Bất kỳ nhà cung cấp nào, quy trình Pydantic, có thể chấp nhận thử lại | Instructor |
| Mô hình cục bộ, cần độ hợp lệ 100%, schema phẳng | Outlines (FSM) |
| Mô hình cục bộ, schema đệ quy | XGrammar hoặc llguidance |
| Máy chủ suy luận tự lưu trữ (self-hosted) | vLLM guided decoding |
| Xử lý hàng loạt (batch) với việc thử lại có thể chấp nhận được | Instructor + mô hình rẻ nhất |

## Triển khai

Lưu dưới dạng `outputs/skill-structured-output-picker.md`:

```markdown
---
name: structured-output-picker
description: Choose a structured output approach, schema design, and validation plan.
version: 1.0.0
phase: 5
lesson: 20
tags: [nlp, llm, structured-output]
---

Given a use case (provider, latency budget, schema complexity, failure tolerance), output:

1. Mechanism. Native vendor structured output, Instructor retries, Outlines FSM, or XGrammar CFG. One-sentence reason.
2. Schema design. Field order (reasoning first, answer last), nullable fields for "unknown", enum vs regex, required fields.
3. Failure strategy. Max retries, fallback model, graceful `null` handling, out-of-distribution refusal.
4. Validation plan. Schema compliance rate (target 100%), semantic validity (LLM-judge), field-coverage rate, latency p50/p99.

Refuse any design that puts `answer` or `decision` before reasoning fields. Refuse to use bare JSON mode without a schema. Flag recursive schemas behind an FSM-only library.
```

## Bài tập

1. **Dễ.** Prompt một mô hình open-weights nhỏ (ví dụ: Llama-3.2-3B) mà không có constrained decoding cho `Review(sentiment, confidence, evidence_span)`. Đo lường tỷ lệ phân tích cú pháp thành JSON hợp lệ trên 100 bài đánh giá.
2. **Trung bình.** Sử dụng cùng tập dữ liệu với Outlines JSON mode. So sánh tỷ lệ tuân thủ, độ trễ và độ chính xác ngữ nghĩa.
3. **Khó.** Triển khai một bộ giải mã bị ràng buộc bởi regex từ đầu cho số điện thoại (`\d{3}-\d{3}-\d{4}`). Xác minh 0 đầu ra không hợp lệ trên 1000 mẫu.

## Thuật ngữ chính

| Thuật ngữ | Mọi người nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| Constrained decoding | Ép buộc đầu ra hợp lệ | Che các logit của token không hợp lệ tại mỗi bước tạo văn bản. |
| Logit processor | Thứ thực hiện ràng buộc | Hàm: `(logits, state) -> masked_logits`. |
| FSM | Máy trạng thái hữu hạn | Biểu diễn ngữ pháp đã biên dịch; tra cứu O(1) cho token hợp lệ tiếp theo. |
| CFG | Ngữ pháp phi ngữ cảnh | Ngữ pháp xử lý đệ quy; chậm hơn nhưng biểu cảm hơn FSM. |
| Schema field order | Có quan trọng không? | Có — trường đầu tiên sẽ cam kết; luôn đặt suy luận trước câu trả lời. |
| Guided decoding | Tên gọi của vLLM | Cùng một khái niệm, tích hợp vào máy chủ suy luận. |
| JSON mode | Phiên bản đầu của OpenAI | Đảm bảo cú pháp JSON; KHÔNG đảm bảo khớp schema. |

## Đọc thêm

- [Willard, Louf (2023). Efficient Guided Generation for LLMs](https://arxiv.org/abs/2307.09702) — bài báo về Outlines.
- [XGrammar paper (2024)](https://arxiv.org/abs/2411.15100) — constrained decoding dựa trên CFG tốc độ cao.
- [vLLM — Structured Outputs](https://docs.vllm.ai/en/latest/features/structured_outputs.html) — tích hợp máy chủ suy luận.
- [OpenAI — Structured Outputs guide](https://platform.openai.com/docs/guides/structured-outputs) — tài liệu tham khảo API + các lưu ý.
- [Instructor library](https://python.useinstructor.com/) — Pydantic + thử lại trên các nhà cung cấp.
- [JSONSchemaBench (2025)](https://arxiv.org/abs/2501.10868) — đánh giá 6 framework constrained decoding.