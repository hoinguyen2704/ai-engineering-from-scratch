# Chatbots — Từ Rule-Based đến Neural và LLM Agents

> ELIZA phản hồi bằng cách khớp mẫu (pattern matching). DialogFlow ánh xạ các ý định (intents). GPT trả lời dựa trên trọng số. Claude chạy các công cụ và xác thực kết quả. Mỗi kỷ nguyên đều giải quyết những thất bại tồi tệ nhất của kỷ nguyên trước đó.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 13 (Question Answering), Phase 5 · 14 (Information Retrieval)
**Time:** ~75 phút

## Vấn đề

Người dùng nói "Tôi muốn đổi chuyến bay." Hệ thống phải tìm ra họ muốn gì, thông tin nào còn thiếu, làm thế nào để lấy thông tin đó và cách hoàn thành hành động. Sau đó người dùng nói "khoan, nếu tôi hủy thì sao?", và hệ thống phải ghi nhớ ngữ cảnh, chuyển đổi tác vụ và duy trì trạng thái.

Đối với một hệ thống ML, hội thoại là một bài toán khó. Đầu vào là mở (open-ended). Đầu ra phải mạch lạc qua nhiều lượt hội thoại. Hệ thống có thể cần tác động vào thế giới thực (đổi chuyến bay, trừ tiền thẻ). Mọi bước sai lầm đều hiển hiện trước mắt người dùng.

Kiến trúc chatbot đã trải qua bốn mô hình, mỗi mô hình ra đời vì mô hình trước đó thất bại quá rõ ràng. Bài học này sẽ đi qua chúng theo thứ tự. Bối cảnh sản xuất năm 2026 là sự kết hợp của hai mô hình cuối cùng.

## Khái niệm

![Chatbot evolution: rule-based → retrieval → neural → agent](../assets/chatbot.svg)

### Nửa thế kỷ của các kịch bản, 1950-2001

Mô hình đầu tiên không chỉ tồn tại trong năm năm. Nó kéo dài năm mươi năm. Hiểu về quỹ đạo của nó rất quan trọng vì mọi hệ thống trong đó đều là cùng một cỗ máy — khớp đầu vào, đưa ra phản hồi có sẵn, cập nhật một chút trạng thái — và năm mươi năm thêm các quy tắc vào cỗ máy đó chưa bao giờ tạo ra được một giải pháp tổng quát. Cái trần đó là lý do tại sao mô hình thứ hai đến thứ tư tồn tại.

**1950.** Turing né tránh câu hỏi "máy móc có thể suy nghĩ không?" bằng cách đề xuất một sự thay thế mang tính vận hành: nếu một người thẩm vấn không thể phân biệt được máy móc với con người qua teletype, thì câu hỏi triết học đó trở nên vô nghĩa. Hội thoại trở thành tiêu chuẩn của lĩnh vực này trước khi lĩnh vực đó có tên gọi.

**1956.** Tên gọi xuất hiện — một hội thảo mùa hè tại Dartmouth đặt tên là "trí tuệ nhân tạo" (artificial intelligence) dựa trên giả thuyết rằng mọi đặc điểm của trí tuệ "về nguyên tắc có thể được mô tả chính xác đến mức máy móc có thể được tạo ra để mô phỏng nó." Đề xuất này dự trù hai tháng để đạt được tiến bộ đáng kể.

**1966.** ELIZA ra mắt thủ thuật phản chiếu mà bạn sẽ xây dựng trong Bước 1: các quy tắc phân tách lấy các mảnh từ đầu vào, các quy tắc lắp ghép phản hồi lại chúng dưới dạng câu hỏi. Tổng cộng khoảng 200 mẫu, không có trạng thái, không có sự hiểu biết — nhưng người dùng vẫn tâm sự với nó. Weizenbaum đã dành phần còn lại của sự nghiệp để lo ngại về việc cần quá ít máy móc để tạo ra nó.

**1972.** PARRY, được xây dựng tại Stanford để mô hình hóa sự hoang tưởng, đã thêm mảnh ghép mà ELIZA còn thiếu: trạng thái nội tại. Các biến số dạng số cho nỗi sợ hãi, sự tức giận và sự nghi ngờ được cập nhật sau mỗi lượt và kiểm soát kịch bản nào sẽ được kích hoạt tiếp theo, vì vậy các đầu vào giống hệt nhau sẽ tạo ra các phản hồi khác nhau tùy thuộc vào cuộc hội thoại cho đến thời điểm đó. Trong một bài kiểm tra mù, các bác sĩ tâm thần đã phân biệt PARRY với bệnh nhân con người ở mức độ ngẫu nhiên. Đây là tổ tiên trực tiếp của việc điều chỉnh persona (persona conditioning) — một system prompt được triển khai dưới dạng ba biến số thực (floats). Cùng năm đó, hai bot được kết nối với nhau qua ARPANET: một kịch bản bác sĩ trị liệu phỏng vấn một máy trạng thái hoang tưởng, cuộc hội thoại bot-với-bot đầu tiên trên mạng.

**1995.** ALICE mở rộng công thức của ELIZA với AIML, một phương ngữ XML cho các cặp mẫu-khuôn mẫu. Khoảng 40.000 danh mục được viết tay, giành ba giải thưởng Loebner. Nó chứng minh quy luật mở rộng của các hệ thống dựa trên quy tắc: nhiều quy tắc hơn chỉ mua được phạm vi bao phủ, không bao giờ mua được tính tổng quát. Mỗi quy tắc là một trách nhiệm mà ai đó phải duy trì.

**2001.** SmarterChild đưa công thức này đến trước 30 triệu người dùng tin nhắn tức thời và thêm các tra cứu backend — thời tiết, chứng khoán, giờ chiếu phim — được ghép vào các khuôn mẫu. Nếu nhìn kỹ, đó chính là tool calling trong trang phục năm 2001: phân tích ý định, gọi dịch vụ, hiển thị kết quả vào phản hồi.

Năm mươi năm, một cơ chế, số lượng quy tắc tăng dần. Mô hình này kết thúc không phải vì ai đó bác bỏ nó mà vì chi phí bảo trì các máy trạng thái viết tay tăng tuyến tính với phạm vi bao phủ, trong khi kỳ vọng của người dùng lại tăng theo bất cứ thứ gì họ thấy vào tuần trước.

```figure
chatbot-lineage
```

**Rule-based (ELIZA, AIML, DialogFlow).** Các mẫu do con người viết khớp với đầu vào của người dùng và tạo ra phản hồi. Các bộ phân loại ý định (intent classifiers) điều hướng đến các luồng được xác định trước. Các máy trạng thái điền vào các ô (slot-filling) thu thập thông tin cần thiết. Hoạt động xuất sắc trong phạm vi hẹp mà nó được thiết kế. Thất bại ngay lập tức khi ra ngoài phạm vi đó. Vẫn được sử dụng trong các lĩnh vực quan trọng về an toàn (xác thực ngân hàng, đặt vé máy bay) nơi không thể chấp nhận sự ảo tưởng (hallucination).

**Retrieval-based.** Một hệ thống kiểu FAQ. Mã hóa mọi cặp (câu nói, phản hồi). Tại thời điểm chạy, mã hóa tin nhắn của người dùng và truy xuất phản hồi được lưu trữ gần nhất. Hãy nghĩ đến tính năng "bài viết tương tự" cổ điển của Zendesk. Xử lý các cách diễn đạt lại tốt hơn quy tắc. Không tạo nội dung mới, vì vậy không có ảo tưởng.

**Neural (seq2seq).** Encoder-decoder được huấn luyện trên các nhật ký hội thoại. Tạo phản hồi từ đầu. Trôi chảy nhưng dễ tạo ra các phản hồi chung chung ("Tôi không biết") và trôi dạt về mặt thực tế. Không bao giờ bám sát chủ đề một cách đáng tin cậy. Đây là lý do tại sao Google, Facebook và Microsoft đều có những chatbot gây thất vọng trong giai đoạn 2016-2019.

**LLM agents.** Một mô hình ngôn ngữ được bao bọc trong một vòng lặp để lập kế hoạch, gọi công cụ và xác thực kết quả. Không phải là một chatbot với một prompt dài. Một vòng lặp agent: lập kế hoạch → gọi công cụ → quan sát kết quả → quyết định bước tiếp theo. Grounding ưu tiên truy xuất (RAG) giúp nó không bị ảo tưởng. Các cuộc gọi công cụ cho phép nó thực sự làm được việc. Đây là kiến trúc của năm 2026.

Bốn mô hình này không phải là sự thay thế tuần tự. Một chatbot sản xuất năm 2026 đi qua cả bốn: rule-based cho xác thực và các hành động phá hủy, truy xuất cho FAQ, tạo nội dung neural cho cách diễn đạt tự nhiên, LLM agent cho các truy vấn mở mơ hồ.

## Xây dựng

### Bước 1: khớp mẫu dựa trên quy tắc

```python
import re


class RulePattern:
    def __init__(self, pattern, response_template):
        self.regex = re.compile(pattern, re.IGNORECASE)
        self.template = response_template


PATTERNS = [
    RulePattern(r"my name is (\w+)", "Nice to meet you, {0}."),
    RulePattern(r"i (need|want) (.+)", "Why do you {0} {1}?"),
    RulePattern(r"i feel (.+)", "Why do you feel {0}?"),
    RulePattern(r"(.*)", "Tell me more about that."),
]


def rule_based_respond(user_input):
    for pattern in PATTERNS:
        m = pattern.regex.match(user_input.strip())
        if m:
            return pattern.template.format(*m.groups())
    return "I don't understand."
```

ELIZA trong 20 dòng. Thủ thuật phản chiếu ("Tôi cảm thấy buồn" → "Tại sao bạn cảm thấy buồn") là bản demo tâm lý trị liệu kinh điển từ Weizenbaum 1966. Vẫn rất mang tính giáo dục.

### Bước 2: dựa trên truy xuất (FAQ)

Đoạn mã minh họa này yêu cầu `pip install sentence-transformers` (thư viện kéo theo torch). Đoạn mã có thể chạy `code/main.py` cho bài học này sử dụng Jaccard similarity từ thư viện chuẩn thay thế, để bài học có thể chạy mà không cần phụ thuộc bên ngoài.

```python
from sentence_transformers import SentenceTransformer
import numpy as np


FAQ = [
    ("how do i reset my password", "Go to Settings > Security > Reset Password."),
    ("how do i cancel my order", "Go to Orders, find the order, click Cancel."),
    ("what is your return policy", "30-day returns on unused items, original packaging."),
]


encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
faq_questions = [q for q, _ in FAQ]
faq_embeddings = encoder.encode(faq_questions, normalize_embeddings=True)


def faq_respond(user_input, threshold=0.5):
    q_emb = encoder.encode([user_input], normalize_embeddings=True)[0]
    sims = faq_embeddings @ q_emb
    best = int(np.argmax(sims))
    if sims[best] < threshold:
        return None
    return FAQ[best][1]
```

Từ chối dựa trên ngưỡng là lựa chọn thiết kế quan trọng. Nếu kết quả khớp tốt nhất không đủ gần, hãy trả về `None` và để hệ thống leo thang.

### Bước 3: tạo nội dung neural (cơ sở)

Sử dụng một encoder-decoder nhỏ đã được tinh chỉnh theo hướng dẫn (FLAN-T5) hoặc một mô hình hội thoại đã được tinh chỉnh. Không thể sử dụng trong sản xuất vào năm 2026 (mâu thuẫn, trôi dạt khỏi chủ đề, vô nghĩa về mặt thực tế), nhưng được sử dụng bên trong các hệ thống lai để tạo cách diễn đạt tự nhiên. Các mô hình decoder-only kiểu DialoGPT cần các dấu phân cách lượt hội thoại và xử lý EOS rõ ràng để tạo ra các phản hồi mạch lạc; một pipeline text2text của FLAN-T5 hoạt động ngay lập tức cho một ví dụ giảng dạy.

```python
from transformers import pipeline

chatbot = pipeline("text2text-generation", model="google/flan-t5-small")

response = chatbot("Respond politely to: Hi there!", max_new_tokens=40)
print(response[0]["generated_text"])
```

### Bước 4: vòng lặp LLM agent

Hình thái sản xuất năm 2026:

```python
def agent_loop(user_message, tools, llm, max_steps=5):
    history = [{"role": "user", "content": user_message}]
    for _ in range(max_steps):
        response = llm(history, tools=tools)
        tool_call = response.get("tool_call")
        if tool_call:
            tool_name = tool_call.get("name")
            args = tool_call.get("arguments")
            if not isinstance(tool_name, str) or tool_name not in tools:
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": str(tool_name), "content": f"error: unknown tool {tool_name!r}"})
                continue
            if not isinstance(args, dict):
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": tool_name, "content": f"error: arguments must be a dict, got {type(args).__name__}"})
                continue
            fn = tools[tool_name]
            result = fn(**args)
            history.append({"role": "assistant", "tool_call": tool_call})
            history.append({"role": "tool", "name": tool_name, "content": result})
        else:
            return response["content"]
    return "I could not complete the task in the step budget."
```

Ba thứ cần đặt tên. Công cụ là các hàm có thể gọi mà LLM có thể kích hoạt. Vòng lặp kết thúc khi LLM trả về câu trả lời cuối cùng thay vì một cuộc gọi công cụ. Ngân sách bước (step budget) ngăn chặn các vòng lặp vô hạn trên các tác vụ mơ hồ.

Sản xuất thực tế bổ sung: grounding ưu tiên truy xuất (chèn các tài liệu liên quan trước mỗi cuộc gọi LLM), guardrails (từ chối các hành động phá hủy mà không có xác nhận), khả năng quan sát (ghi nhật ký mọi bước) và đánh giá (các kiểm tra tự động để đảm bảo hành vi của agent đúng thông số kỹ thuật).

### Bước 5: định tuyến lai (hybrid routing)

```python
def hybrid_chat(user_input):
    if is_destructive_action(user_input):
        return structured_flow(user_input)

    faq_answer = faq_respond(user_input, threshold=0.6)
    if faq_answer:
        return faq_answer

    return agent_loop(user_input, tools, llm)


def is_destructive_action(text):
    danger_words = ["delete", "cancel", "charge", "refund", "transfer"]
    return any(w in text.lower() for w in danger_words)
```

Mô hình: các quy tắc xác định cho bất cứ thứ gì mang tính phá hủy, truy xuất cho các FAQ có sẵn, LLM agents cho mọi thứ khác. Đây là những gì được triển khai trong các hệ thống hỗ trợ khách hàng năm 2026.

## Sử dụng

Stack năm 2026:

| Trường hợp sử dụng | Kiến trúc |
|---------|---------------|
| Đặt chỗ, thanh toán, xác thực | Máy trạng thái Rule-based + điền ô (slot filling) |
| FAQ hỗ trợ khách hàng | Truy xuất trên các câu trả lời đã được biên soạn |
| Chat hỗ trợ mở | LLM agent với RAG + gọi công cụ |
| Công cụ nội bộ / Trợ lý IDE | LLM agent với gọi công cụ (tìm kiếm, đọc, viết) |
| Chatbot đồng hành / nhân vật | LLM đã tinh chỉnh với system prompt persona, truy xuất trên kiến thức |

Luôn sử dụng định tuyến lai trong sản xuất. Không có kiến trúc đơn lẻ nào xử lý tốt mọi yêu cầu. Bản thân lớp định tuyến thường là một bộ phân loại ý định nhỏ.

## Các chế độ thất bại vẫn tồn tại

- **Tự tin bịa đặt.** LLM agent tuyên bố nó đã hoàn thành một hành động mà nó chưa làm. Giảm thiểu: xác thực kết quả, ghi nhật ký các cuộc gọi công cụ, không bao giờ để LLM tuyên bố đã làm điều gì đó mà không có kết quả trả về thành công từ công cụ.
- **Prompt injection.** Người dùng chèn văn bản ghi đè lên system prompt. Được xếp hạng LLM01 trong OWASP Top 10 cho các ứng dụng LLM 2025. Hai loại: injection trực tiếp (dán vào chat) và injection gián tiếp (ẩn trong tài liệu, email hoặc đầu ra công cụ mà agent đọc).

  Tỷ lệ tấn công thay đổi tùy theo kịch bản. Tỷ lệ thành công đo lường dao động ~0.5-8.5% trên các mô hình tiên phong trong các benchmark sử dụng công cụ và lập trình chung. Các thiết lập rủi ro cao cụ thể (tấn công thích ứng vào các AI coding agent, orchestration dễ bị tổn thương) đã đạt tới ~84%. Các CVE sản xuất bao gồm EchoLeak (CVE-2025-32711, CVSS 9.3) — một lỗ hổng rò rỉ dữ liệu zero-click trong Microsoft 365 Copilot được kích hoạt bởi một email do kẻ tấn công kiểm soát.

  Giảm thiểu: coi đầu vào của người dùng là không đáng tin cậy trong suốt vòng lặp; làm sạch trước khi gọi công cụ; cô lập đầu ra công cụ khỏi prompt chính; sử dụng mô hình Plan-Verify-Execute (PVE) nơi agent lập kế hoạch trước, sau đó xác thực từng hành động so với kế hoạch đó trước khi thực thi (điều này ngăn kết quả công cụ chèn các hành động mới chưa được lên kế hoạch); yêu cầu xác nhận của người dùng cho các hành động phá hủy; áp dụng nguyên tắc đặc quyền tối thiểu cho phạm vi công cụ.

  Không có lượng prompt engineering nào loại bỏ hoàn toàn rủi ro này. Các lớp bảo vệ runtime bên ngoài (LLM Guard, xác thực danh sách cho phép, phát hiện bất thường ngữ nghĩa) là bắt buộc.
- **Scope creep.** Agent đi chệch hướng vì một cuộc gọi công cụ trả về thông tin liên quan gián tiếp. Giảm thiểu: thu hẹp hợp đồng công cụ; giữ cho system prompt tập trung; thêm các đánh giá cho tỷ lệ đi chệch hướng.
- **Vòng lặp vô hạn.** Agent liên tục gọi cùng một công cụ. Giảm thiểu: ngân sách bước, khử trùng lặp cuộc gọi công cụ, LLM judge về việc "chúng ta có đang tiến triển không."
- **Cạn kiệt ngữ cảnh (Context window exhaustion).** Các cuộc hội thoại dài đẩy các lượt đầu tiên ra khỏi ngữ cảnh. Giảm thiểu: tóm tắt các lượt cũ, truy xuất các lượt quá khứ liên quan theo độ tương đồng, hoặc sử dụng mô hình ngữ cảnh dài.

## Triển khai

Lưu dưới dạng `outputs/skill-chatbot-architect.md`:

```markdown
---
name: chatbot-architect
description: Design a chatbot stack for a given use case.
version: 1.0.0
phase: 5
lesson: 17
tags: [nlp, agents, chatbot]
---

Given a product context (user need, compliance constraints, available tools, data volume), output:

1. Architecture. Rule-based, retrieval, neural, LLM agent, or hybrid (specify which paths go where).
2. LLM choice if applicable. Name the model family (Claude, GPT-4, Llama-3.1, Mixtral). Match to tool-use quality and cost.
3. Grounding strategy. RAG sources, retrieval method (see lesson 14), tool contracts.
4. Evaluation plan. Task success rate, tool-call correctness, off-task rate, hallucination rate on held-out dialogs.

Refuse to recommend a pure-LLM agent for any destructive action (payments, account deletion, data modification) without a structured confirmation flow. Refuse to skip the prompt-injection audit if the agent has write access to anything.
```

## Bài tập

1. **Dễ.** Triển khai phản hồi dựa trên quy tắc ở trên với 10 mẫu cho một bot đặt hàng quán cà phê. Kiểm tra các trường hợp biên: đặt hàng kép, sửa đổi, hủy bỏ, ý định không rõ ràng.
2. **Trung bình.** Xây dựng một hệ thống FAQ lai + LLM fallback. 50 mục FAQ có sẵn cho một sản phẩm SaaS, LLM fallback với truy xuất trên trang tài liệu. Đo lường tỷ lệ từ chối và độ chính xác trên 100 câu hỏi hỗ trợ thực tế.
3. **Khó.** Triển khai vòng lặp agent ở trên với ba công cụ (tìm kiếm, đọc dữ liệu người dùng, gửi email). Chạy đánh giá với 50 kịch bản thử nghiệm bao gồm các nỗ lực prompt injection. Báo cáo tỷ lệ đi chệch hướng, tỷ lệ tác vụ thất bại và bất kỳ thành công nào của injection.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| Intent | Người dùng muốn gì | Nhãn phân loại (book_flight, reset_password). Được định tuyến đến trình xử lý. |
| Slot | Một mẩu thông tin | Tham số bot cần (ngày, điểm đến). Slot filling là trình tự các câu hỏi. |
| RAG | Truy xuất cộng tạo nội dung | Truy xuất tài liệu liên quan, sau đó grounding phản hồi của LLM. |
| Tool call | Gọi hàm | LLM phát ra một cuộc gọi có cấu trúc với tên + đối số. Runtime thực thi, trả về kết quả. |
| Agent loop | Lập kế hoạch, hành động, xác thực | Bộ điều khiển chạy các cuộc gọi LLM xen kẽ với các cuộc gọi công cụ cho đến khi hoàn thành tác vụ. |
| Prompt injection | Người dùng tấn công prompt | Đầu vào độc hại cố gắng ghi đè system prompt. |

## Đọc thêm

- [Turing (1950). Computing Machinery and Intelligence](https://academic.oup.com/mind/article/LIX/236/433/986238) — bài báo biến hội thoại thành tiêu chuẩn của lĩnh vực.
- [Weizenbaum (1966). ELIZA — A Computer Program For the Study of Natural Language Communication](https://web.stanford.edu/class/cs124/p36-weizenabaum.pdf) — bài báo gốc về chatbot dựa trên quy tắc.
- [Colby, Weber, Hilf (1971). Artificial Paranoia](https://doi.org/10.1016/0004-3702(71)90002-6) — kiến trúc biến số cảm xúc của PARRY, chatbot có trạng thái đầu tiên.
- [Thoppilan et al. (2022). LaMDA: Language Models for Dialog Applications](https://arxiv.org/abs/2201.08239) — bài báo về neural-chatbot của Google, ngay trước khi LLM agents chiếm ưu thế.
- [Yao et al. (2022). ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) — bài báo đặt tên cho mô hình vòng lặp agent.
- [Hướng dẫn của Anthropic về xây dựng các agent hiệu quả](https://www.anthropic.com/research/building-effective-agents) — hướng dẫn sản xuất năm 2024 vẫn còn giá trị vào năm 2026.
- [Greshake et al. (2023). Not what you've signed up for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection](https://arxiv.org/abs/2302.12173) — bài báo về prompt-injection.
- [OWASP Top 10 cho các ứng dụng LLM 2025 — LLM01 Prompt Injection](https://genai.owasp.org/llmrisk/llm01-prompt-injection/) — bảng xếp hạng biến prompt injection thành mối quan tâm bảo mật hàng đầu.
- [AWS — Bảo mật Amazon Bedrock Agents chống lại Indirect Prompt Injections](https://aws.amazon.com/blogs/machine-learning/securing-amazon-bedrock-agents-a-guide-to-safeguarding-against-indirect-prompt-injections/) — các biện pháp phòng thủ lớp orchestration thực tế bao gồm Plan-Verify-Execute và các luồng xác nhận của người dùng.
- [EchoLeak (CVE-2025-32711)](https://www.vectra.ai/topics/prompt-injection) — CVE rò rỉ dữ liệu zero-click kinh điển từ indirect prompt injection. Trường hợp tham chiếu cho lý do tại sao các agent có quyền ghi cần các biện pháp phòng thủ runtime.