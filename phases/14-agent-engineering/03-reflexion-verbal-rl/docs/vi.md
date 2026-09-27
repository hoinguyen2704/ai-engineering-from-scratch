# Reflexion: Verbal Reinforcement Learning

> RL dựa trên gradient cần hàng ngàn lần thử và một cụm GPU để sửa một chế độ lỗi. Reflexion (Shinn và cộng sự, NeurIPS 2023) thực hiện điều đó bằng ngôn ngữ tự nhiên: sau mỗi lần thử thất bại, agent viết một bản phản hồi (reflection), lưu trữ nó vào bộ nhớ tình huống (episodic memory) và điều kiện hóa lần thử tiếp theo dựa trên bộ nhớ đó. Đây là mô hình đứng sau tính năng "sleep-time compute" của Letta, các ghi chú học tập trong CLAUDE.md của Claude Code và lệnh "learn-rule" của pro-workflow.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 02 (ReWOO)
**Time:** ~60 minutes

## Mục tiêu học tập

- Nêu tên ba thành phần của Reflexion (Actor, Evaluator, Self-Reflector) và vai trò của bộ nhớ tình huống.
- Triển khai vòng lặp Reflexion bằng stdlib với bộ đánh giá nhị phân, bộ đệm phản hồi và các lần thử lại mới.
- Lựa chọn giữa các nguồn phản hồi dạng vô hướng (scalar), heuristic và tự đánh giá (self-evaluated) cho một tác vụ cụ thể.
- Giải thích lý do tại sao học tăng cường bằng ngôn ngữ (verbal reinforcement) có thể bắt được các lỗi mà RL dựa trên gradient cần hàng ngàn lần thử mới sửa được.

## Vấn đề

Một agent thất bại trong một tác vụ. Trong RL tiêu chuẩn, bạn sẽ chạy thêm hàng ngàn lần thử, tính toán gradient, cập nhật trọng số. Điều này tốn kém, chậm chạp và hầu hết các agent trong môi trường sản xuất không có ngân sách huấn luyện cho mỗi lần thất bại.

Reflexion (Shinn và cộng sự, arXiv:2303.11366) đặt ra một câu hỏi khác: điều gì sẽ xảy ra nếu agent chỉ cần suy nghĩ về lý do tại sao nó thất bại và thử lại với suy nghĩ đó trong prompt của nó? Không cập nhật trọng số. Không gradient. Chỉ là ngôn ngữ tự nhiên được lưu trữ giữa các lần thử.

Kết quả: trên ALFWorld, nó vượt qua ReAct và các baseline không được fine-tune khác. Trên HotpotQA, nó cải thiện so với ReAct. Trên tác vụ tạo mã (HumanEval/MBPP), nó thiết lập trạng thái tốt nhất tại thời điểm đó. Tất cả đều không cần một bước gradient nào.

## Khái niệm

### Ba thành phần

```
Actor         : generates a trajectory (ReAct-style loop)
Evaluator     : scores the trajectory — binary, heuristic, or self-eval
Self-Reflector: writes a natural-language reflection on the failure
```

Cộng với một cấu trúc dữ liệu:

```
Episodic memory: list of prior reflections, prepended to the next trial's prompt
```

Một lần thử chạy Actor. Evaluator chấm điểm nó. Nếu điểm thấp, Self-Reflector tạo ra một bản phản hồi ("Tôi đã chọn sai công cụ vì tôi đọc nhầm câu hỏi là hỏi về X trong khi nó hỏi về Y"). Bản phản hồi đi vào bộ nhớ tình huống. Lần thử tiếp theo bắt đầu mới nhưng nhìn thấy bản phản hồi đó.

### Ba loại Evaluator

1. **Scalar** — một tín hiệu nhị phân bên ngoài. ALFWorld thành công hoặc thất bại. Các bài kiểm tra HumanEval vượt qua hoặc thất bại. Đơn giản nhất, tín hiệu mạnh nhất.
2. **Heuristic** — các dấu hiệu thất bại được xác định trước. "Nếu agent tạo ra cùng một hành động hai lần liên tiếp, đánh dấu là bị kẹt." "Nếu quỹ đạo vượt quá 50 bước, đánh dấu là không hiệu quả."
3. **Self-evaluated** — LLM tự chấm điểm quỹ đạo của chính nó. Cần thiết khi không có ground truth. Tín hiệu yếu hơn; kết hợp tốt với xác minh dựa trên công cụ (Bài 05 — CRITIC).

Mặc định năm 2026 là sự kết hợp: scalar khi có sẵn, self-eval khi không có, và heuristic làm các rào chắn an toàn.

### Tại sao nó có tính tổng quát

Reflexion không hẳn là một thuật toán mới mà là một mô hình được đặt tên. Hầu như mọi agent "tự chữa lành" (self-healing) trong sản xuất đều chạy một biến thể nào đó:

- Tính năng "sleep-time compute" của Letta (Bài 08): một agent riêng biệt phản hồi về các cuộc hội thoại trước đó và ghi vào các khối bộ nhớ.
- Mô hình `CLAUDE.md` / "save memory" của Claude Code: các phản hồi được ghi lại dưới dạng các bài học, được thêm vào trước các phiên làm việc trong tương lai.
- Lệnh `/learn-rule` của pro-workflow: các chỉnh sửa được ghi lại dưới dạng các quy tắc rõ ràng.
- Các node phản hồi của LangGraph: một node chấm điểm đầu ra và điều hướng để tinh chỉnh nếu cần.

Tất cả đều bắt nguồn từ cùng một hiểu biết: ngôn ngữ tự nhiên là một phương tiện đủ phong phú để mang theo "những gì tôi học được từ thất bại" giữa các lần chạy.

### Khi nào nó hiệu quả và khi nào không

Reflexion hiệu quả khi:

- Có một tín hiệu thất bại rõ ràng (lỗi kiểm tra, lỗi công cụ, câu trả lời sai).
- Lớp tác vụ có thể tái lập (cùng một loại câu hỏi có thể được hỏi lại).
- Bản phản hồi có không gian để cải thiện quỹ đạo (đủ ngân sách hành động).

Reflexion không giúp ích khi:

- Agent đã thành công ngay từ lần thử đầu tiên.
- Lỗi mang tính ngoại cảnh (mạng bị sập, công cụ bị hỏng) — phản hồi về việc "mạng bị sập" không giúp ích cho các lần chạy trong tương lai.
- Bản phản hồi biến thành mê tín — lưu trữ một câu chuyện về một lần chạy lỗi ngẫu nhiên không lặp lại.

Cạm bẫy năm 2026: suy thoái bộ nhớ (memory rot). Các phản hồi tích tụ; một số lỗi thời hoặc sai; các lần chạy lại trở nên chậm hơn khi bộ đệm tình huống lớn dần. Cách giảm thiểu: nén định kỳ (Bài 06), TTL (thời gian sống) trên các phản hồi, hoặc một agent dọn dẹp riêng biệt trong thời gian nghỉ (Letta).

```figure
react-trace
```

## Xây dựng

`code/main.py` triển khai Reflexion trên một câu đố đồ chơi: tạo ra một danh sách 3 phần tử có tổng bằng một mục tiêu. Actor phát ra các danh sách ứng viên; Evaluator kiểm tra tổng; Self-Reflector viết một dòng về những gì đã sai. Bản phản hồi đi vào bộ nhớ tình huống cho lần thử tiếp theo.

Các thành phần:

- `Actor` — một chính sách được viết kịch bản giúp cải thiện khi nhìn thấy các phản hồi.
- `Evaluator.binary()` — kiểm tra đạt/không đạt dựa trên tổng mục tiêu.
- `SelfReflector` — tạo ra một chẩn đoán một dòng về lỗi.
- `EpisodicMemory` — một danh sách có giới hạn với ngữ nghĩa TTL.

Chạy nó:

```
python3 code/main.py
```

Dấu vết (trace) cho thấy ba lần thử. Lần thử 1 thất bại, một phản hồi được lưu trữ, lần thử 2 nhìn thấy phản hồi và cải thiện nhưng vẫn thất bại, lần thử 3 thành công. So sánh với một lần chạy baseline (không có phản hồi) — nó vẫn bị kẹt ở câu trả lời của lần thử 1.

## Sử dụng

LangGraph cung cấp phản hồi dưới dạng mô hình node. Lệnh `/memory` của Claude Code và `/learn-rule` của pro-workflow ngoại hóa bộ đệm tình huống dưới dạng tệp markdown. Tính năng "sleep-time compute" của Letta chạy Self-Reflector trong thời gian nhàn rỗi để agent chính duy trì độ trễ thấp. OpenAI Agents SDK không cung cấp trực tiếp Reflexion; bạn tự xây dựng nó với một Guardrail tùy chỉnh để từ chối các quỹ đạo dựa trên điểm số và một `Session` bộ nhớ tồn tại qua các lần chạy.

## Triển khai

`outputs/skill-reflexion-buffer.md` tạo và duy trì một bộ đệm tình huống với khả năng ghi lại phản hồi, TTL và khử trùng lặp. Với một lớp tác vụ và một lỗi, nó phát ra một bản phản hồi thực sự giúp ích cho lần thử tiếp theo (không phải là một lời khuyên chung chung kiểu "hãy cẩn thận hơn").

## Bài tập

1. Chuyển từ bộ đánh giá nhị phân sang bộ đánh giá vô hướng trả về số liệu khoảng cách (cách mục tiêu bao xa). Nó có hội tụ nhanh hơn không?
2. Thêm TTL là 10 lần thử cho các phản hồi. Các phản hồi cũ hơn có gây hại hay giúp ích sau thời điểm đó không?
3. Triển khai bộ đánh giá heuristic: đánh dấu lần thử là bị kẹt nếu cùng một hành động lặp lại. Điều này tương tác với Self-Reflector như thế nào?
4. Chạy Reflexion với một Actor đối nghịch (adversarial) bỏ qua các phản hồi. Đâu là kỹ thuật prompt tối thiểu để buộc Actor phải chú ý đến chúng?
5. Đọc Phần 4 của bài báo Reflexion về AlfWorld. Tái tạo khái niệm cải thiện tỷ lệ thành công 130%: đâu là sự khác biệt chính so với ReAct thông thường?

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|------------------------|
| Reflexion | "Tự sửa lỗi" | Shinn và cộng sự 2023 — Actor, Evaluator, Self-Reflector cộng với bộ nhớ tình huống |
| Verbal reinforcement | "Học không cần gradient" | Phản hồi bằng ngôn ngữ tự nhiên được thêm vào trước prompt của lần thử tiếp theo |
| Episodic memory | "Phản hồi theo tác vụ" | Bộ đệm giới hạn các phản hồi trước đó cho một lớp tác vụ |
| Scalar evaluator | "Tín hiệu thành công nhị phân" | Đạt/không đạt hoặc điểm số số từ ground truth |
| Heuristic evaluator | "Bộ phát hiện dựa trên mẫu" | Các dấu hiệu thất bại được xác định trước (ví dụ: vòng lặp kẹt, quá nhiều bước) |
| Self-evaluator | "LLM-as-judge trên dấu vết của chính nó" | Phương án dự phòng tín hiệu thấp khi không có ground truth — kết hợp với xác minh dựa trên công cụ |
| Memory rot | "Phản hồi cũ" | Bộ đệm tình huống đầy các mục nhập lỗi thời; sửa bằng cách nén/TTL |
| Sleep-time reflection | "Tự phản hồi bất đồng bộ" | Chạy Self-Reflector ngoài đường dẫn chính để agent chính duy trì tốc độ |

## Đọc thêm

- [Shinn và cộng sự, Reflexion: Language Agents with Verbal Reinforcement Learning (arXiv:2303.11366)](https://arxiv.org/abs/2303.11366) — bài báo gốc
- [Letta, Sleep-time Compute](https://www.letta.com/blog/sleep-time-compute) — phản hồi bất đồng bộ trong sản xuất
- [Anthropic, Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — quản lý bộ đệm tình huống như một phần của ngữ cảnh
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — mô hình node phản hồi