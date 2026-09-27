# ReWOO và Plan-and-Execute: Lập kế hoạch tách rời (Decoupled Planning)

> ReAct đan xen suy nghĩ và hành động trong một luồng duy nhất. ReWOO tách biệt chúng: lập một kế hoạch lớn ngay từ đầu, sau đó mới thực thi. Kết quả là giảm 5 lần số lượng token, tăng 4% độ chính xác trên HotpotQA, và bạn có thể chưng cất (distill) bộ lập kế hoạch vào một mô hình 7B. Plan-and-Execute đã tổng quát hóa phương pháp này; Plan-and-Act đã mở rộng nó cho việc điều hướng web.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop)
**Time:** ~60 phút

## Mục tiêu học tập

- Giải thích lý do tại sao việc phân chia Planner / Worker / Solver của ReWOO giúp tiết kiệm token và cải thiện độ bền vững so với vòng lặp đan xen của ReAct.
- Triển khai một DAG kế hoạch, một bộ thực thi theo thứ tự phụ thuộc và một bộ giải (solver) tổng hợp kết quả từ các worker — tất cả đều bằng stdlib.
- Quyết định khi nào một tác vụ nên chạy theo kiểu plan-then-execute so với ReAct đan xen, sử dụng khung "năm mô hình quy trình làm việc" năm 2026 (Anthropic).
- Nhận biết khi nào cần dữ liệu kế hoạch tổng hợp (synthetic plan data) của Plan-and-Act cho các tác vụ web hoặc di động dài hạn.

## Vấn đề

Vòng lặp suy nghĩ-hành động-quan sát đan xen của ReAct rất đơn giản và linh hoạt, nhưng mỗi lần gọi công cụ đều phải mang theo toàn bộ ngữ cảnh trước đó — bao gồm cả mọi suy nghĩ trước đó. Việc sử dụng token tăng theo cấp số nhân với độ sâu. Tệ hơn nữa: khi một công cụ thất bại giữa chừng, mô hình phải suy luận lại toàn bộ kế hoạch từ quan sát lỗi.

ReWOO (Xu và cộng sự, arXiv:2305.18323, tháng 5 năm 2023) đã nhận ra điều này và đặt cược vào một hướng đi: lập kế hoạch cho toàn bộ mọi thứ ngay từ đầu, tìm kiếm bằng chứng song song, và tổng hợp câu trả lời ở cuối. Một lần gọi LLM để lập kế hoạch, N lần gọi công cụ để lấy bằng chứng (có thể thực hiện song song), và một lần gọi LLM để giải quyết. Sự đánh đổi ở đây là giảm tính linh hoạt (kế hoạch là tĩnh) để đổi lấy hiệu quả token tốt hơn nhiều và các chế độ lỗi rõ ràng hơn.

## Khái niệm

### Ba vai trò

```
Planner:  user_question -> [plan_dag]
Workers:  [plan_dag]     -> [evidence]        (tool calls, possibly parallel)
Solver:   user_question, plan_dag, evidence -> final_answer
```

Planner tạo ra một DAG. Mỗi nút đặt tên cho một công cụ, các đối số của nó và các nút trước đó mà nó phụ thuộc vào (các tham chiếu như `#E1`, `#E2`). Các Worker thực thi các nút theo thứ tự topo. Solver khâu mọi thứ lại với nhau.

### Tại sao tiết kiệm 5 lần token

ReAct làm tăng độ dài prompt tuyến tính theo số bước. Ở bước 10, prompt chứa suy nghĩ 1 cộng với hành động 1 cộng với quan sát 1 cộng với suy nghĩ 2 cộng với hành động 2 cộng với quan sát 2, v.v. Mỗi bước trung gian cũng bao gồm dư thừa prompt gốc.

ReWOO chỉ tốn một prompt cho planner (lớn), N prompt nhỏ cho worker (mỗi prompt chỉ là lệnh gọi công cụ, không có chuỗi suy nghĩ), và một prompt cho solver. Trên HotpotQA, bài báo đo lường được số lượng token ít hơn khoảng 5 lần trong khi đạt độ chính xác tuyệt đối cao hơn 4%.

### Tại sao nó bền vững hơn

Nếu worker 3 thất bại trong ReAct, vòng lặp phải suy luận ra khỏi lỗi ngay giữa luồng. Trong ReWOO, worker 3 trả về một chuỗi lỗi; solver nhìn thấy nó trong ngữ cảnh với kế hoạch gốc và có thể xử lý lỗi một cách nhẹ nhàng. Việc định vị lỗi diễn ra theo từng nút, không phải theo từng bước.

### Chưng cất Planner (Planner distillation)

Kết quả thứ hai của bài báo: vì planner không nhìn thấy các quan sát, bạn có thể tinh chỉnh (fine-tune) một mô hình 7B dựa trên đầu ra của planner từ một giáo viên 175B. Mô hình nhỏ xử lý việc lập kế hoạch; mô hình lớn không cần thiết ở giai đoạn suy luận (inference). Điều này hiện đã trở thành tiêu chuẩn — nhiều tác nhân sản xuất năm 2026 sử dụng một planner nhỏ và một executor lớn hoặc ngược lại.

### Plan-and-Execute (2023)

Bài đăng tháng 8 năm 2023 của nhóm LangChain đã tổng quát hóa ReWOO thành một tên gọi mô hình: Plan-and-Execute. Planner ở bước đầu phát ra danh sách các bước, executor chạy từng bước, và một bộ lập kế hoạch lại (replanner) tùy chọn có thể sửa đổi sau khi quan sát kết quả. Điều này gần với ReAct hơn ReWOO (replanner đưa các quan sát trở lại quá trình lập kế hoạch) nhưng vẫn giữ được khả năng tiết kiệm token.

### Plan-and-Act (Erdogan và cộng sự, arXiv:2503.09572, ICML 2025)

Plan-and-Act mở rộng mô hình này cho các tác nhân web và di động dài hạn. Đóng góp chính là dữ liệu kế hoạch tổng hợp: một trình tạo quỹ đạo được dán nhãn tạo ra dữ liệu huấn luyện trong đó kế hoạch là rõ ràng. Được sử dụng để tinh chỉnh các mô hình planner hoạt động vượt quá 30–50 bước trên các tác vụ giống như WebArena, nơi một quỹ đạo ReAct đơn lẻ mất đi tính mạch lạc.

### Khi nào chọn mô hình nào

| Mô hình | Khi nào |
|---------|------|
| ReAct | Tác vụ ngắn, môi trường chưa biết, cần xử lý ngoại lệ phản ứng |
| ReWOO | Tác vụ có cấu trúc với các công cụ đã biết, nhạy cảm với token, bằng chứng có thể song song hóa |
| Plan-and-Execute | Giống ReWOO nhưng có lập kế hoạch lại sau khi thực thi một phần |
| Plan-and-Act | Dài hạn (>30 bước), web/di động/sử dụng máy tính |
| Tree of Thoughts | Tìm kiếm xứng đáng với chi phí bỏ ra (Bài 04) |

Hướng dẫn tháng 12 năm 2024 của Anthropic: hãy bắt đầu với cái đơn giản nhất. Nếu tác vụ chỉ là một lệnh gọi công cụ cộng với một bản tóm tắt, đừng xây dựng ReWOO. Nếu tác vụ là một bài nghiên cứu dài 40 bước, đừng chỉ dùng ReAct.

```figure
rewoo-plan
```

## Xây dựng

`code/main.py` triển khai một ReWOO đơn giản:

- `Planner` — một chính sách được viết kịch bản phát ra một DAG kế hoạch từ một prompt.
- `Worker` — điều phối lệnh gọi công cụ của từng nút thông qua registry.
- `Solver` — thành phần được viết kịch bản đọc bằng chứng và tạo ra câu trả lời cuối cùng.
- Giải quyết phụ thuộc — các tham chiếu như `#E1` được thay thế bằng đầu ra của các worker trước đó.

Bản demo trả lời câu hỏi "Dân số của thủ đô nước Pháp là bao nhiêu, làm tròn đến hàng triệu?" bằng cách sử dụng kế hoạch hai bước: (1) tra cứu thủ đô, (2) tra cứu dân số, sau đó giải quyết.

Chạy nó:

```
python3 code/main.py
```

Dấu vết (trace) hiển thị toàn bộ kế hoạch trước, sau đó là kết quả của worker, rồi đến việc tổng hợp của solver. Hãy so sánh số lượng token (chúng tôi in ra số lượng ký tự ước tính) với một lần chạy đan xen kiểu ReAct — ReWOO chiến thắng trên loại tác vụ có cấu trúc này.

## Sử dụng

LangGraph cung cấp Plan-and-Execute như một công thức (`create_react_agent` cho ReAct, các đồ thị tùy chỉnh cho plan-execute). Flows của CrewAI mã hóa trực tiếp mô hình này: bạn xác định các tác vụ ngay từ đầu và DAG của Flow sẽ thực thi chúng. Cách tiếp cận dữ liệu tổng hợp của Plan-and-Act chủ yếu vẫn là nghiên cứu; mô hình thời gian chạy (DAG kế hoạch rõ ràng) được triển khai trong sản xuất thông qua LangGraph và CrewAI Flows.

## Triển khai

`outputs/skill-rewoo-planner.md` tạo ra một DAG kế hoạch ReWOO từ yêu cầu của người dùng, dựa trên danh mục công cụ. Nó xác thực kế hoạch (không có chu trình, mọi tham chiếu được giải quyết, mọi công cụ đều tồn tại) trước khi bàn giao cho executor.

## Bài tập

1. Song song hóa việc thực thi worker cho các nút kế hoạch độc lập. Nó mang lại lợi ích gì trên một DAG 6 nút với 2 nhóm song song?
2. Thêm một nút replanner kích hoạt nếu bất kỳ worker nào trả về lỗi. Thay đổi nhỏ nhất đối với ReWOO để biến nó thành Plan-and-Execute là gì?
3. Thay thế `Planner` bằng một mô hình nhỏ (lớp 7B) và giữ `Solver` trên một mô hình tiên phong (frontier model). So sánh chất lượng tổng thể — sự phân chia này thất bại ở đâu?
4. Đọc Phần 4 của bài báo ReWOO về chưng cất planner. Tái tạo kết quả 175B -> 7B về mặt khái niệm: bạn cần dữ liệu huấn luyện nào và bạn đánh giá chất lượng kế hoạch như thế nào?
5. Chuyển đổi bản demo sang hình dạng quỹ đạo của Plan-and-Act: kế hoạch là một chuỗi, không phải một DAG. Những sự đánh đổi nào thay đổi?

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|------------------------|
| ReWOO | "Suy luận không cần quan sát" | Lập kế hoạch, sau đó lấy bằng chứng song song, sau đó giải quyết — không có quan sát trong prompt lập kế hoạch |
| Plan-and-Execute | "Mô hình plan-execute của LangChain" | ReWOO với một nút replanner tùy chọn sau khi thực thi |
| Plan-and-Act | "Plan-execute mở rộng" | Phân chia planner/executor rõ ràng với dữ liệu huấn luyện kế hoạch tổng hợp cho các tác vụ dài hạn |
| Evidence reference | "#E1, #E2, ..." | Trình giữ chỗ nút kế hoạch được thay thế bằng đầu ra của worker trước đó tại thời điểm điều phối |
| Planner distillation | "Planner nhỏ, executor lớn" | Tinh chỉnh một mô hình nhỏ dựa trên các dấu vết planner từ một giáo viên lớn |
| Token efficiency | "Ít lượt đi về hơn" | Ít hơn 5 lần token trên HotpotQA so với ReAct trong bài báo |
| DAG executor | "Bộ điều phối topo" | Chạy các nút kế hoạch theo thứ tự phụ thuộc; song song ở mỗi cấp |

## Đọc thêm

- [Xu và cộng sự, ReWOO: Decoupling Reasoning from Observations (arXiv:2305.18323)](https://arxiv.org/abs/2305.18323) — bài báo gốc
- [Erdogan và cộng sự, Plan-and-Act (arXiv:2503.09572)](https://arxiv.org/abs/2503.09572) — planner-executor mở rộng với các kế hoạch tổng hợp
- [Hướng dẫn LangGraph Plan-and-Execute](https://docs.langchain.com/oss/python/langgraph/overview) — công thức của framework
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — chọn mô hình đơn giản nhất hoạt động hiệu quả