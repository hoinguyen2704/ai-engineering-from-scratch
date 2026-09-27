# Tree of Thoughts và LATS: Tìm kiếm có chủ đích

> Một chuỗi suy luận (chain-of-thought) đơn lẻ không có khả năng quay lui. ToT (Yao và cộng sự, 2023) biến suy luận thành một cây với cơ chế tự đánh giá tại mỗi nút. LATS (Zhou và cộng sự, 2024) hợp nhất ToT với ReAct và Reflexion dưới dạng Monte Carlo Tree Search. Game of 24 tăng từ 4% (CoT) lên 74% (ToT); LATS đạt 92.7% pass@1 trên HumanEval.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 03 (Reflexion)
**Time:** ~75 phút

## Mục tiêu học tập

- Định hình suy luận như một bài toán tìm kiếm: các nút là "ý tưởng" (thoughts), các cạnh là "sự mở rộng" (expansions), giá trị là "độ hứa hẹn".
- Triển khai tìm kiếm cây BFS theo phong cách ToT bằng stdlib với điểm số tự đánh giá.
- Mở rộng sang vòng lặp MCTS kiểu LATS với các bước select / expand / simulate / backpropagate.
- Quyết định khi nào việc tìm kiếm xứng đáng với chi phí token (Game of 24, tạo mã) và khi nào một chuỗi suy luận đơn lẻ là đủ (hỏi đáp đơn giản).

## Vấn đề

Chain-of-thought là một bước đi tuyến tính. Nếu bước đầu tiên sai, mọi bước tiếp theo đều dựa trên một tiền đề sai lầm. Trong Game of 24 (sử dụng bốn chữ số với + − × ÷ để tạo ra 24), GPT-4 CoT chỉ đạt độ chính xác 4%. Mô hình chọn sai biểu thức con ngay từ đầu và không thể phục hồi.

Điều mà suy luận cần là khả năng đề xuất nhiều ứng viên, đánh giá chúng, chọn những ứng viên hứa hẹn và quay lui khi gặp ngõ cụt. Đó chính là tìm kiếm. Tree of Thoughts và LATS là hai công thức kinh điển.

## Khái niệm

### Tree of Thoughts (Yao và cộng sự, NeurIPS 2023)

Mỗi nút là một bước trung gian mạch lạc ("một ý tưởng"). Mỗi nút có thể mở rộng thành K ý tưởng con. LLM tự đánh giá mỗi nút bằng một prompt chấm điểm. Quá trình tìm kiếm sẽ khám phá cây — theo BFS, DFS hoặc beam search.

```
                     (root: "find 24 from 4 6 4 1")
                    /               |            \
           ("6 - 4 = 2")    ("4 + 1 = 5")    ("4 * 6 = 24")  <- Score: HIGH
              /   \              |                  |
          ...    ...          ...                finish
```

Tự đánh giá là phần quan trọng nhất. Bài báo chỉ ra ba biến thể: `sure / likely / impossible` phân loại, `1..10` điểm số dạng số, và bỏ phiếu giữa các ứng viên. Cả ba đều vượt trội hơn hẳn CoT trong Game of 24 (tăng từ 4% lên 74% với GPT-4).

### LATS (Zhou và cộng sự, ICML 2024)

LATS hợp nhất ToT, ReAct và Reflexion dưới dạng MCTS. LLM đóng ba vai trò:

- **Policy**: đề xuất các hành động tiếp theo (phong cách ReAct).
- **Value function**: chấm điểm một chuỗi suy luận một phần (tự đánh giá phong cách ToT).
- **Self-reflector**: khi thất bại, viết một phản hồi bằng ngôn ngữ tự nhiên (phong cách Reflexion) và sử dụng nó để tái tạo các lần thử nghiệm (rollouts) trong tương lai.

Phản hồi từ môi trường (quan sát) được trộn vào hàm giá trị để quá trình tìm kiếm được thông tin từ kết quả thực tế, không chỉ là ý kiến của mô hình. Kết quả tại thời điểm công bố: HumanEval pass@1 đạt 92.7% với GPT-4 (SOTA), WebShop đạt trung bình 75.9 với GPT-3.5 (tiệm cận với fine-tuning dựa trên gradient).

### MCTS, tối giản

Bốn giai đoạn mỗi vòng lặp:

1. **Select** — đi từ gốc đến lá bằng UCT (upper confidence bound for trees).
2. **Expand** — tạo K nút con thông qua policy.
3. **Simulate** — thực hiện rollout từ một nút con bằng policy, chấm điểm nút lá bằng hàm giá trị (hoặc phần thưởng từ môi trường).
4. **Backpropagate** — cập nhật số lần truy cập và ước tính giá trị ngược lên đường đi.

Công thức UCT: `Q(s, a) + c * sqrt(ln N(s) / N(s, a))`. Số hạng đầu là khai thác (exploitation); số hạng sau là khám phá (exploration). Điều chỉnh `c` cho từng tác vụ.

### Thực tế về chi phí

Tìm kiếm làm bùng nổ số lượng token. ToT trong Game of 24 sử dụng gấp 100–1000 lần số token so với CoT. LATS cũng tương tự. Điều này không miễn phí; hãy dành việc tìm kiếm cho:

- Các tác vụ mà một chuỗi suy luận đơn lẻ rõ ràng là không đủ (Game of 24, mã nguồn phức tạp).
- Các tác vụ mà thời gian thực thi không quan trọng bằng tính chính xác.
- Các tác vụ có hàm giá trị rẻ và đáng tin cậy (unit test cho mã nguồn, mục tiêu rõ ràng cho toán học).

Nếu tác vụ của bạn có một câu trả lời đúng duy nhất và bộ đánh giá nhiễu, tìm kiếm thường làm mọi thứ tệ hơn — nó tìm ra một câu trả lời sai nhưng có "điểm số cao".

### Định vị năm 2026

Hầu hết các agent trong môi trường production không chạy LATS. Chúng chạy ReAct với xác thực dựa trên công cụ (CRITIC, Bài 05). Tìm kiếm xuất hiện trong các ngách chuyên biệt:

- Các agent lập trình chạy test làm hàm giá trị (phong cách HumanEval).
- Các agent nghiên cứu chuyên sâu khám phá nhiều hướng truy vấn.
- Các quy trình lập kế hoạch phức tạp bên trong các subgraph của LangGraph.

AlphaEvolve (Bài 11) là thái cực của năm 2025: tìm kiếm tiến hóa trên mã nguồn, độ phù hợp có thể kiểm tra bằng máy, đạt được những bước tiến đột phá (cải thiện matmul 4x4 đầu tiên sau 56 năm).

```figure
tree-of-thoughts
```

## Triển khai

`code/main.py` triển khai:

- Một ToT BFS nhỏ trên tác vụ "chọn toán tử số học".
- Một vòng lặp LATS MCTS nhỏ trên cùng tác vụ (Select / Expand / Simulate / Backpropagate) với lựa chọn UCT.
- Một hàm giá trị kết hợp điểm số biểu tượng cộng với điểm số tự đánh giá.

Chạy thử:

```
python3 code/main.py
```

Dấu vết (trace) cho thấy ToT mở rộng ba ứng viên mỗi nút với BFS, so với LATS hội tụ vào rollout tốt nhất thông qua MCTS. Số lượng token được in cho cả hai.

## Sử dụng

LangGraph cung cấp các mẫu khám phá kiểu ToT dưới dạng subgraph; blog của đội ngũ LangChain về LATS (tháng 5 năm 2024) là tài liệu hướng dẫn tham khảo. LlamaIndex cung cấp một agent `TreeOfThoughts`. Đối với hầu hết các agent production năm 2026, mô hình này nằm sau cổng `if task_complexity > threshold: use_search()` — xem mẫu evaluator-optimizer trong Bài 05.

## Triển khai thực tế

`outputs/skill-search-policy.md` lựa chọn giữa ReAct tuyến tính, ToT, LATS và tìm kiếm tiến hóa dựa trên hình thái tác vụ, ngân sách và độ tin cậy của bộ đánh giá.

## Bài tập

1. Chạy LATS nhỏ với UCT c=0.1 so với c=2.0. Điều gì thay đổi trong trace?
2. Thay thế hàm giá trị bằng một bộ chấm điểm nhiễu hơn (thêm jitter ngẫu nhiên). MCTS có còn tìm thấy lá tốt nhất không? Tín hiệu trên nhiễu tối thiểu mà nó có thể chịu đựng là bao nhiêu?
3. Triển khai beam-search ToT (giữ top-k tại mỗi cấp) và so sánh với BFS. Cái nào tốt hơn khi ngân sách token hạn hẹp?
4. Đọc phần 5.1 của LATS. Tái tạo số lượng rollout của HumanEval: cần bao nhiêu rollout để đạt được pass@1 như báo cáo?
5. Đọc phần thảo luận của bài báo LATS về "khi nào LATS ít giúp ích hơn". Viết một quy tắc quyết định dài một đoạn ánh xạ hình thái tác vụ với chiến lược tìm kiếm.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|------------------------|
| Tree of Thoughts | "Branching CoT" | Yao và cộng sự — cây các nút ý tưởng với tự đánh giá |
| LATS | "MCTS cho LLMs" | Zhou và cộng sự — hợp nhất ToT + ReAct + Reflexion dưới MCTS |
| UCT | "Upper confidence bound" | Công thức lựa chọn cân bằng giữa khai thác (Q) và khám phá (ln N / n) |
| Value function | "Độ tốt của trạng thái" | Điểm số LLM được prompt hoặc phần thưởng môi trường; dùng để backprop |
| Policy | "Đề xuất hành động" | Bộ tạo kiểu ReAct; phát ra các ý tưởng/hành động tiếp theo |
| Rollout | "Chuỗi suy luận mô phỏng" | Đi từ một nút đến lá bằng policy, chấm điểm bằng value |
| Backpropagate | "Cập nhật tổ tiên" | Đẩy phần thưởng của lá lên đường đi, cập nhật số lần truy cập và Q |
| Search cost | "Bùng nổ token" | 100-1000x CoT trong Game of 24; ngân sách trước khi áp dụng |

## Đọc thêm

- [Yao và cộng sự, Tree of Thoughts (arXiv:2305.10601)](https://arxiv.org/abs/2305.10601) — bài báo kinh điển
- [Zhou và cộng sự, LATS (arXiv:2310.04406)](https://arxiv.org/abs/2310.04406) — MCTS với phản hồi Reflexion
- [Tổng quan về LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) — các mẫu subgraph cho tìm kiếm
- [AlphaEvolve (arXiv:2506.13131)](https://arxiv.org/abs/2506.13131) — tìm kiếm tiến hóa với bộ đánh giá lập trình được