# Theory of Mind và Sự phối hợp mới nổi (Emergent Coordination)

> Li và cộng sự (arXiv:2310.10701) đã chỉ ra rằng các tác nhân LLM trong một trò chơi hợp tác bằng văn bản thể hiện **Theory of Mind (ToM) bậc cao mới nổi** — khả năng suy luận về những gì một tác nhân khác tin tưởng về niềm tin của tác nhân thứ ba — nhưng lại thất bại trong việc lập kế hoạch dài hạn do vấn đề quản lý ngữ cảnh và ảo giác (hallucination). Riedl (arXiv:2510.10701) đã đo lường sự hiệp đồng bậc cao trong một quần thể và phát hiện ra rằng **chỉ có** điều kiện prompt ToM mới tạo ra sự phân hóa gắn liền với danh tính và tính bổ trợ hướng tới mục tiêu; các LLM có năng lực thấp hơn chỉ cho thấy sự xuất hiện giả tạo. Nói cách khác, sự phối hợp mới nổi phụ thuộc vào prompt và mô hình, chứ không phải tự nhiên mà có. Bài học này triển khai một tác nhân nhận thức ToM tối giản, thực hiện một nhiệm vụ hợp tác với và không có prompt ToM, đồng thời đo lường sự khác biệt về phối hợp dựa trên giao thức Riedl 2025.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 07 (Society of Mind and Debate), Phase 16 · 17 (Generative Agents)
**Time:** ~75 phút

## Vấn đề

Sự phối hợp đa tác nhân thường trông có vẻ kỳ diệu: các tác nhân phân chia công việc, dự đoán hành động của nhau, tránh trùng lặp. Thông thường, sự "mới nổi" này là sản phẩm của kỹ thuật prompt — ai đó đã yêu cầu các tác nhân "phối hợp với nhau". Loại bỏ prompt, sự phối hợp cũng biến mất.

Phát hiện năm 2025 của Riedl nghiêm ngặt hơn: trong các điều kiện được kiểm soát, sự phối hợp chỉ xuất hiện khi các tác nhân được yêu cầu suy luận về **tâm trí của các tác nhân khác** (ToM). Nếu không có prompt ToM, ngay cả các mô hình mạnh cũng chỉ cho thấy các mô hình phối hợp không vượt qua được các kiểm soát thống kê. Điều này rất quan trọng trong sản xuất: các đội ngũ tung ra các tính năng "phối hợp đa tác nhân" vốn phụ thuộc vào prompt và rất dễ đổ vỡ.

Bài học này coi ToM là một khả năng cụ thể (suy luận về niềm tin của niềm tin), xây dựng một tác nhân nhận thức ToM tối giản và đo lường sự phối hợp thực sự trông như thế nào so với vẻ ngoài do prompt tạo ra.

## Khái niệm

### ToM nghĩa là gì

Tâm lý học phát triển: một đứa trẻ 3 tuổi nghĩ rằng thế giới nội tâm của bất kỳ ai cũng giống như của chúng. Một đứa trẻ 5 tuổi hiểu rằng người khác có những niềm tin khác biệt. Một đứa trẻ 7 tuổi có thể suy luận về niềm tin của niềm tin ("cô ấy nghĩ rằng tôi nghĩ quả bóng nằm dưới cái cốc"). Đây lần lượt là ToM bậc 0, bậc 1 và bậc 2.

Đối với các tác nhân LLM, các bậc ToM tương ứng với:

- **Bậc 0:** không có mô hình về người khác. Tác nhân chỉ hành động dựa trên quan sát của chính mình.
- **Bậc 1:** tác nhân có mô hình về niềm tin của mỗi tác nhân khác. "Alice tin rằng X."
- **Bậc 2:** tác nhân mô hình hóa các niềm tin đệ quy. "Alice tin rằng Bob tin rằng X."

Li và cộng sự 2023 phát hiện ra rằng ToM bậc 1 và bậc 2 xuất hiện ở các tác nhân LLM trong các trò chơi hợp tác nhưng bị suy giảm khi tầm nhìn dài hạn và giao tiếp không đáng tin cậy.

### Tóm tắt bài kiểm tra Sally-Anne

Một bài kiểm tra niềm tin sai lệch năm 1985: Sally đặt một viên bi vào giỏ A rồi rời đi. Anne chuyển nó sang giỏ B. Sally sẽ tìm ở đâu khi quay lại? Một đứa trẻ có ToM bậc 1 sẽ nói giỏ A (niềm tin của Sally khác với thực tế). Một đứa trẻ không có ToM sẽ nói giỏ B.

Các LLM thời kỳ GPT-4 vượt qua các bài kiểm tra kiểu Sally-Anne khi được đặt câu hỏi đơn giản. Chúng thất bại khi câu chuyện dài, bối cảnh thay đổi nhiều lần hoặc câu hỏi được đặt ra gián tiếp. Đó là trạng thái thực tế của ToM trong các LLM sản xuất năm 2026.

### Đo lường sự phối hợp của Riedl

Riedl (arXiv:2510.05174) đã xây dựng một bài kiểm tra quy mô quần thể: N tác nhân, một mục tiêu hợp tác, các điều kiện prompt biến đổi. Đo lường:

1. **Phân hóa gắn liền với danh tính (Identity-linked differentiation).** Các tác nhân có phát triển sự phân biệt vai trò ổn định theo thời gian không?
2. **Tính bổ trợ hướng tới mục tiêu (Goal-directed complementarity).** Hành động của các tác nhân có bổ trợ cho nhau (các nhiệm vụ phụ khác nhau) thay vì trùng lặp không?
3. **Sự hiệp đồng bậc cao (Higher-order synergy).** Một thước đo thống kê về việc liệu nhóm có đạt được điều mà không tập hợp con nào có thể làm được hay không.

Kết quả: chỉ trong điều kiện prompt ToM, cả ba chỉ số mới tạo ra tín hiệu vượt trên mức cơ sở. Nếu không có prompt ToM, các chỉ số dao động gần mức ngẫu nhiên đối với các mô hình có năng lực trung bình. Các mô hình lớn cho thấy một số sự phối hợp mà không cần prompt ToM rõ ràng, nhưng hiệu ứng nhỏ hơn so với khi có prompt rõ ràng.

### Ảo tưởng về sự phối hợp

Nếu không có các kiểm soát thống kê, "sự phối hợp mới nổi" trong các bản demo thường phản ánh:

- Kỹ thuật prompt cài cắm sự phối hợp (system prompt yêu cầu "hãy làm việc cùng nhau").
- Thiên kiến người quan sát (chúng ta nhìn thấy các mô hình mà chúng ta mong đợi).
- Lựa chọn hậu kiểm các lần chạy thành công.

Các hệ thống sản xuất quảng cáo "sự phối hợp mới nổi" mà không có tín hiệu đo lường được nên được coi là tiếp thị. Hãy đo lường trước khi khẳng định.

### Tác nhân nhận thức ToM tối giản

Cấu trúc:

```
agent state:
  own_beliefs:    {facts the agent believes}
  other_models:   {other_agent_id -> {beliefs_the_agent_attributes_to_them}}
  actions_last_N: [history of others' actions]

observation update:
  - update own_beliefs from direct observation
  - update other_models[agent_id] from their action + prior beliefs

action selection:
  - enumerate candidate actions
  - for each, predict what each other agent will do next given their modeled beliefs
  - pick action that maximizes joint outcome under those predictions
```

Thuộc tính `other_models` là trạng thái ToM. ToM bậc 1 chỉ giữ một cấp độ. ToM bậc 2 thêm `other_models[i][other_models_of_j]` — những gì tôi nghĩ tác nhân i nghĩ tác nhân j tin tưởng.

### Tại sao tầm nhìn dài hạn lại gây hại

Li và cộng sự ghi nhận: giới hạn ngữ cảnh khiến các tác nhân quên mất niềm tin nào thuộc về ai. Ảo giác thêm các niềm tin sai lệch vào mô hình của các tác nhân khác. Cả hai đều tạo ra các lỗi "Tôi nghĩ anh ấy nghĩ X" tích tụ theo thời gian.

Các biện pháp giảm thiểu được ghi nhận trong bài báo và các nghiên cứu tiếp theo giai đoạn 2024-2026:

- **Trạng thái ToM rõ ràng trong prompt.** Định dạng cấu trúc: `{agent_id: belief_list}`. Buộc việc truy xuất phải bảo toàn sự liên kết giữa danh tính và niềm tin.
- **Chuỗi suy luận ngắn hơn.** Ít cập nhật ToM hơn mỗi lượt giúp giảm thiểu ảo giác tích tụ.
- **Kho lưu trữ ToM bên ngoài.** Duy trì mô hình bên ngoài ngữ cảnh LLM; chỉ đưa vào các phần liên quan mỗi lượt.

### Nơi ToM thất bại trong sản xuất

- **Môi trường đối kháng.** Các tác nhân có ToM tốt dễ bị thao túng hơn (bạn có thể mô hình hóa những gì họ mô hình hóa về bạn, sau đó khai thác).
- **Các đội ngũ không đồng nhất.** Khi các mô hình khác nhau, mô hình ToM hoạt động cho đối thủ này có thể không tổng quát hóa được cho đối thủ khác.
- **Các nhiệm vụ phụ thuộc vào sự thật khách quan.** ToM nói về niềm tin; nếu sự chính xác phụ thuộc vào dữ kiện, ToM có thể gây xao nhãng.

### Sự phối hợp bạn thực sự có thể đo lường

Ba tín hiệu thực tế cho thấy sự phối hợp của một đội ngũ là có thật thay vì chỉ là vẻ ngoài do prompt tạo ra:

1. **Tính bổ trợ theo thời gian.** Trong một nhiệm vụ nhiều lượt, các hành động của tác nhân có bao phủ các nhiệm vụ phụ rời rạc không?
2. **Sự dự đoán.** Hành động của tác nhân A ở lượt T+1 có phụ thuộc vào dự đoán về hành động của B ở lượt T+2 mà hóa ra là đúng không?
3. **Sự điều chỉnh.** Khi A đọc sai niềm tin của B ở lượt T, A có điều chỉnh vào lượt T+2 không?

Những điều này có thể đo lường được trong một hệ thống đa tác nhân có ghi nhật ký. Chúng là phiên bản thực chất của câu chuyện "phối hợp".

```figure
sw-theory-of-mind
```

## Xây dựng

`code/main.py` triển khai:

- `ToMAgent` — theo dõi niềm tin của chính mình và các mô hình niềm tin của từng tác nhân khác.
- Một nhiệm vụ hợp tác: ba tác nhân phải thu thập ba token từ ba chiếc hộp; mỗi hộp chỉ chứa được một token. Các tác nhân không thể giao tiếp; chúng suy luận ý định từ hành động của nhau.
- Hai cấu hình: `zeroth_order` (không ToM) và `first_order` (ToM với mô hình niềm tin một cấp độ).
- Đo lường qua 200 thử nghiệm ngẫu nhiên: tỷ lệ hoàn thành, tỷ lệ trùng lặp (hai tác nhân nhắm vào cùng một hộp), số lượt trung bình để hoàn thành.

Chạy:

```
python3 code/main.py
```

Kết quả mong đợi: các tác nhân bậc 0 trùng lặp nỗ lực ở mức ~35% và hoàn thành ~60% các thử nghiệm trong 10 lượt. Các tác nhân ToM bậc 1 trùng lặp ở mức ~5% và hoàn thành ~95%. Sự khác biệt chính là hiệu ứng phối hợp có thể đo lường được.

## Sử dụng

`outputs/skill-tom-auditor.md` là một kỹ năng kiểm định tuyên bố về "sự phối hợp mới nổi" của một hệ thống đa tác nhân. Kiểm tra vẻ ngoài do prompt tạo ra, ý nghĩa thống kê so với nhóm đối chứng và tính bổ trợ được đo lường.

## Triển khai

Danh sách kiểm tra các tuyên bố về phối hợp:

- **Điều kiện đối chứng.** Một phiên bản hệ thống của bạn không có prompt phối hợp. Hãy đo lường cả hai.
- **Kiểm tra thống kê.** Sự khác biệt giữa hệ thống và đối chứng có ý nghĩa ở mức `p < 0.05` trên chỉ số của bạn không?
- **Thước đo tính bổ trợ.** Sự rời rạc của hành động theo thời gian, không chỉ là thành công cuối cùng.
- **Nhật ký trường hợp thất bại.** Khi các tác nhân phối hợp sai, trạng thái ToM trông như thế nào?
- **Công bố năng lực mô hình.** Nếu hiệu ứng biến mất trên các mô hình nhỏ hơn, hãy nói rõ điều đó.

## Bài tập

1. Chạy `code/main.py`. Xác nhận ToM bậc 1 giảm tỷ lệ trùng lặp khoảng 7 lần. Khoảng cách này có duy trì khi bạn mở rộng lên 5 tác nhân và 5 hộp không?
2. Triển khai ToM bậc 2 (tác nhân A mô hình hóa những gì B nghĩ về C). Nó có cải thiện so với bậc 1 không? Trên các nhiệm vụ nào?
3. Chèn một **ảo giác** vào trạng thái ToM: lật ngẫu nhiên một niềm tin mỗi lượt. Điều này làm suy giảm hiệu suất bậc 1 bao nhiêu?
4. Đọc Li và cộng sự (arXiv:2310.10701). Tái tạo phát hiện "suy giảm tầm nhìn dài hạn": khi số lượt tăng từ 10 lên 30, hiệu suất ToM bậc 1 của bạn thay đổi như thế nào?
5. Đọc Riedl 2025 (arXiv:2510.05174). Triển khai thống kê hiệp đồng bậc cao trên nhật ký mô phỏng của bạn. Hiệu ứng có hiện diện mà không cần điều kiện prompt ToM không?

## Thuật ngữ chính

| Thuật ngữ | Mọi người nói | Ý nghĩa thực tế |
|------|----------------|------------------------|
| Theory of Mind | "Hiểu tâm trí người khác" | Khả năng mô hình hóa niềm tin của tác nhân khác. Được phân bậc (0, 1, 2+). |
| Sally-Anne test | "Bài kiểm tra niềm tin sai lệch" | Tâm lý học phát triển 1985; LLM vượt qua các phiên bản đơn giản, thất bại ở các phiên bản phức tạp. |
| First-order ToM | "A tin rằng X" | Mô hình hóa niềm tin của một người khác về các sự kiện. |
| Second-order ToM | "A tin rằng B tin rằng X" | Mô hình hóa đệ quy sâu hơn một cấp. |
| Identity-linked differentiation | "Vai trò ổn định theo thời gian" | Chỉ số của Riedl: vai trò tồn tại bền vững, không ngẫu nhiên. |
| Goal-directed complementarity | "Hành động rời rạc" | Các tác nhân nhắm vào các nhiệm vụ phụ khác nhau, không phải cùng một nhiệm vụ. |
| Higher-order synergy | "Nhóm vượt trội hơn bất kỳ tập hợp con nào" | Thước đo thống kê của Riedl cho sự phối hợp thực sự. |
| Coordination illusion | "Trông có vẻ phối hợp" | Vẻ ngoài phối hợp do prompt tạo ra mà không có tín hiệu đo lường được. |

## Đọc thêm

- [Li và cộng sự — Theory of Mind for Multi-Agent Collaboration via Large Language Models](https://arxiv.org/abs/2310.10701) — ToM mới nổi trong các trò chơi hợp tác; các chế độ thất bại tầm nhìn dài hạn
- [Riedl — Emergent Coordination in Multi-Agent Language Models](https://arxiv.org/abs/2510.05174) — đo lường quy mô quần thể; prompt ToM là điều kiện then chốt
- [Premack & Woodruff — Does the chimpanzee have a theory of mind?](https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/does-the-chimpanzee-have-a-theory-of-mind/1E96B02CD9850E69AF20F81FA7EB3595) — nguồn gốc năm 1978 của khái niệm ToM
- [Baron-Cohen, Leslie, Frith — Does the autistic child have a theory of mind?](https://doi.org/10.1016/0010-0277(85) — bài báo Sally-Anne (1985)