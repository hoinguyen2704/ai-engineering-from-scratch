# Chuyên môn hóa vai trò — Planner, Critic, Executor, Verifier

> Phân rã đa tác nhân (multi-agent) phổ biến nhất vào năm 2026: một tác nhân lập kế hoạch, một tác nhân thực thi, và một tác nhân phê bình hoặc xác minh. MetaGPT (arXiv:2308.00352) chính thức hóa điều này thành các SOP (Quy trình vận hành tiêu chuẩn) được mã hóa vào các prompt vai trò — Product Manager, Architect, Project Manager, Engineer, QA Engineer — tuân theo `Code = SOP(Team)`. ChatDev (arXiv:2307.07924) liên kết designer, programmer, reviewer, tester thông qua một "chuỗi trò chuyện" với cơ chế "communicative dehallucination" (các tác nhân chủ động yêu cầu làm rõ các chi tiết còn thiếu). Vai trò verifier đóng vai trò then chốt: Cemri và cộng sự (MAST, arXiv:2503.13657) chỉ ra rằng mọi thất bại của hệ thống đa tác nhân đều có thể bắt nguồn từ việc thiếu hoặc hỏng khâu xác minh. PwC báo cáo mức tăng độ chính xác gấp 7 lần (10% → 70%) nhờ các vòng lặp xác thực có cấu trúc trong CrewAI.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 04 (Primitive Model), Phase 16 · 05 (Supervisor)
**Time:** ~60 phút

## Vấn đề

Các hệ thống đa tác nhân chung chung thường tạo ra kết quả chung chung. Ba lập trình viên trong một nhóm chat sẽ viết ra ba phiên bản của cùng một đoạn mã tầm thường. Bạn có thể thêm nhiều tác nhân hơn, thêm nhiều vòng lặp hơn, nhưng vẫn không vượt qua được ngưỡng chất lượng mong muốn.

Giải pháp không phải là thêm nhiều tác nhân — mà là các tác nhân *khác biệt*. Hãy gán các vai trò riêng biệt. Cung cấp cho critic các công cụ mà planner không có. Cung cấp cho verifier một bộ kiểm thử khách quan. Lúc này, hệ thống sẽ có sự bất đồng nội bộ với các điều chỉnh dựa trên thực tế, thay vì chỉ là những phỏng đoán song song.

## Khái niệm

### Bốn vai trò kinh điển

**Planner.** Đọc mục tiêu, tạo ra danh sách các bước hoặc đặc tả kỹ thuật. Công cụ: truy xuất tri thức, tài liệu. Đầu ra: kế hoạch có cấu trúc.

**Executor.** Đọc từng bước của kế hoạch, tạo ra sản phẩm (artifact). Công cụ: các công cụ làm việc thực tế (trình biên dịch mã, shell, API client). Đầu ra: sản phẩm.

**Critic.** Đọc đầu ra của executor và đối chiếu với ý định của planner. Công cụ: quyền truy cập chỉ đọc vào sản phẩm, phân tích tĩnh. Đầu ra: chấp nhận/từ chối kèm lý do.

**Verifier.** Đọc sản phẩm và chạy kiểm tra xác định (deterministic). Công cụ: trình chạy kiểm thử (test runner), trình kiểm tra kiểu dữ liệu (type checker), trình xác thực schema. Đầu ra: đạt/không đạt kèm bằng chứng.

Critic mang tính chủ quan, có quan điểm riêng và thường dựa trên LLM. Verifier mang tính khách quan, xác định và thường dựa trên mã nguồn. Chúng không phải là cùng một vai trò.

### Mô hình SOP của MetaGPT

MetaGPT (arXiv:2308.00352) mã hóa các SOP kỹ thuật phần mềm thành các prompt vai trò:

- **Product Manager** viết PRD.
- **Architect** tạo thiết kế hệ thống.
- **Project Manager** chia nhỏ nhiệm vụ.
- **Engineer** thực hiện triển khai.
- **QA Engineer** chạy kiểm thử.

Mỗi vai trò có một schema đầu vào/đầu ra nghiêm ngặt. Prompt vai trò xác định vai trò đó *là gì* và *phải tạo ra cái gì*. Công thức `Code = SOP(Team)` — các SOP xác định biến một đội ngũ LLM thành một quy trình có thể dự đoán được.

### Communicative dehallucination của ChatDev

ChatDev bổ sung một bước quan trọng: khi một executor cần một chi tiết cụ thể không có trong kế hoạch, nó sẽ chủ động hỏi designer trước khi tiếp tục. Điều này ngăn chặn lỗi kinh điển của LLM là tự bịa ra chi tiết một cách hợp lý.

Triển khai: prompt vai trò bao gồm "khi bạn cần thông tin cụ thể mà bạn không được cung cấp, hãy hỏi vai trò liên quan theo tên trước khi tạo đầu ra."

### Tại sao verifier quan trọng nhất

Cemri và cộng sự (MAST) đã truy vết 1642 thất bại thực thi của hệ thống đa tác nhân. 21,3% là do lỗ hổng xác minh — hệ thống đã xuất ra một câu trả lời mà chưa ai kiểm tra. 79% còn lại thường bắt nguồn từ việc "có một bước kiểm tra nhưng bị lỗi âm thầm hoặc không bao giờ được chạy". Xác minh là vai trò chịu tải chính.

PwC báo cáo (triển khai CrewAI, 2025) rằng việc thêm một vòng lặp xác thực có cấu trúc đã nâng độ chính xác từ 10% lên 70%. Mức tăng gấp 7 lần chỉ từ một vai trò.

### Critic so với verifier

- Critic là một LLM đánh giá sản phẩm về mặt chất lượng. Mang tính chủ quan. Có thể bị đánh lừa bởi văn phong thuyết phục.
- Verifier là một chương trình xác định chạy trên sản phẩm. Mang tính khách quan. Đưa ra kết quả đạt/không đạt kèm bằng chứng.

Hãy sử dụng cả hai. Critic bắt được các vấn đề về thẩm mỹ mà verifier không thể diễn đạt. Verifier bắt được các lỗi mà critic không thể thấy vì chúng chỉ xuất hiện khi chạy thực tế (runtime).

### Anti-pattern (Mẫu thiết kế xấu)

Mọi vai trò trong hệ thống của bạn đều là LLM và đầu ra của mọi vai trò đều là "trông có vẻ ổn". Đây là kiểu thất bại MAST kinh điển. Hãy thêm ít nhất một verifier mà kết quả đạt/không đạt được quyết định bởi mã nguồn, không phải bởi LLM.

### Ánh xạ khung làm việc (Framework mappings)

- **CrewAI** — `Agent(role, goal, backstory)` là bề mặt chuyên môn hóa chuẩn mực.
- **LangGraph** — các node có thể có các prompt chuyên biệt; các cạnh (edges) thực thi quy trình.
- **AutoGen** — các ConversableAgent chuyên biệt theo vai trò với tên gọi một từ trong GroupChat.
- **OpenAI Agents SDK** — các công cụ bàn giao (handoff) giữa các Agent chuyên biệt vai trò.

```figure
swarm-roles
```

## Xây dựng

`code/main.py` triển khai một quy trình 4 vai trò để xây dựng một hàm Python đơn giản:

- **Planner** tạo ra đặc tả.
- **Executor** tạo ra chuỗi mã.
- **Critic** (mô phỏng bởi LLM) gắn cờ các vấn đề rõ ràng.
- **Verifier** chạy mã đã tạo trong sandbox (`exec`) đối chiếu với một trường hợp kiểm thử.

Demo chạy hai lần: một lần executor tạo mã đúng (critic + verifier đều đạt), một lần executor tạo mã sai đặc tả (critic bỏ lỡ lỗi vì nó trông có vẻ hợp lý, verifier bắt được vì kiểm thử thất bại).

Chạy:

```
python3 code/main.py
```

## Sử dụng

`outputs/skill-role-designer.md` nhận một nhiệm vụ và tạo ra danh sách vai trò (3-5 vai trò), schema đầu vào/đầu ra cho mỗi vai trò và kiểm tra xác minh. Hãy sử dụng công cụ này trước khi kết nối các tác nhân vào một framework.

## Triển khai

Danh sách kiểm tra:

- **Ít nhất một verifier xác định.** Không bao giờ dùng toàn bộ là LLM.
- **Schema I/O rõ ràng cho mỗi vai trò.** Planner trả về đặc tả, không phải văn xuôi; executor đọc schema đó.
- **Communicative dehallucination.** Executor phải hỏi planner khi thiếu thông tin; không bao giờ tự bịa ra.
- **Thứ tự Critic/Verifier.** Chạy critic trước (chi phí thấp, bắt lỗi thiết kế), verifier sau (chậm, bắt lỗi logic).
- **Ngân sách vòng lặp.** Tối đa 2 vòng lặp sửa đổi critic-executor trước khi leo thang lên con người.

## Bài tập

1. Chạy `code/main.py` và quan sát cách verifier bắt được lỗi mà critic bỏ lỡ. Thêm một kiểm tra phân tích tĩnh (đếm số lần xuất hiện của `return`) như một verifier bổ sung. Nó bắt được những gì mà kiểm thử runtime bỏ lỡ?
2. Thêm vai trò thứ 5: "requirements analyst" (chuyên viên phân tích yêu cầu) để chuyển đổi mong muốn của người dùng thành đặc tả cho planner. Những yêu cầu communicative dehallucination nào nên được gửi lên vai trò này?
3. Đọc MetaGPT Phần 3 ("Agents"). Liệt kê schema đầu vào/đầu ra của 5 vai trò trong MetaGPT.
4. Đọc sơ đồ chuỗi trò chuyện của ChatDev (arXiv:2307.07924 Hình 3). Xác định nơi communicative dehallucination phá vỡ một vòng lặp vốn dĩ sẽ là vô tận.
5. Mức tăng độ chính xác gấp 7 lần của PwC đến từ các vòng lặp xác thực. Hãy giả định ba nhiệm vụ mà việc thêm verifier sẽ không giúp ích — nơi mà việc kiểm tra tính đúng đắn một cách xác định là không thể hoặc quá tốn kém.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|------------------------|
| Chuyên môn hóa vai trò | "Các tác nhân khác nhau, công việc khác nhau" | Các system prompt riêng biệt được tinh chỉnh cho vai trò planner/executor/critic/verifier. |
| Mô hình SOP | "Quy trình vận hành tiêu chuẩn được mã hóa" | Cách tiếp cận của MetaGPT: schema I/O nghiêm ngặt cho mỗi vai trò biến một đội ngũ thành một quy trình. |
| Communicative dehallucination | "Hỏi trước khi bịa đặt" | Mô hình ChatDev: executor hỏi planner khi thiếu chi tiết thay vì tự bịa ra. |
| Critic | "Người đánh giá LLM" | Người đánh giá chủ quan, có quan điểm. Bắt các vấn đề về thẩm mỹ. Có thể bị đánh lừa bởi văn phong thuyết phục. |
| Verifier | "Kiểm tra xác định" | Kết quả đạt/không đạt dựa trên mã nguồn. Trình chạy kiểm thử, trình kiểm tra kiểu dữ liệu, trình xác thực schema. Không thể bị đánh lừa. |
| Lỗ hổng xác minh | "Không ai kiểm tra" | 21,3% các thất bại MAST. Sản phẩm được xuất xưởng mà không có bước kiểm tra nào có thể bắt được lỗi. |
| Vòng lặp sửa đổi | "Critic gửi lại" | Việc critic từ chối kích hoạt executor chạy lại với phản hồi. Cần có ngân sách. |
| Anti-pattern toàn LLM | "Trông có vẻ ổn" | Mọi vai trò đều là LLM, không có kiểm tra xác định. Kiểu thất bại MAST kinh điển. |

## Đọc thêm

- [Hong và cộng sự — MetaGPT: Meta Programming for Multi-Agent Collaboration](https://arxiv.org/abs/2308.00352) — bài báo tham khảo về SOP-as-role-prompt
- [Qian và cộng sự — Communicative Agents for Software Development (ChatDev)](https://arxiv.org/abs/2307.07924) — chuỗi trò chuyện + communicative dehallucination
- [Cemri và cộng sự — Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657) — phân loại MAST; lỗ hổng xác minh chiếm 21,3% các thất bại
- [Tài liệu CrewAI — Agent roles](https://docs.crewai.com/en/introduction) — bề mặt đặc tả vai trò trong môi trường sản xuất