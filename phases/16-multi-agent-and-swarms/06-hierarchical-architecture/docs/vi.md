# Kiến trúc phân cấp (Hierarchical Architecture) và các chế độ lỗi

> Kiến trúc phân cấp là sự lồng ghép giữa các supervisor. Các agent quản lý (manager) điều hành các sub-manager, và các sub-manager điều hành các worker. CrewAI `Process.hierarchical` là phiên bản chuẩn trong sách giáo khoa: một `manager_llm` thực hiện phân bổ nhiệm vụ động và xác thực kết quả đầu ra. Tương đương trong LangGraph là `create_supervisor(create_supervisor(...))`. Đây là mô hình tự nhiên khi nhiệm vụ phản ánh đúng sơ đồ tổ chức thực tế. Tuy nhiên, đây cũng là mô hình dễ rơi vào tình trạng "vòng lặp quản lý" — các agent quản lý phân công công việc kém, hiểu sai kết quả từ cấp dưới hoặc không đạt được sự đồng thuận. Mô hình tuần tự (Sequential) thường hiệu quả hơn.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 05 (Supervisor Pattern)
**Time:** ~60 phút

## Vấn đề

Khi đã nắm vững mô hình supervisor, bước tiếp theo tự nhiên là: "Điều gì xảy ra nếu các worker cũng là các supervisor?". Các nhóm có các nhóm con; các công ty có các phòng ban trong phòng ban. Kiến trúc phân cấp phản ánh điều đó.

Vấn đề là: LLM manager không giống như quản lý con người. Một quản lý con người có những hiểu biết ổn định về năng lực của cấp dưới. Một LLM manager phải suy luận lại toàn bộ cấu trúc tổ chức ở mỗi lượt dựa trên bất cứ thứ gì có trong ngữ cảnh (context). Chỉ cần một chút sai lệch nhỏ trong ngữ cảnh đó, toàn bộ cây phân cấp sẽ phân bổ công việc sai lệch.

## Khái niệm

### Hình thái

```
                 Manager
                 ┌─────┐
                 └──┬──┘
           ┌────────┴────────┐
           ▼                 ▼
       Sub-Mgr A         Sub-Mgr B
       ┌─────┐           ┌─────┐
       └──┬──┘           └──┬──┘
         ┌┴──┬──┐          ┌┴──┐
         ▼   ▼  ▼          ▼   ▼
       W1  W2  W3         W4  W5
```

Mỗi nút nội bộ (internal node) thực hiện lập kế hoạch, ủy quyền và tổng hợp. Chỉ các nút lá (leaves) mới thực hiện công việc.

### Khi nào nên dùng

- **Ánh xạ tổ chức rõ ràng.** Nếu nhiệm vụ thực tế mang tính chất phòng ban ("pháp chế xem xét tài liệu, tài chính xem xét tài liệu, kỹ thuật xem xét tài liệu, sau đó tóm tắt cho lãnh đạo"), thì cấu trúc phân cấp là rất rõ ràng.
- **Tóm tắt cục bộ.** Mỗi sub-manager tổng hợp kết quả của nhóm mình trước khi top manager nhìn thấy. Top manager chỉ thấy ba bản tóm tắt từ sub-manager thay vì mười lăm kết quả đầu ra từ worker.

### Khi nào nó thất bại

Ba chế độ lỗi mà các báo cáo hậu kiểm năm 2026 liên tục ghi nhận:

1. **Lỗi phân công nhiệm vụ.** Manager đọc mục tiêu, suy diễn sai cách phân rã và ủy quyền cho sai sub-manager. Vì sub-manager tuân thủ làm những gì được giao, lỗi chỉ xuất hiện ở bước tổng hợp cuối cùng — cách xa một cấp so với nơi con người có thể phát hiện ra.
2. **Hiểu sai kết quả đầu ra.** Sub-manager trả về "không thể xác minh tuyên bố X". Top manager tóm tắt thành "tuyên bố X không được xác nhận". Ý nghĩa bị sai lệch ở mỗi cấp độ.
3. **Vòng lặp đồng thuận.** Hai sub-manager không đồng ý với nhau; top manager yêu cầu họ hòa giải; họ lại ủy quyền xuống dưới; worker chạy lại; sub-manager trả về kết quả hơi khác; vòng lặp tiếp diễn. `Process.hierarchical` của CrewAI ngăn chặn điều này bằng giới hạn bước (step limits), nhưng bản thân giới hạn đó lại trở thành một siêu tham số (hyperparameter).

### Câu hỏi quyết định

Tuần tự (pipeline tuyến tính) so với phân cấp: nhiệm vụ của bạn thực sự có các nhóm con độc lập, hay đó chỉ là một luồng tuyến tính đang giả vờ là một cái cây? Nếu là vế sau, hãy dùng tuần tự. Nếu là vế trước, hãy dùng phân cấp nhưng phải có ngân sách hòa giải (reconciliation rules) rõ ràng.

### Triển khai theo framework Role

`Process.hierarchical` của CrewAI kết nối một LLM manager phía trên các nhóm chuyên gia. Manager sẽ:

- nhận nhiệm vụ cấp cao nhất,
- phân công nhiệm vụ con cho các nhóm,
- đánh giá kết quả đầu ra của nhóm,
- quyết định chấp nhận, ủy quyền lại hoặc lặp lại.

Tài liệu: https://docs.crewai.com/en/introduction (tìm mục "Hierarchical Process" trong Core Concepts).

### Triển khai theo framework Graph

LangGraph sử dụng các lệnh gọi `create_supervisor` lồng nhau. Supervisor bên trong có đồ thị riêng của nó; supervisor bên ngoài coi đồ thị bên trong như một nút hộp đen (opaque node). Cách này sạch sẽ hơn CrewAI khi cần gỡ lỗi (bạn có thể bước qua từng đồ thị riêng biệt) nhưng khó thể hiện việc thay đổi cấu trúc cây một cách linh hoạt.

Tham khảo: https://reference.langchain.com/python/langgraph-supervisor.

```figure
swarm-hierarchy-token
```

## Xây dựng

`code/main.py` chạy một hệ thống phân cấp 3 cấp:

- top manager: chia nhiệm vụ thành các nhánh "kỹ thuật" và "pháp lý",
- sub-manager kỹ thuật: chia thành các worker "frontend" và "backend",
- sub-manager pháp lý: một worker.

Bản demo đối chiếu giữa lộ trình thành công (mọi người đồng ý) với **lộ trình bị nhiễu** nơi top manager phân rã sai, gán nhãn "pháp lý" thành "tài chính" và quan sát lỗi lan truyền — sub-manager tuân thủ làm công việc tài chính, bộ tổng hợp cấp cao báo cáo kết quả tài chính, còn câu hỏi pháp lý ban đầu không được trả lời.

Chạy:

```
python3 code/main.py
```

Kết quả đầu ra hiển thị cả hai lộ trình với sự so sánh rõ ràng giữa "những gì được yêu cầu" và "những gì được thực hiện".

## Sử dụng

`outputs/skill-hierarchy-fitness.md` đánh giá xem một nhiệm vụ nhất định nên sử dụng mô hình phân cấp, tuần tự hay supervisor phẳng. Đầu vào: mô tả nhiệm vụ, cấu trúc tổ chức, ngân sách hòa giải. Đầu ra: khuyến nghị mô hình kèm theo các chế độ lỗi cụ thể cần đề phòng.

## Triển khai thực tế

Nếu bạn triển khai phân cấp:

- **Giới hạn độ sâu cây ở mức 2.** Ba cấp độ trở lên đã che giấu hầu hết các lỗi khỏi khả năng quan sát.
- **Ngân sách hòa giải rõ ràng.** Thiết lập số vòng lặp tối đa trước khi top manager phải đưa ra quyết định. Thường là 2.
- **Nguồn gốc trên mỗi bản tổng hợp.** Bản tóm tắt của mỗi nút phải trích dẫn kết quả đầu ra của nút lá nào đã tạo ra nó.
- **Cảnh báo về sự sai lệch phân rã (decomposition drift).** Ghi nhật ký việc phân rã của manager theo từng bước; so sánh với truy vấn của người dùng. Nếu việc phân rã không còn bao hàm truy vấn, hãy kích hoạt cảnh báo.

## Bài tập

1. Chạy `code/main.py` và so sánh lộ trình thành công với lộ trình bị nhiễu. Cần bao nhiêu cấp độ chuyển giao quản lý trước khi kết quả đầu ra cuối cùng lệch hoàn toàn so với câu hỏi của người dùng?
2. Thêm cấp độ thứ ba (top → sub → sub-sub → worker). Đo lường tần suất lộ trình bị nhiễu tự sửa lỗi so với việc lệch hoàn toàn khi độ sâu tăng lên.
3. Triển khai một worker "canary" tại mỗi sub-manager, worker này luôn được hỏi câu hỏi gốc của người dùng mà không thay đổi. Sử dụng câu trả lời của canary để phát hiện sự sai lệch phân rã. Manager nên phản ứng thế nào khi canary không đồng ý với câu trả lời đã tổng hợp?
4. Đọc tài liệu `Process.hierarchical` của CrewAI. Xác định một rào chắn (guardrail) cụ thể mà CrewAI áp dụng (giới hạn bước, ràng buộc manager_llm) và mô tả chế độ lỗi mà nó nhắm tới.
5. So sánh supervisor lồng nhau của LangGraph với phân cấp của CrewAI. Cái nào giúp việc phát hiện các vòng lặp hòa giải dễ dàng hơn?

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|------------------------|
| Hierarchical | "Mô hình sơ đồ tổ chức" | Các supervisor quản lý supervisor; chỉ các nút lá mới làm việc. |
| Manager LLM | "Sếp" | LLM thực hiện phân rã, phân công và xác thực tại một nút nội bộ. |
| Decomposition drift | "Sếp mất phương hướng" | Việc phân chia của top manager không còn bao hàm câu hỏi gốc. |
| Reconciliation loop | "Họp hành vô tận" | Các sub-manager không đồng ý; top manager ủy quyền lại; worker chạy lại; lặp cho đến khi hết ngân sách. |
| Depth-2 ceiling | "Đừng sâu quá 2 cấp" | Rào chắn thực nghiệm: 3 cấp độ trở lên làm mất khả năng quan sát. |
| Canary question | "Sự thật gốc tại mọi cấp" | Một worker luôn được hỏi truy vấn gốc để phát hiện sự sai lệch. |
| Provenance chain | "Ai nói gì" | Truy xuất nguồn gốc từ mỗi bản tổng hợp ngược lại các kết quả đầu ra của nút lá. |

## Đọc thêm

- [Giới thiệu CrewAI — Process.hierarchical](https://docs.crewai.com/en/introduction) — phân cấp chuẩn với LLM manager
- [Tham khảo supervisor LangGraph](https://reference.langchain.com/python/langgraph-supervisor) — supervisor lồng nhau qua `create_supervisor`
- [Kỹ thuật Anthropic — Hệ thống nghiên cứu](https://www.anthropic.com/engineering/multi-agent-research-system) — lý do Anthropic cố tình chọn supervisor phẳng thay vì phân cấp
- [Cemri và cộng sự — Tại sao các hệ thống LLM đa tác nhân thất bại?](https://arxiv.org/abs/2503.13657) — phân loại MAST; phần về lỗi phối hợp ghi lại sự sai lệch phân rã