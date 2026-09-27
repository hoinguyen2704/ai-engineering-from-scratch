# Generative Agents and Emergent Simulation

> Park và cộng sự 2023 (UIST '23, arXiv:2304.03442) đã xây dựng **Smallville**, một môi trường sandbox gồm 25 tác nhân (agent) với kiến trúc ba phần: **memory stream** (nhật ký ngôn ngữ tự nhiên), **reflection** (các tổng hợp cấp cao mà tác nhân tự tạo ra về luồng dữ liệu của chính mình), và **plan** (hành vi theo ngày, sau đó là các kế hoạch con). Kết quả mang tính bước ngoặt là sự xuất hiện của bữa tiệc Valentine: một tác nhân được gieo ý tưởng "muốn tổ chức tiệc Valentine", mà không cần kịch bản chi tiết, đã tạo ra các lời mời lan truyền trong cộng đồng, phối hợp thời gian và bữa tiệc đã thực sự diễn ra — từ 24 tác nhân ban đầu không hề biết gì về nó. Các thử nghiệm cắt bỏ (ablation) cho thấy cả ba thành phần đều cần thiết để tạo ra sự tin cậy (believability). Những lỗi được ghi nhận là sai lệch về chuẩn mực không gian (đi vào các cửa hàng đã đóng cửa, dùng chung nhà vệ sinh đơn). Đây là kiến trúc tham chiếu cho các mô phỏng tác nhân và đánh giá xã hội đa tác nhân vào năm 2026.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 04 (Primitive Model), Phase 16 · 13 (Shared Memory)
**Time:** ~75 phút

## Vấn đề

Hầu hết các hệ thống đa tác nhân đều là các nhóm được kịch bản hóa chặt chẽ: người lập kế hoạch lập kế hoạch, người viết mã viết mã, người đánh giá đánh giá. Điều đó hiệu quả với các tác vụ được xác định rõ ràng. Nó không nắm bắt được hành vi tự phát, không theo kịch bản nảy sinh khi các tác nhân có trí nhớ, ưu tiên và một thế giới mở. Nghiên cứu, mô phỏng xã hội và ngày càng nhiều AI trong trò chơi cần loại thứ hai này.

Kiến trúc Smallville là chuẩn mực cho điều đó. Cho đến trước nghiên cứu của Park năm 2023, các mô phỏng tác nhân tốt nhất chỉ là những kẻ tuân theo kịch bản nông cạn; sau đó, mô hình này trở thành mặc định cho các tác nhân tạo sinh trong thế giới mở. Nếu bạn xây dựng một mô phỏng tác nhân vào năm 2026, bạn đang sử dụng ba thành phần của Smallville hoặc phải giải trình rõ ràng lý do tại sao không.

## Khái niệm

### Ba thành phần chính

**Memory stream.** Một nhật ký chỉ thêm (append-only) chứa các quan sát, hành động, suy ngẫm và kế hoạch. Mỗi mục có dấu thời gian, loại, mô tả (ngôn ngữ tự nhiên) và siêu dữ liệu dẫn xuất: **recency** (độ gần đây), **importance** (tầm quan trọng, do tác nhân tự đánh giá 1-10) và **relevance** (độ liên quan, tính bằng cosine similarity với truy vấn hiện tại).

```
[2026-02-14 09:12:03] observation: Isabella Rodriguez asked me if I like jazz
[2026-02-14 09:14:22] reflection:   I enjoy long conversations about music
[2026-02-14 10:05:00] plan:         Attend Isabella's Valentine's Day party tonight
```

Việc truy xuất bộ nhớ kết hợp cả ba điểm số: `score = w_recency * e^(-decay * age) + w_importance * importance + w_relevance * cos_sim`. Các mục có điểm cao nhất (top-k) sẽ được đưa vào prompt hiện tại.

**Reflection.** Định kỳ (sau mỗi N ký ức hoặc khi có sự kiện quan trọng), tác nhân tạo ra các tổng hợp bậc cao từ những ký ức gần đây. Các mục reflection được đưa ngược lại vào luồng bộ nhớ và có thể truy xuất như bất kỳ ký ức nào khác. Đây là cách các tác nhân xây dựng "sự hiểu biết" — tương đương với niềm tin dài hạn trong kiến trúc này.

**Plan.** Phân rã từ trên xuống dưới (top-down). Đầu tiên là kế hoạch theo ngày với các nét chính ("đi làm, ăn tối với Klaus"). Sau đó là kế hoạch theo giờ. Cuối cùng là kế hoạch theo hành động. Các kế hoạch có thể sửa đổi: khi một quan sát mâu thuẫn với kế hoạch, tác nhân sẽ lập kế hoạch lại cho phân đoạn bị ảnh hưởng.

### Tại sao cả ba đều quan trọng (ablation)

Park và cộng sự đã thực hiện các thử nghiệm cắt bỏ bằng cách loại bỏ lần lượt quan sát, suy ngẫm và kế hoạch. Mỗi lần cắt bỏ đều làm giảm độ tin cậy:

- Không có **quan sát**, tác nhân bỏ lỡ ngữ cảnh và hành động dựa trên niềm tin lỗi thời.
- Không có **suy ngẫm**, tác nhân không thể hình thành niềm tin bậc cao; các tương tác vẫn nông cạn.
- Không có **kế hoạch**, hành vi trở thành sự nhiễu loạn phản ứng; các mục tiêu tan biến.

Điểm tin cậy từ người đánh giá là cao nhất khi có đủ cả ba; việc loại bỏ bất kỳ thành phần nào cũng tạo ra sự suy giảm có thể đo lường được.

### Sự xuất hiện của bữa tiệc Valentine

Một tác nhân, Isabella Rodriguez, được gieo mục tiêu "muốn tổ chức tiệc Valentine tại Hobbs Cafe vào ngày 14 tháng 2 lúc 5 giờ chiều." 24 tác nhân còn lại không nhận được mục tiêu này. Qua các ngày mô phỏng:

1. Kế hoạch của Isabella bao gồm việc mời mọi người.
2. Mỗi lời mời trở thành một quan sát trong luồng bộ nhớ của hàng xóm.
3. Suy ngẫm của hàng xóm đó tạo ra niềm tin: "Isabella đang tổ chức tiệc."
4. Kế hoạch của hàng xóm kết hợp thêm "tham dự tiệc vào ngày 14 tháng 2."
5. Hàng xóm kể cho những người hàng xóm khác. Lời mời lan truyền mà không cần sự điều phối trung tâm.
6. Vào lúc 5 giờ chiều ngày 14 tháng 2, một vài tác nhân tập trung tại Hobbs Cafe.

Đây là sự xuất hiện (emergence) theo nghĩa kỹ thuật: hành vi cấp hệ thống (một bữa tiệc) nảy sinh từ các tương tác cục bộ (lời mời song phương + lập kế hoạch cá nhân) mà không cần người điều phối trung tâm.

### Các dạng lỗi được ghi nhận

Park và cộng sự đã ghi nhận rõ ràng:

- **Lỗi chuẩn mực không gian.** Các tác nhân đi vào các cửa hàng đã đóng cửa. Các tác nhân cố gắng sử dụng cùng một nhà vệ sinh đơn. Các tác nhân ăn trong các phòng không dành cho việc ăn uống. Mô hình không tự suy luận ra các chuẩn mực xã hội-vật lý chỉ từ môi trường.
- **Tràn bộ nhớ.** Các mô phỏng sâu khiến chi phí truy xuất bộ nhớ tăng lên. Giải pháp thực tế: nén bộ nhớ định kỳ (tóm tắt và cắt tỉa) và làm suy giảm các mục có tầm quan trọng thấp.
- **Ảo tưởng suy ngẫm (Reflection hallucination).** Các suy ngẫm có thể tạo ra các mối quan hệ không tồn tại trong luồng bộ nhớ. Cách giảm thiểu: bao gồm ID nguồn của ký ức trong các prompt suy ngẫm và xác minh tại thời điểm truy xuất.

Đây là những dạng lỗi liên quan đến sản xuất: bất kỳ mô phỏng tác nhân nào vào năm 2026 đều kế thừa chúng.

### Quy tắc triển khai ba thành phần

1. **Bộ nhớ chỉ thêm (append-only).** Không bao giờ sửa đổi một mục bộ nhớ. Các chỉnh sửa là các mục mới.
2. **Điểm tầm quan trọng rất rẻ.** Gọi LLM để đánh giá tầm quan trọng 1-10 tại thời điểm ghi. Lưu trữ điểm số đó.
3. **Truy xuất được xếp hạng, không lọc.** Lấy top-k theo điểm tổng hợp; không sử dụng bộ lọc cứng (vì sẽ làm mất ngữ cảnh).
4. **Suy ngẫm chạy định kỳ.** Kích hoạt khi tổng tầm quan trọng của các ký ức chưa xử lý vượt quá ngưỡng (ví dụ: 150).
5. **Kế hoạch có thể sửa đổi.** Khi một quan sát mới mâu thuẫn với kế hoạch, chỉ tạo lại phân đoạn bị ảnh hưởng, không phải toàn bộ kế hoạch.

### Các tác nhân tạo sinh ngoài Smallville

Tài liệu nghiên cứu giai đoạn 2024-2026 mở rộng kiến trúc này:

- **Mô phỏng xã hội đa tác nhân cho nghiên cứu chính sách / thị trường.** Các quần thể giống Smallville mô phỏng hành vi người dùng để phản hồi các tính năng. Nhanh hơn A/B test; độ chính xác vẫn còn gây tranh cãi.
- **AI NPC cho trò chơi.** Các game RPG với tác nhân Smallville tạo ra các cốt truyện tự phát thay vì các nhiệm vụ theo kịch bản.
- **Các benchmark đánh giá tác nhân tạo sinh.** Thay vì độ chính xác của tác vụ, thước đo trở thành độ tin cậy + tính nhất quán của hành vi trong các lần chạy dài.

Kiến trúc này là tham chiếu. Các phần mở rộng thay thế các thành phần (vector store cho bộ nhớ, suy ngẫm tăng cường truy xuất, kế hoạch neurosymbolic) nhưng vẫn giữ cấu trúc ba phần.

### Tại sao điều này quan trọng đối với kỹ thuật đa tác nhân

Smallville là bằng chứng cho thấy sự xuất hiện đa tác nhân rất rẻ khi các thành phần phù hợp. Kiến trúc này hiện đã được sao chép trên các mô hình mã nguồn mở (các LLM nhỏ hơn mất đi độ tin cậy một cách từ từ, không đột ngột). Bất kỳ hệ thống sản xuất nào cần **hành vi xã hội tự phát** đều sử dụng hình thái này. Bất kỳ hệ thống nào cần **thực thi tác vụ chặt chẽ** đều sử dụng các mô hình giám sát / vai trò / nguyên thủy từ các giai đoạn trước.

```figure
a5-memory-reflection
```

## Xây dựng

`code/main.py` triển khai ba thành phần trong Python stdlib với các chính sách tác nhân được kịch bản hóa (không dùng LLM thực). Bản demo tái tạo sự xuất hiện của bữa tiệc Valentine ở quy mô nhỏ:

- `MemoryStream` — nhật ký chỉ thêm với truy xuất theo độ gần đây/tầm quan trọng/độ liên quan.
- `reflect(stream)` — suy ngẫm được kịch bản hóa trên các ký ức quan trọng gần đây.
- `plan(agent_state)` — kế hoạch theo ngày và giờ dựa trên niềm tin hiện tại.
- Kịch bản: 5 tác nhân. Tác nhân 1 bắt đầu với "tổ chức tiệc lúc 5 giờ chiều." Qua các tick mô phỏng, lời mời lan truyền và các tác nhân hội tụ.

Chạy:

```
python3 code/main.py
```

Kết quả mong đợi: dấu vết theo từng tick. Đến tick cuối cùng, ít nhất 3 trong số 5 tác nhân hiển thị bữa tiệc trong kế hoạch của họ và họ hội tụ tại địa điểm tổ chức tiệc. Hạt giống duy nhất đã tạo ra sự phối hợp đến nơi mà không cần bất kỳ người điều phối nào.

## Sử dụng

`outputs/skill-simulation-designer.md` thiết kế một mô phỏng tác nhân tạo sinh: số lượng tác nhân, lược đồ bộ nhớ, nhịp độ suy ngẫm, tầm nhìn kế hoạch và thước đo đánh giá.

## Triển khai

Các quy tắc cho mô phỏng sản xuất:

- **Bộ nhớ là cơ sở dữ liệu.** Chọn một kho lưu trữ thực (vector DB, Postgres) ở quy mô lớn. Stdlib trong bộ nhớ chỉ dành cho nguyên mẫu.
- **Ghi nhật ký dấu vết truy xuất.** Đối với mỗi hành động, hãy ghi lại top-k ký ức đã thúc đẩy nó. Đây là khả năng gỡ lỗi của bạn.
- **Ngân sách token cho mỗi tác nhân.** Mỗi lần truy xuất + suy ngẫm + lập kế hoạch của tác nhân trên mỗi tick là O(k) lệnh gọi LLM. N tác nhân × T tick × số lệnh gọi mỗi tick có thể vượt quá ngân sách của bạn.
- **Nén bộ nhớ định kỳ.** Tóm tắt và cắt tỉa các mục có tầm quan trọng thấp. Chính sách lưu giữ là một quyết định thiết kế, không phải là chi tiết nhỏ.
- **Phát hiện vi phạm chuẩn mực không gian / xã hội** một cách rõ ràng. Kiến trúc không tự học được chúng.

## Bài tập

1. Chạy `code/main.py`. Xác nhận 3+ tác nhân hội tụ tại bữa tiệc. Tăng số lượng tác nhân lên 10 — sự xuất hiện có còn xảy ra không?
2. Loại bỏ bước suy ngẫm. Hành vi trông như thế nào? Ánh xạ tới kết quả cắt bỏ trong nghiên cứu của Park năm 2023.
3. Giới thiệu một mục tiêu cạnh tranh ("Klaus muốn thực hiện một bài nói chuyện nghiên cứu lúc 5 giờ chiều"). Các tác nhân có bị chia rẽ không, hay một mục tiêu chiếm ưu thế? Điều gì quyết định điều đó?
4. Thêm các ràng buộc không gian: Hobbs Cafe chứa tối đa 4 tác nhân. Mô phỏng có xử lý tình trạng tràn một cách ổn thỏa không, hay nó gặp phải lỗi "nhà vệ sinh đơn"?
5. Đọc Park và cộng sự (arXiv:2304.03442) Phần 6 (các thí nghiệm về hành vi tự phát). Xác định một hành vi không thể tái tạo trong mô phỏng nhỏ của bạn. Bạn cần nâng cao thành phần nào của kiến trúc?

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|------------------------|
| Memory stream | "Nhật ký của tác nhân" | Nhật ký chỉ thêm chứa các quan sát, hành động, suy ngẫm, kế hoạch. |
| Recency | "Ký ức mới đến mức nào" | Điểm số suy giảm theo hàm mũ dựa trên độ tuổi. |
| Importance | "Tác nhân quan tâm đến mức nào" | Tự đánh giá 1-10 tại thời điểm ghi. Được lưu trữ. |
| Relevance | "Liên quan thế nào đến truy vấn" | Cosine similarity (dựa trên embedding). |
| Reflection | "Niềm tin bậc cao" | Tổng hợp được tạo từ các ký ức gần đây, được nạp lại như một ký ức mới. |
| Plan | "Phân rã ngày/giờ/hành động" | Cây kế hoạch từ trên xuống dưới. Có thể sửa đổi khi quan sát mâu thuẫn. |
| Smallville | "Sandbox của Park 2023" | Mô phỏng 25 tác nhân tạo ra sự xuất hiện của bữa tiệc Valentine. |
| Believability | "Thước đo chất lượng" | Điểm đánh giá của con người về việc hành vi có giống một tác nhân hợp lý hay không. |

## Đọc thêm

- [Park và cộng sự — Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442) — kiến trúc tham chiếu
- [Trang bài báo UIST '23](https://dl.acm.org/doi/10.1145/3586183.3606763) — nơi xuất bản
- [Phát hành mã nguồn Smallville](https://github.com/joonspk-research/generative_agents) — triển khai Python tham chiếu
- [Hayes-Roth 1985 — A Blackboard Architecture for Control](https://www.sciencedirect.com/science/article/abs/pii/0004370285900639) — nghiên cứu tiền đề cho các tác nhân có bộ nhớ cấu trúc