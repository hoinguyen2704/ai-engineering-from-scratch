# Parallel / Swarm / Networked Architectures

> Đối lập với supervisor: không có bộ quyết định trung tâm. Các agent đọc từ một event bus chung, lấy công việc một cách bất đồng bộ và ghi kết quả trở lại. LangGraph hỗ trợ rõ ràng "Swarm Architecture" cho các môi trường phi tập trung, năng động. Matrix (arXiv:2511.21686) biểu diễn cả luồng điều khiển và luồng dữ liệu dưới dạng các thông điệp được tuần tự hóa, truyền qua các hàng đợi phân tán để loại bỏ nút thắt cổ chai của bộ điều phối (orchestrator). Sự đánh đổi ở đây rất rõ ràng: tính xác định và khả năng truy xuất nguồn gốc để đổi lấy khả năng mở rộng. Swarm phù hợp với các tác vụ có nhiều bài toán con độc lập; nó không phù hợp với các tác vụ cần một kế hoạch tổng thể nhất quán.

**Type:** Learn + Build
**Languages:** Python (stdlib, `threading`, `queue`)
**Prerequisites:** Phase 16 · 05 (Supervisor Pattern), Phase 16 · 04 (Primitive Model)
**Time:** ~75 minutes

## Problem

Supervisor có thể mở rộng cho một vài worker. Nhưng còn hàng trăm thì sao? Bản thân supervisor trở thành nút thắt cổ chai: mọi quyết định về việc ai làm gì đều phải thông qua một agent duy nhất. Một bước kế hoạch chậm trễ sẽ làm đình trệ toàn bộ hệ thống.

Kiến trúc Swarm đảo ngược thiết kế này. Thay vì một bộ lập kế hoạch trung tâm phân phối công việc, các worker tự lấy công việc từ một hàng đợi chung. "Sự phối hợp" được tích hợp sẵn vào ngữ nghĩa của event bus. Không có bộ điều phối; hệ thống mở rộng cho đến khi hàng đợi đạt giới hạn.

## Concept

### The shape

```
                ┌──── shared queue ────┐
                │                      │
       ┌────────┼────────┐  ◄──────┬───┘
       ▼        ▼        ▼         │
     Worker  Worker  Worker   Worker
      A       B       C        D
       │        │        │         │
       └────────┴────────┴─────────┘
                 │
                 ▼
            results pool
```

Không có bộ điều phối. Mỗi worker lặp lại quy trình: lấy tác vụ, xử lý, ghi kết quả (và tùy chọn thêm các tác vụ tiếp theo vào hàng đợi).

### When swarm fits

- **Nhiều tác vụ độc lập.** Cào dữ liệu (scraping), chuyển đổi, phân loại. Các tác vụ không phụ thuộc lẫn nhau.
- **Công việc có thời lượng biến thiên.** Nếu một số tác vụ mất 100ms và số khác mất 10s, swarm sẽ tự động cân bằng tải — các worker nhanh sẽ lấy các công việc tiếp theo. Một supervisor phải dự đoán trước thời lượng.
- **Thông lượng quan trọng hơn tính xác định.** Bạn quan tâm đến tổng thời gian hoàn thành, không phải thứ tự nghiêm ngặt.

### When swarm fails

- **Quy trình làm việc theo thứ tự.** Nếu bước 3 cần đầu ra của bước 2, swarm có nguy cơ thực thi bước 3 trước khi bước 2 hoàn thành.
- **Các tác vụ cần kế hoạch toàn cục.** Các câu hỏi nghiên cứu phức tạp cần một bộ lập kế hoạch. Một swarm các nhà nghiên cứu sẽ tạo ra các sự kiện độc lập, không phải một báo cáo nhất quán.
- **Gỡ lỗi (Debugging).** Không có nhật ký trung tâm và công việc diễn ra bất đồng bộ, việc tái hiện lỗi rất tốn kém.

### Matrix (arXiv:2511.21686)

Matrix là bài báo năm 2025 đưa swarm đến kết luận tự nhiên của nó: cả luồng điều khiển và luồng dữ liệu đều là các thông điệp được tuần tự hóa trên các hàng đợi phân tán. Không có bộ điều phối trung tâm. Khả năng chịu lỗi đến từ độ bền của thông điệp. Khả năng mở rộng là vấn đề của message broker, không phải của hệ thống.

Đóng góp: một mô hình lập trình nơi sự phối hợp đa agent là "agent này đăng ký vào chủ đề (topic) thông điệp nào?" thay vì "supervisor chọn agent nào tiếp theo?". Điều này làm cho hệ thống trông giống như một lưới sự kiện pub/sub.

### Swarm in graph frameworks

Tài liệu LangGraph 2025 mô tả rõ ràng "Swarm Architecture" là một trong những mô hình đa agent: các agent là các node, nhưng các cạnh tạo thành một đồ thị có hướng với các chu trình và bất kỳ node nào cũng có thể được kích hoạt từ pool. Một worker chọn từ công việc khả dụng dựa trên điều kiện, không phải do supervisor chỉ định.

### Failure mode: starvation and hot-spotting

Nếu tất cả các worker đều lấy tác vụ nhanh nhất có thể, các tác vụ chạy lâu sẽ không bao giờ được chọn cho đến khi chúng là những tác vụ cuối cùng còn lại. Đây là hiện tượng đói tài nguyên (starvation) trong hàng đợi cổ điển.

Các biện pháp giảm thiểu:
- Hàng đợi ưu tiên với cơ chế aging (tăng độ ưu tiên theo thời gian chờ).
- Chuyên môn hóa worker: một số worker chỉ nhận các tác vụ "dài".
- Back-pressure: giới hạn số lượng tác vụ nhanh đi vào hàng đợi.

### The content-based routing link

Swarm kết hợp tự nhiên với định tuyến dựa trên nội dung (Bài 22). Thay vì một hàng đợi chung, hãy có một hàng đợi cho mỗi loại thông điệp. Các worker chuyên biệt chỉ đăng ký loại của chúng. Đây là cơ sở cho các kiến trúc message-bus có thể mở rộng lên hàng nghìn agent.

```figure
sw-work-stealing
```

## Build It

`code/main.py` triển khai một swarm gồm 4 luồng worker lấy công việc từ một `queue.Queue` chung. Các tác vụ có thời lượng biến thiên (một số nhanh, một số chậm). Bản demo đối chiếu:

- **Sequential baseline:** một worker xử lý tất cả tác vụ tuần tự.
- **Fixed assignment:** mỗi tác vụ được gán trước cho một worker cụ thể (kiểu supervisor).
- **Swarm:** các worker lấy công việc từ hàng đợi chung.

Swarm tự động cân bằng tải; gán cố định khiến các worker nhanh phải nhàn rỗi khi tác vụ được gán cho chúng bị chậm.

Chạy:

```
python3 code/main.py
```

Đầu ra hiển thị số lượng tác vụ trên mỗi worker (swarm phân phối không đều nhưng tối ưu) và thời gian thực tế (wall-clock time).

## Use It

`outputs/skill-swarm-fit.md` đánh giá xem một tác vụ nên sử dụng swarm hay supervisor. Các đầu vào: tính độc lập của tác vụ, sự biến thiên thời lượng, yêu cầu về thứ tự, nhu cầu gỡ lỗi.

## Ship It

Danh sách kiểm tra:

- **Hàng đợi ưu tiên với aging.** Ngăn chặn tình trạng đói tài nguyên của tác vụ dài.
- **Tính lũy đẳng (Idempotency) của worker.** Một tác vụ có thể bị lấy nhiều lần nếu worker gặp sự cố giữa chừng. Các worker phải có tính lũy đẳng.
- **Hàng đợi bền vững (Durable queue).** Sử dụng Kafka, Redis Streams hoặc hàng đợi dựa trên cơ sở dữ liệu cho môi trường production. `queue.Queue` chỉ nằm trong bộ nhớ.
- **Khả năng quan sát theo tác vụ.** Mỗi tác vụ có một trace ID; mỗi worker ghi nhật ký bắt đầu/kết thúc với ID đó.
- **Back-pressure.** Nếu hàng đợi tăng nhanh hơn tốc độ xử lý của worker, hãy làm chậm producer.

## Exercises

1. Chạy `code/main.py`. Swarm nhanh hơn bao nhiêu so với tuần tự trên khối lượng công việc có thời lượng biến thiên? Nhanh hơn bao nhiêu so với gán cố định?
2. Thêm biến thể hàng đợi ưu tiên (sử dụng `queue.PriorityQueue`). Gán ưu tiên theo trường "độ quan trọng" của tác vụ. Quan sát xem các tác vụ ưu tiên thấp có bao giờ bị đói tài nguyên dưới tải liên tục hay không.
3. Triển khai bộ phát hiện điểm nóng (hot-spot detector): ghi nhật ký khi bất kỳ worker nào xử lý số tác vụ gấp 3 lần worker chậm nhất. Điều đó cho thấy gì về phân phối thời lượng tác vụ?
4. Đọc phần tóm tắt và Mục 3 của bài báo Matrix (arXiv:2511.21686). Xác định một sự đánh đổi cụ thể mà Matrix chấp nhận (đạt được khả năng mở rộng) và một thứ mà nó từ bỏ (khả năng truy xuất, tính xác định).
5. Chuyển đổi bản demo swarm để sử dụng `queue.Queue` gồm các tuple (task_type, payload), với các worker chỉ đăng ký các loại cụ thể. Những quy tắc định tuyến nào hợp lý khi các tác vụ không đồng nhất?

## Key Terms

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|------------------------|
| Swarm architecture | "Agent phi tập trung" | Worker lấy công việc từ hàng đợi chung; không có bộ điều phối trung tâm. |
| Event bus | "Agent đăng ký chủ đề" | Message broker định tuyến tác vụ đến worker theo loại hoặc nội dung. |
| Starvation | "Tác vụ không bao giờ chạy" | Tác vụ ưu tiên thấp không bao giờ được chọn vì công việc ưu tiên cao liên tục đến. |
| Hot-spotting | "Một worker bị quá tải" | Mất cân bằng tải khiến một worker nhận hầu hết các tác vụ. |
| Back-pressure | "Làm chậm producer" | Cơ chế báo hiệu ngược lên trên để dừng sản xuất khi hàng đợi đầy. |
| Idempotent worker | "An toàn để chạy lại" | Tác vụ được xử lý hai lần vẫn cho ra kết quả như nhau. Cần thiết vì worker có thể crash giữa chừng. |
| Durable queue | "Chống crash" | Hàng đợi được lưu trữ trên đĩa hoặc lưu trữ sao chép; tác vụ không bị mất khi worker crash. |
| Matrix framework | "Swarm truyền thông điệp đầy đủ" | Cả luồng dữ liệu và điều khiển đều là các thông điệp tuần tự hóa trên hàng đợi phân tán. |

## Further Reading

- [LangGraph workflows and agents — Swarm Architecture](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — hỗ trợ swarm rõ ràng
- [Matrix — A Decentralized Framework for Multi-Agent Systems](https://arxiv.org/abs/2511.21686) — swarm truyền thông điệp đầy đủ
- [Anthropic engineering — why supervisor not swarm in Research](https://www.anthropic.com/engineering/multi-agent-research-system) — tại sao một hệ thống production cụ thể chọn supervisor thay vì swarm
- [AutoGen v0.4 actor-model docs](https://microsoft.github.io/autogen/stable/) — bản viết lại actor-model hướng sự kiện, gần với swarm hơn so với GroupChat của v0.2