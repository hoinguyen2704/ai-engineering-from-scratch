# Pipeline Parallel và Phân tích Bubble

> Tensor parallelism chia nhỏ phép nhân ma trận trên các rank. Pipeline parallelism chia nhỏ mô hình trên các rank, mỗi rank đảm nhận một stage. Các microbatch chảy qua pipeline. Khoảng thời gian trống ở đầu và cuối được gọi là bubble; việc tối thiểu hóa nó chính là kỹ năng cốt lõi.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 19 Track C lessons 42-49
**Time:** ~90 min

## Mục tiêu học tập

- Chia một mô hình tuần tự thành N stage và mô phỏng một pipeline forward trên N rank.
- Lên lịch M microbatch qua pipeline sử dụng lịch trình GPipe (fill forward, sau đó mới backward) và tính toán tỷ lệ bubble.
- So sánh bubble với lịch trình 1F1B xen kẽ được sử dụng trong Megatron-LM và PipeDream.
- Bảo vệ việc phân bổ stage: tính toán cân bằng trên mỗi stage quan trọng hơn số lượng tham số cân bằng trên mỗi stage.

## Vấn đề

Một mô hình 70B-parameter ở định dạng fp16 cần tới 140 GB chỉ riêng cho các tham số. Không GPU tiêu dùng nào chứa nổi. ZeRO-3 phân mảnh tham số trên các rank nhưng vẫn yêu cầu mỗi rank phải allgather toàn bộ layer cho mỗi bước forward, tốn log(N) hop mỗi layer. Pipeline parallel đi theo một hướng khác: cắt mô hình thành N stage và đặt mỗi stage trên một rank. Forward của layer 1 hoàn tất trên rank 0 và chuyển tensor activation sang rank 1; rank 1 chạy layer 2 và chuyển sang rank 2; và cứ tiếp tục như vậy. Backward chảy theo chiều ngược lại. Bộ nhớ giảm tuyến tính vì mỗi rank chỉ giữ một stage; tính toán mang tính tuần tự, đó chính là vấn đề bubble.

Bubble là thời gian nhàn rỗi ở đầu pipeline (chờ microbatch đầu tiên đến stage cuối) và ở cuối (chờ microbatch cuối cùng thoát ra). Với M microbatch và N stage, tỷ lệ bubble trên mỗi stage là (N-1)/(M+N-1). Tại M=8, N=4, tỷ lệ này là 27%. Tại M=64, N=4, tỷ lệ là 4.5%. Bubble thu hẹp khi bạn có nhiều microbatch trên mỗi bước, đồng nghĩa với kích thước batch trên mỗi microbatch nhỏ, đây chính là ràng buộc thúc đẩy thiết kế microbatch.

## Khái niệm

```mermaid
flowchart LR
  R0[rank 0: stage 0 / layer 0] --> R1[rank 1: stage 1 / layer 1]
  R1 --> R2[rank 2: stage 2 / layer 2]
  R2 --> R3[rank 3: stage 3 / loss]
  R3 -.backward.-> R2
  R2 -.backward.-> R1
  R1 -.backward.-> R0
```

### Lịch trình GPipe

Lấp đầy pipeline forward với tất cả M microbatch trước khi bắt đầu bất kỳ backward nào; sau đó xả ngược lại. Activation từ mọi microbatch phải được giữ lại cho đến khi thực hiện backward, vì vậy bộ nhớ tăng tuyến tính theo M. Forward mất M+N-1 chu kỳ, backward mất thêm M+N-1 chu kỳ. Công việc hữu ích trên mỗi stage là 2M chu kỳ; bubble trên mỗi stage là 2(N-1) chu kỳ. Tỷ lệ bubble là (N-1)/(M+N-1) khi mỗi forward và backward mất một đơn vị thời gian. Chọn M lớn hơn nhiều so với N sẽ giúp ẩn đi bubble.

### Lịch trình 1F1B

Xen kẽ: ngay khi forward của một microbatch đến stage cuối, hãy bắt đầu backward của nó và để nó truyền ngược lại. Lịch trình xen kẽ một forward và một backward trên mỗi stage. Bubble vẫn là N-1 nhưng bộ nhớ activation bị giới hạn bởi độ sâu của pipeline, không phải số lượng microbatch. Các pipeline sản xuất sử dụng 1F1B (Megatron, PipeDream). Bài học này triển khai GPipe trước vì nó đơn giản hơn, và 1F1B như một bài tập.

### Tại sao tính toán cân bằng trên mỗi stage lại quan trọng

Nếu stage 0 mất 50 ms và stage 1 mất 100 ms, mọi chu kỳ đều bị chặn bởi stage 1. Các stage khác nhàn rỗi 50 ms mỗi chu kỳ để chờ stage 1 giải phóng. Số lượng tham số bằng nhau là trục sai lầm: tính toán của một transformer bị chi phối bởi attention cộng với MLP trên mỗi layer, và các layer embedding có nhiều tham số nhưng ít tính toán. Việc phân bổ stage nên cân bằng FLOPs trên mỗi stage, không phải trọng số trên mỗi stage.

### Microbatch so với batch

Một pipeline chạy M microbatch với kích thước B mỗi cái. Kích thước batch hiệu dụng là M*B. Gradient ở cuối một bước pipeline là gradient trên tổng M*B ví dụ. Tỷ lệ bubble phụ thuộc vào M; bộ tối ưu hóa (optimizer) nhìn thấy M*B. Điều chỉnh M nghĩa là đánh đổi giữa bubble (thấp hơn với M cao) và bộ nhớ trên mỗi microbatch (bộ nhớ activation cao hơn với M cao đối với GPipe).

```figure
cd-pipeline-bubble
```

## Xây dựng

`code/main.py` triển khai:

- `PipelineStage`: một `nn.Module` nhỏ giữ các tham số của một stage và hiển thị `forward(activation)`.
- `Pipeline(stages, num_microbatches)`: điều phối lịch trình GPipe trên các stage mô phỏng bằng cách sử dụng thời gian thực mô phỏng trên mỗi stage.
- `bubble_fraction(num_stages, num_microbatches)`: công thức đóng (N-1)/(M+N-1).
- Một bản demo 4 stage in ra dấu vết (trace) trên mỗi microbatch và tỷ lệ bubble đo được.

Chạy nó:

```bash
python3 code/main.py
```

Đầu ra: biểu đồ Gantt theo stage-microbatch và tỷ lệ bubble so với dự đoán từ công thức đóng.

## Các mô hình sản xuất thực tế

Ba mô hình giúp pipeline parallel đủ ổn định để triển khai.

**Activation checkpointing kết hợp với pipeline.** Với M microbatch đang chạy trên GPipe, bộ nhớ activation bằng M lần một microbatch. Activation checkpointing tính toán lại forward tại thời điểm backward, đánh đổi tính toán lấy bộ nhớ; sự kết hợp này làm cho pipeline trở nên khả thi đối với các chuỗi dài.

**Sự cân bằng stage được đo lường, không phải giả định.** Các đội ngũ sản xuất chạy một lượt profiling để đo lường tính toán thực tế trên mỗi layer (FLOPs và thời gian thực) trên phần cứng mục tiêu, sau đó phân vùng theo phép đo đó. Cờ `--num-layers-per-stage` của Megatron-LM chấp nhận một danh sách để cho phép số lượng layer không đều khi các stage có chi phí trên mỗi layer khác nhau.

**Lịch trình send-recv phải tránh deadlock.** Một pipeline mà mọi stage đều gửi trước khi nhận sẽ bị deadlock trên đường truyền. Cách sửa lỗi tiêu chuẩn là xen kẽ: các stage rank chẵn gửi trước rồi nhận, các stage rank lẻ nhận trước rồi gửi. Bài học này lập lịch các rank một cách rõ ràng để mô hình này có thể quan sát được.

## Sử dụng

Các mô hình sản xuất:

- **Megatron-LM.** Tài liệu tham khảo cho pipeline parallel ở quy mô lớn. Sử dụng 1F1B và hỗ trợ kết hợp tensor + pipeline + data parallel.
- **DeepSpeed Pipeline.** Tích hợp với ZeRO; ZeRO-1 + pipeline là sự kết hợp phổ biến cho các mô hình mở lớn nhất.
- **PyTorch Pipe.** Trình bao bọc pipeline gốc của PyTorch, được xây dựng trên `torch.distributed.pipeline.sync.Pipe`.

## Triển khai

Bài học 80 lưu trữ các mảnh tham số trên mỗi stage vào checkpoint đã phân mảnh. Bài học 81 kết hợp DDP + ZeRO + pipeline trên bản demo end-to-end (về mặt tinh thần; bản demo giữ pipeline mô phỏng cho thời gian chạy).

## Bài tập

1. Triển khai 1F1B và xác minh tỷ lệ bubble khớp với GPipe nhưng bộ nhớ activation bị giới hạn.
2. Profile thời gian thực trên mỗi stage của một mô hình sâu hơn và cân bằng lại các stage theo thời gian thực đo được.
3. Thêm tích lũy gradient (gradient accumulation) qua các microbatch của pipeline và kiểm tra xem gradient có bằng với gradient của forward full-batch tương đương hay không.
4. Kết hợp pipeline với activation checkpointing và đo lường sự sụt giảm bộ nhớ so với chi phí tính toán.
5. Kết hợp pipeline với DDP (mỗi pipeline rank được sao chép trên một nhóm data-parallel) và suy luận qua lịch trình 2D.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|------------------------|
| Pipeline | "Model parallel theo chiều sâu" | Một stage trên mỗi rank, activation chảy từ stage này sang stage khác |
| Bubble | "Thời gian nhàn rỗi của pipeline" | (N-1) bước ở đầu + cuối nơi một số stage không có việc làm |
| Microbatch | "Lát cắt của batch" | Một đơn vị forward/backward; bubble thu hẹp khi M tăng |
| GPipe | "Lấp đầy rồi xả" | Tất cả M forward trước khi có bất kỳ backward nào; bộ nhớ activation cao |
| 1F1B | "Lịch trình xen kẽ" | Một forward một backward trên mỗi stage; bộ nhớ activation bị giới hạn |

## Đọc thêm

- [Huang et al, GPipe: Efficient Training of Giant Neural Networks](https://arxiv.org/abs/1811.06965)
- [Narayanan et al, PipeDream: Generalized Pipeline Parallelism for DNN Training](https://arxiv.org/abs/1806.03377)
- [Tài liệu pipeline parallel của Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
- Phase 19 Lesson 76 - các primitive send/recv mà lịch trình sử dụng
- Phase 19 Lesson 78 - ZeRO trực giao với pipeline và thường được kết hợp cùng nhau