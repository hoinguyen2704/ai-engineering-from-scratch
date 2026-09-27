# DualPipe Parallelism

> DeepSeek-V3 được huấn luyện trên 2.048 GPU H800 với các MoE expert nằm rải rác trên các node. Chi phí giao tiếp all-to-all giữa các node cho expert tốn 1 giờ GPU cho mỗi 1 giờ tính toán. Các GPU bị nhàn rỗi một nửa thời gian. DualPipe (DeepSeek, tháng 12 năm 2024) là một pipeline hai chiều giúp chồng lấp (overlap) quá trình tính toán forward và backward với các giao tiếp all-to-all mà chúng kích hoạt. Các bubble (khoảng trống) giảm đi, thông lượng tăng lên, và việc giữ hai bản sao tham số mô hình (từ "dual" tạo nên tên gọi) trở nên rẻ hơn khi Expert Parallelism đã phân tán các expert trên các rank. Bài học này là một hướng dẫn dạng Learn về những gì DualPipe thực sự làm và lý do tại sao cải tiến DualPipeV của Sea AI Lab loại bỏ chi phí gấp đôi tham số với cái giá là một bubble chặt hơn một chút.

**Type:** Learn
**Languages:** Python (stdlib, schedule simulator)
**Prerequisites:** Phase 10 · 05 (distributed training, FSDP, DeepSpeed), Phase 10 · 14 (open-model architectures and MoE)
**Time:** ~60 phút

## Mục tiêu học tập

- Nêu tên bốn thành phần của một chunk forward-backward trong DualPipe và lý do tại sao mỗi thành phần lại có cửa sổ overlap riêng.
- Giải thích vấn đề pipeline bubble ở quy mô lớn, và "không có bubble" (bubble-free) nghĩa là gì trong thực tế so với marketing.
- Truy vết lịch trình DualPipe bằng tay cho 8 rank PP và 16 micro-batch, đồng thời xác nhận các luồng forward và reverse lấp đầy các khe nhàn rỗi của nhau.
- Nêu sự đánh đổi mà DualPipeV (Sea AI Lab, 2025) thực hiện: loại bỏ việc sao chép tham số 2x với cái giá là bubble lớn hơn một chút khi Expert Parallelism không hoạt động.

## Vấn đề

Việc huấn luyện mô hình MoE 671B trên 2k GPU H800 gặp phải ba nút thắt cổ chai cộng hưởng:

1. **Áp lực bộ nhớ.** Mỗi GPU giữ một phần của mô hình. Bộ nhớ activation ở độ dài chuỗi 8k trên 61 layer và 128 head là rất lớn.
2. **Pipeline bubbles.** Pipeline parallelism truyền thống (GPipe, 1F1B) khiến các GPU nhàn rỗi trong khi chờ đợi đầu vào hoặc gradient của stage đó. Ở 8 stage, khoảng 12% thời gian GPU có thể là bubble ngay cả với lịch trình 1F1B.
3. **Cross-node all-to-all.** MoE với expert parallelism phân tán các expert trên các node. Mỗi lượt forward kích hoạt một all-to-all để điều phối token đến các expert của chúng, và một lượt khác để kết hợp. Ở quy mô 2k GPU, điều này dễ dàng trở thành tỷ lệ tính toán-trên-giao tiếp là 1:1.

Mỗi vấn đề này đều có các giải pháp riêng: gradient checkpointing cho bộ nhớ, Zero Bubble (Sea AI Lab, 2023) cho pipeline bubbles, các kernel giao tiếp expert-parallel cho all-to-all. Những gì DualPipe làm là kết hợp chúng lại với nhau. Lịch trình chồng lấp tính toán và giao tiếp trong một chunk forward-backward duy nhất, đưa các micro-batch từ cả hai đầu của pipeline vào cùng lúc, và sử dụng lịch trình thu được để ẩn các giao tiếp all-to-all bên trong các cửa sổ tính toán.

Kết quả được báo cáo: gần như loại bỏ hoàn toàn pipeline bubbles, đạt hiệu suất sử dụng GPU trên 95% trong quá trình huấn luyện 14,8T-token của DeepSeek-V3.

## Khái niệm

### Ôn tập về pipeline parallelism

Chia một mô hình N-layer trên P thiết bị. Thiết bị `i` giữ các layer `i * N/P .. (i+1) * N/P - 1`. Một micro-batch chảy forward qua các thiết bị từ 0 đến P-1, sau đó backward từ P-1 về 0. Mỗi thiết bị chỉ có thể bắt đầu stage forward khi thiết bị trước đó gửi đầu ra và chỉ có thể bắt đầu backward khi thiết bị phía sau gửi gradient ngược lên.

GPipe (Huang và cộng sự, 2019) lập lịch từng micro-batch một, gây lãng phí phần lớn thời gian GPU. 1F1B (Narayanan và cộng sự, 2021) xen kẽ các lượt forward và backward cho nhiều micro-batch. Zero Bubble (Qi và cộng sự, 2023) chia lượt backward thành hai phần — backward-cho-input (B) và backward-cho-weights (W) — và lập lịch chúng để lấp đầy bubble. Sau Zero Bubble, pipeline gần như đã khít.

DualPipe là bước tiếp theo. Nó bổ sung hai ý tưởng:

### Ý tưởng 1: phân rã chunk

Mỗi chunk forward được chia thành bốn thành phần:

- **Attention.** Các phép chiếu Q/K/V, attention, phép chiếu đầu ra.
- **All-to-all dispatch.** Giao tiếp giữa các node để gửi token đến các expert của chúng.
- **MLP.** Tính toán của MoE expert.
- **All-to-all combine.** Giao tiếp giữa các node để mang đầu ra của expert trở lại.

Một chunk backward bổ sung các phiên bản gradient của từng thành phần này. DualPipe lập lịch chúng sao cho all-to-all dispatch diễn ra song song với tính toán attention của chunk tiếp theo, và all-to-all combine diễn ra song song với tính toán MLP của chunk sau đó.

### Ý tưởng 2: lập lịch hai chiều

Hầu hết các lịch trình pipeline đưa micro-batch vào từ stage 0 và chảy về phía stage P-1. DualPipe đưa micro-batch vào từ CẢ HAI đầu. Stage 0 thấy các micro-batch forward bắt nguồn từ đó; stage P-1 cũng thấy các micro-batch forward bắt nguồn từ đó. Hai luồng gặp nhau ở giữa.

Để làm được điều này, thiết bị `i` phải giữ CẢ layer đầu pipeline `i` VÀ layer cuối pipeline `P - 1 - i`. Đó là phần "dual" của DualPipe: mỗi thiết bị giữ hai bản sao của các layer mô hình mà nó cần phục vụ (một cho mỗi hướng). Ở quy mô của DeepSeek-V3, đây là chi phí sao chép tham số 2x. Nó có thể chấp nhận được vì Expert Parallelism đã phân tán các MoE expert mỏng đến mức việc sao chép các layer không phải expert hai lần là chuyện nhỏ.

Quan trọng là, luồng forward theo một hướng và luồng backward theo hướng kia chồng lấp chính xác tại nơi mà các bubble sẽ xuất hiện trong lịch trình một chiều. Các bubble biến mất.

### Lịch trình truy vết bằng tay

Xét P = 4 rank, 8 micro-batch, chia thành 4 forward / 4 reverse. Thời gian di chuyển từ trái sang phải; các hàng là các rank thiết bị.

```
           Time →
rank 0:  F1 F2 F3 F4  F5R F6R F7R F8R  B1 B2 B3 B4  ...
rank 1:     F1 F2 F3  F4/F5R F6R F7R   B1 B2 ...
rank 2:        F1 F2  F3/F5R F4/F6R    B1 ...
rank 3:           F1  F2/F5R F3/F6R    ...
```

Đọc ký hiệu "F4/F5R": rank 1 đang chạy forward của micro-batch 4 (đi từ trái sang phải trong pipeline) VÀ forward của micro-batch 5 (đi từ phải sang trái) trong cùng một khe thời gian. Đó là ý nghĩa của "hai chiều" trong vận hành.

Tại rank 2, các luồng chéo chồng lấp sớm hơn, tại rank 0 và P-1 chúng chồng lấp muộn nhất. Trong giai đoạn ổn định của lịch trình, mỗi rank chạy forward-của-hướng-X chồng lấp với backward-của-hướng-Y. Tính toán luôn bận rộn. Các all-to-all dispatch cho lượt forward ẩn bên trong tính toán backward. Các all-to-all combine ẩn bên trong tính toán forward. Các bubble bị ép ra ngoài.

### Tính toán bubble

Pipeline bubble tiêu chuẩn của 1F1B (thời gian lãng phí trên mỗi rank):

```
bubble_1F1B = (P - 1) * forward_chunk_time
```

Cải tiến Zero Bubble giảm nó xuống nhưng không về 0. DualPipe, trong giai đoạn ổn định, có bubble bằng 0 nếu số lượng micro-batch chia hết cho 2 lần độ sâu pipeline. Ngoài giai đoạn ổn định (warmup và cooldown), vẫn có một số bubble nhưng chúng không tăng theo số lượng micro-batch — một đặc tính quan trọng mà bài báo nhấn mạnh.

Theo thuật ngữ marketing: "không có bubble". Theo thuật ngữ kỹ thuật: các bubble không tăng theo số lượng micro-batch. Phân tích tiếp theo của Sea AI Lab (DualPipeV / Cut-in-half) cho thấy bubble bằng 0 hoàn toàn chỉ khi Expert Parallelism không phải là nút thắt; với all-to-all do EP điều khiển, luôn có sự thỏa hiệp về lập lịch.

### DualPipeV — sự tinh chỉnh

Sea AI Lab (2025) nhận thấy rằng việc sao chép tham số 2x là lãng phí khi overlap giao tiếp EP không phải là mục tiêu chính. Lịch trình DualPipeV của họ gập việc đưa micro-batch hai chiều thành lịch trình "hình chữ V" chạy trên một bản sao tham số duy nhất. Bubble lớn hơn một chút so với DualPipe, nhưng tiết kiệm bộ nhớ đáng kể. DeepSeek đã áp dụng DualPipeV trong triển khai DualPipe mã nguồn mở của họ như một chế độ tắt EP.

Sự đánh đổi:

| Tính năng | DualPipe | DualPipeV | 1F1B | Zero Bubble |
|---------|---------|-----------|------|------------|
| Bản sao tham số mỗi thiết bị | 2 | 1 | 1 | 1 |
| Bubble so với micro-batch | hằng số | tăng nhẹ | tăng | tăng |
| Overlap tính toán-giao tiếp | đầy đủ | một phần | tối thiểu | một phần |
| Sử dụng khi | MoE nặng EP | dày đặc hoặc nhẹ EP | cơ sở | bất kỳ pipeline nào |

### Ý nghĩa đối với lượt chạy 14,8T-token

Quá trình tiền huấn luyện của DeepSeek-V3 tiêu thụ 14,8T token trên 2.048 GPU H800 trong khoảng 2,8 triệu giờ GPU. Với 1F1B ngây thơ, họ sẽ mất 12-15% thời gian đó cho pipeline bubbles — khoảng 340-420K giờ GPU, đủ để huấn luyện một mô hình 70B hoàn chỉnh. DualPipe đã khôi phục phần lớn số đó. Việc định lượng trực tiếp đóng góp là khó khăn nếu không có nhật ký nội bộ, nhưng tuyên bố trong bài báo là hiệu suất sử dụng GPU trung bình trên 95% trong suốt quá trình huấn luyện.

Đối với các lượt chạy nhỏ hơn (dưới 1k GPU), DualPipe là quá mức cần thiết — pipeline bubbles nhỏ hơn so với tổng chi phí, và việc huấn luyện mô hình dày đặc hiếm khi gặp nút thắt all-to-all. Đối với huấn luyện MoE tiên phong ở quy mô hàng nghìn GPU, nó thực sự là bắt buộc.

### Vị trí trong stack

- Bổ sung cho **FSDP** (Phase 10 · 05). FSDP chia nhỏ các tham số mô hình trên các rank; DualPipe lập lịch tính toán trên các rank. Chúng kết hợp với nhau.
- Tương thích với **ZeRO-3** gradient sharding. Việc quản lý cho sao chép hai bản sao cần phối hợp với các gradient đã chia nhỏ của ZeRO.
- Yêu cầu **các kernel all-to-all tùy chỉnh** được tinh chỉnh cho cấu trúc liên kết cụm cụ thể. Các kernel mã nguồn mở của DeepSeek là triển khai tham chiếu.

```figure
expert-capacity
```

## Sử dụng

`code/main.py` là một trình mô phỏng lịch trình pipeline. Nó lấy `(P, n_micro_batches, schedule)` và in ra hiệu suất sử dụng giai đoạn ổn định cho từng loại 1F1B, Zero Bubble, DualPipe và DualPipeV. Đây là một công cụ giảng dạy — các con số khớp với các tuyên bố định tính trong các bài báo, chúng không phải là tuyên bố về tốc độ tăng tốc đo được trong sản xuất.

Giá trị của trình mô phỏng: chạy nó với các số lượng P và micro-batch khác nhau và quan sát cách tỷ lệ bubble tăng lên đối với 1F1B nhưng không phải đối với DualPipe.

Các cân nhắc tích hợp cho một lượt huấn luyện thực tế:

- Chọn độ sâu pipeline-parallel chia hết cho số lượng micro-batch của bạn.
- Đảm bảo mesh expert-parallel của bạn hỗ trợ all-to-all hai chiều. Các kernel của DeepSeek là tham chiếu.
- Hãy chuẩn bị tinh thần dành một tuần để gỡ lỗi lịch trình trong lần đầu tiên. Việc quản lý rất phức tạp.
- Theo dõi hiệu suất sử dụng GPU trên mỗi rank, không chỉ tổng hợp. Lợi ích của DualPipe đến từ việc thắt chặt các rank chậm chạp.

## Triển khai

Bài học này tạo ra `outputs/skill-dualpipe-planner.md`. Với thông số kỹ thuật của cụm huấn luyện (số lượng GPU, cấu trúc liên kết, kết nối, hình dạng mô hình), nó đề xuất chiến lược pipeline parallelism, thuật toán lập lịch để sử dụng và tỷ lệ bubble dự kiến ở quy mô mục tiêu.

## Bài tập

1. Chạy `code/main.py` trên `(P=8, micro_batches=16, schedule=dualpipe)` và `(P=8, micro_batches=16, schedule=1f1b)`. Tính toán sự khác biệt về hiệu suất sử dụng GPU và biểu thị nó dưới dạng giờ GPU được khôi phục trên mỗi triệu token huấn luyện.

2. Phác thảo bảng lịch trình cho `(P=4, micro_batches=8, schedule=dualpipe)` bằng tay. Đánh dấu mỗi khe thời gian với ID micro-batch và hướng. Xác định khe thời gian đầu tiên mà các bubble không xuất hiện.

3. Đọc Hình 5 của báo cáo kỹ thuật DeepSeek-V3 (arXiv:2412.19437). Xác định cửa sổ overlap cho all-to-all dispatch bên trong một chunk forward của DualPipe. Giải thích cách lịch trình tính toán ẩn nó đi.

4. Tính toán chi phí tham số 2x của DualPipe cho một mô hình dày đặc 70B với P=8 stage pipeline và một mô hình MoE 671B với P=16 stage pipeline. Chỉ ra lý do tại sao chi phí của trường hợp MoE lại nhỏ hơn theo tỷ lệ (hầu hết các tham số là expert, được chia nhỏ trên một nhóm EP lớn).

5. So sánh DualPipe với Chimera (một bộ lập lịch hai chiều cạnh tranh từ năm 2021). Xác định hai đặc tính cụ thể mà DualPipe đã thêm vào mà Chimera không có, sử dụng Mục 3.4 của bài báo làm tham chiếu.

## Thuật ngữ chính

| Thuật ngữ | Mọi người nói gì | Ý nghĩa thực sự |
|------|----------------|------------------------|
| Pipeline bubble | "Thời gian nhàn rỗi mỗi rank" | Các chu kỳ GPU bị lãng phí vì một stage pipeline đang chờ đầu vào hoặc gradient của nó |
| 1F1B | "Lịch trình pipeline mặc định" | Lập lịch xen kẽ một forward / một backward; cơ sở mà DualPipe đánh bại |
| Zero Bubble | "Sea AI Lab 2023" | Chia backward thành B (gradient đầu vào) và W (gradient trọng số); thắt chặt pipeline gần như hoàn toàn |
| DualPipe | "Lịch trình DeepSeek-V3" | Pipeline hai chiều + overlap tính toán-giao tiếp; các bubble không tăng theo số lượng micro-batch |
| DualPipeV | "Cắt làm đôi" | Tinh chỉnh hình chữ V giúp loại bỏ việc sao chép tham số 2x với cái giá là các bubble lớn hơn một chút |
| Chunk | "Đơn vị công việc pipeline" | Một lượt forward hoặc backward của một micro-batch qua một stage pipeline |
| All-to-all dispatch | "Gửi token đến expert" | Giao tiếp giữa các node định tuyến token đến các MoE expert được chỉ định của chúng |
| All-to-all combine | "Mang đầu ra expert trở lại" | Giao tiếp giữa các node thu thập đầu ra của expert sau MLP |
| Expert Parallelism (EP) | "Expert trên các GPU" | Chia nhỏ các MoE expert trên các rank để các GPU khác nhau giữ các expert khác nhau |
| Pipeline Parallelism (PP) | "Layer trên các GPU" | Chia nhỏ các layer mô hình trên các rank; chiều mà DualPipe lập lịch |
| Tỷ lệ bubble | "Thời gian GPU lãng phí" | (bubble_time / total_time); tỷ lệ mà DualPipe hướng tới bằng 0 |

## Đọc thêm

- [DeepSeek-AI — Báo cáo kỹ thuật DeepSeek-V3 (arXiv:2412.19437), Mục 3.3.2 và Hình 5](https://arxiv.org/abs/2412.19437) — tài liệu tham khảo chính về DualPipe
- [DeepSeek — Kho lưu trữ GitHub DualPipe](https://github.com/deepseek-ai/DualPipe) — triển khai tham chiếu mã nguồn mở, bao gồm chế độ DualPipeV (Cut-in-half)
- [Qi và cộng sự — Zero Bubble Pipeline Parallelism (arXiv:2401.10241, Sea AI Lab 2023)](https://arxiv.org/abs/2401.10241) — tiền thân của Zero Bubble
- [Sea AI Lab — DualPipe có thể tốt hơn nếu không có Dual](https://sail.sea.com/blog/articles/63) — phân tích DualPipeV đã thông báo cho chế độ tắt EP của DeepSeek
- [Narayanan và cộng sự — PipeDream / 1F1B (arXiv:1806.03377, 2018-2021)](https://arxiv.org/abs/1806.03377) — lịch trình 1F1B mà DualPipe so sánh cùng
- [Huang và cộng sự — GPipe (arXiv:1811.06965, 2018)](https://arxiv.org/abs/1811.06965) — bài báo gốc về pipeline parallelism và vấn đề bubble