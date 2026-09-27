# Serving Engine Internals — PagedAttention, Continuous Batching, Chunked Prefill

> Thông lượng của các engine phục vụ (serving-engine) hiện đại dựa trên ba mặc định kết hợp, chứ không phải một thủ thuật đơn lẻ. PagedAttention luôn được bật. Continuous batching chèn các yêu cầu mới vào batch đang hoạt động giữa các vòng lặp decode. Chunked prefill cắt nhỏ các prompt dài để các token decode không bị "đói" tài nguyên. Khi bật cả ba, một model Llama 3.3 70B FP8 trên một card H100 SXM5 đạt 2.200-2.400 tok/s ở mức 128 yêu cầu đồng thời — cao hơn khoảng 25% so với mặc định của vLLM và gấp 3-4 lần so với vòng lặp PyTorch thông thường. Bài học này sẽ đi sâu vào scheduler và attention kernel của vLLM — engine tham chiếu cho cả ba kỹ thuật này — ở mức độ bạn có thể vẽ sơ đồ, và kết thúc với một bộ continuous batcher mô phỏng trong `code/main.py`, thực hiện lập lịch prefill và decode theo cách vLLM vận hành.

**Type:** Learn
**Languages:** Python (stdlib, toy continuous batching scheduler)
**Prerequisites:** Phase 17 · 01 (Model Serving), Phase 11 (LLM Engineering)
**Time:** ~75 minutes

## Mục tiêu học tập

- Giải thích PagedAttention như một bộ cấp phát KV cache: các block, bảng block (block table), và lý do tại sao phân mảnh bộ nhớ duy trì dưới 4% ở tải sản xuất.
- Vẽ sơ đồ continuous batching ở cấp độ vòng lặp: cách các chuỗi (sequence) hoàn thành rời khỏi batch và các chuỗi mới tham gia mà không làm gián đoạn hệ thống.
- Mô tả chunked prefill trong một câu và nêu tên chỉ số độ trễ mà nó bảo vệ (gợi ý: đó là TTFT tail, không phải thông lượng trung bình).
- Nêu tên vấn đề (gotcha) trong vLLM v0.18.0 năm 2026 mà các đội ngũ thường gặp phải khi kích hoạt tất cả các tối ưu hóa cùng lúc.

## Vấn đề

Một vòng lặp serve PyTorch thông thường chạy từng yêu cầu một: tokenize, prefill, decode cho đến khi gặp EOS, rồi trả về. Với một người dùng, cách này ổn. Với một trăm người dùng, nó trở thành một hàng đợi của những người kiên nhẫn. Giải pháp hiển nhiên — static batching — đệm (pad) mọi yêu cầu theo prompt dài nhất trong cửa sổ, đệm mọi decode theo đầu ra dự kiến dài nhất, và làm đình trệ toàn bộ batch vì chuỗi chậm nhất. Bạn phải trả phí cho phần đệm không bao giờ dùng tới, và các yêu cầu nhanh phải chờ đợi các yêu cầu chậm.

vLLM giải quyết ba vấn đề cùng lúc. PagedAttention ngăn chặn việc phân mảnh KV cache chiếm dụng 60-80% bộ nhớ GPU như cách cấp phát liên tục (contiguous) cổ điển. Continuous batching cho phép các yêu cầu tham gia và rời khỏi batch giữa mỗi vòng lặp decode, giúp batch luôn đầy ắp công việc thực tế. Chunked prefill chia một prompt 32k-token thành các lát cắt ~512-token xen kẽ với decode, để một prompt dài không làm đóng băng mọi token decode trên GPU.

Mặc định sản xuất năm 2026 là bật cả ba. Bạn cần hiểu mỗi tính năng làm gì vì các chế độ lỗi (failure modes) đều nằm ở scheduler, không phải ở model.

## Khái niệm

### PagedAttention như một hệ thống bộ nhớ ảo

KV cache là `num_layers × 2 × num_heads × head_dim × seq_len × bytes_per_element` cho mỗi chuỗi. Với Llama 3.3 70B ở 8192 token, con số này xấp xỉ 1.25 GB mỗi chuỗi ở định dạng BF16. Nếu bạn đặt trước 8192 slot cho mỗi yêu cầu nhưng yêu cầu trung bình chỉ dùng 1500 token, bạn lãng phí khoảng 82% HBM đã đặt trước. Batching cổ điển phải trả giá cho sự lãng phí này.

PagedAttention mượn ý tưởng từ bộ nhớ ảo của hệ điều hành. KV cache không liên tục theo từng chuỗi. Nó được cấp phát theo các block có kích thước cố định (mặc định là 16 token). Mỗi chuỗi có một bảng block ánh xạ các vị trí token logic của nó tới các ID block vật lý. Khi một chuỗi phát triển vượt quá các block đã cấp, một block nữa sẽ được thêm vào. Khi nó kết thúc, các block của nó được trả về pool.

Phân mảnh giảm từ 60-80% (cổ điển) xuống dưới 4% (PagedAttention). Bạn không bật PagedAttention bằng một flag — đó là bộ cấp phát duy nhất mà vLLM cung cấp. Tham số điều chỉnh là `--gpu-memory-utilization` (mặc định 0.9), cho vLLM biết bao nhiêu HBM cần dành riêng cho các KV block sau khi đã tải trọng số và các activation.

### Continuous batching ở cấp độ vòng lặp

"Dynamic batching" cũ chờ đợi một cửa sổ (ví dụ 10 ms) để lấp đầy batch, sau đó chạy prefill + decode + decode + decode cho đến khi mọi chuỗi kết thúc. Các chuỗi nhanh rời đi sớm và ngồi không trong khi GPU hoàn thành các chuỗi chậm.

Continuous batching hoạt động giữa mỗi bước decode. Gọi tập hợp các chuỗi đang chạy là danh sách `RUNNING`. Tại mỗi vòng lặp:

1. Bất kỳ chuỗi nào trong `RUNNING` vừa đạt EOS hoặc max_tokens sẽ bị loại bỏ.
2. Scheduler xem xét hàng đợi chờ. Nếu có KV block trống, nó nhận các chuỗi mới (prefill hoặc tiếp tục).
3. Forward pass chạy trên bất cứ thứ gì hiện có trong `RUNNING`, phát ra một token mới cho mỗi chuỗi.

Kích thước batch không bao giờ bị đệm đến một con số cố định. Các chuỗi ở các vị trí khác nhau trong đầu ra của chúng chia sẻ một forward pass hợp nhất. Trong vLLM 2026, đây được gọi là `V1 scheduler`. Bất biến chính: scheduler chạy một lần mỗi vòng lặp decode, không phải một lần mỗi yêu cầu.

### Chunked prefill bảo vệ TTFT tail

Prefill bị giới hạn bởi tính toán (compute-bound). Một prompt 32k-token trên Llama 3.3 70B mất ~800 ms prefill thuần túy trên một H100. Trong khi prefill chạy, các token decode cho mọi chuỗi khác trong batch phải chờ. Trong vòng lặp phục vụ, độ trễ token đầu tiên (TTFT) của một prompt dài trở thành độ trễ giữa các token (ITL) cho hàng chục người dùng khác.

Chunked prefill chia prefill thành các chunk kích thước cố định (mặc định 512 token) và lập lịch mỗi chunk như một đơn vị. Giữa các chunk, scheduler có thể đẩy các chuỗi decode tiến thêm một token. Bạn đánh đổi một chút độ trễ prefill tuyệt đối (vài ms mỗi chunk) để có độ rung (jitter) thời gian decode thấp hơn nhiều. P99 ITL dưới tải hỗn hợp giảm từ ~50 ms xuống ~15 ms trong các benchmark công bố.

### Sự tương tác của ba mặc định

Cả ba tính năng đều giả định sự tồn tại của nhau. PagedAttention cung cấp cho scheduler một tài nguyên KV chi tiết để tối ưu hóa. Continuous batching cần tài nguyên chi tiết đó để việc nhận một chuỗi mới không buộc phải sắp xếp lại toàn bộ hệ thống. Chunked prefill là một quyết định mà scheduler đưa ra trên cùng danh sách `RUNNING` — đó là một chính sách scheduler khác, không phải một hệ thống riêng biệt.

Bạn không cần biết mọi flag. Bạn cần biết scheduler tối ưu hóa cái gì: goodput dưới ngân sách KV-block, tuân theo việc cắt lát chunked prefill.

### Vấn đề v0.18.0 năm 2026

Trong vLLM v0.18.0, bạn không thể kết hợp `--enable-chunked-prefill` với draft-model speculative decoding (`--speculative-model`). Ngoại lệ được ghi nhận là N-gram GPU speculative decoding trong scheduler V1. Các đội ngũ bật mọi flag mà không đọc ghi chú phát hành sẽ gặp lỗi runtime khi khởi động, thay vì một sự suy giảm hiệu năng nhẹ. Nếu lợi ích từ speculative decoding đáng để bạn bật chunked prefill, hãy xem xét lại lựa chọn — câu trả lời đúng vào năm 2026 thường là EAGLE-3 không có chunked prefill, thay vì một draft model cộng với chunked prefill không thể biên dịch.

### Các con số bạn nên nhớ

- Llama 3.3 70B FP8, H100 SXM5, 128 đồng thời, bật cả ba: 2.200-2.400 tok/s.
- Cùng model, vLLM mặc định (không chunked prefill): ~1.800 tok/s.
- Cùng model, vòng lặp forward PyTorch thông thường: ~600 tok/s.
- Lãng phí phân mảnh KV cache với PagedAttention ở tải sản xuất: <4%.
- P99 ITL dưới tải hỗn hợp: ~15 ms với chunked prefill, ~50 ms không có.

### Scheduler trông như thế nào

```
while True:
    finished = [s for s in RUNNING if s.is_done()]
    for s in finished: release_blocks(s); RUNNING.remove(s)

    while WAITING and have_free_blocks_for(WAITING[0]):
        s = WAITING.pop(0)
        allocate_initial_blocks(s)
        RUNNING.append(s)

    # schedule prefill chunks + decode in one batch
    batch = []
    for s in RUNNING:
        if s.in_prefill:
            batch.append(next_prefill_chunk(s))   # e.g. 512 tokens
        else:
            batch.append(decode_one_token(s))     # 1 token

    run_forward(batch)                            # one fused GPU call
```

`code/main.py` chính xác là vòng lặp này trong Python stdlib với số lượng token giả lập và độ trễ forward giả lập. Chạy nó cho thấy cách chunked prefill giữ cho các chuỗi decode hoạt động trong suốt quá trình prefill dài.

```figure
tensor-parallel
```

## Sử dụng

`code/main.py` mô phỏng một scheduler kiểu vLLM với các tính năng có thể bật/tắt. Chạy nó để thấy:

- Chế độ `NAIVE`: từng yêu cầu một, không batching.
- Chế độ `STATIC`: đệm và chờ, batching cổ điển.
- Chế độ `CONTINUOUS`: nhận và giải phóng ở cấp độ vòng lặp.
- Chế độ `CONTINUOUS + CHUNKED`: các lát cắt prefill xen kẽ với decode.

Đầu ra hiển thị tổng thông lượng (token mỗi giây ảo), TTFT trung bình, và P99 ITL. Hàng `CONTINUOUS + CHUNKED` sẽ chiếm ưu thế trên lưu lượng hỗn hợp.

## Triển khai

Bài học này tạo ra `outputs/skill-vllm-scheduler-reader.md`. Với một cấu hình phục vụ (kích thước batch, sử dụng bộ nhớ KV, kích thước chunked prefill, cấu hình speculative), nó tạo ra một chẩn đoán scheduler chỉ ra đâu là mặc định trong ba mặc định đang gây nghẽn và cần điều chỉnh gì.

## Bài tập

1. Chạy `code/main.py`. So sánh `STATIC` với `CONTINUOUS` trên khối lượng công việc có các yêu cầu ngắn và dài hỗn hợp. Khoảng cách thông lượng đến từ đâu — hiệu suất prefill, hiệu suất decode, hay độ trễ đuôi (tail latency)?
2. Sửa đổi scheduler mô phỏng để thêm `--max-num-batched-tokens`. Giá trị phù hợp cho một H100 chạy Llama 3.3 70B FP8 là bao nhiêu? (Gợi ý: nó là hàm của kích thước KV block và số lượng block trống, không phải HBM thô.)
3. Đọc lại ghi chú phát hành vLLM v0.18.0. Những tổ hợp flag nào loại trừ lẫn nhau? Hãy liệt kê chúng.
4. Tính toán sự lãng phí phân mảnh KV cache cho một trace gồm 1.000 yêu cầu với trung bình 1.500 token đầu ra, độ lệch chuẩn 600 token, dưới (a) cấp phát liên tục mỗi yêu cầu ở mức tối đa 8192, (b) PagedAttention với các block 16-token.
5. Giải thích trong một đoạn văn tại sao chunked prefill giúp P99 ITL nhưng không giúp thông lượng nếu xét riêng lẻ. Trong thực tế, thông lượng tăng lên từ đâu?

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|------------------------|
| PagedAttention | "thủ thuật KV" | Bộ cấp phát block kích thước cố định cho KV cache; phân mảnh <4% |
| Block table | "bảng trang" | Bản đồ theo từng chuỗi từ vị trí token logic đến block KV vật lý |
| Continuous batching | "dynamic batching, nhưng đúng" | Quyết định nhận/giải phóng được đưa ra mỗi vòng lặp decode |
| Chunked prefill | "chia nhỏ prefill" | Chia prefill dài thành các lát cắt 512-token xen kẽ với decode |
| TTFT | "thời gian token đầu tiên" | Prefill + hàng đợi + mạng; bị chi phối bởi prefill ở các prompt dài |
| ITL | "độ trễ giữa các token" | Thời gian giữa các token decode liên tiếp; bị chi phối bởi kích thước batch |
| Goodput | "thông lượng đạt SLO" | Token/giây mà tại đó mọi yêu cầu vẫn đạt mục tiêu TTFT và ITL |
| V1 scheduler | "scheduler mới" | Scheduler vLLM 2026; N-gram spec decode là đường dẫn tương thích với chunked-prefill |
| `--gpu-memory-utilization` | "núm vặn bộ nhớ" | Tỷ lệ HBM dành riêng cho các KV block sau khi đã tải trọng số và activation |

## Đọc thêm

- [Tài liệu vLLM — Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode/) — nguồn chính thức về tính tương thích của chunked-prefill và speculative-decoding.
- [Ghi chú phát hành vLLM (NVIDIA)](https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/index.html) — nhịp độ phát hành năm 2026 và hành vi cụ thể theo phiên bản.
- [Blog vLLM — PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html) — bài viết gốc vẫn định nghĩa cách tư duy về bộ cấp phát.
- [Bài báo PagedAttention (arXiv:2309.06180)](https://arxiv.org/abs/2309.06180) — phân tích phân mảnh và thiết kế scheduler.
- [Aleksa Gordic — Bên trong vLLM](https://www.aleksagordic.com/blog/vllm) — hướng dẫn chi tiết về scheduler V1 với flame graph.