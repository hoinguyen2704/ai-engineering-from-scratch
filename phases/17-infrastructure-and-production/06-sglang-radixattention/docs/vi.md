# Prefix-Cache Serving — RadixAttention và KV Reuse

> Hãy coi KV cache là một tài nguyên hạng nhất, có thể tái sử dụng được lưu trữ trong một radix tree, và lập lịch thay đổi theo nó: thay vì FCFS (đến trước phục vụ trước) như cách vLLM lập lịch, một bộ lập lịch nhận biết cache (cache-aware scheduler) sẽ ưu tiên các yêu cầu có tiền tố (prefix) chia sẻ dài hơn — thực hiện duyệt radix theo chiều sâu (depth-first) để các nhánh "nóng" luôn nằm trong HBM. SGLang là engine xây dựng việc phục vụ (serving) xoay quanh ý tưởng này. Trên Llama 3.1 8B với các prompt 1K kiểu ShareGPT, SGLang đạt ~16.200 tok/s so với ~12.500 của vLLM, lợi thế khoảng 29%. Trên các workload RAG có tiền tố dày đặc, lợi thế lên tới 6,4 lần. Trên các workload dạng voice-cloning, tỷ lệ cache hit đạt trên 86%. Được triển khai trên hơn 400.000 GPU vào năm 2026 tại xAI, LinkedIn, Cursor, Oracle, GCP, Azure, AWS. Điểm cần lưu ý là con số 6,4 lần sẽ biến mất khi thứ tự tiền tố không nhất quán — thứ tự chính là đòn bẩy của kỹ sư.

**Type:** Learn
**Languages:** Python (stdlib, toy radix-tree cache + cache-aware scheduler)
**Prerequisites:** Phase 17 · 04 (Serving Engine Internals), Phase 14 (Agentic RAG)
**Time:** ~75 phút

## Mục tiêu học tập

- Vẽ sơ đồ RadixAttention: cách các tiền tố được lưu trữ trong một radix tree và cách các KV block được chia sẻ giữa các chuỗi có cùng gốc tại một nhánh.
- Giải thích lập lịch nhận biết cache và tại sao FCFS lại sai lầm đối với lưu lượng truy cập có tiền tố dày đặc.
- Tính toán tốc độ tăng tốc dự kiến cho một workload dựa trên tỷ lệ hit của prefix-cache và phân phối độ dài prompt.
- Gọi tên quy tắc sắp xếp prompt giúp con số 6,4 lần trở thành hiện thực thay vì lãng phí tiềm năng.

## Vấn đề

Cách phục vụ cổ điển coi prompt của mỗi yêu cầu là một khối không thể tách rời. Ngay cả khi 5.000 yêu cầu RAG đều bắt đầu bằng cùng một system prompt 2.000 token cộng với cùng một phần mở đầu truy xuất, vLLM vẫn thực hiện prefill cho 2.000 token tiền tố đó 5.000 lần. GPU thực hiện cùng một công việc lặp đi lặp lại.

Quan sát: các prompt trong workload agentic và RAG hầu như luôn chia sẻ các tiền tố dài. System prompt, tool schema, few-shot example, tiêu đề truy xuất, lịch sử hội thoại — tất cả đều lặp lại giữa các yêu cầu. Nếu bạn lưu trữ KV cache cho tiền tố đó một lần và tái sử dụng nó, bạn sẽ không cần prefill lại nữa.

RadixAttention thực hiện chính xác điều này. Các token được đánh chỉ mục trong một radix tree; mỗi node sở hữu các KV block cho chuỗi token trên đường đi từ gốc. Một yêu cầu mới sẽ đi dọc theo cây: bất kỳ node nào có token khớp sẽ tái sử dụng các KV block của node đó. Chi phí prefill trở nên tỷ lệ thuận với phần hậu tố "mới", không phải toàn bộ prompt.

Thách thức nằm ở việc lập lịch. Nếu hai yêu cầu chia sẻ một tiền tố 2.000 token và yêu cầu thứ ba chỉ chia sẻ 200 token của cùng tiền tố đó, bạn muốn phục vụ hai yêu cầu chia sẻ dài cùng nhau để tiền tố dài đó nằm lại trong HBM. FCFS làm điều ngược lại — nó phục vụ bất cứ ai đến trước, có khả năng đẩy (evict) nhánh "nóng" ra ngoài trước khi yêu cầu có tiền tố dài tiếp theo đến.

## Khái niệm

### Radix tree như một KV index

Một radix tree (compact trie) lưu trữ các chuỗi token. Mỗi node sở hữu một dải token và các KV block được tính toán cho dải đó. Các node con mở rộng chuỗi thêm một hoặc nhiều token.

```
root
 |- "You are a helpful assistant..."  (2,000 tokens, 124 KV blocks)
      |- "Context: <doc A>..."        (500 tokens, 31 blocks)
           |- "Question: Alice..."    (80 tokens, 5 blocks)
           |- "Question: Bob..."      (95 tokens, 6 blocks)
      |- "Context: <doc B>..."        (520 tokens, 33 blocks)
```

Một yêu cầu mới đến với system prompt + "Context: <doc A>" + "Question: Carol". Bộ lập lịch duyệt: tiền tố hệ thống khớp (124 block được tái sử dụng), nhánh doc-A khớp (31 block được tái sử dụng), sau đó chỉ cấp phát các block mới cho "Question: Carol" (4 block). Chi phí prefill: 4 block token mới. Nếu không có cây: 160 block. Tiết kiệm ~40 lần chi phí prefill.

### Lập lịch nhận biết cache (Cache-aware scheduling)

Việc tái sử dụng dựa trên radix-tree sẽ vô nghĩa nếu cache bị xáo trộn liên tục. Hai chính sách chính:

1. **Điều phối theo chiều sâu (Depth-first dispatch)**. Khi chọn yêu cầu tiếp theo từ hàng đợi, hãy ưu tiên các yêu cầu có cùng gốc với tập hợp đang chạy. Điều này giữ cho nhánh "nóng" được ghim lại.
2. **LRU ở cấp độ nhánh, không phải cấp độ block**. Đẩy toàn bộ các nhánh (bắt đầu từ các lá được sử dụng ít nhất) thay vì từng block riêng lẻ, để hình dạng cache khớp với hình dạng radix.

FCFS vi phạm cả hai. Một yêu cầu chia sẻ 2.000 token nằm sau một yêu cầu chia sẻ 50 token, sau đó nhánh 2.000 token bị đẩy ra để nhường chỗ cho nhánh 50 token.

### Các con số benchmark bạn nên ghi nhớ

- Llama 3.1 8B, H100, ShareGPT 1K prompts: SGLang ~16.200 tok/s so với vLLM ~12.500 (lợi thế ~29%).
- RAG có tiền tố dày đặc (cùng hệ thống + cùng tài liệu, câu hỏi khác nhau): lên tới 6,4 lần trên SGLang.
- Workload voice cloning: tỷ lệ prefix-cache hit đạt 86,4%.
- Tỷ lệ hit trong sản xuất trên các khách hàng của SGLang: 50-99% tùy thuộc vào kỷ luật prompt.
- Được triển khai trên hơn 400.000 GPU vào năm 2026.

### Điểm cần lưu ý về thứ tự

Con số 6,4 lần dựa vào thứ tự template prompt nhất quán. Nếu client của bạn xây dựng prompt dưới dạng `[system, tools, context, history, question]` trong một số yêu cầu và `[system, context, tools, history, question]` trong các yêu cầu khác, cây sẽ không thể tìm thấy tiền tố được chia sẻ. Những gì trông giống như một tiền tố được chia sẻ đối với con người lại là hai chuỗi riêng biệt đối với radix tree.

Đòn bẩy của kỹ sư: template prompt của bạn chính là một cache key. Hãy cố định thứ tự. Đặt mọi thứ bất biến (hệ thống, công cụ, schema) lên đầu. Đặt ngữ cảnh truy xuất tiếp theo. Đặt câu hỏi của người dùng cuối cùng. Không chèn nội dung động vào tiền tố.

Trường hợp thực tế từ nghiên cứu: việc di chuyển nội dung động ra khỏi tiền tố có thể cache đã giúp một lần triển khai tăng tỷ lệ cache hit từ 7% lên 74% chỉ với một thay đổi.

### Nơi RadixAttention thắng và thua

Thắng:
- RAG (cùng tiêu đề truy xuất, câu hỏi khác nhau).
- Agent (cùng tool schema, truy vấn khác nhau).
- Chat với system prompt dài.
- Workload giọng nói / hình ảnh với các tiêu đề lặp lại.

Thua (trở lại mức thông lượng của vLLM):
- Tạo nội dung single-shot với các prompt độc nhất (hoàn thành code, chat mở không có system prompt).
- Prompt động nơi mọi yêu cầu đều chèn nội dung độc nhất vào tiền tố.

### Tại sao đây là vấn đề của bộ lập lịch, không chỉ là vấn đề của kernel

Bạn có thể triển khai tái sử dụng KV như một thủ thuật kernel. Sự thấu hiểu của SGLang là việc tái sử dụng chỉ mang lại hiệu quả nếu bộ lập lịch giữ cho nhánh "nóng" luôn thường trú. Chính sách "tái sử dụng nếu có" ngây thơ sẽ làm xáo trộn cache dưới tải trọng hỗn hợp. Bộ lập lịch được đánh chỉ mục bằng radix-tree chính là thứ biến thủ thuật kernel thành lợi thế sản xuất 29%.

### Tương tác với vLLM

Hai hệ thống này không phải là đối thủ cạnh tranh gay gắt. Năm 2026, vLLM đã thêm prefix caching (`--enable-prefix-caching`) và bộ định tuyến nhận biết cache (vLLM Router bằng Rust). Khoảng cách đã thu hẹp nhưng không hoàn toàn biến mất — toàn bộ stack của SGLang là radix-first; vLLM đã tích hợp thêm vào. Đối với các workload bị chi phối bởi việc tái sử dụng tiền tố, SGLang vẫn là lựa chọn mặc định. Đối với việc phục vụ mục đích chung mà không có các mẫu tiền tố mạnh, vLLM vẫn ngang bằng hoặc tốt hơn.

```figure
roofline
```

## Sử dụng

`code/main.py` triển khai một KV cache radix-tree đơn giản cộng với bộ lập lịch có hai chính sách: FCFS và nhận biết cache. Chạy cùng một workload qua cả hai, báo cáo tỷ lệ prefix-cache hit và sự thay đổi thông lượng. Sau đó chạy một workload "thứ tự xáo trộn" để cho thấy sự sụp đổ 6,4 lần.

## Triển khai

Bài học này tạo ra `outputs/skill-radix-scheduler-advisor.md`. Với mô tả workload (hình dạng template prompt, mẫu truy xuất, số lượng tenant đồng thời), nó tạo ra một quy định về thứ tự prompt và quyết định có/không cho việc áp dụng SGLang.

## Bài tập

1. Chạy `code/main.py`. So sánh FCFS và nhận biết cache trên cùng một workload. Sự khác biệt đến từ đâu — tiết kiệm prefill, tiết kiệm decode, hay độ trễ hàng đợi?
2. Sửa đổi workload để các prompt hoán vị ngẫu nhiên `[system, tools, context]`. Chạy lại. Điều gì xảy ra với tỷ lệ hit? Tại sao?
3. Tính toán chi phí HBM để giữ một system prompt 2.000 token thường trú như một nhánh radix trên Llama 3.1 8B. So sánh với chi phí của một batch 16 chuỗi mà không tái sử dụng tiền tố.
4. Đọc bài báo SGLang RadixAttention. Giải thích trong ba câu tại sao việc đẩy (eviction) LRU theo hình cây lại tốt hơn LRU theo block dưới tải trọng tiền tố dày đặc.
5. Một khách hàng báo cáo tỷ lệ cache hit chỉ 8%. Hãy nêu ba nguyên nhân có khả năng xảy ra và chẩn đoán bạn sẽ thực hiện cho từng nguyên nhân.

## Thuật ngữ chính

| Thuật ngữ | Mọi người nói | Ý nghĩa thực tế |
|------|----------------|------------------------|
| RadixAttention | "thứ của SGLang" | KV cache được đánh chỉ mục như một radix tree để các tiền tố chia sẻ tái sử dụng block |
| Radix tree | "compact trie" | Cây nơi mỗi node sở hữu một dải token và các KV block của nó |
| Cache-aware scheduler | "ưu tiên nhánh nóng" | Bộ lập lịch ưu tiên các yêu cầu chia sẻ nhánh đang thường trú |
| Prefix-cache hit rate | "bao nhiêu phần trăm prompt của bạn là miễn phí" | Tỷ lệ token prompt được phục vụ từ các KV block tái sử dụng |
| FCFS | "đến trước phục vụ trước" | Lập lịch mặc định làm phá vỡ tính cục bộ của tiền tố |
| Branch-level LRU | "đẩy lá" | Chính sách đẩy phù hợp với hình dạng radix |
| Prompt template ordering | "cache key" | Thứ tự các thành phần của prompt quyết định những gì cây có thể chia sẻ |
| System prompt pinning | "tiền tố thường trú" | Giữ phần hệ thống bất biến được ghim để tránh xáo trộn do đẩy cache |

## Đọc thêm

- [SGLang GitHub](https://github.com/sgl-project/sglang) — mã nguồn và tài liệu.
- [SGLang documentation](https://sgl-project.github.io/) — chi tiết về RadixAttention và lập lịch.
- [SGLang paper — Efficiently Programming Large Language Models (arXiv:2312.07104)](https://arxiv.org/abs/2312.07104) — tài liệu tham khảo thiết kế.
- [LMSYS blog — SGLang with RadixAttention](https://www.lmsys.org/blog/2024-01-17-sglang/) — các con số benchmark và lý do lập lịch.
- [vLLM — Prefix Caching](https://docs.vllm.ai/en/latest/features/prefix_caching.html) — triển khai kiểu radix của riêng vLLM, để so sánh.