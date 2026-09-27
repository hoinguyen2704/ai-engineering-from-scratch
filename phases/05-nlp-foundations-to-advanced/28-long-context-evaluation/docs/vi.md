# Đánh giá Long-Context — NIAH, RULER, LongBench, MRCR

> Gemini 3 Pro quảng cáo ngữ cảnh 10M token. Ở mức 1M token, MRCR 8-needle giảm xuống còn 26,3%. Quảng cáo ≠ khả dụng. Đánh giá long-context cho bạn biết năng lực thực tế của mô hình mà bạn đang triển khai.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 13 (Question Answering), Phase 5 · 23 (Chunking Strategies)
**Time:** ~60 phút

## Vấn đề

Bạn có một hợp đồng dài 200 trang. Mô hình tuyên bố có ngữ cảnh 1M token. Bạn dán hợp đồng vào và hỏi: "Điều khoản chấm dứt hợp đồng là gì?" Mô hình trả lời — nhưng lại trả lời dựa trên trang bìa vì điều khoản chấm dứt nằm ở độ sâu 120k token, vượt quá phạm vi mà mô hình thực sự chú ý (attend).

Đây là khoảng cách năng lực ngữ cảnh năm 2026. Bảng thông số kỹ thuật ghi 1M hoặc 10M. Thực tế cho thấy 60-70% trong số đó là khả dụng, và "khả dụng" phụ thuộc vào tác vụ.

- **Retrieval (một chiếc kim trong đống cỏ):** gần như hoàn hảo cho đến mức tối đa được quảng cáo trên các mô hình tiên phong.
- **Multi-hop / aggregation:** suy giảm mạnh sau khoảng 128k trên hầu hết các mô hình.
- **Reasoning over dispersed facts:** tác vụ đầu tiên bị thất bại.

Đánh giá long-context đo lường các trục này. Bài học này liệt kê các benchmark, ý nghĩa thực sự của từng loại và cách xây dựng bài kiểm tra needle tùy chỉnh cho lĩnh vực của bạn.

## Khái niệm

![NIAH baseline, RULER multi-task, LongBench holistic](../assets/long-context-eval.svg)

**Needle-in-a-Haystack (NIAH, 2023).** Đặt một sự kiện ("từ ma thuật là pineapple") tại một độ sâu được kiểm soát trong ngữ cảnh dài. Yêu cầu mô hình truy xuất nó. Quét theo độ sâu × độ dài. Đây là benchmark long-context gốc. Các mô hình tiên phong hiện đã bão hòa bài kiểm tra này; nó là một tiêu chuẩn cần thiết nhưng chưa đủ.

**RULER (Nvidia, 2024).** 13 loại tác vụ trong 4 danh mục: retrieval (đơn / đa khóa / đa giá trị), multi-hop tracing (theo dõi biến), aggregation (tần suất từ phổ biến), QA. Độ dài ngữ cảnh có thể cấu hình (4k đến 128k+). Tiết lộ các mô hình bão hòa NIAH nhưng thất bại ở các tác vụ multi-hop. Trong bản phát hành năm 2024, chỉ một nửa trong số 17 mô hình tuyên bố ngữ cảnh 32k+ duy trì được chất lượng ở mức 32k.

**LongBench v2 (2024).** 503 câu hỏi trắc nghiệm, ngữ cảnh từ 8k-2M từ, sáu danh mục tác vụ: QA đơn tài liệu, QA đa tài liệu, long in-context learning, hội thoại dài, kho mã nguồn, dữ liệu cấu trúc dài. Benchmark sản xuất cho hành vi long-context trong thế giới thực.

**MRCR (Multi-Round Coreference Resolution).** Giải quyết tham chiếu đa vòng ở quy mô lớn. Các biến thể 8-needle, 24-needle, 100-needle. Phơi bày số lượng sự kiện mà mô hình có thể xử lý trước khi cơ chế attention bị suy giảm.

**NoLiMa.** "Non-lexical needle." Needle và truy vấn không có sự trùng lặp về từ ngữ; việc truy xuất đòi hỏi một bước suy luận ngữ nghĩa. Khó hơn NIAH.

**HELMET.** Nối nhiều tài liệu lại với nhau, đặt câu hỏi từ bất kỳ tài liệu nào. Kiểm tra khả năng selective attention.

**BABILong.** Nhúng các chuỗi suy luận bAbI vào bên trong các đống cỏ không liên quan. Kiểm tra khả năng reasoning-in-a-haystack, không chỉ là truy xuất.

### Những gì cần báo cáo thực tế

- **Advertised context window.** Con số trên bảng thông số kỹ thuật.
- **Effective retrieval length.** Điểm vượt qua NIAH ở một ngưỡng nhất định (ví dụ: 90%).
- **Effective reasoning length.** Điểm vượt qua multi-hop hoặc aggregation ở ngưỡng đó.
- **Degradation curve.** Độ chính xác so với độ dài ngữ cảnh, được vẽ biểu đồ theo từng loại tác vụ.

Hai con số cho bảng thông số của bạn: hiệu quả truy xuất và hiệu quả suy luận. Thông thường, hiệu quả suy luận chỉ bằng 25-50% cửa sổ ngữ cảnh được quảng cáo.

```figure
gx-niah-decay
```

## Xây dựng

### Bước 1: NIAH tùy chỉnh cho lĩnh vực của bạn

Xem `code/main.py`. Khung sườn:

```python
def build_haystack(filler_text, needle, depth_ratio, total_tokens):
    if not (0.0 <= depth_ratio <= 1.0):
        raise ValueError(f"depth_ratio must be in [0, 1], got {depth_ratio}")
    if total_tokens <= 0:
        raise ValueError(f"total_tokens must be positive, got {total_tokens}")

    filler_tokens = tokenize(filler_text)
    needle_tokens = tokenize(needle)
    if not filler_tokens:
        raise ValueError("filler_text produced no tokens")

    # Repeat filler until long enough to fill the haystack body.
    body_len = max(total_tokens - len(needle_tokens), 0)
    while len(filler_tokens) < body_len:
        filler_tokens = filler_tokens + filler_tokens
    filler_tokens = filler_tokens[:body_len]

    insert_at = min(int(body_len * depth_ratio), body_len)
    haystack = filler_tokens[:insert_at] + needle_tokens + filler_tokens[insert_at:]
    return " ".join(haystack)


def score_niah(model, haystack, question, expected):
    answer = model.complete(f"Context: {haystack}\nQ: {question}\nA:", max_tokens=50)
    return 1 if expected.lower() in answer.lower() else 0
```

Quét `depth_ratio` ∈ {0, 0.25, 0.5, 0.75, 1.0} × `total_tokens` ∈ {1k, 4k, 16k, 64k}. Vẽ biểu đồ heatmap. Đó chính là thẻ NIAH cho mô hình mục tiêu của bạn.

### Bước 2: Biến thể multi-needle

```python
def build_multi_needle(filler, needles, total_tokens):
    depths = [0.1, 0.4, 0.7]
    chunks = [filler[:int(total_tokens * 0.1)]]
    for depth, needle in zip(depths, needles):
        chunks.append(needle)
        next_chunk = filler[int(total_tokens * depth): int(total_tokens * (depth + 0.3))]
        chunks.append(next_chunk)
    return " ".join(chunks)
```

Các câu hỏi như "Ba từ ma thuật là gì?" đòi hỏi phải truy xuất cả ba. Thành công với single-needle không dự đoán được thành công với multi-needle.

### Bước 3: Multi-hop variable tracing (kiểu RULER)

```python
haystack = """X1 = 42. ... (filler) ... X2 = X1 + 10. ... (filler) ... X3 = X2 * 2."""
question = "What is X3?"
```

Câu trả lời đòi hỏi phải xâu chuỗi ba phép gán. Các mô hình tiên phong ở mức 128k thường giảm độ chính xác xuống còn 50-70% ở đây.

### Bước 4: LongBench v2 trên stack của bạn

```python
from datasets import load_dataset
longbench = load_dataset("THUDM/LongBench-v2")

def eval_model_on_longbench(model, subset="single-doc-qa"):
    tasks = [x for x in longbench["test"] if x["task"] == subset]
    correct = 0
    for x in tasks:
        answer = model.complete(x["context"] + "\n\nQ: " + x["question"], max_tokens=20)
        if normalize(answer) == normalize(x["answer"]):
            correct += 1
    return correct / len(tasks)
```

Báo cáo độ chính xác theo từng danh mục. Điểm tổng hợp thường che giấu những khác biệt lớn ở cấp độ tác vụ.

## Các cạm bẫy

- **Chỉ đánh giá bằng NIAH.** Vượt qua NIAH ở 1M token không nói lên điều gì về multi-hop. Luôn chạy RULER hoặc bài kiểm tra multi-hop tùy chỉnh.
- **Lấy mẫu độ sâu đồng nhất.** Nhiều triển khai chỉ kiểm tra độ sâu=0.5. Hãy kiểm tra độ sâu=0, 0.25, 0.5, 0.75, 1.0 — hiệu ứng "lost in the middle" là có thật.
- **Trùng lặp từ vựng với filler.** Nếu needle chia sẻ từ khóa với filler, việc truy xuất trở nên quá dễ dàng. Hãy sử dụng các needle không trùng lặp kiểu NoLiMa.
- **Bỏ qua độ trễ.** Các prompt 1M-token mất 30-120 giây để prefill. Hãy đo thời gian đến token đầu tiên (time-to-first-token) cùng với độ chính xác.
- **Số liệu do nhà cung cấp tự báo cáo.** OpenAI, Google, Anthropic đều công bố điểm số của riêng họ. Luôn chạy lại độc lập trên trường hợp sử dụng của bạn.

## Sử dụng

Stack năm 2026:

| Tình huống | Benchmark |
|-----------|-----------|
| Kiểm tra nhanh | NIAH tùy chỉnh tại 3 độ sâu × 3 độ dài |
| Chọn mô hình cho sản xuất | RULER (13 tác vụ) tại độ dài mục tiêu |
| Chất lượng QA thực tế | Tập con LongBench v2 single-doc-QA |
| Suy luận multi-hop | BABILong hoặc variable-tracing tùy chỉnh |
| Hội thoại / đối thoại | MRCR 8-needle tại độ dài mục tiêu |
| Hồi quy nâng cấp mô hình | NIAH nội bộ cố định + bộ công cụ RULER, chạy trên mọi mô hình mới |

Quy tắc ngón tay cái cho sản xuất: không bao giờ tin vào cửa sổ ngữ cảnh cho đến khi bạn đã chạy NIAH + 1 tác vụ suy luận ở độ dài dự định.

## Triển khai

Lưu dưới dạng `outputs/skill-long-context-eval.md`:

```markdown
---
name: long-context-eval
description: Design a long-context evaluation battery for a given model and use case.
version: 1.0.0
phase: 5
lesson: 28
tags: [nlp, long-context, evaluation]
---

Given a target model, target context length, and use case, output:

1. Tests. NIAH depth × length grid; RULER multi-hop; custom domain task.
2. Sampling. Depths 0, 0.25, 0.5, 0.75, 1.0 at each length.
3. Metrics. Retrieval pass rate; reasoning pass rate; time-to-first-token; cost-per-query.
4. Cutoff. Effective retrieval length (90% pass) and effective reasoning length (70% pass). Report both.
5. Regression. Fixed harness, rerun on every model upgrade, surface deltas.

Refuse to trust a context window from the model card alone. Refuse NIAH-only evaluation for any multi-hop workload. Refuse vendor self-reported long-context scores as independent evidence.
```

## Bài tập

1. **Dễ.** Xây dựng một NIAH với 3 độ sâu (0.25, 0.5, 0.75) × 3 độ dài (1k, 4k, 16k). Chạy trên bất kỳ mô hình nào. Vẽ biểu đồ tỷ lệ vượt qua dưới dạng heatmap 3×3.
2. **Trung bình.** Thêm biến thể 3-needle. Đo lường khả năng truy xuất cả 3 tại mỗi độ dài. So sánh với tỷ lệ vượt qua single-needle ở cùng độ dài.
3. **Khó.** Xây dựng một tác vụ variable-tracing (X1 → X2 → X3, với 3 bước nhảy) nhúng trong 64k filler. Đo độ chính xác trên 3 mô hình tiên phong. Báo cáo độ dài suy luận hiệu quả trên mỗi mô hình.

## Thuật ngữ chính

| Thuật ngữ | Mọi người nói | Ý nghĩa thực sự |
|------|-----------------|-----------------------|
| NIAH | Needle in haystack | Đặt một sự kiện vào filler, yêu cầu mô hình truy xuất. |
| RULER | NIAH nâng cấp | 13 loại tác vụ bao gồm truy xuất / multi-hop / aggregation / QA. |
| Effective context | Năng lực thực tế | Độ dài mà tại đó độ chính xác vẫn giữ trên ngưỡng. |
| Lost in the middle | Thiên kiến độ sâu | Các mô hình ít chú ý đến nội dung ở giữa các đầu vào dài. |
| Multi-needle | Nhiều sự kiện cùng lúc | Nhiều needle; kiểm tra khả năng xử lý attention, không chỉ truy xuất. |
| MRCR | Multi-round coref | Coreference 8, 24, hoặc 100-needle; phơi bày sự bão hòa attention. |
| NoLiMa | Non-lexical needle | Needle và truy vấn không trùng lặp token; đòi hỏi suy luận. |

## Đọc thêm

- [Kamradt (2023). Needle in a Haystack analysis](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) — repo NIAH gốc.
- [Hsieh et al. (2024). RULER: What's the Real Context Size of Your Long-Context LMs?](https://arxiv.org/abs/2404.06654) — benchmark đa tác vụ.
- [Bai et al. (2024). LongBench v2](https://arxiv.org/abs/2412.15204) — đánh giá long-context thực tế.
- [Modarressi et al. (2024). NoLiMa: Non-lexical needles](https://arxiv.org/abs/2404.06666) — các needle khó hơn.
- [Kuratov et al. (2024). BABILong](https://arxiv.org/abs/2406.10149) — suy luận trong đống cỏ.
- [Liu et al. (2024). Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) — bài báo về thiên kiến độ sâu.