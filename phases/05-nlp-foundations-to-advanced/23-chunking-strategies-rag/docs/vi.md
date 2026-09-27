# Các chiến lược Chunking cho RAG

> Cấu hình chunking ảnh hưởng đến chất lượng truy xuất nhiều như việc lựa chọn embedding model (Vectara NAACL 2025). Nếu thực hiện chunking sai, thì dù có reranking cũng không thể cứu vãn được.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 14 (Information Retrieval), Phase 5 · 22 (Embedding Models)
**Time:** ~60 phút

## Vấn đề

Bạn đưa một hợp đồng dài 50 trang vào hệ thống RAG. Người dùng hỏi: "Điều khoản chấm dứt hợp đồng là gì?". Hệ thống truy xuất trả về trang bìa. Tại sao? Bởi vì mô hình được huấn luyện trên các chunk 512-token và điều khoản chấm dứt nằm ở trang 20, bị cắt ngang bởi ngắt trang, không có từ khóa cục bộ nào liên kết nó với truy vấn.

Giải pháp không phải là "mua một embedding model tốt hơn". Giải pháp là chunking. Kích thước bao nhiêu? Độ chồng lấp (overlap) thế nào? Cắt ở đâu? Có cần ngữ cảnh xung quanh không?

Các benchmark tháng 2/2026 cho thấy những kết quả đáng ngạc nhiên:

- Nghiên cứu năm 2026 của Vectara: recursive chunking 512-token đánh bại semantic chunking với độ chính xác 69% → 54%.
- SPLADE + Mistral-8B trên Natural Questions: overlap không mang lại lợi ích đo lường được.
- Context cliff: chất lượng phản hồi giảm mạnh ở khoảng 2.500 token ngữ cảnh.

Câu trả lời "hiển nhiên" (semantic chunking, 20% overlap, 1000 token) thường là sai. Bài học này xây dựng trực giác cho sáu chiến lược và chỉ cho bạn khi nào nên dùng chiến lược nào.

## Khái niệm

![Six chunking strategies visualized on one passage](../assets/chunking.svg)

**Fixed chunking.** Chia mỗi N ký tự hoặc token. Baseline đơn giản nhất. Cắt ngang câu. Nén tốt, nhưng tính mạch lạc kém.

**Recursive.** `RecursiveCharacterTextSplitter` của LangChain. Thử chia theo `\n\n` trước, sau đó là `\n`, rồi `.`, và cuối cùng là khoảng trắng. Dự phòng một cách sạch sẽ. Đây là mặc định của năm 2026.

**Semantic.** Embed từng câu. Tính cosine similarity giữa các câu liền kề. Chia tại nơi độ tương đồng giảm xuống dưới ngưỡng. Bảo toàn tính mạch lạc của chủ đề. Chậm hơn; đôi khi tạo ra các đoạn nhỏ 40-token làm giảm hiệu quả truy xuất.

**Sentence.** Chia theo ranh giới câu. Một câu mỗi chunk hoặc một cửa sổ gồm N câu. Tương đương với semantic chunking lên đến ~5k token với chi phí thấp hơn nhiều.

**Parent-document.** Lưu trữ các chunk con nhỏ để truy xuất *và* chunk cha lớn hơn để lấy ngữ cảnh. Truy xuất bằng chunk con; trả về chunk cha. Suy giảm hiệu năng một cách có kiểm soát: các chunk con kém vẫn trả về các chunk cha hợp lý.

**Late chunking (2024).** Embed toàn bộ tài liệu ở cấp độ token trước, sau đó gộp (pool) các token embedding thành chunk embedding. Bảo toàn ngữ cảnh xuyên suốt các chunk. Hoạt động với các embedder có ngữ cảnh dài (BGE-M3, Jina v3). Tốn tài nguyên tính toán hơn.

**Contextual retrieval (Anthropic, 2024).** Thêm vào trước mỗi chunk một bản tóm tắt do LLM tạo ra về vị trí của nó trong tài liệu ("Chunk này là phần 3.2 của các điều khoản chấm dứt..."). Cải thiện 35-50% khả năng truy xuất trong benchmark của Anthropic. Tốn kém khi đánh chỉ mục (index).

### Quy tắc đánh bại mọi mặc định

Khớp kích thước chunk với loại truy vấn:

| Loại truy vấn | Kích thước chunk |
|------------|-----------|
| Factoid ("tên CEO là gì?") | 256-512 token |
| Phân tích / multi-hop | 512-1024 token |
| Hiểu toàn bộ phần nội dung | 1024-2048 token |

Benchmark năm 2026 của NVIDIA. Chunk nên đủ lớn để chứa câu trả lời cộng với ngữ cảnh cục bộ, và đủ nhỏ để top-K của bộ truy xuất tập trung vào câu trả lời thay vì nhiễu ngữ cảnh.

```figure
n5-chunk-cuts
```

## Xây dựng

### Bước 1: fixed và recursive chunking

```python
def chunk_fixed(text, size=512, overlap=0):
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step)]


def chunk_recursive(text, size=512, seps=("\n\n", "\n", ". ", " ")):
    if len(text) <= size:
        return [text]
    for sep in seps:
        if sep not in text:
            continue
        parts = text.split(sep)
        chunks = []
        buf = ""
        for p in parts:
            if len(p) > size:
                if buf:
                    chunks.append(buf)
                    buf = ""
                chunks.extend(chunk_recursive(p, size=size, seps=seps[1:] or (" ",)))
                continue
            candidate = buf + sep + p if buf else p
            if len(candidate) <= size:
                buf = candidate
            else:
                if buf:
                    chunks.append(buf)
                buf = p
        if buf:
            chunks.append(buf)
        return [c for c in chunks if c.strip()]
    return chunk_fixed(text, size)
```

### Bước 2: semantic chunking

```python
def chunk_semantic(text, encoder, threshold=0.6, min_chars=200, max_chars=2048):
    sentences = split_sentences(text)
    if not sentences:
        return []
    embs = encoder.encode(sentences, normalize_embeddings=True)
    chunks = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = float(embs[i] @ embs[i - 1])
        current_len = sum(len(s) for s in chunks[-1])
        if sim < threshold and current_len >= min_chars:
            chunks.append([sentences[i]])
        else:
            chunks[-1].append(sentences[i])

    result = []
    for group in chunks:
        text_group = " ".join(group)
        if len(text_group) > max_chars:
            result.extend(chunk_recursive(text_group, size=max_chars))
        else:
            result.append(text_group)
    return result
```

Điều chỉnh `threshold` cho phù hợp với miền dữ liệu của bạn. Quá cao → các đoạn rời rạc. Quá thấp → một chunk khổng lồ.

### Bước 3: parent-document

```python
def chunk_parent_child(text, parent_size=2048, child_size=256):
    parents = chunk_recursive(text, size=parent_size)
    mapping = []
    for p_idx, parent in enumerate(parents):
        children = chunk_recursive(parent, size=child_size)
        for child in children:
            mapping.append({"child": child, "parent_idx": p_idx, "parent": parent})
    return mapping


def retrieve_parent(child_query, mapping, encoder, top_k=3):
    child_embs = encoder.encode([m["child"] for m in mapping], normalize_embeddings=True)
    q_emb = encoder.encode([child_query], normalize_embeddings=True)[0]
    scores = child_embs @ q_emb
    top = np.argsort(-scores)[:top_k]
    seen, parents = set(), []
    for i in top:
        if mapping[i]["parent_idx"] not in seen:
            parents.append(mapping[i]["parent"])
            seen.add(mapping[i]["parent_idx"])
    return parents
```

Thông tin quan trọng: khử trùng lặp (dedupe) các chunk cha. Nhiều chunk con có thể ánh xạ tới cùng một chunk cha; việc trả về tất cả sẽ gây lãng phí ngữ cảnh.

### Bước 4: contextual retrieval (mô hình Anthropic)

```python
def contextualize_chunks(document, chunks, llm):
    context_prompts = [
        f"""<document>{document}</document>
Here is the chunk to situate: <chunk>{c}</chunk>
Write 50-100 words placing this chunk in the document's context."""
        for c in chunks
    ]
    contexts = llm.batch(context_prompts)
    return [f"{ctx}\n\n{c}" for ctx, c in zip(contexts, chunks)]
```

Đánh chỉ mục các chunk đã được thêm ngữ cảnh. Tại thời điểm truy vấn, việc truy xuất được hưởng lợi từ tín hiệu xung quanh bổ sung.

### Bước 5: đánh giá

```python
def recall_at_k(queries, corpus_chunks, encoder, k=5):
    chunk_embs = encoder.encode(corpus_chunks, normalize_embeddings=True)
    hits = 0
    for q_text, gold_idxs in queries:
        q_emb = encoder.encode([q_text], normalize_embeddings=True)[0]
        top = np.argsort(-(chunk_embs @ q_emb))[:k]
        if any(i in gold_idxs for i in top):
            hits += 1
    return hits / len(queries)
```

Luôn luôn benchmark. Chiến lược "tốt nhất" cho kho dữ liệu của bạn có thể không giống với bất kỳ bài blog nào.

## Các cạm bẫy

- **Chunking chỉ được đánh giá trên các truy vấn factoid.** Các truy vấn multi-hop sẽ cho thấy những người chiến thắng rất khác biệt. Hãy sử dụng tập đánh giá phân tầng theo loại truy vấn.
- **Semantic chunking không có kích thước tối thiểu.** Tạo ra các đoạn 40-token làm giảm hiệu quả truy xuất. Luôn áp đặt `min_tokens`.
- **Overlap như một "tín ngưỡng".** Các nghiên cứu năm 2026 cho thấy overlap thường không mang lại lợi ích gì và làm tăng gấp đôi chi phí index. Hãy đo lường, đừng giả định.
- **Không áp đặt min/max.** Các chunk 5 token hoặc 5000 token đều làm hỏng việc truy xuất. Hãy giới hạn (clamp).
- **Chunking xuyên tài liệu.** Không bao giờ để một chunk trải dài qua hai tài liệu. Luôn chunk theo từng tài liệu, sau đó mới hợp nhất.

## Sử dụng

Stack năm 2026:

| Tình huống | Chiến lược |
|-----------|----------|
| Xây dựng lần đầu, kho dữ liệu chưa biết | Recursive, 512 token, không overlap |
| Factoid QA | Recursive, 256-512 token |
| Phân tích / multi-hop | Recursive, 512-1024 token + parent-document |
| Tham chiếu chéo nặng (hợp đồng, bài báo) | Late chunking hoặc contextual retrieval |
| Kho dữ liệu hội thoại / đối thoại | Chunk theo lượt + metadata người nói |
| Các phát biểu ngắn (tweet, đánh giá) | Một tài liệu = một chunk |

Bắt đầu với recursive 512. Đo lường recall@5 trên tập đánh giá 50 truy vấn. Sau đó điều chỉnh từ đó.

## Triển khai

Lưu dưới dạng `outputs/skill-chunker.md`:

```markdown
---
name: chunker
description: Pick a chunking strategy, size, and overlap for a given corpus and query distribution.
version: 1.0.0
phase: 5
lesson: 23
tags: [nlp, rag, chunking]
---

Given a corpus (document types, avg length, domain) and query distribution (factoid / analytical / multi-hop), output:

1. Strategy. Recursive / sentence / semantic / parent-document / late / contextual. Reason.
2. Chunk size. Token count. Reason tied to query type.
3. Overlap. Default 0; justify if >0.
4. Min/max enforcement. `min_tokens`, `max_tokens` guards.
5. Evaluation plan. Recall@5 on 50-query stratified eval set (factoid, analytical, multi-hop).

Refuse any chunking strategy without min/max chunk size enforcement. Refuse overlap above 20% without an ablation showing it helps. Flag semantic chunking recommendations without a min-token floor.
```

## Bài tập

1. **Dễ.** Chunk một tài liệu 20 trang với fixed(512, 0), recursive(512, 0), và recursive(512, 100). So sánh số lượng chunk và chất lượng ranh giới.
2. **Trung bình.** Xây dựng tập đánh giá 30 truy vấn trên 5 tài liệu. Đo lường recall@5 cho recursive, semantic, và parent-document. Cái nào thắng? Nó có khớp với các bài blog không?
3. **Khó.** Triển khai contextual retrieval. Đo lường mức cải thiện MRR so với baseline recursive. Báo cáo chi phí index (số lần gọi LLM) so với mức tăng độ chính xác.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| Chunk | Một phần của tài liệu | Đơn vị con của tài liệu được embed, index và truy xuất. |
| Overlap | Biên độ an toàn | N token được chia sẻ giữa các chunk liền kề; thường vô dụng trong các benchmark 2026. |
| Semantic chunking | Chunking thông minh | Chia tại nơi độ tương đồng embedding của các câu liền kề giảm xuống. |
| Parent-document | Truy xuất hai cấp | Truy xuất các chunk con nhỏ, trả về các chunk cha lớn hơn. |
| Late chunking | Chunk sau khi embed | Embed toàn bộ tài liệu ở cấp độ token, gộp thành các vector chunk. |
| Contextual retrieval | Mẹo của Anthropic | Bản tóm tắt do LLM tạo ra được thêm vào trước mỗi chunk trước khi index. |
| Context cliff | Bức tường 2500-token | Sự sụt giảm chất lượng quan sát được ở khoảng 2.5k token ngữ cảnh trong RAG (tháng 1/2026). |

## Đọc thêm

- [Yepes et al. / LangChain — Tài liệu về Recursive Character Splitting](https://python.langchain.com/docs/how_to/recursive_text_splitter/) — mặc định trong môi trường production.
- [Vectara (2024, NAACL 2025). Phân tích cấu hình chunking](https://arxiv.org/abs/2410.13070) — chunking quan trọng ngang với việc chọn embedding.
- [Jina AI — Late Chunking trong các Embedding Model có ngữ cảnh dài (2024)](https://jina.ai/news/late-chunking-in-long-context-embedding-models/) — bài báo về late chunking.
- [Anthropic — Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) — cải thiện 35-50% khả năng truy xuất với các tiền tố ngữ cảnh do LLM tạo ra.
- [Benchmark kích thước chunk của NVIDIA 2026 — Tóm tắt bởi Premai](https://blog.premai.io/rag-chunking-strategies-the-2026-benchmark-guide/) — kích thước chunk theo loại truy vấn.