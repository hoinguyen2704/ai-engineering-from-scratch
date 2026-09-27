# Information Retrieval and Search

> BM25 chính xác nhưng cứng nhắc. Dense bao quát rộng nhưng lại bỏ lỡ các từ khóa. Hybrid là tiêu chuẩn mặc định của năm 2026. Mọi thứ khác chỉ là tinh chỉnh.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 04 (GloVe, FastText, Subword)
**Time:** ~75 phút

## Vấn đề

Người dùng nhập "what happens if someone lies to get money" và mong đợi tìm thấy điều luật thực sự bao hàm hành vi đó: "Section 420 IPC." Tìm kiếm theo từ khóa (keyword search) hoàn toàn bỏ lỡ kết quả này (do không có từ vựng chung). Tìm kiếm ngữ nghĩa (semantic search) cũng sẽ bỏ lỡ nếu các embedding không được huấn luyện trên văn bản pháp luật. Tìm kiếm thực tế phải xử lý được cả hai.

IR (Information Retrieval) là pipeline nằm dưới mọi hệ thống RAG, mọi thanh tìm kiếm, mọi tính năng tra cứu thông minh trên các trang tài liệu. Kiến trúc năm 2026 hoạt động hiệu quả trong môi trường production không phải là một phương pháp đơn lẻ. Đó là một chuỗi các phương pháp bổ trợ cho nhau, mỗi phương pháp sẽ khắc phục những điểm yếu của phương pháp trước đó.

Bài học này sẽ xây dựng từng thành phần và chỉ rõ mỗi thành phần giải quyết những lỗi nào.

## Khái niệm

![Hybrid retrieval: BM25 + dense + RRF + cross-encoder rerank](../assets/retrieval.svg)

Bốn lớp. Hãy chọn những lớp bạn cần.

1. **Sparse retrieval (BM25).** Nhanh, chính xác với các khớp nối chính xác (exact matches), nhưng rất tệ về ngữ nghĩa. Chạy trên một inverted index. Tốc độ dưới 10ms mỗi truy vấn trên hàng triệu tài liệu. Giúp bạn tìm đúng các tham chiếu điều luật, mã sản phẩm, thông báo lỗi, và các thực thể có tên (named entities).
2. **Dense retrieval.** Mã hóa truy vấn và tài liệu thành các vector. Tìm kiếm láng giềng gần nhất (nearest neighbor search). Nắm bắt được các cách diễn đạt khác nhau và sự tương đồng về ngữ nghĩa. Bỏ lỡ các khớp nối từ khóa chính xác nếu chỉ khác nhau một ký tự. Tốc độ 50-200ms mỗi truy vấn với FAISS hoặc vector DB.
3. **Fusion.** Hợp nhất các danh sách đã xếp hạng từ sparse và dense. Reciprocal Rank Fusion (RRF) là lựa chọn mặc định dễ dàng vì nó bỏ qua các điểm số thô (vốn nằm ở các thang đo khác nhau) và chỉ sử dụng vị trí xếp hạng. Weighted fusion là một tùy chọn khi bạn biết một tín hiệu nào đó chiếm ưu thế trong lĩnh vực của mình.
4. **Cross-encoder rerank.** Lấy top-30 từ fusion. Chạy một cross-encoder (truy vấn + tài liệu cùng nhau, chấm điểm từng cặp). Giữ lại top-5. Cross-encoder chậm hơn trên mỗi cặp so với bi-encoder nhưng chính xác hơn nhiều. Bạn tối ưu hóa bằng cách chỉ chạy chúng trên top-30.

Tìm kiếm ba chiều (BM25 + dense + learned-sparse như SPLADE) vượt trội hơn hai chiều trong các benchmark năm 2026 nhưng cần cơ sở hạ tầng cho các chỉ mục learned-sparse. Đối với hầu hết các đội ngũ, hai chiều cộng với cross-encoder rerank là điểm cân bằng lý tưởng.

```figure
gx-hybrid-retrieval
```

## Xây dựng

### Bước 1: BM25 từ đầu

```python
import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        if not corpus:
            raise ValueError("corpus must not be empty")
        self.corpus = [tokenize(d) for d in corpus]
        self.k1 = k1
        self.b = b
        self.n_docs = len(self.corpus)
        self.avg_dl = sum(len(d) for d in self.corpus) / self.n_docs
        self.df = Counter()
        for doc in self.corpus:
            for term in set(doc):
                self.df[term] += 1

    def idf(self, term):
        n = self.df.get(term, 0)
        return math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))

    def score(self, query, doc_idx):
        q_tokens = tokenize(query)
        doc = self.corpus[doc_idx]
        dl = len(doc)
        freq = Counter(doc)
        score = 0.0
        for term in q_tokens:
            f = freq.get(term, 0)
            if f == 0:
                continue
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
            score += self.idf(term) * numerator / denominator
        return score

    def rank(self, query, top_k=10):
        scored = [(self.score(query, i), i) for i in range(self.n_docs)]
        scored.sort(reverse=True)
        return scored[:top_k]
```

Hai tham số cần biết. `k1=1.5` kiểm soát độ bão hòa tần suất thuật ngữ (term-frequency saturation); giá trị cao hơn nghĩa là trọng số lớn hơn cho việc lặp lại thuật ngữ. `b=0.75` kiểm soát chuẩn hóa độ dài; 0 bỏ qua độ dài tài liệu, 1 chuẩn hóa hoàn toàn. Các giá trị mặc định là khuyến nghị của Robertson từ bài báo gốc và hiếm khi cần tinh chỉnh.

### Bước 2: dense retrieval với bi-encoder

```python
from sentence_transformers import SentenceTransformer
import numpy as np


def build_dense_index(corpus, model_id="sentence-transformers/all-MiniLM-L6-v2"):
    encoder = SentenceTransformer(model_id)
    embeddings = encoder.encode(corpus, normalize_embeddings=True)
    return encoder, embeddings


def dense_search(encoder, embeddings, query, top_k=10):
    q_emb = encoder.encode([query], normalize_embeddings=True)
    sims = (embeddings @ q_emb.T).flatten()
    order = np.argsort(-sims)[:top_k]
    return [(float(sims[i]), int(i)) for i in order]
```

L2-normalize các embedding để tích vô hướng bằng với cosine similarity. `all-MiniLM-L6-v2` có 384 chiều, nhanh và đủ mạnh cho hầu hết các truy vấn tiếng Anh. Đối với công việc đa ngôn ngữ, hãy sử dụng `paraphrase-multilingual-MiniLM-L12-v2`. Để có độ chính xác cao nhất, hãy dùng `bge-large-en-v1.5` hoặc `e5-large-v2`.

### Bước 3: Reciprocal Rank Fusion

```python
def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, (_, doc_idx) in enumerate(ranking):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(score, doc_idx) for doc_idx, score in fused]
```

Hằng số `k=60` đến từ bài báo RRF gốc. `k` cao hơn sẽ làm phẳng sự đóng góp của các khác biệt về thứ hạng; `k` thấp hơn làm cho các thứ hạng đầu chiếm ưu thế. 60 là giá trị mặc định được công bố và hiếm khi cần tinh chỉnh.

### Bước 4: hybrid search + rerank

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def hybrid_search(query, bm25, encoder, dense_embeddings, corpus, top_k=5, pool_size=30, reranker=reranker):
    sparse_ranking = bm25.rank(query, top_k=pool_size)
    dense_ranking = dense_search(encoder, dense_embeddings, query, top_k=pool_size)
    fused = reciprocal_rank_fusion([sparse_ranking, dense_ranking])[:pool_size]

    pairs = [(query, corpus[doc_idx]) for _, doc_idx in fused]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(scores, [doc_idx for _, doc_idx in fused]), reverse=True)
    return reranked[:top_k]
```

Ba giai đoạn kết hợp. BM25 tìm các khớp nối từ vựng. Dense tìm các khớp nối ngữ nghĩa. RRF hợp nhất hai bảng xếp hạng mà không cần hiệu chỉnh điểm số. Cross-encoder chấm điểm lại top-30 bằng cách sử dụng các cặp truy vấn-tài liệu cùng nhau, giúp nắm bắt sự liên quan chi tiết mà bi-encoder đã bỏ lỡ. Giữ lại top-5.

### Bước 5: đánh giá

| Chỉ số | Ý nghĩa |
|--------|---------|
| Recall@k | Trong số các truy vấn có tài liệu đúng tồn tại, bao nhiêu lần nó nằm trong top-k? |
| MRR (Mean Reciprocal Rank) | Trung bình của 1/thứ hạng của tài liệu liên quan đầu tiên. |
| nDCG@k | Tính đến các mức độ liên quan, không chỉ là nhị phân liên quan/không liên quan. |

Đối với RAG nói riêng, **Recall@k** của bộ truy xuất là con số quan trọng nhất. Người đọc của bạn không thể trả lời nếu đoạn văn đúng không nằm trong tập hợp được truy xuất.

Mẹo gỡ lỗi: đối với các truy vấn thất bại, hãy so sánh sự khác biệt giữa bảng xếp hạng sparse và dense. Nếu một bên tìm thấy tài liệu đúng còn bên kia thì không, bạn đang gặp vấn đề về lệch từ vựng (cách sửa: thêm phần còn thiếu) hoặc mơ hồ về ngữ nghĩa (cách sửa: embedding tốt hơn hoặc reranker).

## Sử dụng

Stack năm 2026:

| Quy mô | Stack |
|-------|-------|
| 1k-100k tài liệu | In-memory BM25 + `all-MiniLM-L6-v2` embeddings + RRF. Không cần DB riêng. |
| 100k-10M tài liệu | FAISS hoặc pgvector cho dense + Elasticsearch / OpenSearch cho BM25. Chạy song song. |
| 10M+ tài liệu | Qdrant / Weaviate / Vespa / Milvus với hỗ trợ hybrid. Cross-encoder rerank trên top-30. |
| Chất lượng tốt nhất | Ba chiều (BM25 + dense + SPLADE) + ColBERT late-interaction reranking |

Dù bạn chọn gì, hãy dành ngân sách cho việc đánh giá. Hãy benchmark recall của việc truy xuất trước khi benchmark độ chính xác của RAG end-to-end. Người đọc không thể sửa chữa những gì bộ truy xuất đã bỏ lỡ.

### Những bài học đắt giá từ RAG production năm 2026

- **80% lỗi RAG bắt nguồn từ khâu nhập liệu và chunking, không phải từ model.** Các đội ngũ dành hàng tuần để thay đổi LLM và tinh chỉnh prompt trong khi việc truy xuất vẫn âm thầm trả về ngữ cảnh sai ở mỗi truy vấn thứ ba. Hãy sửa chunking trước.
- **Chiến lược chunking quan trọng hơn kích thước chunk.** Chia nhỏ theo kích thước cố định làm hỏng các bảng, mã nguồn và các tiêu đề lồng nhau. Chunking nhận biết câu (sentence-aware) là mặc định; chunking dựa trên ngữ nghĩa hoặc LLM mang lại hiệu quả cao cho tài liệu kỹ thuật và hướng dẫn sản phẩm.
- **Mô hình parent-doc.** Truy xuất các "child" chunk nhỏ để có độ chính xác. Khi nhiều child từ cùng một phần parent xuất hiện, hãy thay thế bằng khối parent để bảo toàn ngữ cảnh. Điều này giúp nâng cao chất lượng câu trả lời một cách nhất quán mà không cần huấn luyện lại.
- **k_rerank=3 thường là tối ưu.** Mỗi chunk thêm vào sau đó chỉ làm tăng chi phí token và độ trễ tạo phản hồi mà không nâng cao chất lượng câu trả lời. Nếu k=8 vẫn tốt hơn k=3 đối với bạn, nghĩa là reranker đang hoạt động kém hiệu quả.
- **HyDE / mở rộng truy vấn.** Tạo một câu trả lời giả định từ truy vấn, nhúng nó, sau đó truy xuất. Thu hẹp khoảng cách diễn đạt giữa các câu hỏi ngắn và tài liệu dài. Tăng độ chính xác miễn phí mà không cần huấn luyện.
- **Ngân sách ngữ cảnh dưới 8K token.** Nếu các kết quả khớp nhất quán ở giới hạn đó, nghĩa là ngưỡng reranker đang quá lỏng lẻo.
- **Version mọi thứ.** Prompt, quy tắc chunking, mô hình embedding, reranker. Bất kỳ sự trôi dạt (drift) nào cũng sẽ âm thầm làm hỏng chất lượng câu trả lời. Các cổng CI kiểm tra tính trung thực, độ chính xác ngữ cảnh và tỷ lệ câu hỏi không được trả lời sẽ ngăn chặn các lỗi hồi quy trước khi người dùng nhìn thấy.
- **Truy xuất ba chiều (BM25 + dense + learned-sparse như SPLADE) vượt trội hơn hai chiều** trên các benchmark năm 2026, đặc biệt là với các truy vấn kết hợp danh từ riêng với ngữ nghĩa. Hãy triển khai khi cơ sở hạ tầng hỗ trợ các chỉ mục SPLADE.

Thiết kế truy xuất đúng cách giúp giảm 70-90% hiện tượng ảo giác (hallucinations) theo các đo lường trong ngành năm 2026. Hầu hết các cải thiện hiệu suất RAG đến từ việc truy xuất tốt hơn, không phải từ việc tinh chỉnh model.

## Triển khai

Lưu dưới dạng `outputs/skill-retrieval-picker.md`:

```markdown
---
name: retrieval-picker
description: Pick a retrieval stack for a given corpus and query pattern.
version: 1.0.0
phase: 5
lesson: 14
tags: [nlp, retrieval, rag, search]
---

Given requirements (corpus size, query pattern, latency budget, quality bar, infra constraints), output:

1. Stack. BM25 only, dense only, hybrid (BM25 + dense + RRF), hybrid + cross-encoder rerank, or three-way (BM25 + dense + learned-sparse).
2. Dense encoder. Name the specific model. Match to language(s), domain, and context length.
3. Reranker. Name the specific cross-encoder model if used. Flag that rerank adds 30-100ms latency on top-30.
4. Evaluation plan. Recall@10 is the primary retriever metric. MRR for multi-answer. Baseline first, incremental improvements measured against it.

Refuse to recommend dense-only for corpora with named entities, error codes, or product SKUs unless the user has evidence dense handles exact matches. Refuse to skip reranking for high-stakes retrieval (legal, medical) where the final top-5 decides the user's answer.
```

## Bài tập

1. **Dễ.** Triển khai `hybrid_search` ở trên trên một tập dữ liệu 500 tài liệu. Kiểm tra 20 truy vấn. So sánh recall tại 5 giữa BM25-only, dense-only và hybrid.
2. **Trung bình.** Thêm tính toán MRR. Đối với mỗi truy vấn kiểm tra có tài liệu đúng đã biết, hãy tìm thứ hạng của tài liệu đúng trong các bảng xếp hạng BM25, dense và hybrid. Báo cáo MRR cho từng loại.
3. **Khó.** Tinh chỉnh một dense encoder trên lĩnh vực của bạn bằng MultipleNegativesRankingLoss (Sentence Transformers). Xây dựng tập huấn luyện từ 500 cặp truy vấn-tài liệu. So sánh recall trước và sau khi tinh chỉnh.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| BM25 | Tìm kiếm từ khóa | Okapi BM25. Chấm điểm tài liệu theo tần suất thuật ngữ, IDF và độ dài. |
| Dense retrieval | Tìm kiếm vector | Mã hóa truy vấn + tài liệu thành vector, tìm láng giềng gần nhất. |
| Bi-encoder | Mô hình embedding | Mã hóa truy vấn và tài liệu độc lập. Nhanh tại thời điểm truy vấn. |
| Cross-encoder | Mô hình reranker | Mã hóa truy vấn + tài liệu cùng nhau. Chậm nhưng chính xác. |
| RRF | Rank fusion | Kết hợp hai bảng xếp hạng bằng cách cộng `1/(k + rank)`. |
| Recall@k | Chỉ số truy xuất | Tỷ lệ các truy vấn mà tài liệu liên quan nằm trong top-k. |

## Đọc thêm

- [Robertson and Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) — tài liệu chuẩn mực về BM25.
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) — DPR, bi-encoder kinh điển.
- [Formal et al. (2021). SPLADE: Sparse Lexical and Expansion Model](https://arxiv.org/abs/2107.05720) — bộ truy xuất sparse đã học giúp thu hẹp khoảng cách với dense.
- [Cormack, Clarke, Büttcher (2009). Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) — bài báo về RRF.
- [Khattab and Zaharia (2020). ColBERT: Efficient and Effective Passage Search](https://arxiv.org/abs/2004.12832) — truy xuất tương tác muộn (late-interaction).