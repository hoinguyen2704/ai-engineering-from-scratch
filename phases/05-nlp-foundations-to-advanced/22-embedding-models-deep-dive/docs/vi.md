# Embedding Models — The 2026 Deep Dive

> Word2Vec cung cấp cho bạn một vector cho mỗi từ. Các embedding model hiện đại cung cấp cho bạn một vector cho mỗi đoạn văn, hỗ trợ đa ngôn ngữ, với các dạng sparse, dense và multi-vector, được tùy chỉnh kích thước để phù hợp với index của bạn. Chọn sai model đồng nghĩa với việc RAG của bạn sẽ truy xuất sai thông tin.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 03 (Word2Vec), Phase 5 · 14 (Information Retrieval)
**Time:** ~60 minutes

## Vấn đề

Hệ thống RAG của bạn truy xuất sai đoạn văn trong 40% trường hợp. Thủ phạm hiếm khi là vector database hay prompt, mà chính là embedding model.

Việc chọn một embedding vào năm 2026 đồng nghĩa với việc cân nhắc trên năm trục:

1. **Dense vs sparse vs multi-vector.** Một vector cho mỗi đoạn văn, hoặc một vector cho mỗi token, hoặc một túi từ (bag of words) thưa thớt có trọng số.
2. **Độ bao phủ ngôn ngữ.** Các model đơn ngữ tiếng Anh vẫn thắng thế trong các tác vụ chỉ dùng tiếng Anh. Các model đa ngôn ngữ thắng thế khi dữ liệu đầu vào hỗn hợp.
3. **Độ dài ngữ cảnh (Context length).** 512 token so với 8,192 hoặc 32,768 — và dung lượng thực tế thường chỉ đạt 60-70% so với mức tối đa được quảng cáo.
4. **Ngân sách chiều dữ liệu (Dimension budget).** 3,072 số thực ở độ chính xác đầy đủ = 12 KB mỗi vector. Với 100 triệu vector, chi phí lưu trữ là $1,300/tháng. Kỹ thuật cắt tỉa Matryoshka giúp giảm con số này xuống 4 lần.
5. **Open vs hosted.** Open-weight nghĩa là bạn kiểm soát stack và dữ liệu. Hosted nghĩa là bạn đánh đổi quyền kiểm soát để luôn có công nghệ mới nhất.

Bài học này nêu tên các sự đánh đổi để bạn có thể lựa chọn dựa trên bằng chứng, thay vì chọn theo xu hướng của quý trước.

## Khái niệm

![Dense, sparse, and multi-vector embeddings](../assets/embedding-modes.svg)

**Dense embeddings.** Một vector cho mỗi đoạn văn (thường là 384-3,072 chiều). Độ tương đồng cosine xếp hạng các đoạn văn theo sự gần gũi về ngữ nghĩa. OpenAI `text-embedding-3-large`, BGE-M3 chế độ dense, Voyage-3. Lựa chọn mặc định.

**Sparse embeddings.** Kiểu SPLADE. Một Transformer dự đoán trọng số cho mỗi token trong từ vựng, sau đó đặt hầu hết chúng về 0. Kết quả là một vector thưa thớt có kích thước |từ vựng|. Nắm bắt được sự khớp từ vựng (giống BM25) nhưng với trọng số từ được học. Rất mạnh với các truy vấn chứa nhiều từ khóa.

**Multi-vector (late interaction).** ColBERTv2, Jina-ColBERT. Một vector cho mỗi token. Chấm điểm bằng MaxSim: với mỗi token truy vấn, tìm token tài liệu tương đồng nhất, sau đó cộng các điểm số lại. Tốn kém hơn về lưu trữ và chấm điểm, nhưng thắng thế ở các truy vấn dài và dữ liệu chuyên biệt.

**BGE-M3: cả ba trong một.** Một model duy nhất xuất ra các biểu diễn dense, sparse và multi-vector cùng lúc. Mỗi loại có thể được truy vấn độc lập; điểm số được hợp nhất thông qua tổng có trọng số. Lựa chọn mặc định năm 2026 khi bạn muốn sự linh hoạt từ một checkpoint.

**Matryoshka Representation Learning.** Được huấn luyện sao cho N chiều đầu tiên của vector tạo thành một embedding độc lập hữu ích. Cắt tỉa một vector 1,536 chiều xuống còn 256 chiều và chỉ mất ~1% độ chính xác để tiết kiệm 6 lần dung lượng lưu trữ. Được hỗ trợ bởi OpenAI text-3, Cohere v4, Voyage-4, Jina v5, Gemini Embedding 2, Nomic v1.5+.

### Bảng xếp hạng MTEB chỉ kể một phần câu chuyện

Massive Text Embedding Benchmark — 56 tác vụ trên 8 loại tác vụ khi ra mắt (2022), mở rộng lên hơn 100 tác vụ trong MTEB v2. Đầu năm 2026, Gemini Embedding 2 đứng đầu về truy xuất (67.71 MTEB-R). Cohere embed-v4 dẫn đầu về tổng quát (65.2 MTEB). BGE-M3 dẫn đầu về open-weight đa ngôn ngữ (63.0). Bảng xếp hạng là cần thiết nhưng chưa đủ — hãy luôn benchmark trên chính dữ liệu của bạn.

### Mô hình ba tầng

| Use case | Pattern |
|----------|---------|
| Fast first-pass | Dense bi-encoder (BGE-M3, text-3-small) |
| Recall boost | Sparse (SPLADE, BGE-M3 sparse) + RRF fuse |
| Precision on top-50 | Multi-vector (ColBERTv2) hoặc cross-encoder reranker |

Hầu hết các stack sản xuất đều sử dụng cả ba.

```figure
gx-matryoshka
```

## Xây dựng

### Bước 1: baseline — dense embeddings với Sentence-BERT

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
corpus = [
    "The first iPhone launched in 2007.",
    "Apple released the iPod in 2001.",
    "Android is an operating system from Google.",
]
emb = encoder.encode(corpus, normalize_embeddings=True)

query = "When was the iPhone released?"
q_emb = encoder.encode([query], normalize_embeddings=True)[0]
scores = emb @ q_emb
print(sorted(enumerate(scores), key=lambda x: -x[1]))
```

`normalize_embeddings=True` làm cho tích vô hướng bằng với độ tương đồng cosine. Luôn luôn thiết lập nó.

### Bước 2: Cắt tỉa Matryoshka

```python
def truncate(vectors, dim):
    out = vectors[:, :dim]
    return out / np.linalg.norm(out, axis=1, keepdims=True)

emb_256 = truncate(emb, 256)
emb_128 = truncate(emb, 128)
```

Chuẩn hóa lại sau khi cắt tỉa. Nomic v1.5, OpenAI text-3 và Voyage-4 được huấn luyện để việc này không gây mất mát thông tin ở các cấp độ đầu. Các model không phải Matryoshka (Sentence-BERT gốc) sẽ bị suy giảm độ chính xác nghiêm trọng khi bị cắt tỉa.

### Bước 3: Đa chức năng của BGE-M3

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

output = model.encode(
    corpus,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=True,
)
# output["dense_vecs"]:    (n_docs, 1024)
# output["lexical_weights"]: list of dict {token_id: weight}
# output["colbert_vecs"]:  list of (n_tokens, 1024) arrays
```

Ba index, một lệnh gọi inference. Hợp nhất điểm số:

```python
dense_score = ... # cosine over dense_vecs
sparse_score = model.compute_lexical_matching_score(q_lex, d_lex)
colbert_score = model.colbert_score(q_col, d_col)
final = 0.4 * dense_score + 0.2 * sparse_score + 0.4 * colbert_score
```

Điều chỉnh trọng số trên dữ liệu của bạn.

### Bước 4: Đánh giá MTEB trên tác vụ tùy chỉnh

```python
from mteb import MTEB

tasks = ["ArguAna", "SciFact", "NFCorpus"]
evaluation = MTEB(tasks=tasks)
results = evaluation.run(encoder, output_folder="./mteb-results")
```

Chạy các model ứng viên của bạn trên một tập con *đại diện*. Đừng chỉ tin vào thứ hạng trên bảng xếp hạng — dữ liệu của bạn mới là quan trọng.

### Bước 5: Tự viết hàm cosine từ đầu

Xem `code/main.py`. Embedding theo Averaged Hashing Trick (chỉ dùng stdlib). Không cạnh tranh được với các embedding từ Transformer, nhưng cho thấy hình thái: tokenize → vector → chuẩn hóa → tích vô hướng.

## Những cạm bẫy

- **Dùng cùng một model cho truy vấn và tài liệu.** Một số model (Voyage, Jina-ColBERT) sử dụng mã hóa bất đối xứng (asymmetric encoding) — truy vấn và tài liệu đi qua các đường dẫn khác nhau. Luôn kiểm tra model card.
- **Thiếu tiền tố (prefix).** Các model `bge-*` cần `"Represent this sentence for searching relevant passages: "` được thêm vào trước các truy vấn. Sẽ mất 3-5 điểm recall nếu bạn quên.
- **Cắt tỉa Matryoshka quá mức.** 1,536 → 256 thường là an toàn. 1,536 → 64 thì không. Hãy kiểm chứng trên tập đánh giá của bạn.
- **Cắt ngắn ngữ cảnh.** Hầu hết các model sẽ âm thầm cắt ngắn đầu vào vượt quá độ dài tối đa. Các tài liệu dài cần được chia nhỏ (xem bài 23).
- **Bỏ qua độ trễ đuôi (latency tail).** Điểm MTEB che giấu độ trễ p99. Một model 600M có thể thắng model 335M 2 điểm nhưng tốn kém gấp 3 lần cho mỗi truy vấn.

## Sử dụng

Stack năm 2026:

| Tình huống | Lựa chọn |
|-----------|------|
| Chỉ tiếng Anh, nhanh, API | `text-embedding-3-large` hoặc `voyage-3-large` |
| Open-weight, tiếng Anh | `BAAI/bge-large-en-v1.5` |
| Open-weight, đa ngôn ngữ | `BAAI/bge-m3` hoặc `Qwen3-Embedding-8B` |
| Ngữ cảnh dài (32k+) | Voyage-3-large, Cohere embed-v4, Qwen3-Embedding-8B |
| Triển khai chỉ dùng CPU | Nomic Embed v2 (137M tham số, MoE) |
| Hạn chế lưu trữ | Matryoshka-truncated + int8 quantization |
| Truy vấn nhiều từ khóa | Thêm SPLADE sparse, RRF-fuse với dense |

Mô hình 2026: bắt đầu với BGE-M3 hoặc text-3-large, đánh giá trên dữ liệu của bạn với MTEB, thay thế nếu một model chuyên biệt thắng hơn 3 điểm.

## Triển khai

Lưu dưới dạng `outputs/skill-embedding-picker.md`:

```markdown
---
name: embedding-picker
description: Pick embedding model, dimension, and retrieval mode for a given corpus and deployment.
version: 1.0.0
phase: 5
lesson: 22
tags: [nlp, embeddings, retrieval]
---

Given a corpus (size, languages, domain, avg length), deployment target (cloud / edge / on-prem), latency budget, and storage budget, output:

1. Model. Named checkpoint or API. One-sentence reason.
2. Dimension. Full / Matryoshka-truncated / int8-quantized. Reason tied to storage budget.
3. Mode. Dense / sparse / multi-vector / hybrid. Reason.
4. Query prefix / template if required by the model card.
5. Evaluation plan. MTEB tasks relevant to domain + held-out domain eval with nDCG@10.

Refuse recommendations that truncate Matryoshka to <64 dims without domain validation. Refuse ColBERTv2 for corpora under 10k passages (overhead not justified). Flag long-document corpora (>8k tokens) routed to models with 512-token windows.
```

## Bài tập

1. **Dễ.** Encode 100 câu với `bge-small-en-v1.5` ở kích thước đầy đủ (384), sau đó ở Matryoshka 128. Đo lường mức giảm MRR trên 10 truy vấn.
2. **Trung bình.** So sánh BGE-M3 dense, sparse và colbert trên 500 đoạn văn từ dữ liệu của bạn. Cái nào thắng về recall@10? Liệu RRF fusion có thắng được chế độ đơn lẻ tốt nhất không?
3. **Khó.** Chạy MTEB trên ba model ứng viên cho hai tác vụ hàng đầu trong lĩnh vực của bạn. Báo cáo điểm MTEB, độ trễ p99 trên batch 100 truy vấn và chi phí $/1M truy vấn. Chọn model tối ưu nhất theo Pareto.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| Dense embedding | Vector | Một vector kích thước cố định cho mỗi văn bản. Dùng độ tương đồng cosine để xếp hạng. |
| Sparse embedding | BM25 đã học | Một trọng số cho mỗi token từ vựng; hầu hết là số 0; được huấn luyện end-to-end. |
| Multi-vector | Kiểu ColBERT | Một vector cho mỗi token; chấm điểm MaxSim; index lớn hơn, recall tốt hơn. |
| Matryoshka | Thủ thuật búp bê Nga | N chiều đầu tiên tự nó đã là một embedding nhỏ hợp lệ. |
| MTEB | Benchmark | Massive Text Embedding Benchmark — 56 tác vụ khi ra mắt, 100+ trong v2. |
| BEIR | Benchmark truy xuất | 18 tác vụ truy xuất zero-shot; thường được trích dẫn về độ bền vững xuyên lĩnh vực. |
| Asymmetric encoding | Đường dẫn truy vấn ≠ tài liệu | Model sử dụng các phép chiếu khác nhau cho truy vấn và tài liệu. |

## Đọc thêm

- [Reimers, Gurevych (2019). Sentence-BERT](https://arxiv.org/abs/1908.10084) — bài báo về bi-encoder.
- [Muennighoff et al. (2022). MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316) — bài báo về bảng xếp hạng.
- [Chen et al. (2024). BGE-M3: Multi-lingual, Multi-functionality, Multi-granularity](https://arxiv.org/abs/2402.03216) — model ba chế độ thống nhất.
- [Kusupati et al. (2022). Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) — mục tiêu huấn luyện theo thang đo chiều.
- [Santhanam et al. (2022). ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction](https://arxiv.org/abs/2112.01488) — late interaction trong sản xuất.
- [Bảng xếp hạng MTEB trên Hugging Face](https://huggingface.co/spaces/mteb/leaderboard) — xếp hạng trực tiếp.