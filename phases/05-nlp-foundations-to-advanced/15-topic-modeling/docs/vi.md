# Topic Modeling — LDA và BERTopic

> LDA: các tài liệu là hỗn hợp của các chủ đề, các chủ đề là phân phối trên các từ. BERTopic: các tài liệu được phân cụm trong không gian embedding, các cụm là các chủ đề. Cùng mục tiêu, khác cách phân rã.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 03 (Word2Vec)
**Time:** ~45 minutes

## Vấn đề

Bạn có 10.000 phiếu hỗ trợ khách hàng, 50.000 bài báo hoặc 200.000 tweet. Bạn cần biết tập dữ liệu đó nói về cái gì mà không cần đọc hết. Bạn không có các danh mục được gán nhãn. Bạn thậm chí không biết có bao nhiêu danh mục tồn tại.

Topic modeling giải quyết vấn đề đó mà không cần giám sát (unsupervised). Hãy cung cấp cho nó một corpus, bạn sẽ nhận lại một tập hợp nhỏ các chủ đề mạch lạc và, đối với mỗi tài liệu, một phân phối trên các chủ đề đó.

Hai họ thuật toán chiếm ưu thế. LDA (2003) coi mỗi tài liệu là một hỗn hợp của các chủ đề tiềm ẩn và mỗi chủ đề là một phân phối trên các từ. Suy luận mang tính Bayesian. Nó vẫn được sử dụng trong môi trường production nơi bạn cần gán chủ đề theo kiểu hỗn hợp (mixed-membership) và các phân phối xác suất ở cấp độ từ ngữ có thể giải thích được.

BERTopic (2020) mã hóa các tài liệu bằng BERT, giảm chiều dữ liệu bằng UMAP, phân cụm bằng HDBSCAN và trích xuất các từ chủ đề thông qua class-based TF-IDF. Nó vượt trội trên văn bản ngắn, mạng xã hội và bất kỳ trường hợp nào mà sự tương đồng về ngữ nghĩa quan trọng hơn sự trùng lặp từ ngữ. Một tài liệu chỉ nhận được một chủ đề, đây là một hạn chế đối với nội dung dài.

Bài học này xây dựng trực giác cho cả hai và chỉ ra cách chọn thuật toán cho một corpus nhất định.

## Khái niệm

![LDA mixture model vs BERTopic clustering](../assets/topic-modeling.svg)

**Câu chuyện tạo sinh của LDA.** Mỗi chủ đề là một phân phối trên các từ. Mỗi tài liệu là một hỗn hợp của các chủ đề. Để tạo ra một từ trong tài liệu, hãy lấy mẫu một chủ đề từ hỗn hợp của tài liệu đó, sau đó lấy mẫu một từ từ phân phối của chủ đề đó. Suy luận thực hiện ngược lại: dựa trên các từ quan sát được, suy ra phân phối chủ đề cho mỗi tài liệu và phân phối từ cho mỗi chủ đề. Collapsed Gibbs sampling hoặc variational Bayes thực hiện các phép tính này.

Đầu ra chính của LDA:

- `doc_topic`: ma trận `(n_docs, n_topics)`, mỗi hàng có tổng bằng 1 (hỗn hợp chủ đề của tài liệu).
- `topic_word`: ma trận `(n_topics, vocab_size)`, mỗi hàng có tổng bằng 1 (phân phối từ của chủ đề).

**Pipeline của BERTopic.**

1. Mã hóa mỗi tài liệu bằng một sentence transformer (ví dụ: `all-MiniLM-L6-v2`). Vector 384 chiều.
2. Giảm chiều dữ liệu bằng UMAP xuống khoảng 5 chiều. Các embedding của BERT có số chiều quá cao để phân cụm.
3. Phân cụm bằng HDBSCAN. Dựa trên mật độ, tạo ra các cụm có kích thước thay đổi và một nhãn "outlier" (ngoại lai).
4. Đối với mỗi cụm, tính toán class-based TF-IDF trên các tài liệu của cụm đó để trích xuất các từ hàng đầu.

Đầu ra là một chủ đề cho mỗi tài liệu (cộng với nhãn outlier -1). Tùy chọn, có thể lấy membership mềm thông qua vector xác suất của HDBSCAN.

```figure
topic-drift
```

## Xây dựng

### Bước 1: LDA thông qua scikit-learn

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np


def fit_lda(documents, n_topics=5, max_features=1000):
    cv = CountVectorizer(
        max_features=max_features,
        stop_words="english",
        min_df=2,
        max_df=0.9,
    )
    X = cv.fit_transform(documents)
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=50,
        learning_method="online",
    )
    doc_topic = lda.fit_transform(X)
    feature_names = cv.get_feature_names_out()
    return lda, cv, doc_topic, feature_names


def print_top_words(lda, feature_names, n_top=10):
    for idx, topic in enumerate(lda.components_):
        top_idx = np.argsort(-topic)[:n_top]
        words = [feature_names[i] for i in top_idx]
        print(f"topic {idx}: {' '.join(words)}")
```

Lưu ý: các stopwords đã được loại bỏ, min_df và max_df lọc các thuật ngữ hiếm và phổ biến, sử dụng CountVectorizer (không phải TfidfVectorizer) vì LDA mong đợi các số đếm thô.

### Bước 2: BERTopic (production)

```python
from bertopic import BERTopic

topic_model = BERTopic(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    min_topic_size=15,
    verbose=True,
)

topics, probs = topic_model.fit_transform(documents)
info = topic_model.get_topic_info()
print(info.head(20))
valid_topics = info[info["Topic"] != -1]["Topic"].tolist()
for topic_id in valid_topics[:5]:
    print(f"topic {topic_id}: {topic_model.get_topic(topic_id)[:10]}")
```

Bộ lọc trên `Topic != -1` loại bỏ các bucket outlier của BERTopic (các tài liệu mà HDBSCAN không thể phân cụm). `min_topic_size` kiểm soát kích thước cụm tối thiểu của HDBSCAN; mặc định của thư viện BERTopic là 10. Ví dụ này đặt rõ ràng là 15 cho quy mô của bài học. Đối với các corpus trên 10.000 tài liệu, hãy tăng lên 50 hoặc 100.

### Bước 3: Đánh giá

Cả hai phương pháp đều xuất ra các từ chủ đề. Câu hỏi là liệu các từ đó có mạch lạc hay không.

- **Topic coherence (c_v).** Kết hợp NPMI (normalized pointwise mutual information) của các cặp từ hàng đầu trong các ngữ cảnh cửa sổ trượt, tổng hợp các điểm số thành các vector chủ đề và so sánh các vector đó thông qua cosine similarity. Điểm càng cao càng tốt. Sử dụng `gensim.models.CoherenceModel` với `coherence="c_v"`.
- **Topic diversity.** Tỷ lệ các từ duy nhất trên tất cả các từ hàng đầu của các chủ đề. Càng cao càng tốt (các chủ đề không trùng lặp).
- **Kiểm tra định tính.** Đọc các từ hàng đầu của mỗi chủ đề. Chúng có gọi tên một thứ gì đó thực tế không? Đánh giá của con người vẫn là tuyến phòng thủ cuối cùng.

## Khi nào chọn phương pháp nào

| Tình huống | Chọn |
|-----------|------|
| Văn bản ngắn (tweet, đánh giá, tiêu đề) | BERTopic |
| Tài liệu dài với hỗn hợp chủ đề | LDA |
| Không có GPU / tài nguyên tính toán hạn chế | LDA hoặc NMF |
| Cần phân phối đa chủ đề ở cấp độ tài liệu | LDA |
| Tích hợp LLM để gán nhãn chủ đề | BERTopic (hỗ trợ trực tiếp) |
| Triển khai edge bị hạn chế tài nguyên | LDA |
| Độ mạch lạc ngữ nghĩa tối đa | BERTopic |

Cân nhắc thực tế lớn nhất là độ dài tài liệu. Các BERT embedding bị cắt bớt; các số đếm của LDA hoạt động trên mọi độ dài. Đối với các tài liệu dài hơn ngữ cảnh của mô hình embedding, hãy chia nhỏ + tổng hợp hoặc sử dụng LDA.

## Sử dụng

Stack năm 2026:

- **BERTopic.** Mặc định cho văn bản ngắn và bất cứ thứ gì mà ngữ nghĩa quan trọng.
- **`gensim.models.LdaModel`.** LDA cổ điển cho production, trưởng thành, đã được kiểm chứng qua thực tế.
- **`sklearn.decomposition.LatentDirichletAllocation`.** LDA dễ sử dụng cho các thử nghiệm.
- **NMF.** Non-negative matrix factorization. Giải pháp thay thế nhanh cho LDA, chất lượng tương đương trên văn bản ngắn.
- **Top2Vec.** Thiết kế tương tự BERTopic. Cộng đồng nhỏ hơn nhưng tốt trên một số benchmark.
- **FASTopic.** Mới hơn, nhanh hơn BERTopic trên các corpus rất lớn.
- **Gán nhãn dựa trên LLM.** Chạy bất kỳ thuật toán phân cụm nào, sau đó yêu cầu một mô hình đặt tên cho mỗi cụm.

## Triển khai

Lưu dưới dạng `outputs/skill-topic-picker.md`:

```markdown
---
name: topic-picker
description: Pick LDA or BERTopic for a corpus. Specify library, knobs, evaluation.
version: 1.0.0
phase: 5
lesson: 15
tags: [nlp, topic-modeling]
---

Given a corpus description (document count, avg length, domain, language, compute budget), output:

1. Algorithm. LDA / NMF / BERTopic / Top2Vec / FASTopic. One-sentence reason.
2. Configuration. Number of topics: `recommended = max(5, round(sqrt(n_docs)))`, clamped to 200 for corpora under 40,000 docs; permit >200 only when the corpus is genuinely large (>40k) and note the increased compute cost. `min_df` / `max_df` filters and embedding model for neural approaches also belong here.
3. Evaluation. Topic coherence (c_v) via `gensim.models.CoherenceModel`, topic diversity, and a 20-sample human read.
4. Failure mode to probe. For LDA, "junk topics" absorbing stopwords and frequent terms. For BERTopic, the -1 outlier cluster swallowing ambiguous documents.

Refuse BERTopic on documents longer than the embedding model's context window without a chunking strategy. Refuse LDA on very short text (tweets, reviews under 10 tokens) as coherence collapses. Flag any n_topics choice below 5 as likely wrong; flag >200 on corpora under 40k docs as likely over-splitting.
```

## Bài tập

1. **Dễ.** Fit LDA với 5 chủ đề trên tập dữ liệu 20 Newsgroups. In ra 10 từ hàng đầu mỗi chủ đề. Gán nhãn thủ công cho mỗi chủ đề. Thuật toán có tìm ra các danh mục thực tế không?
2. **Trung bình.** Fit BERTopic trên cùng tập con 20 Newsgroups. So sánh số lượng chủ đề tìm thấy, các từ hàng đầu và độ mạch lạc định tính so với LDA. Phương pháp nào làm nổi bật các danh mục thực tế rõ ràng hơn?
3. **Khó.** Tính độ mạch lạc c_v cho cả LDA và BERTopic trên corpus của bạn. Chạy mỗi cái với 5, 10, 20, 50 chủ đề. Vẽ biểu đồ độ mạch lạc so với số lượng chủ đề. Báo cáo phương pháp nào ổn định hơn trên các số lượng chủ đề khác nhau.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| Topic | Một chủ đề của corpus | Một phân phối xác suất trên các từ (LDA) hoặc một cụm các tài liệu tương tự (BERTopic). |
| Mixed membership | Tài liệu thuộc nhiều chủ đề | LDA gán cho mỗi tài liệu một phân phối trên tất cả các chủ đề. |
| UMAP | Giảm chiều dữ liệu | Học đa tạp (manifold learning) bảo toàn cấu trúc cục bộ; được sử dụng trong BERTopic. |
| HDBSCAN | Phân cụm mật độ | Tìm các cụm có kích thước thay đổi; tạo nhãn "nhiễu" (-1) cho các outlier. |
| c_v coherence | Chỉ số chất lượng chủ đề | Trung bình pointwise mutual information của các từ chủ đề hàng đầu trong các cửa sổ trượt. |

## Đọc thêm

- [Blei, Ng, Jordan (2003). Latent Dirichlet Allocation](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) — bài báo gốc về LDA.
- [Grootendorst (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure](https://arxiv.org/abs/2203.05794) — bài báo về BERTopic.
- [Röder, Both, Hinneburg (2015). Exploring the Space of Topic Coherence Measures](https://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf) — bài báo giới thiệu c_v và các chỉ số liên quan.
- [Tài liệu BERTopic](https://maartengr.github.io/BERTopic/) — tài liệu tham khảo cho production. Các ví dụ rất xuất sắc.