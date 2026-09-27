# Bag of Words, TF-IDF và Biểu diễn văn bản

> Đếm trước, nghĩ sau. TF-IDF vẫn vượt trội hơn các embedding trên các tác vụ được xác định rõ ràng vào năm 2026.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (Xử lý văn bản), Phase 2 · 02 (Hồi quy tuyến tính từ đầu)
**Time:** ~75 phút

## Vấn đề

Mô hình cần các con số. Bạn lại có các chuỗi ký tự.

Mọi pipeline NLP đều phải trả lời cùng một câu hỏi: Làm thế nào để biến một luồng token có độ dài thay đổi thành một vector có kích thước cố định mà bộ phân loại có thể tiêu thụ được? Câu trả lời đầu tiên mà lĩnh vực này tìm ra là cách đơn giản nhất nhưng hiệu quả. Đếm các từ. Tạo một vector.

Vector đó đã gánh vác nhiều tác vụ NLP thực tế hơn bất kỳ mô hình embedding nào. Bộ lọc thư rác, bộ phân loại chủ đề, phát hiện bất thường trong log, xếp hạng tìm kiếm (trước khi có BM25), làn sóng phân tích cảm xúc đầu tiên, thập kỷ đầu tiên của các benchmark NLP học thuật. Các kỹ sư năm 2026 vẫn ưu tiên sử dụng nó cho các tác vụ phân loại hẹp. Nó nhanh, dễ giải thích và thường không thể phân biệt được với một mô hình embedding 400 triệu tham số trên các tác vụ mà sự hiện diện của từ ngữ mới là yếu tố quan trọng.

Bài học này xây dựng Bag of Words, sau đó là TF-IDF, từ đầu. Sau đó, nó cho thấy scikit-learn thực hiện điều tương tự chỉ trong ba dòng code. Cuối cùng, nó chỉ ra các trường hợp thất bại khiến bạn phải tìm đến các embedding.

## Khái niệm

**Bag of Words (BoW)** loại bỏ thứ tự. Đối với mỗi tài liệu, hãy đếm số lần mỗi từ trong từ vựng xuất hiện. Độ dài vector chính là kích thước từ vựng. Vị trí `i` là số lần xuất hiện của từ `i`.

**TF-IDF** tái trọng số BoW. Một từ xuất hiện trong mọi tài liệu là từ không mang nhiều thông tin, vì vậy hãy giảm trọng số của nó. Một từ hiếm trong toàn bộ corpus nhưng lại xuất hiện thường xuyên trong một tài liệu đơn lẻ là tín hiệu quan trọng, vì vậy hãy tăng trọng số của nó.

```
TF-IDF(w, d) = TF(w, d) * IDF(w)
             = count(w in d) / |d| * log(N / df(w))
```

Trong đó `TF` là tần suất thuật ngữ trong tài liệu, `df` là tần suất tài liệu (số lượng tài liệu chứa từ đó), `N` là tổng số tài liệu. `log` giữ cho trọng số bị chặn đối với các từ phổ biến.

Đặc tính chính: cả hai đều tạo ra các vector thưa (sparse) với các trục có thể giải thích được. Bạn có thể nhìn vào trọng số của một bộ phân loại đã huấn luyện và đọc xem từ nào đẩy một tài liệu về phía lớp nào. Bạn không thể làm điều này với một BERT embedding 768 chiều.

```figure
bow-tfidf
```

## Xây dựng

### Bước 1: xây dựng từ vựng

```python
def build_vocab(docs):
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab
```

Đầu vào: danh sách các tài liệu đã được token hóa (bất kỳ bộ token hóa cấp từ nào cũng được; `code/main.py` trong bài học này sử dụng một biến thể viết thường đơn giản hóa). Đầu ra: dict `{word: index}`. Thứ tự chèn ổn định có nghĩa là chỉ số từ 0 là từ đầu tiên được nhìn thấy trong tài liệu đầu tiên. Quy ước có thể thay đổi; scikit-learn sắp xếp theo thứ tự bảng chữ cái.

### Bước 2: bag of words

```python
def bag_of_words(docs, vocab):
    matrix = [[0] * len(vocab) for _ in docs]
    for i, doc in enumerate(docs):
        for token in doc:
            if token in vocab:
                matrix[i][vocab[token]] += 1
    return matrix
```

```python
>>> docs = [["cat", "sat", "on", "mat"], ["cat", "cat", "ran"]]
>>> vocab = build_vocab(docs)
>>> bag_of_words(docs, vocab)
[[1, 1, 1, 1, 0], [2, 0, 0, 0, 1]]
```

Các hàng là tài liệu. Các cột là chỉ số từ vựng. Mục `[i][j]` là "số lần từ `j` xuất hiện trong tài liệu `i`." Tài liệu 1 có `cat` hai lần vì nó thực sự xuất hiện như vậy. Tài liệu 0 có `ran` không lần nào vì nó không xuất hiện.

### Bước 3: tần suất thuật ngữ và tần suất tài liệu

```python
import math


def term_frequency(doc_bow, doc_length):
    return [c / doc_length if doc_length else 0 for c in doc_bow]


def document_frequency(bow_matrix):
    df = [0] * len(bow_matrix[0])
    for row in bow_matrix:
        for j, count in enumerate(row):
            if count > 0:
                df[j] += 1
    return df


def inverse_document_frequency(df, n_docs):
    return [math.log((n_docs + 1) / (d + 1)) + 1 for d in df]
```

Hai thủ thuật làm mịn cần lưu ý. `(n+1)/(d+1)` tránh `log(x/0)`. `+1` ở cuối đảm bảo một từ xuất hiện trong mọi tài liệu vẫn có IDF là 1 (không phải 0), khớp với mặc định của scikit-learn. Các triển khai khác sử dụng `log(N/df)` thô. Cả hai đều hoạt động; phiên bản đã làm mịn thân thiện hơn.

### Bước 4: TF-IDF

```python
def tfidf(bow_matrix):
    n_docs = len(bow_matrix)
    df = document_frequency(bow_matrix)
    idf = inverse_document_frequency(df, n_docs)
    out = []
    for row in bow_matrix:
        length = sum(row)
        tf = term_frequency(row, length)
        out.append([tf_j * idf_j for tf_j, idf_j in zip(tf, idf)])
    return out
```

```python
>>> docs = [
...     ["the", "cat", "sat"],
...     ["the", "dog", "sat"],
...     ["the", "cat", "ran"],
... ]
>>> vocab = build_vocab(docs)
>>> bow = bag_of_words(docs, vocab)
>>> tfidf(bow)
```

Ba tài liệu, năm từ vựng (`the`, `cat`, `sat`, `dog`, `ran`). `the` xuất hiện trong cả ba, vì vậy IDF của nó thấp. `dog` xuất hiện trong một, vì vậy IDF của nó cao. Các vector rất thưa (hầu hết các mục đều nhỏ) và các từ mang tính phân biệt cao sẽ nổi bật.

### Bước 5: L2-normalize các hàng

```python
def l2_normalize(matrix):
    out = []
    for row in matrix:
        norm = math.sqrt(sum(x * x for x in row))
        out.append([x / norm if norm else 0 for x in row])
    return out
```

Nếu không chuẩn hóa, một tài liệu dài hơn sẽ có vector lớn hơn và làm lu mờ các điểm số tương đồng. Chuẩn hóa L2 đưa mọi tài liệu lên siêu cầu đơn vị. Độ tương đồng Cosine giữa các hàng bây giờ chỉ đơn giản là tích vô hướng.

## Sử dụng

scikit-learn cung cấp phiên bản sẵn sàng cho sản xuất.

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

docs = ["the cat sat on the mat", "the dog sat on the mat", "the cat ran"]

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(docs)
print(bow_vectorizer.get_feature_names_out())
print(bow.toarray())

tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(docs)
print(tfidf.toarray().round(3))
```

`CountVectorizer` thực hiện token hóa, từ vựng và BoW trong một lệnh gọi. `TfidfVectorizer` thêm trọng số IDF và chuẩn hóa L2. Cả hai đều trả về các ma trận thưa. Đối với 100 nghìn tài liệu, phiên bản dày (dense) sẽ không vừa bộ nhớ; hãy giữ ở dạng thưa cho đến khi bộ phân loại yêu cầu dạng dày.

Các núm điều chỉnh thay đổi mọi thứ:

| Tham số | Hiệu ứng |
|-----|--------|
| `ngram_range=(1, 2)` | Bao gồm bigram. Thường cải thiện phân loại. |
| `min_df=2` | Loại bỏ các từ xuất hiện trong ít hơn 2 tài liệu. Cắt tỉa từ vựng trên dữ liệu nhiễu. |
| `max_df=0.95` | Loại bỏ các từ xuất hiện trong hơn 95% tài liệu. Xấp xỉ việc loại bỏ stopword mà không cần danh sách cứng. |
| `stop_words="english"` | Danh sách stopword tích hợp của scikit-learn. Phụ thuộc vào tác vụ — phân tích cảm xúc *không nên* loại bỏ các từ phủ định. |
| `sublinear_tf=True` | Sử dụng `1 + log(tf)` thay vì `tf` thô. Hữu ích khi một thuật ngữ lặp lại nhiều lần trong một tài liệu. |

### Khi nào TF-IDF vẫn thắng (tính đến năm 2026)

- Phát hiện thư rác, gắn nhãn chủ đề, gắn cờ bất thường trong log. Sự hiện diện của từ ngữ là điều quan trọng; sắc thái ngữ nghĩa không quan trọng.
- Các chế độ dữ liệu thấp (hàng trăm ví dụ được gắn nhãn). TF-IDF cộng với hồi quy logistic không tốn chi phí tiền huấn luyện.
- Bất cứ nơi nào độ trễ là quan trọng. TF-IDF cộng với mô hình tuyến tính trả về kết quả trong micro giây. Việc nhúng một tài liệu qua transformer mất 10-100ms.
- Các hệ thống phải giải thích được dự đoán của chúng. Kiểm tra các hệ số của bộ phân loại. Các từ tích cực hàng đầu chính là lý do.

### Khi nào TF-IDF thất bại

Sự thất bại do mù ngữ nghĩa. Hãy xem xét hai tài liệu này:

- "The movie was not good at all."
- "The movie was excellent."

Một là đánh giá tiêu cực. Một là tích cực. Sự trùng lặp TF-IDF của chúng chính xác là `{the, movie, was}`. Một bộ phân loại bag-of-words phải ghi nhớ rằng từ `not` gần `good` sẽ đảo ngược nhãn. Nó có thể học điều này nếu có đủ dữ liệu, nhưng không bao giờ tinh tế bằng một mô hình hiểu được cú pháp.

Sự thất bại khác: các từ nằm ngoài từ vựng (out-of-vocabulary) khi suy luận. Một mô hình BoW được huấn luyện trên các đánh giá IMDb không biết phải làm gì với `Zoomer-approved` nếu token đó chưa bao giờ xuất hiện trong quá trình huấn luyện. Các subword embedding (bài học 04) xử lý được điều này. TF-IDF thì không.

### Hybrid: TF-IDF weighted embeddings

Lựa chọn mặc định thực dụng năm 2026 cho phân loại dữ liệu trung bình: sử dụng trọng số TF-IDF làm cơ chế attention trên các word embedding.

```python
def tfidf_weighted_embedding(doc, tfidf_scores, embedding_table, dim):
    vec = [0.0] * dim
    total_weight = 0.0
    for token in doc:
        if token not in embedding_table or token not in tfidf_scores:
            continue
        weight = tfidf_scores[token]
        emb = embedding_table[token]
        for i in range(dim):
            vec[i] += weight * emb[i]
        total_weight += weight
    if total_weight == 0:
        return vec
    return [v / total_weight for v in vec]
```

Bạn có được khả năng ngữ nghĩa từ embedding và sự nhấn mạnh vào các từ hiếm từ TF-IDF. Bộ phân loại huấn luyện trên vector đã được gộp (pooled). Cách tiếp cận này vượt trội hơn cả hai phương pháp riêng lẻ đối với phân tích cảm xúc, chủ đề và phân loại ý định dưới khoảng 50 nghìn ví dụ được gắn nhãn.

## Triển khai

Lưu dưới dạng `outputs/prompt-vectorization-picker.md`:

```markdown
---
name: vectorization-picker
description: Given a text-classification task, recommend BoW, TF-IDF, embeddings, or a hybrid.
phase: 5
lesson: 02
---

You recommend a text-vectorization strategy. Given a task description, output:

1. Representation (BoW, TF-IDF, transformer embeddings, or a hybrid). Explain why in one sentence.
2. Specific vectorizer configuration. Name the library. Quote the arguments (`ngram_range`, `min_df`, `max_df`, `sublinear_tf`, `stop_words`).
3. One failure mode to test before shipping.

Refuse to recommend embeddings when the user has under 500 labeled examples unless they show evidence of semantic failure in a TF-IDF baseline. Refuse to remove stopwords for sentiment analysis (negations carry signal). Flag class imbalance as needing more than a vectorizer change.

Example input: "Classifying 30k customer support tickets into 12 categories. Most tickets are 2-3 sentences. English only. Need explainability for audit logs."

Example output:

- Representation: TF-IDF. 30k examples is not small; explainability requirement rules out dense embeddings.
- Config: `TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True, stop_words=None)`. Keep stopwords because category keywords sometimes are stopwords ("not working" vs "working").
- Failure to test: verify `min_df=3` does not drop rare category keywords. Run `get_feature_names_out` filtered by class and eyeball.
```

## Bài tập

1. **Dễ.** Triển khai `cosine_similarity(doc_vec_a, doc_vec_b)` trên đầu ra TF-IDF đã chuẩn hóa L2. Xác minh rằng các tài liệu giống hệt nhau có điểm số 1.0 và các tài liệu có từ vựng rời rạc có điểm số 0.0.
2. **Trung bình.** Thêm hỗ trợ `n-gram` vào `bag_of_words`. Tham số `n` tạo ra các số đếm trên `n`-gram. Kiểm tra xem `n=2` trên `["the", "cat", "sat"]` tạo ra số đếm bigram cho `["the cat", "cat sat"]`.
3. **Khó.** Xây dựng mô hình hybrid TF-IDF-weighted-embedding ở trên bằng cách sử dụng các vector GloVe 100d (tải xuống một lần, lưu vào bộ nhớ đệm). So sánh độ chính xác phân loại với TF-IDF thuần túy và mean-pooled embedding thuần túy trên tập dữ liệu 20 Newsgroups. Báo cáo phương pháp nào thắng ở đâu.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| BoW | Vector tần suất từ | Số lần xuất hiện của các từ vựng trong một tài liệu. Loại bỏ thứ tự. |
| TF | Tần suất thuật ngữ | Số lần xuất hiện của một từ trong tài liệu, tùy chọn chuẩn hóa theo độ dài tài liệu. |
| DF | Tần suất tài liệu | Số lượng tài liệu chứa từ đó ít nhất một lần. |
| IDF | Tần suất tài liệu nghịch đảo | `log(N / df)` đã làm mịn. Giảm trọng số các từ xuất hiện ở khắp mọi nơi. |
| Sparse vector | Hầu hết là số 0 | Từ vựng thường là 10k-100k từ; hầu hết đều vắng mặt trong bất kỳ tài liệu nào. |
| Cosine similarity | Góc vector | Tích vô hướng của các vector đã chuẩn hóa L2. 1 là giống hệt, 0 là trực giao. |

## Đọc thêm

- [scikit-learn — trích xuất đặc trưng từ văn bản](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — tài liệu tham khảo API chính thức, cộng với ghi chú về mọi núm điều chỉnh.
- [Salton, G., & Buckley, C. (1988). Các phương pháp trọng số thuật ngữ trong truy xuất văn bản tự động](https://www.sciencedirect.com/science/article/pii/0306457388900210) — bài báo đã biến TF-IDF thành mặc định trong một thập kỷ.
- ["Tại sao TF-IDF vẫn vượt trội hơn Embeddings" — Ashfaque Thonikkadavan (Medium)](https://medium.com/@cmtwskb/why-tf-idf-still-beats-embeddings-ad85c123e1b2) — góc nhìn năm 2026 về thời điểm phương pháp cũ chiến thắng và lý do tại sao.