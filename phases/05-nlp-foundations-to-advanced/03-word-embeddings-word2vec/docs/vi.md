# Word Embeddings — Word2Vec từ đầu

> Một từ được định nghĩa bởi những từ đi cùng nó. Hãy huấn luyện một mạng thần kinh nông dựa trên ý tưởng đó và hình học sẽ xuất hiện.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 3 · 03 (Backpropagation from Scratch)
**Time:** ~75 phút

## Vấn đề

TF-IDF biết `dog` và `puppy` là hai từ khác nhau. Nó không biết rằng chúng có nghĩa gần như tương đương. Một bộ phân loại được huấn luyện trên `dog` không thể tổng quát hóa cho một bài đánh giá về `puppy`. Bạn có thể giải quyết tạm thời bằng cách liệt kê các từ đồng nghĩa, nhưng cách này thất bại với các thuật ngữ hiếm, biệt ngữ chuyên ngành và mọi ngôn ngữ mà bạn không dự đoán trước được.

Bạn muốn một biểu diễn mà ở đó `dog` và `puppy` nằm gần nhau trong không gian. Nơi mà `king - man + woman` nằm gần `queen`. Nơi mà một mô hình được huấn luyện trên `dog` truyền tải một phần tín hiệu sang `puppy` một cách miễn phí.

Word2Vec đã mang lại cho chúng ta không gian đó. Một mạng thần kinh hai lớp, các đợt huấn luyện trên hàng nghìn tỷ token, được công bố vào năm 2013. Kiến trúc này đơn giản đến mức đáng kinh ngạc. Kết quả của nó đã định hình lại NLP trong suốt một thập kỷ.

## Khái niệm

**Giả thuyết phân phối** (Firth, 1957): "Bạn sẽ biết một từ qua những từ đi cùng nó." Nếu hai từ xuất hiện trong các ngữ cảnh tương tự, chúng có khả năng mang ý nghĩa tương tự nhau.

Word2Vec có hai biến thể, cả hai đều khai thác ý tưởng đó.

- **Skip-gram.** Cho trước một từ trung tâm, dự đoán các từ xung quanh. `cat -> (the, sat, on)` với kích thước cửa sổ là 2.
- **CBOW (continuous bag of words).** Cho trước các từ xung quanh, dự đoán từ trung tâm. `(the, sat, on) -> cat`.

Skip-gram huấn luyện chậm hơn nhưng xử lý các từ hiếm tốt hơn. Nó đã trở thành lựa chọn mặc định.

Mạng này có một lớp ẩn không có tính phi tuyến. Đầu vào là một vector one-hot trên từ vựng. Đầu ra là một softmax trên từ vựng. Sau khi huấn luyện, bạn loại bỏ lớp đầu ra. Trọng số của lớp ẩn chính là các embedding.

```
one-hot(center) ── W ──▶ hidden (d-dim) ── W' ──▶ softmax(vocab)
                          ^
                          this is the embedding
```

Mẹo nhỏ: softmax trên 100 nghìn từ là cực kỳ tốn kém. Word2Vec sử dụng **negative sampling** để biến nó thành một tác vụ phân loại nhị phân. Dự đoán "từ ngữ cảnh này có xuất hiện gần từ trung tâm này không, có hay không". Lấy mẫu một vài từ âm (không cùng xuất hiện) cho mỗi cặp huấn luyện thay vì tính toán softmax trên toàn bộ từ vựng.

```figure
word-vector-arithmetic
```

## Xây dựng

### Bước 1: các cặp huấn luyện từ một corpus

```python
def skipgram_pairs(docs, window=2):
    pairs = []
    for doc in docs:
        for i, center in enumerate(doc):
            for j in range(max(0, i - window), min(len(doc), i + window + 1)):
                if i == j:
                    continue
                pairs.append((center, doc[j]))
    return pairs
```

```python
>>> skipgram_pairs([["the", "cat", "sat", "on", "mat"]], window=2)
[('the', 'cat'), ('the', 'sat'),
 ('cat', 'the'), ('cat', 'sat'), ('cat', 'on'),
 ('sat', 'the'), ('sat', 'cat'), ('sat', 'on'), ('sat', 'mat'),
 ...]
```

Mỗi cặp (từ trung tâm, từ ngữ cảnh) trong một cửa sổ là một ví dụ huấn luyện dương tính.

### Bước 2: bảng embedding

Hai ma trận. `W` là bảng embedding cho từ trung tâm (bảng bạn sẽ giữ lại). `W'` là bảng cho từ ngữ cảnh (thường bị loại bỏ, đôi khi được lấy trung bình với `W`).

```python
import numpy as np


def init_embeddings(vocab_size, dim, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(vocab_size, dim))
    W_prime = rng.normal(0, 0.1, size=(vocab_size, dim))
    return W, W_prime
```

Khởi tạo ngẫu nhiên nhỏ. Kích thước từ vựng 10 nghìn và chiều 100 là thực tế; để giảng dạy, 50 từ vựng x 16 chiều là đủ để thấy hình học.

### Bước 3: mục tiêu negative sampling

Với mỗi cặp dương tính `(center, context)`, lấy mẫu `k` từ ngẫu nhiên từ từ vựng làm các từ âm. Huấn luyện mô hình sao cho tích vô hướng `W[center] · W'[context]` cao đối với các cặp dương tính và thấp đối với các cặp âm.

```python
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_pair(W, W_prime, center_idx, context_idx, negative_indices, lr):
    v_c = W[center_idx]
    u_pos = W_prime[context_idx]
    u_negs = W_prime[negative_indices]

    pos_score = sigmoid(v_c @ u_pos)
    neg_scores = sigmoid(u_negs @ v_c)

    grad_center = (pos_score - 1) * u_pos
    for i, u in enumerate(u_negs):
        grad_center += neg_scores[i] * u

    W[context_idx] = W[context_idx]
    W_prime[context_idx] -= lr * (pos_score - 1) * v_c
    for i, neg_idx in enumerate(negative_indices):
        W_prime[neg_idx] -= lr * neg_scores[i] * v_c
    W[center_idx] -= lr * grad_center
```

Công thức kỳ diệu: logistic loss trên cặp dương tính (muốn sigmoid gần bằng 1) cộng với logistic loss trên các cặp âm (muốn sigmoid gần bằng 0). Gradient chảy vào cả hai bảng. Cách dẫn xuất đầy đủ nằm trong bài báo gốc; hãy thử tự viết lại một lần bằng giấy bút nếu bạn muốn ghi nhớ nó.

### Bước 4: huấn luyện trên một corpus đồ chơi

```python
def train(docs, dim=16, window=2, k_neg=5, epochs=100, lr=0.05, seed=0):
    vocab = build_vocab(docs)
    vocab_size = len(vocab)
    rng = np.random.default_rng(seed)
    W, W_prime = init_embeddings(vocab_size, dim, seed=seed)
    pairs = skipgram_pairs(docs, window=window)

    for epoch in range(epochs):
        rng.shuffle(pairs)
        for center, context in pairs:
            c_idx = vocab[center]
            ctx_idx = vocab[context]
            negs = rng.integers(0, vocab_size, size=k_neg)
            negs = [n for n in negs if n != ctx_idx and n != c_idx]
            train_pair(W, W_prime, c_idx, ctx_idx, negs, lr)
    return vocab, W
```

Sau đủ số epoch trên một corpus lớn, các từ chia sẻ ngữ cảnh sẽ có các embedding trung tâm tương tự nhau. Trên một corpus đồ chơi, bạn thấy hiệu ứng này khá mờ nhạt. Trên hàng tỷ token, bạn sẽ thấy nó một cách rõ rệt.

### Bước 5: mẹo tương tự (analogy)

```python
def nearest(vocab, W, target_vec, topk=5, exclude=None):
    exclude = exclude or set()
    inv_vocab = {i: w for w, i in vocab.items()}
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-9
    W_norm = W / norms
    target = target_vec / (np.linalg.norm(target_vec) + 1e-9)
    sims = W_norm @ target
    order = np.argsort(-sims)
    out = []
    for i in order:
        if i in exclude:
            continue
        out.append((inv_vocab[i], float(sims[i])))
        if len(out) == topk:
            break
    return out


def analogy(vocab, W, a, b, c, topk=5):
    v = W[vocab[b]] - W[vocab[a]] + W[vocab[c]]
    return nearest(vocab, W, v, topk=topk, exclude={vocab[a], vocab[b], vocab[c]})
```

Trên các vector Google News 300d đã được huấn luyện sẵn:

```python
>>> analogy(vocab, W, "man", "king", "woman")
[('queen', 0.71), ('monarch', 0.62), ('princess', 0.59), ...]
```

`king - man + woman = queen`. Không phải vì mô hình biết "hoàng gia" là gì. Mà vì vector `(king - man)` nắm bắt được thứ gì đó giống như "hoàng gia", và việc cộng nó vào `woman` sẽ dẫn đến vùng gần với "hoàng gia-nữ".

## Sử dụng

Viết Word2Vec từ đầu là để học tập. NLP trong sản xuất sử dụng `gensim`.

```python
from gensim.models import Word2Vec

sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "ran", "across", "the", "room"],
]

model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,
    negative=5,
    workers=4,
    epochs=30,
)

print(model.wv["cat"])
print(model.wv.most_similar("cat", topn=3))
```

Đối với công việc thực tế, bạn hầu như không bao giờ tự huấn luyện Word2Vec. Bạn tải xuống các vector đã được huấn luyện sẵn.

- **GloVe** — Cách tiếp cận phân rã ma trận đồng xuất hiện của Stanford. Các checkpoint 50d, 100d, 200d, 300d. Độ bao phủ chung tốt. Bài 04 đề cập cụ thể về GloVe.
- **fastText** — Phần mở rộng Word2Vec của Facebook, nhúng các n-gram ký tự. Xử lý các từ ngoài từ vựng bằng cách kết hợp các subword. Bài 04.
- **Pretrained Word2Vec trên Google News** — 300d, từ vựng 3 triệu từ, xuất bản năm 2013. Vẫn được tải xuống hàng ngày.

### Khi nào Word2Vec vẫn thắng thế vào năm 2026

- Truy xuất chuyên biệt theo miền (domain-specific) nhẹ. Huấn luyện trên các bản tóm tắt y tế trong một giờ trên laptop, nhận được các vector chuyên biệt mà không mô hình tổng quát nào nắm bắt được.
- Kỹ thuật đặc trưng kiểu tương tự. `gender_vector = mean(man - woman pairs)`. Trừ nó khỏi các từ khác để có được một trục trung tính về giới tính. Vẫn được sử dụng trong nghiên cứu về tính công bằng (fairness).
- Khả năng diễn giải. 100d đủ nhỏ để vẽ biểu đồ qua PCA hoặc t-SNE và thực sự thấy các cụm hình thành.
- Bất cứ nơi nào suy luận phải chạy trên thiết bị không có GPU. Tra cứu Word2Vec chỉ là một thao tác lấy một hàng dữ liệu.

### Nơi Word2Vec thất bại

Bức tường đa nghĩa (polysemy). `bank` có một vector. `river bank` và `financial bank` chia sẻ nó. `table` (bảng tính vs. đồ nội thất) cũng chia sẻ nó. Một bộ phân loại ở hạ nguồn không thể phân biệt các nghĩa từ vector này.

Các embedding theo ngữ cảnh (ELMo, BERT, mọi transformer sau này) đã giải quyết vấn đề này bằng cách tạo ra một vector khác nhau cho mỗi lần xuất hiện của từ dựa trên ngữ cảnh xung quanh. Đó là bước nhảy từ Word2Vec sang BERT: từ tĩnh sang ngữ cảnh. Phase 7 đề cập đến phần transformer.

Vấn đề ngoài từ vựng (out-of-vocabulary) là thất bại khác. Word2Vec chưa bao giờ thấy `Zoomer-approved` nếu nó không có trong dữ liệu huấn luyện. Không có phương án dự phòng. fastText khắc phục điều này bằng cách kết hợp subword (bài 04).

## Triển khai

Lưu dưới dạng `outputs/skill-embedding-probe.md`:

```markdown
---
name: embedding-probe
description: Inspect a word2vec model. Run analogies, find neighbors, diagnose quality.
version: 1.0.0
phase: 5
lesson: 03
tags: [nlp, embeddings, debugging]
---

You probe trained word embeddings to verify they are working. Given a `gensim.models.KeyedVectors` object and a vocabulary, you run:

1. Three canonical analogy tests. `king : man :: queen : woman`. `paris : france :: tokyo : japan`. `walking : walked :: swimming : ?`. Report the top-1 result and its cosine.
2. Five nearest-neighbor tests on domain-specific words the user supplies. Print top-5 neighbors with cosines.
3. One symmetry check. `similarity(a, b) == similarity(b, a)` to within float precision.
4. One degenerate check. If any embedding has a norm below 0.01 or above 100, the model has a training bug. Flag it.

Refuse to declare a model good on analogy accuracy alone. Analogy benchmarks are gameable and do not transfer to downstream tasks. Recommend intrinsic + downstream evaluation together.
```

## Bài tập

1. **Dễ.** Chạy vòng lặp huấn luyện trên một corpus nhỏ (20 câu về mèo và chó). Sau 200 epoch, xác minh `nearest(vocab, W, W[vocab["cat"]])` trả về `dog` trong top 3 của nó. Nếu không, hãy tăng số epoch hoặc từ vựng.
2. **Trung bình.** Thêm lấy mẫu phụ (subsampling) các từ thường gặp. Các từ có tần suất trên `10^-5` sẽ bị loại khỏi các cặp huấn luyện với xác suất tỷ lệ thuận với tần suất của chúng. Đo lường hiệu quả đối với sự tương đồng của các từ hiếm.
3. **Khó.** Huấn luyện một mô hình trên corpus 20 Newsgroups. Tính toán hai trục thiên kiến: `he - she` và `doctor - nurse`. Chiếu các từ chỉ nghề nghiệp lên cả hai trục. Báo cáo nghề nghiệp nào có khoảng cách thiên kiến lớn nhất. Đây là loại thăm dò mà các nhà nghiên cứu về tính công bằng sử dụng.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| Word embedding | Từ dưới dạng vector | Một biểu diễn dày, số chiều thấp (thường là 100-300) được học từ ngữ cảnh. |
| Skip-gram | Mẹo Word2Vec | Dự đoán các từ ngữ cảnh từ từ trung tâm. Chậm hơn CBOW, tốt hơn cho các từ hiếm. |
| Negative sampling | Phím tắt huấn luyện | Thay thế softmax trên toàn bộ từ vựng bằng phân loại nhị phân với `k` từ ngẫu nhiên. |
| Static embedding | Một vector mỗi từ | Cùng một vector bất kể ngữ cảnh. Thất bại với từ đa nghĩa. |
| Contextual embedding | Vector nhạy cảm ngữ cảnh | Vector khác nhau cho mỗi lần xuất hiện dựa trên các từ xung quanh. Những gì transformer tạo ra. |
| OOV | Ngoài từ vựng | Từ không thấy trong huấn luyện. Word2Vec không thể tạo vector cho các từ này. |

## Đọc thêm

- [Mikolov et al. (2013). Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) — bài báo về negative-sampling. Ngắn gọn và dễ đọc.
- [Rong, X. (2014). word2vec Parameter Learning Explained](https://arxiv.org/abs/1411.2738) — cách dẫn xuất gradient rõ ràng nhất, nếu toán học trong bài báo gốc cảm thấy quá dày đặc.
- [gensim Word2Vec tutorial](https://radimrehurek.com/gensim/models/word2vec.html) — các thiết lập huấn luyện trong sản xuất thực sự hiệu quả.