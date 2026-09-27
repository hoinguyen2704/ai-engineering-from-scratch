# GloVe, FastText và Subword Embeddings

> Word2Vec huấn luyện một embedding cho mỗi từ. GloVe phân rã ma trận đồng xuất hiện (co-occurrence matrix). FastText nhúng các thành phần của từ. BPE tạo cầu nối đến các Transformer.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 03 (Word2Vec from Scratch)
**Time:** ~45 phút

## Vấn đề

Word2Vec để lại hai câu hỏi chưa được giải đáp.

Thứ nhất, có một hướng nghiên cứu song song tập trung vào việc phân rã trực tiếp ma trận đồng xuất hiện (LSA, HAL) thay vì thực hiện cập nhật skip-gram trực tuyến. Liệu cách tiếp cận lặp của Word2Vec có thực sự tốt hơn về mặt bản chất, hay sự khác biệt chỉ là kết quả của cách hai phương pháp xử lý các tần suất? **GloVe** đã trả lời câu hỏi đó: phân rã ma trận với hàm mất mát (loss function) được lựa chọn kỹ lưỡng có hiệu suất ngang bằng hoặc vượt trội hơn Word2Vec, đồng thời chi phí huấn luyện thấp hơn.

Thứ hai, cả hai phương pháp đều không có giải pháp cho các từ chưa từng xuất hiện. `Zoomer-approved`, `dogecoin`, bất kỳ danh từ riêng nào mới xuất hiện tuần trước, hay mọi dạng biến thể của một từ gốc hiếm gặp. **FastText** đã giải quyết vấn đề này bằng cách nhúng các character n-gram: một từ là tổng các thành phần của nó, bao gồm cả các hình vị (morpheme), vì vậy ngay cả những từ nằm ngoài từ vựng (out-of-vocabulary) cũng có được vector hợp lý.

Thứ ba, khi các Transformer xuất hiện, câu hỏi lại thay đổi. Từ vựng ở cấp độ từ (word-level) bị giới hạn ở mức khoảng một triệu mục; ngôn ngữ thực tế mở rộng hơn thế nhiều. **Byte-pair encoding (BPE)** và các biến thể của nó đã giải quyết vấn đề này bằng cách học một từ vựng gồm các đơn vị subword phổ biến bao quát mọi trường hợp. Mọi tokenizer hiện đại cho mọi LLM hiện đại đều là subword tokenizer.

Bài học này sẽ đi qua cả ba phương pháp, sau đó giải thích khi nào nên sử dụng phương pháp nào.

## Khái niệm

**GloVe (Global Vectors).** Xây dựng ma trận đồng xuất hiện từ-từ `X`, trong đó `X[i][j]` là tần suất từ `j` xuất hiện trong ngữ cảnh của từ `i`. Huấn luyện các vector sao cho `v_i · v_j + b_i + b_j ≈ log(X[i][j])`. Trọng số hóa hàm mất mát để các cặp từ xuất hiện quá thường xuyên không làm lu mờ các cặp khác. Xong.

**FastText.** Một từ là tổng các character n-gram của nó cộng với chính từ đó. `where` trở thành `<wh, whe, her, ere, re>, <where>`. Vector của từ là tổng các vector thành phần này. Huấn luyện tương tự như Word2Vec. Lợi ích: các từ chưa biết (`whereupon`) được cấu thành từ các n-gram đã biết.

**BPE (Byte-pair encoding).** Bắt đầu với một từ vựng gồm các byte (hoặc ký tự) riêng lẻ. Đếm mọi cặp liền kề trong tập dữ liệu. Hợp nhất cặp phổ biến nhất thành một token mới. Lặp lại trong `k` lần. Kết quả: một từ vựng gồm `k + 256` token, trong đó các chuỗi phổ biến (`ing`, `tion`, `the`) là các token đơn lẻ và các từ hiếm được chia nhỏ thành các mảnh quen thuộc. Mọi câu đều có thể được token hóa thành các đơn vị có nghĩa.

```figure
n5-subword-merge
```

## Xây dựng

### GloVe: phân rã ma trận đồng xuất hiện

```python
import numpy as np
from collections import Counter


def build_cooccurrence(docs, window=5):
    pair_counts = Counter()
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    for doc in docs:
        indexed = [vocab[t] for t in doc]
        for i, center in enumerate(indexed):
            for j in range(max(0, i - window), min(len(indexed), i + window + 1)):
                if i != j:
                    distance = abs(i - j)
                    pair_counts[(center, indexed[j])] += 1.0 / distance
    return vocab, pair_counts


def glove_train(vocab, pair_counts, dim=16, epochs=100, lr=0.05, x_max=100, alpha=0.75, seed=0):
    n = len(vocab)
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(n, dim))
    W_tilde = rng.normal(0, 0.1, size=(n, dim))
    b = np.zeros(n)
    b_tilde = np.zeros(n)

    for epoch in range(epochs):
        for (i, j), x_ij in pair_counts.items():
            weight = (x_ij / x_max) ** alpha if x_ij < x_max else 1.0
            diff = W[i] @ W_tilde[j] + b[i] + b_tilde[j] - np.log(x_ij)
            coef = weight * diff

            grad_W_i = coef * W_tilde[j]
            grad_W_tilde_j = coef * W[i]
            W[i] -= lr * grad_W_i
            W_tilde[j] -= lr * grad_W_tilde_j
            b[i] -= lr * coef
            b_tilde[j] -= lr * coef

    return W + W_tilde
```

Có hai thành phần quan trọng cần lưu ý. Hàm trọng số `f(x) = (x/x_max)^alpha` làm giảm trọng số của các cặp xuất hiện quá thường xuyên (như `(the, and)`) để chúng không chi phối hàm mất mát. Embedding cuối cùng là tổng của bảng `W` (từ trung tâm) và `W_tilde` (ngữ cảnh). Việc cộng cả hai là một thủ thuật đã được công bố, thường mang lại hiệu quả tốt hơn so với việc chỉ sử dụng một bảng.

### FastText: embedding nhận diện subword

```python
def char_ngrams(word, n_min=3, n_max=6):
    wrapped = f"<{word}>"
    grams = {wrapped}
    for n in range(n_min, n_max + 1):
        for i in range(len(wrapped) - n + 1):
            grams.add(wrapped[i:i + n])
    return grams
```

```python
>>> char_ngrams("where")
{'<where>', '<wh', 'whe', 'her', 'ere', 're>', '<whe', 'wher', 'here', 'ere>', '<wher', 'where', 'here>'}
```

Mỗi từ được biểu diễn bằng tập hợp các n-gram của nó (thường từ 3 đến 6 ký tự). Embedding của từ là tổng các embedding của các n-gram. Đối với huấn luyện skip-gram, hãy thay thế vector đơn lẻ của Word2Vec bằng tổng này.

```python
def fasttext_vector(word, ngram_table):
    grams = char_ngrams(word)
    vecs = [ngram_table[g] for g in grams if g in ngram_table]
    if not vecs:
        return None
    return np.sum(vecs, axis=0)
```

Đối với một từ chưa biết, bạn vẫn nhận được một vector miễn là một số n-gram của nó đã được biết đến. `whereupon` chia sẻ `<wh`, `her`, `ere`, và `<where` với `where`, vì vậy hai từ này sẽ nằm gần nhau trong không gian vector.

### BPE: học từ vựng subword

```python
def learn_bpe(corpus, k_merges):
    vocab = Counter()
    for word, freq in corpus.items():
        tokens = tuple(word) + ("</w>",)
        vocab[tokens] = freq

    merges = []
    for _ in range(k_merges):
        pair_freq = Counter()
        for tokens, freq in vocab.items():
            for a, b in zip(tokens, tokens[1:]):
                pair_freq[(a, b)] += freq
        if not pair_freq:
            break
        best = pair_freq.most_common(1)[0][0]
        merges.append(best)

        new_vocab = Counter()
        for tokens, freq in vocab.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_vocab[tuple(new_tokens)] = freq
        vocab = new_vocab
    return merges


def apply_bpe(word, merges):
    tokens = list(word) + ["</w>"]
    for a, b in merges:
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
                new_tokens.append(a + b)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens
```

```python
>>> corpus = Counter({"low": 5, "lower": 2, "newest": 6, "widest": 3})
>>> merges = learn_bpe(corpus, k_merges=10)
>>> apply_bpe("lowest", merges)
['low', 'est</w>']
```

Lần lặp đầu tiên hợp nhất cặp liền kề phổ biến nhất. Sau đủ số lần lặp, các chuỗi con phổ biến (`low`, `est`, `tion`) trở thành các token đơn lẻ và các từ hiếm được chia nhỏ một cách hợp lý.

Các tokenizer thực tế của GPT / BERT / T5 học từ 30k đến 100k lần hợp nhất. Kết quả: bất kỳ văn bản nào cũng được token hóa thành một chuỗi có độ dài giới hạn gồm các ID đã biết, không bao giờ gặp lỗi OOV.

## Sử dụng

Trong thực tế, bạn hiếm khi tự huấn luyện những mô hình này. Bạn thường tải các checkpoint đã được huấn luyện sẵn.

```python
import fasttext.util
fasttext.util.download_model("en", if_exists="ignore")
ft = fasttext.load_model("cc.en.300.bin")
print(ft.get_word_vector("whereupon").shape)
print(ft.get_word_vector("zoomerapproved").shape)
```

Đối với token hóa subword kiểu BPE trong kỷ nguyên Transformer:

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
print(tok.tokenize("unbelievably tokenized"))
```

```
['un', 'bel', 'iev', 'ably', 'Ġtoken', 'ized']
```

Tiền tố `Ġ` đánh dấu ranh giới từ (một quy ước của GPT-2). Mọi tokenizer hiện đại đều là biến thể của BPE, WordPiece (BERT), hoặc SentencePiece (T5, LLaMA).

### Khi nào chọn phương pháp nào

| Tình huống | Lựa chọn |
|-----------|------|
| Vector từ đa năng đã huấn luyện sẵn, không cần xử lý OOV | GloVe 300d |
| Vector từ đa năng đã huấn luyện sẵn, cần xử lý lỗi chính tả / từ mới / ngôn ngữ giàu hình thái | FastText |
| Bất cứ thứ gì đưa vào Transformer (huấn luyện hoặc suy luận) | Sử dụng tokenizer đi kèm với mô hình. Không bao giờ thay đổi. |
| Huấn luyện mô hình ngôn ngữ từ đầu | Huấn luyện tokenizer BPE hoặc SentencePiece trên tập dữ liệu của bạn trước |
| Phân loại văn bản trong sản xuất với mô hình tuyến tính | Vẫn là TF-IDF. Bài học 02. |

## Triển khai

Lưu dưới dạng `outputs/skill-embeddings-picker.md`:

```markdown
---
name: tokenizer-picker
description: Pick a tokenization approach for a new language model or text pipeline.
version: 1.0.0
phase: 5
lesson: 04
tags: [nlp, tokenization, embeddings]
---

Given a task and dataset description, you output:

1. Tokenization strategy (word-level, BPE, WordPiece, SentencePiece, byte-level). One-sentence reason.
2. Vocabulary size target (e.g., 32k for an English-only LM, 64k-100k for multilingual).
3. Library call with the exact training command. Name the library. Quote the arguments.
4. One reproducibility pitfall. Tokenizer-model mismatch is the single most common silent production bug; call out which pair must be used together.

Refuse to recommend training a custom tokenizer when the user is fine-tuning a pretrained LLM. Refuse to recommend word-level tokenization for any model targeting production inference. Flag non-English / multi-script corpora as needing SentencePiece with byte fallback.
```

## Bài tập

1. **Dễ.** Chạy `char_ngrams("playing")` và `char_ngrams("played")`. Tính độ chồng lấp Jaccard của hai tập hợp n-gram. Bạn sẽ thấy các thành phần chia sẻ đáng kể (`pla`, `lay`, `play`), đó là lý do tại sao FastText chuyển đổi tốt giữa các biến thể hình thái.
2. **Trung bình.** Mở rộng `learn_bpe` để theo dõi sự tăng trưởng của từ vựng. Vẽ biểu đồ số token trên mỗi ký tự của tập dữ liệu theo số lần hợp nhất. Bạn sẽ thấy sự nén dữ liệu nhanh chóng lúc đầu, sau đó tiệm cận mức ~2-3 ký tự mỗi token.
3. **Khó.** Huấn luyện BPE với 1k lần hợp nhất trên toàn bộ tác phẩm của Shakespeare. So sánh việc token hóa các từ phổ biến với các danh từ riêng hiếm gặp. Đo số token trung bình trên mỗi từ trước và sau khi huấn luyện. Viết ra những điều khiến bạn ngạc nhiên.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| Co-occurrence matrix | Bảng tần suất từ-từ | `X[i][j]` = tần suất từ `j` xuất hiện trong cửa sổ xung quanh từ `i`. |
| Subword | Một phần của từ | Một character n-gram (FastText) hoặc token đã học (BPE/WordPiece/SentencePiece). |
| BPE | Byte-pair encoding | Hợp nhất lặp đi lặp lại các cặp liền kề phổ biến nhất cho đến khi từ vựng đạt kích thước mục tiêu. |
| OOV | Out of vocabulary | Từ mà mô hình chưa từng thấy. Word2Vec/GloVe thất bại. FastText và BPE xử lý được. |
| Byte-level BPE | BPE trên byte thô | Cơ chế của GPT-2. Từ vựng bắt đầu với 256 byte, vì vậy không bao giờ có OOV. |

## Đọc thêm

- [Pennington, Socher, Manning (2014). GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf) — bài báo về GloVe, bảy trang, vẫn là cách dẫn dắt hàm mất mát tốt nhất.
- [Bojanowski et al. (2017). Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606) — FastText.
- [Sennrich, Haddow, Birch (2016). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — bài báo giới thiệu BPE vào NLP hiện đại.
- [Hugging Face tokenizer summary](https://huggingface.co/docs/transformers/tokenizer_summary) — cách BPE, WordPiece và SentencePiece thực sự khác biệt trong thực tế.