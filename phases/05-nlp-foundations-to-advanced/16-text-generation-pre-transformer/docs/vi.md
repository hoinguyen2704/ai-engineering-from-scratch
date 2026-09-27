# Tạo văn bản trước thời đại Transformer — Mô hình ngôn ngữ N-gram

> Nếu một từ gây ngạc nhiên, mô hình đó tệ. Perplexity biến sự ngạc nhiên thành một con số. Smoothing giữ cho con số đó hữu hạn.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (Text Processing), Phase 2 · 14 (Naive Bayes)
**Time:** ~45 phút

## Vấn đề

Trước thời đại Transformer, trước RNN, trước word embedding, một mô hình ngôn ngữ dự đoán từ tiếp theo bằng cách đếm tần suất từ đó xuất hiện sau `n-1` từ trước đó. Đếm "the cat" → "sat" 47 lần, "the cat" → "jumped" 12 lần, "the cat" → "refrigerator" 0 lần. Chuẩn hóa để có được phân phối xác suất.

Đó chính là mô hình ngôn ngữ n-gram. Nó vận hành mọi hệ thống nhận dạng giọng nói, kiểm tra chính tả và dịch máy dựa trên cụm từ từ năm 1980 đến 2015. Nó vẫn được sử dụng khi bạn cần mô hình ngôn ngữ nhẹ, chạy trực tiếp trên thiết bị.

Vấn đề thú vị là phải làm gì với các n-gram chưa từng thấy. Một mô hình dựa trên đếm thô sẽ gán xác suất bằng 0 cho bất kỳ thứ gì nó chưa thấy, điều này là thảm họa vì các câu thường dài và hầu như mọi câu dài đều chứa ít nhất một chuỗi chưa từng thấy. 50 năm nghiên cứu về smoothing đã giải quyết vấn đề này. Kneser-Ney smoothing là kết quả, và deep learning hiện đại đã kế thừa truyền thống thực nghiệm đó.

## Khái niệm

![N-gram model: count, smooth, generate](../assets/ngram.svg)

### Trò chơi dự đoán

Trước khi bất kỳ cơ chế nào tồn tại, một thí nghiệm đã định nghĩa mô hình ngôn ngữ là gì. Hãy che chữ cái tiếp theo của một câu tiếng Anh. Yêu cầu ai đó đoán, từng chữ một, cho đến khi họ đoán đúng. Ghi lại số lần đoán. Lặp lại vài trăm lần.

Số lần đoán không phải là chuyện tầm phào. Chúng là một cách mã hóa lại văn bản không mất dữ liệu: đưa chuỗi số lần đoán cho một người đoán thứ hai giống hệt, họ có thể khôi phục lại từng chữ cái, vì tại mỗi vị trí họ biết chính xác những dự đoán nào đến trước. Một thông điệp có thể mã hóa lại bằng ít ký hiệu hơn sẽ mang ít thông tin hơn trên mỗi ký hiệu, vì vậy thống kê số lần đoán đặt ra giới hạn trên cho entropy của tiếng Anh.

Shannon đã thực hiện thí nghiệm này vào năm 1951 và thu được một con số vẫn còn chi phối lĩnh vực này. Một bảng chữ cái 27 ký hiệu (26 chữ cái cộng với dấu cách) có thể mang `log2(27) ≈ 4.75` bit trên mỗi chữ cái. Những người đoán với ngữ cảnh 100 chữ cái đạt mức từ 0,6 đến 1,3 bit trên mỗi chữ cái. Tiếng Anh có khoảng ba phần tư là các bước đi bắt buộc. Cấu trúc mà một mô hình phải học đã được đo lường trước khi bất kỳ mô hình nào có thể học được nó.

Mọi mô hình ngôn ngữ kể từ đó đều là một người chơi cơ học của trò chơi này, và mọi con số đánh giá trong bài học này đều là điểm số của trò chơi:

- **Cross-entropy loss** là số bit trung bình mà mô hình cần cho mỗi ký hiệu. Huấn luyện một LM thực chất là tối thiểu hóa điểm số của nó trong trò chơi dự đoán.
- **Perplexity** là `2^bits` (hoặc `e^nats`): hệ số phân nhánh mà mô hình vẫn phải đối mặt sau khi dự đoán. Việc đoán đồng nhất trên 27 ký hiệu có perplexity là 27; một người chơi với 1 bit/chữ cái có perplexity là 2.
- **Độ dài ngữ cảnh là bộ nhớ của người chơi.** Mô hình trigram chơi với bộ nhớ 2 token. Transformer chơi cùng trò chơi đó với 100K token. Các quy tắc không bao giờ thay đổi; người chơi đã trở nên giỏi hơn.

Một đơn vị cần lưu ý: điểm số trò chơi trên mỗi chữ cái tính bằng bit (`log2`), trong khi các công thức n-gram bên dưới tính trên mỗi word token bằng nats (log tự nhiên) — và vì perplexity `e^H` tính bằng nats bằng `2^H` tính bằng bit, hai góc nhìn này là cùng một phép đo ở các đơn vị khác nhau.

```figure
prediction-game
```

**Xác suất N-gram:** `P(w_i | w_{i-n+1}, ..., w_{i-1})`. Cố định `n` (thường là 3 cho trigram, 4 cho 4-gram). Tính toán từ các số đếm:

```text
P(w | context) = count(context, w) / count(context)
```

**Vấn đề đếm bằng 0.** Bất kỳ n-gram nào không thấy trong quá trình huấn luyện đều có xác suất bằng 0. Một nghiên cứu năm 2007 trên Brown corpus cho thấy ngay cả mô hình 4-gram cũng có 30% số 4-gram trong tập kiểm tra chưa từng thấy khi huấn luyện. Bạn không thể đánh giá trên bất kỳ văn bản thực tế nào mà không có smoothing.

**Các phương pháp smoothing, theo thứ tự tinh vi:**

1. **Laplace (add-one).** Cộng 1 vào mọi số đếm. Đơn giản, nhưng rất tệ với các sự kiện hiếm.
2. **Good-Turing.** Phân bổ lại khối xác suất từ các sự kiện tần suất cao sang các sự kiện chưa thấy dựa trên tần suất của tần suất.
3. **Interpolation.** Kết hợp các ước tính n-gram, (n-1)-gram, v.v., với các trọng số có thể điều chỉnh.
4. **Backoff.** Nếu n-gram có số đếm bằng 0, quay lại (n-1)-gram. Katz backoff chuẩn hóa điều này.
5. **Absolute discounting.** Trừ một giá trị chiết khấu cố định `D` khỏi tất cả các số đếm, phân phối lại cho các sự kiện chưa thấy.
6. **Kneser-Ney.** Absolute discounting cộng với một lựa chọn thông minh cho mô hình bậc thấp hơn: sử dụng *xác suất tiếp nối* (continuation probability - một từ xuất hiện trong bao nhiêu ngữ cảnh) thay vì tần suất thô.

Sự thấu hiểu của Kneser-Ney rất sâu sắc. "San Francisco" là một bigram phổ biến. Unigram "Francisco" xuất hiện chủ yếu sau "San". Absolute discounting ngây thơ gán cho "Francisco" xác suất unigram cao (vì số đếm cao). Kneser-Ney nhận thấy rằng "Francisco" chỉ xuất hiện trong một ngữ cảnh và hạ thấp xác suất tiếp nối của nó tương ứng. Kết quả: một bigram mới kết thúc bằng "Francisco" nhận được xác suất thấp phù hợp.

**Đánh giá: perplexity.** Số mũ của log-likelihood âm trung bình trên mỗi từ trong tập kiểm tra (held-out test set). Càng thấp càng tốt. Perplexity bằng 100 nghĩa là mô hình bối rối như khi chọn ngẫu nhiên giữa 100 từ.

```text
perplexity = exp(- (1/N) * Σ log P(w_i | context_i))
```

```figure
ngram-backoff
```

## Xây dựng

### Bước 1: đếm trigram

```python
from collections import Counter, defaultdict


def train_ngram(corpus_tokens, n=3):
    ngrams = Counter()
    contexts = Counter()
    for sentence in corpus_tokens:
        padded = ["<s>"] * (n - 1) + sentence + ["</s>"]
        for i in range(len(padded) - n + 1):
            ctx = tuple(padded[i:i + n - 1])
            word = padded[i + n - 1]
            ngrams[ctx + (word,)] += 1
            contexts[ctx] += 1
    return ngrams, contexts


def raw_probability(ngrams, contexts, context, word):
    ctx = tuple(context)
    if contexts.get(ctx, 0) == 0:
        return 0.0
    return ngrams.get(ctx + (word,), 0) / contexts[ctx]
```

Đầu vào là danh sách các câu đã được token hóa. Đầu ra là số đếm n-gram và số đếm ngữ cảnh. `<s>` và `</s>` là các ranh giới câu.

### Bước 2: Laplace smoothing

```python
def laplace_probability(ngrams, contexts, vocab_size, context, word):
    ctx = tuple(context)
    numerator = ngrams.get(ctx + (word,), 0) + 1
    denominator = contexts.get(ctx, 0) + vocab_size
    return numerator / denominator
```

Cộng 1 vào mọi số đếm. Làm mượt nhưng phân bổ quá nhiều khối xác suất cho các sự kiện chưa thấy, gây hại cho các sự kiện hiếm đã biết.

### Bước 3: Kneser-Ney (bigram, nội suy)

```python
def kneser_ney_bigram_model(corpus_tokens, discount=0.75):
    unigrams = Counter()
    bigrams = Counter()
    unigram_contexts = defaultdict(set)

    for sentence in corpus_tokens:
        padded = ["<s>"] + sentence + ["</s>"]
        for i, w in enumerate(padded):
            unigrams[w] += 1
            if i > 0:
                prev = padded[i - 1]
                bigrams[(prev, w)] += 1
                unigram_contexts[w].add(prev)

    total_unique_bigrams = sum(len(ctx_set) for ctx_set in unigram_contexts.values())
    continuation_prob = {
        w: len(ctx_set) / total_unique_bigrams for w, ctx_set in unigram_contexts.items()
    }

    context_totals = Counter()
    for (prev, w), count in bigrams.items():
        context_totals[prev] += count

    unique_follow = defaultdict(set)
    for (prev, w) in bigrams:
        unique_follow[prev].add(w)

    def prob(prev, w):
        count = bigrams.get((prev, w), 0)
        denom = context_totals.get(prev, 0)
        if denom == 0:
            return continuation_prob.get(w, 1e-9)
        first_term = max(count - discount, 0) / denom
        lambda_prev = discount * len(unique_follow[prev]) / denom
        return first_term + lambda_prev * continuation_prob.get(w, 1e-9)

    return prob
```

Ba phần chuyển động. `continuation_prob` nắm bắt "từ này xuất hiện trong bao nhiêu ngữ cảnh khác nhau?" (sự đổi mới của Kneser-Ney). `lambda_prev` là khối lượng được giải phóng bởi chiết khấu, dùng để làm trọng số cho backoff. Xác suất cuối cùng là số hạng chính đã chiết khấu cộng với số hạng tiếp nối có trọng số.

### Bước 4: tạo văn bản với lấy mẫu (sampling)

```python
import random


def generate(prob_fn, vocab, prefix, max_len=30, seed=0):
    rng = random.Random(seed)
    tokens = list(prefix)
    for _ in range(max_len):
        candidates = [(w, prob_fn(tokens[-1], w)) for w in vocab]
        total = sum(p for _, p in candidates)
        r = rng.random() * total
        acc = 0.0
        for w, p in candidates:
            acc += p
            if r <= acc:
                tokens.append(w)
                break
        if tokens[-1] == "</s>":
            break
    return tokens
```

Lấy mẫu tỷ lệ thuận với xác suất. Luôn cho đầu ra khác nhau theo mỗi seed. Đối với đầu ra kiểu beam-search, chọn argmax tại mỗi bước (greedy) và thêm một núm điều chỉnh ngẫu nhiên nhỏ (temperature).

### Bước 5: perplexity

```python
import math


def perplexity(prob_fn, sentences):
    total_log_prob = 0.0
    total_tokens = 0
    for sentence in sentences:
        padded = ["<s>"] + sentence + ["</s>"]
        for i in range(1, len(padded)):
            p = prob_fn(padded[i - 1], padded[i])
            total_log_prob += math.log(max(p, 1e-12))
            total_tokens += 1
    return math.exp(-total_log_prob / total_tokens)
```

Càng thấp càng tốt. Đối với Brown corpus, mô hình 4-gram KN được tinh chỉnh tốt đạt perplexity khoảng 140. Transformer LM đạt 15-30 trên cùng tập kiểm tra. Khoảng cách là khoảng 10 lần. Khoảng cách đó là lý do tại sao lĩnh vực này đã chuyển dịch.

## Sử dụng

- **Giảng dạy NLP cổ điển.** Sự tiếp cận rõ ràng nhất về smoothing, MLE và perplexity mà bạn có thể có.
- **KenLM.** Thư viện n-gram cho sản xuất. Được sử dụng làm bộ chấm điểm lại (rescorer) trong các hệ thống giọng nói và dịch máy nơi độ trễ thấp là quan trọng.
- **Tự động hoàn thành trên thiết bị.** Các mô hình trigram trong bàn phím. Vẫn còn đó.
- **Baseline.** Luôn tính perplexity của n-gram LM trước khi tuyên bố neural LM của bạn tốt. Nếu Transformer của bạn không vượt trội hơn KN một khoảng cách lớn, có gì đó không ổn.

## Triển khai

Lưu dưới dạng `outputs/prompt-lm-baseline.md`:

```markdown
---
name: lm-baseline
description: Build a reproducible n-gram language model baseline before training a neural LM.
phase: 5
lesson: 16
---

Given a corpus and target use (next-word prediction, rescoring, perplexity baseline), output:

1. N-gram order. Trigram for general English, 4-gram if corpus is large, 5-gram for speech rescoring.
2. Smoothing. Modified Kneser-Ney is the default; Laplace only for teaching.
3. Library. `kenlm` for production, `nltk.lm` for teaching, roll your own only to learn.
4. Evaluation. Held-out perplexity with consistent tokenization between train and test sets.

Refuse to report perplexity computed with different tokenization between systems being compared — perplexity numbers are comparable only under identical tokenization. Flag OOV rate in test set; KN handles OOV poorly unless you reserve a special <UNK> token during training.
```

## Bài tập

1. **Dễ.** Huấn luyện một trigram LM trên tập dữ liệu Shakespeare 1.000 câu. Tạo 20 câu. Chúng sẽ hợp lý ở cấp độ cục bộ nhưng không mạch lạc ở cấp độ toàn cục. Đây là bản demo kinh điển.
2. **Trung bình.** Triển khai perplexity cho mô hình KN của bạn trên tập kiểm tra Shakespeare. So sánh với Laplace. Bạn sẽ thấy KN giảm perplexity từ 30-50%.
3. **Khó.** Xây dựng bộ sửa lỗi chính tả trigram: với một từ sai chính tả và ngữ cảnh của nó, tạo các từ sửa lỗi và xếp hạng theo xác suất ngữ cảnh dưới LM. Đánh giá trên tập dữ liệu chính tả Birkbeck (công khai).

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| N-gram | Chuỗi từ | Chuỗi gồm `n` token liên tiếp. |
| Smoothing | Tránh số 0 | Phân bổ lại khối xác suất để các sự kiện chưa thấy có xác suất khác 0. |
| Perplexity | Chỉ số chất lượng LM | `exp(-average log-prob)` trên dữ liệu held-out. Càng thấp càng tốt. |
| Backoff | Quay lại ngữ cảnh ngắn hơn | Nếu số đếm trigram bằng 0, dùng bigram. Katz backoff chuẩn hóa điều này. |
| Kneser-Ney | Smoothing tốt nhất cho n-gram | Absolute discounting + xác suất tiếp nối cho mô hình bậc thấp hơn. |
| Continuation probability | Đặc thù của KN | `P(w)` được tính trọng số bởi số lượng ngữ cảnh mà `w` xuất hiện, không phải theo số đếm thô. |
| Entropy của văn bản | Thông tin trên mỗi ký hiệu | Số bit trung bình cần thiết để mã hóa ký hiệu tiếp theo dựa trên ngữ cảnh. Ước tính năm 1951 của Shannon cho tiếng Anh in với tối đa 100 chữ cái ngữ cảnh: 0,6-1,3 bit/chữ cái, được đo trước khi bất kỳ mô hình nào tồn tại. |

## Đọc thêm

- [Shannon (1951). Prediction and Entropy of Printed English](https://www.princeton.edu/~wbialek/rome/refs/shannon_51.pdf) — thí nghiệm trò chơi đoán chữ đã định nghĩa mục tiêu mà mọi mô hình ngôn ngữ vẫn đang tối ưu hóa.
- [Jurafsky and Martin — Speech and Language Processing, Chapter 3 (2026 draft)](https://web.stanford.edu/~jurafsky/slp3/3.pdf) — tài liệu kinh điển về n-gram LM và smoothing.
- [Chen and Goodman (1998). An Empirical Study of Smoothing Techniques for Language Modeling](https://dash.harvard.edu/handle/1/25104739) — bài báo xác lập Kneser-Ney là phương pháp smoothing n-gram tốt nhất.
- [Kneser and Ney (1995). Improved Backing-off for M-gram Language Modeling](https://ieeexplore.ieee.org/document/479394) — bài báo gốc về KN.
- [KenLM](https://kheafield.com/code/kenlm/) — n-gram LM tốc độ cao cho sản xuất, vẫn được sử dụng vào năm 2026 cho các ứng dụng nhạy cảm với độ trễ.