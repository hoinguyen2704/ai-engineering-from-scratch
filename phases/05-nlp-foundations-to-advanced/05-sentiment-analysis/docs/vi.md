# Phân tích cảm xúc (Sentiment Analysis)

> Tác vụ NLP kinh điển. Hầu hết những gì bạn cần biết về phân loại văn bản cổ điển đều xuất hiện ở đây.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 2 · 14 (Naive Bayes)
**Time:** ~75 phút

## Vấn đề

"Đồ ăn không ngon lắm." Tích cực hay tiêu cực?

Phân tích cảm xúc nghe có vẻ đơn giản. Một người đánh giá nói rằng họ thích hoặc không thích điều gì đó. Hãy gán nhãn cho câu đó. Lý do nó trở thành tác vụ NLP kinh điển là vì mọi trường hợp trông có vẻ dễ dàng đều ẩn chứa những khó khăn. Phủ định làm đảo ngược ý nghĩa. Mỉa mai làm thay đổi hoàn toàn ý nghĩa. "Không tệ chút nào" là tích cực mặc dù có hai từ mang nghĩa tiêu cực. Emoji mang nhiều tín hiệu hơn cả văn bản xung quanh. Từ vựng chuyên ngành rất quan trọng (`tight` trong đánh giá âm nhạc so với `tight` trong đánh giá thời trang).

Phân tích cảm xúc là một phòng thí nghiệm thực tế cho NLP cổ điển. Nếu bạn hiểu tại sao mọi mô hình cơ sở (baseline) ngây thơ đều có một chế độ lỗi cụ thể, bạn sẽ hiểu tại sao mọi mô hình phức tạp hơn lại được phát minh. Bài học này xây dựng một baseline Naive Bayes từ đầu, thêm hồi quy logistic (logistic regression) và chỉ ra những cái bẫy khiến việc phân tích cảm xúc trong môi trường thực tế trở thành một vấn đề đòi hỏi sự tuân thủ nghiêm ngặt.

## Khái niệm

Phân tích cảm xúc cổ điển là một công thức gồm hai bước.

1. **Biểu diễn (Represent).** Chuyển văn bản thành vector đặc trưng. Sử dụng BoW, TF-IDF hoặc n-grams.
2. **Phân loại (Classify).** Khớp một mô hình tuyến tính (Naive Bayes, logistic regression, SVM) trên các ví dụ đã được gán nhãn.

Naive Bayes là mô hình "ngây thơ" nhất mà vẫn hoạt động hiệu quả. Giả định rằng mọi đặc trưng là độc lập với nhau khi biết nhãn. Ước tính `P(word | positive)` và `P(word | negative)` từ các tần suất đếm. Khi suy luận, nhân các xác suất lại với nhau. Giả định độc lập "ngây thơ" này sai một cách nực cười nhưng kết quả lại gây sốc vì độ chính xác cao. Lý do: với các đặc trưng văn bản thưa thớt và dữ liệu vừa phải, bộ phân loại quan tâm đến việc mỗi từ nghiêng về phía nào hơn là mức độ nghiêng bao nhiêu.

Logistic regression khắc phục giả định độc lập. Nó học một trọng số cho mỗi đặc trưng, bao gồm cả các trọng số âm. `not good` dưới dạng một đặc trưng bigram sẽ nhận được một trọng số âm. Naive Bayes không thể làm điều đó đối với các bigram mà nó chưa từng được gán nhãn.

```figure
sentiment-logits
```

## Xây dựng

### Bước 1: một tập dữ liệu mini thực tế

```python
POSITIVE = [
    "absolutely loved this movie",
    "beautiful cinematography and a great story",
    "one of the best films of the year",
    "brilliant acting from the lead",
    "heartwarming and funny",
]

NEGATIVE = [
    "boring and far too long",
    "not worth your time",
    "the plot made no sense",
    "terrible acting, awful script",
    "i want my two hours back",
]
```

Cố tình chọn tập dữ liệu nhỏ. Công việc thực tế sử dụng hàng chục nghìn ví dụ (IMDb, SST-2, Yelp polarity). Toán học là như nhau.

### Bước 2: multinomial Naive Bayes từ đầu

```python
import math
from collections import Counter


def train_nb(docs_by_class, vocab, alpha=1.0):
    class_priors = {}
    class_word_probs = {}
    total_docs = sum(len(d) for d in docs_by_class.values())

    for cls, docs in docs_by_class.items():
        class_priors[cls] = len(docs) / total_docs
        counts = Counter()
        for doc in docs:
            for token in doc:
                counts[token] += 1
        total = sum(counts.values()) + alpha * len(vocab)
        class_word_probs[cls] = {
            w: (counts[w] + alpha) / total for w in vocab
        }
    return class_priors, class_word_probs


def predict_nb(doc, class_priors, class_word_probs):
    scores = {}
    for cls in class_priors:
        s = math.log(class_priors[cls])
        for token in doc:
            if token in class_word_probs[cls]:
                s += math.log(class_word_probs[cls][token])
        scores[cls] = s
    return max(scores, key=scores.get)
```

Làm mịn cộng (additive smoothing, alpha=1.0) chính là Laplace smoothing. Nếu không có nó, một từ chưa từng xuất hiện trong một lớp sẽ có xác suất bằng 0 và log sẽ bị lỗi. `alpha=0.01` là phổ biến trong thực tế. `alpha=1.0` là mặc định trong giảng dạy.

### Bước 3: logistic regression từ đầu

```python
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_lr(X, y, epochs=500, lr=0.05, l2=0.01):
    n_features = X.shape[1]
    w = np.zeros(n_features)
    b = 0.0
    for _ in range(epochs):
        logits = X @ w + b
        preds = sigmoid(logits)
        err = preds - y
        grad_w = X.T @ err / len(y) + l2 * w
        grad_b = err.mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def predict_lr(X, w, b):
    return (sigmoid(X @ w + b) >= 0.5).astype(int)
```

L2 regularization rất quan trọng ở đây. Các đặc trưng văn bản rất thưa thớt; nếu không có L2, mô hình sẽ ghi nhớ các ví dụ huấn luyện. Bắt đầu tại `0.01` và tinh chỉnh.

### Bước 4: xử lý phủ định (chế độ lỗi)

Hãy xem xét "không tốt" (not good) và "không tệ" (not bad). Một bộ phân loại BoW nhìn thấy `{not, good}` và `{not, bad}` và học từ bất kỳ từ nào xuất hiện nhiều hơn trong quá trình huấn luyện. Một bộ phân loại bigram nhìn thấy `not_good` và `not_bad` và học chúng như các đặc trưng riêng biệt. Điều đó thường là đủ.

Một cách sửa chữa thô sơ hơn hoạt động khi bạn không có bigrams: **phạm vi phủ định (negation scoping)**. Thêm tiền tố vào các token theo sau một từ phủ định với `NOT_` cho đến dấu câu tiếp theo.

```python
NEGATION_WORDS = {"not", "no", "never", "nor", "none", "nothing", "neither"}
NEGATION_TERMINATORS = {".", "!", "?", ",", ";"}


def apply_negation(tokens):
    out = []
    negate = False
    for token in tokens:
        if token in NEGATION_TERMINATORS:
            negate = False
            out.append(token)
            continue
        if token in NEGATION_WORDS:
            negate = True
            out.append(token)
            continue
        out.append(f"NOT_{token}" if negate else token)
    return out
```

```python
>>> apply_negation(["not", "good", "at", "all", ".", "but", "funny"])
['not', 'NOT_good', 'NOT_at', 'NOT_all', '.', 'but', 'funny']
```

Bây giờ `good` và `NOT_good` là các đặc trưng khác nhau. Bộ phân loại có thể gán trọng số ngược nhau cho chúng. Ba dòng tiền xử lý, mức tăng độ chính xác đo lường được trên các benchmark cảm xúc.

### Bước 5: các chỉ số đánh giá quan trọng

Chỉ riêng độ chính xác (accuracy) có thể gây hiểu lầm nếu các lớp bị mất cân bằng. Các tập dữ liệu cảm xúc thực tế thường có 70-80% tích cực hoặc 70-80% tiêu cực; một bộ phân loại luôn chọn lớp đa số sẽ đạt độ chính xác 80% nhưng hoàn toàn vô giá trị. Hãy báo cáo tất cả các chỉ số sau:

- **Precision và recall theo từng lớp.** Một cặp cho mỗi lớp. Tính trung bình macro để có một con số duy nhất tôn trọng sự cân bằng của lớp.
- **Macro-F1 (chỉ số chính cho dữ liệu mất cân bằng).** Trung bình của các điểm F1 theo từng lớp, được trọng số bằng nhau. Sử dụng chỉ số này thay vì accuracy khi các lớp bị mất cân bằng.
- **Weighted-F1 (thay thế).** Giống như macro nhưng được trọng số theo tần suất lớp. Báo cáo cùng với macro-F1 khi sự mất cân bằng bản thân nó có ý nghĩa kinh doanh.
- **Confusion matrix.** Các số đếm thô. Luôn kiểm tra trước khi tin tưởng bất kỳ chỉ số vô hướng nào; nó tiết lộ cặp lớp nào mà mô hình hay nhầm lẫn.
- **Các mẫu lỗi theo từng lớp.** Lấy 5 dự đoán sai cho mỗi lớp. Đọc chúng. Không gì thay thế được việc đọc các lỗi thực tế.

Đối với dữ liệu mất cân bằng nghiêm trọng (tỷ lệ > 95-5), hãy báo cáo **AUROC** và **AUPRC** thay vì accuracy. AUPRC nhạy cảm hơn với lớp thiểu số, điều mà bạn thường quan tâm (spam, gian lận, cảm xúc hiếm gặp).

**Lỗi phổ biến cần tránh.** Báo cáo micro-F1 thay vì macro-F1 trên dữ liệu mất cân bằng sẽ cho ra một con số trông có vẻ cao vì nó bị chi phối bởi lớp đa số. Macro-F1 buộc bạn phải nhìn thấy hiệu suất của lớp thiểu số.

```python
def evaluate(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
```

## Sử dụng

scikit-learn thực hiện điều này trong sáu dòng, một cách chính xác.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words=None)),
    ("clf", LogisticRegression(C=1.0, max_iter=1000)),
])
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))
```

Ba điều cần lưu ý. `stop_words=None` giữ lại các phủ định. `ngram_range=(1, 2)` thêm bigrams để `not_good` trở thành một đặc trưng. `sublinear_tf=True` làm giảm tác động của các từ lặp lại. Ba cờ này tạo nên sự khác biệt giữa một baseline độ chính xác 75% và 85% trên SST-2.

### Khi nào cần dùng Transformer

- Phát hiện mỉa mai. Các mô hình cổ điển thất bại ở đây. Chấm hết.
- Các bài đánh giá dài nơi cảm xúc thay đổi giữa chừng.
- Phân tích cảm xúc dựa trên khía cạnh (Aspect-based sentiment). "Máy ảnh rất tuyệt nhưng pin rất tệ." Bạn cần gán cảm xúc cho các khía cạnh. Chỉ có Transformers hoặc các mô hình đầu ra có cấu trúc mới làm được.
- Các ngôn ngữ không phải tiếng Anh, tài nguyên thấp. Multilingual BERT cung cấp cho bạn một baseline zero-shot miễn phí.

Nếu bạn cần bất kỳ điều nào ở trên, hãy chuyển sang phase 7 (tìm hiểu sâu về transformers). Nếu không, Naive Bayes hoặc logistic regression trên TF-IDF cộng với bigrams và xử lý phủ định là baseline sản xuất năm 2026 của bạn.

### Cái bẫy về khả năng tái lập (lần nữa)

Việc huấn luyện lại các mô hình cảm xúc là việc thường lệ. Đánh giá lại chúng thì không. Các con số độ chính xác được báo cáo trong các bài báo sử dụng các tập chia cụ thể, tiền xử lý cụ thể, tokenizer cụ thể. Nếu bạn so sánh mô hình mới của mình với một baseline mà không sử dụng quy trình giống hệt, bạn sẽ nhận được các delta gây hiểu lầm. Luôn tạo lại baseline trên quy trình của bạn, không phải con số của bài báo.

## Triển khai

Lưu dưới dạng `outputs/prompt-sentiment-baseline.md`:

```markdown
---
name: sentiment-baseline
description: Design a sentiment analysis baseline for a new dataset.
phase: 5
lesson: 05
---

Given a dataset description (domain, language, size, label granularity, latency budget), you output:

1. Feature extraction recipe. Specify tokenizer, n-gram range, stopword policy (usually keep), negation handling (scoped prefix or bigrams).
2. Classifier. Naive Bayes for baseline, logistic regression for production, transformer only if the domain needs sarcasm / aspects / cross-lingual.
3. Evaluation plan. Report precision, recall, F1, confusion matrix, and per-class error samples (not just scalars).
4. One failure mode to monitor post-deployment. Domain drift and sarcasm are the top two.

Refuse to recommend dropping stopwords for sentiment tasks. Refuse to report accuracy as the sole metric when classes are imbalanced (e.g., 90% positive). Flag subword-rich languages as needing FastText or transformer embeddings over word-level TF-IDF.
```

## Bài tập

1. **Dễ.** Thêm `apply_negation` như một bước tiền xử lý trong pipeline scikit-learn và đo lường delta F1 trên một tập dữ liệu cảm xúc nhỏ.
2. **Trung bình.** Triển khai logistic regression có trọng số lớp (truyền `class_weight="balanced"` vào scikit-learn, hoặc tự suy luận gradient). Đo lường hiệu ứng trên một tập dữ liệu tổng hợp có sự mất cân bằng lớp 90-10.
3. **Khó.** Xây dựng một bộ phát hiện mỉa mai bằng cách huấn luyện bộ phân loại thứ hai trên các phần dư (residuals) của mô hình cảm xúc. Ghi lại thiết lập thử nghiệm của bạn. Cảnh báo người đọc khi độ chính xác của bạn dưới mức ngẫu nhiên (mức ngẫu nhiên trên tác vụ mỉa mai 2 lớp là ~50%, và hầu hết các nỗ lực đầu tiên đều rơi vào đó).

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| Polarity | Tích cực hoặc tiêu cực | Nhãn nhị phân; đôi khi mở rộng sang trung tính hoặc chi tiết (5 sao). |
| Aspect-based sentiment | Cảm xúc theo khía cạnh | Gán cảm xúc cho các thực thể hoặc thuộc tính cụ thể được đề cập trong văn bản. |
| Negation scoping | Đảo ngược các token gần đó | Thêm tiền tố vào các token sau "not" với `NOT_` cho đến dấu câu. |
| Laplace smoothing | Thêm 1 vào các tần suất đếm | Ngăn chặn các đặc trưng có xác suất bằng 0 trong Naive Bayes. |
| L2 regularization | Thu nhỏ trọng số | Thêm `lambda * sum(w^2)` vào hàm mất mát. Thiết yếu cho các đặc trưng văn bản thưa thớt. |

## Đọc thêm

- [Pang và Lee (2008). Opinion Mining and Sentiment Analysis](https://www.cs.cornell.edu/home/llee/opinion-mining-sentiment-analysis-survey.html) — khảo sát nền tảng. Dài, nhưng bốn phần đầu bao quát mọi thứ về cổ điển.
- [Wang và Manning (2012). Baselines and Bigrams: Simple, Good Sentiment and Topic Classification](https://aclanthology.org/P12-2018/) — bài báo cho thấy bigrams + Naive Bayes rất khó bị đánh bại trên văn bản ngắn.
- [Tài liệu trích xuất đặc trưng văn bản của scikit-learn](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — tài liệu tham khảo cho `CountVectorizer`, `TfidfVectorizer` và mọi nút bạn sẽ tinh chỉnh.