# Xác suất và Phân phối

> Xác suất là ngôn ngữ mà AI sử dụng để biểu đạt sự không chắc chắn.

**Type:** Learn
**Language:** Python
**Prerequisites:** Phase 1, Lessons 01-04
**Time:** ~75 minutes

## Learning Objectives

- Triển khai PMF và PDF từ đầu cho các phân phối Bernoulli, categorical, Poisson, uniform, và normal
- Tính toán giá trị kỳ vọng (expected value), phương sai (variance), và sử dụng Định lý Giới hạn Trung tâm (Central Limit Theorem) để giải thích tại sao phân phối Gaussian lại chiếm ưu thế
- Xây dựng hàm softmax và log-softmax với kỹ thuật ổn định số học (numerical stability trick - trừ đi logit lớn nhất)
- Tính toán hàm mất mát cross-entropy từ logits và kết nối nó với negative log-likelihood

## The Problem

Một bộ phân loại (classifier) xuất ra `[0.03, 0.91, 0.06]`. Một mô hình ngôn ngữ chọn từ tiếp theo từ 50,000 ứng viên. Một mô hình khuếch tán (diffusion model) tạo ra hình ảnh bằng cách lấy mẫu từ các phân phối đã học. Tất cả những điều này đều là xác suất đang hoạt động.

Mọi dự đoán mà một mô hình đưa ra đều là một phân phối xác suất. Mọi hàm mất mát (loss function) đều đo lường mức độ sai lệch giữa phân phối dự đoán và phân phối thực tế. Mọi bước huấn luyện đều điều chỉnh các tham số để làm cho phân phối này trông giống phân phối kia hơn. Nếu không có xác suất, bạn không thể đọc bất kỳ bài báo ML nào, gỡ lỗi (debug) một mô hình, hay hiểu tại sao loss khi huấn luyện của bạn lại là NaN.

## The Concept

### Biến cố, Không gian mẫu và Xác suất

Không gian mẫu S là tập hợp tất cả các kết quả có thể xảy ra. Một biến cố (event) là một tập con của không gian mẫu. Xác suất ánh xạ các biến cố thành các con số từ 0 đến 1.

```
Coin flip:
  S = {H, T}
  P(H) = 0.5,  P(T) = 0.5

Single die roll:
  S = {1, 2, 3, 4, 5, 6}
  P(even) = P({2, 4, 6}) = 3/6 = 0.5
```

Ba tiên đề định nghĩa toàn bộ xác suất:
1. P(A) >= 0 cho bất kỳ biến cố A nào
2. P(S) = 1 (điều gì đó luôn luôn xảy ra)
3. P(A hoặc B) = P(A) + P(B) khi A và B không thể cùng xảy ra

Mọi thứ khác (định lý Bayes, kỳ vọng, phân phối) đều bắt nguồn từ ba quy tắc này.

### Xác suất có điều kiện và Tính độc lập

P(A|B) là xác suất của A với điều kiện B đã xảy ra.

```
P(A|B) = P(A and B) / P(B)

Example: deck of cards
  P(King | Face card) = P(King and Face card) / P(Face card)
                      = (4/52) / (12/52)
                      = 4/12 = 1/3
```

Hai biến cố là độc lập khi việc biết một biến cố không cho bạn biết gì về biến cố kia:

```
Independent:   P(A|B) = P(A)
Equivalent to: P(A and B) = P(A) * P(B)
```

Tung đồng xu là các biến cố độc lập. Rút bài không hoàn lại thì không.

### Hàm khối xác suất (PMF) so với Hàm mật độ xác suất (PDF)

Các biến ngẫu nhiên rời rạc có hàm khối xác suất (PMF). Mỗi kết quả có một xác suất cụ thể mà bạn có thể đọc trực tiếp.

```
PMF: P(X = k)

Fair die:
  P(X = 1) = 1/6
  P(X = 2) = 1/6
  ...
  P(X = 6) = 1/6

  Sum of all probabilities = 1
```

Các biến ngẫu nhiên liên tục có hàm mật độ xác suất (PDF). Mật độ tại một điểm duy nhất không phải là xác suất. Xác suất có được từ việc tích phân mật độ trên một khoảng.

```
PDF: f(x)

P(a <= X <= b) = integral of f(x) from a to b

f(x) can be greater than 1 (density, not probability)
integral from -inf to +inf of f(x) dx = 1
```

Sự khác biệt này rất quan trọng trong ML. Đầu ra của bài toán phân loại là PMF (các lựa chọn rời rạc). Không gian ẩn (latent spaces) của VAE sử dụng PDF (liên tục).

### Các phân phối phổ biến

**Bernoulli:** một lần thử, hai kết quả. Mô hình hóa phân loại nhị phân (binary classification).

```
P(X = 1) = p
P(X = 0) = 1 - p
Mean = p,  Variance = p(1-p)
```

**Categorical:** một lần thử, k kết quả. Mô hình hóa phân loại đa lớp (multi-class classification - đầu ra softmax).

```
P(X = i) = p_i,  where sum of p_i = 1
Example: P(cat) = 0.7,  P(dog) = 0.2,  P(bird) = 0.1
```

**Uniform:** tất cả các kết quả có khả năng xảy ra như nhau. Được sử dụng để khởi tạo ngẫu nhiên (random initialization).

```
Discrete: P(X = k) = 1/n for k in {1, ..., n}
Continuous: f(x) = 1/(b-a) for x in [a, b]
```

**Normal (Gaussian):** đường cong hình chuông. Được tham số hóa bởi trung bình (mu) và phương sai (sigma^2).

```
f(x) = (1 / sqrt(2*pi*sigma^2)) * exp(-(x - mu)^2 / (2*sigma^2))

Standard normal: mu = 0, sigma = 1
  68% of data within 1 sigma
  95% within 2 sigma
  99.7% within 3 sigma
```

**Poisson:** đếm các biến cố hiếm gặp trong một khoảng thời gian cố định. Mô hình hóa tỷ lệ biến cố.

```
P(X = k) = (lambda^k * e^(-lambda)) / k!
Mean = lambda,  Variance = lambda
```

### Giá trị kỳ vọng và Phương sai

Giá trị kỳ vọng (expected value) là kết quả trung bình có trọng số.

```
Discrete:   E[X] = sum of x_i * P(X = x_i)
Continuous: E[X] = integral of x * f(x) dx
```

Phương sai (variance) đo lường mức độ phân tán quanh giá trị trung bình.

```
Var(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2
Standard deviation = sqrt(Var(X))
```

Trong ML, giá trị kỳ vọng xuất hiện dưới dạng hàm mất mát (loss trung bình trên phân phối dữ liệu). Phương sai cho bạn biết về độ ổn định của mô hình. Phương sai cao trong gradient có nghĩa là quá trình huấn luyện bị nhiễu (noisy).

### Phân phối đồng thời và Phân phối lề

Một phân phối đồng thời (joint distribution) P(X, Y) mô tả hai biến ngẫu nhiên cùng nhau.

Ví dụ về PMF đồng thời (X = thời tiết, Y = ô):

| | Y=0 (không mang ô) | Y=1 (mang ô) | Phân phối lề P(X) |
|---|---|---|---|
| X=0 (nắng) | 0.40 | 0.10 | P(X=0) = 0.50 |
| X=1 (mưa) | 0.05 | 0.45 | P(X=1) = 0.50 |
| **Phân phối lề P(Y)** | P(Y=0) = 0.45 | P(Y=1) = 0.55 | 1.00 |

Phân phối lề (marginal distribution) loại bỏ biến còn lại bằng cách lấy tổng:

```
P(X = x) = sum over all y of P(X = x, Y = y)
```

Tổng các hàng và cột trong bảng trên chính là các phân phối lề.

### Tại sao phân phối chuẩn xuất hiện ở khắp mọi nơi

Định lý Giới hạn Trung tâm (Central Limit Theorem): tổng (hoặc trung bình) của nhiều biến ngẫu nhiên độc lập hội tụ về phân phối chuẩn, bất kể phân phối ban đầu là gì.

```
Roll 1 die:  uniform distribution (flat)
Average of 2 dice:  triangular (peaked)
Average of 30 dice: nearly perfect bell curve

This works for ANY starting distribution.
```

Đây là lý do tại sao:
- Sai số đo lường xấp xỉ chuẩn (nhiều nguồn độc lập nhỏ)
- Khởi tạo trọng số trong mạng nơ-ron sử dụng phân phối chuẩn
- Nhiễu gradient trong SGD xấp xỉ chuẩn (tổng của nhiều gradient mẫu)
- Phân phối chuẩn là phân phối có entropy cực đại cho một giá trị trung bình và phương sai cho trước

### Log xác suất (Log Probabilities)

Xác suất thô gây ra các vấn đề về số học. Nhân nhiều xác suất nhỏ với nhau sẽ nhanh chóng bị tràn dưới (underflow) về không.

```
P(sentence) = P(word1) * P(word2) * ... * P(word_n)
            = 0.01 * 0.003 * 0.02 * ...
            -> 0.0 (underflow after ~30 terms)
```

Log xác suất khắc phục điều này. Phép nhân trở thành phép cộng.

```
log P(sentence) = log P(word1) + log P(word2) + ... + log P(word_n)
                = -4.6 + -5.8 + -3.9 + ...
                -> finite number (no underflow)
```

Quy tắc:
- log(a * b) = log(a) + log(b)
- log xác suất luôn <= 0 (vì 0 < P <= 1)
- Càng âm = càng ít khả năng xảy ra
- Cross-entropy loss là log xác suất âm của lớp đúng

### Softmax như một phân phối xác suất

Mạng nơ-ron xuất ra các điểm số thô (logits). Softmax chuyển đổi chúng thành một phân phối xác suất hợp lệ.

```
softmax(z_i) = exp(z_i) / sum(exp(z_j) for all j)

Properties:
  - All outputs are in (0, 1)
  - All outputs sum to 1
  - Preserves relative ordering of inputs
  - exp() amplifies differences between logits
```

Mẹo softmax: trừ đi logit lớn nhất trước khi lũy thừa để ngăn chặn tràn số (overflow).

```
z = [100, 101, 102]
exp(102) = overflow

z_shifted = z - max(z) = [-2, -1, 0]
exp(0) = 1  (safe)

Same result, no overflow.
```

Log-softmax kết hợp softmax và log để ổn định số học. PyTorch sử dụng điều này bên trong cho cross-entropy loss.

### Lấy mẫu (Sampling)

Lấy mẫu nghĩa là rút ra các giá trị ngẫu nhiên từ một phân phối. Trong ML:
- Dropout lấy mẫu ngẫu nhiên các nơ-ron để triệt tiêu (về 0)
- Tăng cường dữ liệu (Data augmentation) lấy mẫu các phép biến đổi ngẫu nhiên
- Mô hình ngôn ngữ lấy mẫu token tiếp theo từ phân phối dự đoán
- Mô hình khuếch tán lấy mẫu nhiễu và khử nhiễu dần dần

Lấy mẫu từ các phân phối tùy ý đòi hỏi các kỹ thuật như inverse transform sampling, rejection sampling, hoặc reparameterization trick (được sử dụng trong VAE).

```figure
gaussian-pdf
```

## Build It

### Bước 1: Cơ bản về xác suất

```python
import math
import random

def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def combinations(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))

def conditional_probability(p_a_and_b, p_b):
    return p_a_and_b / p_b

p_king_given_face = conditional_probability(4/52, 12/52)
print(f"P(King | Face card) = {p_king_given_face:.4f}")
```

### Bước 2: PMF và PDF từ đầu

```python
def bernoulli_pmf(k, p):
    return p if k == 1 else (1 - p)

def categorical_pmf(k, probs):
    return probs[k]

def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / factorial(k)

def uniform_pdf(x, a, b):
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0

def normal_pdf(x, mu, sigma):
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exponent)
```

### Bước 3: Giá trị kỳ vọng và phương sai

```python
def expected_value(values, probabilities):
    return sum(v * p for v, p in zip(values, probabilities))

def variance(values, probabilities):
    mu = expected_value(values, probabilities)
    return sum(p * (v - mu) ** 2 for v, p in zip(values, probabilities))

die_values = [1, 2, 3, 4, 5, 6]
die_probs = [1/6] * 6
mu = expected_value(die_values, die_probs)
var = variance(die_values, die_probs)
print(f"Die: E[X] = {mu:.4f}, Var(X) = {var:.4f}, SD = {var**0.5:.4f}")
```

### Bước 4: Lấy mẫu từ các phân phối

```python
def sample_bernoulli(p, n=1):
    return [1 if random.random() < p else 0 for _ in range(n)]

def sample_categorical(probs, n=1):
    cumulative = []
    total = 0
    for p in probs:
        total += p
        cumulative.append(total)
    samples = []
    for _ in range(n):
        r = random.random()
        for i, c in enumerate(cumulative):
            if r <= c:
                samples.append(i)
                break
    return samples

def sample_normal_box_muller(mu, sigma, n=1):
    samples = []
    for _ in range(n):
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        samples.append(mu + sigma * z)
    return samples
```

### Bước 5: Softmax và log xác suất

```python
def softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    exps = [math.exp(z) for z in shifted]
    total = sum(exps)
    return [e / total for e in exps]

def log_softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = max_logit + math.log(sum(math.exp(z) for z in shifted))
    return [z - log_sum_exp for z in logits]

def cross_entropy_loss(logits, target_index):
    log_probs = log_softmax(logits)
    return -log_probs[target_index]
```

### Bước 6: Minh họa Định lý Giới hạn Trung tâm

```python
def demonstrate_clt(dist_fn, n_samples, n_averages):
    averages = []
    for _ in range(n_averages):
        samples = [dist_fn() for _ in range(n_samples)]
        averages.append(sum(samples) / len(samples))
    return averages
```

### Bước 7: Trực quan hóa

```python
import matplotlib.pyplot as plt

xs = [mu + sigma * (i - 500) / 100 for i in range(1001)]
ys = [normal_pdf(x, mu, sigma) for x, mu, sigma in ...]
plt.plot(xs, ys)
```

Các bản triển khai đầy đủ với tất cả các hình ảnh trực quan có trong `code/probability.py`.

## Use It

Với NumPy và SciPy, mọi thứ ở trên chỉ là các câu lệnh một dòng:

```python
import numpy as np
from scipy import stats

normal = stats.norm(loc=0, scale=1)
samples = normal.rvs(size=10000)
print(f"Mean: {np.mean(samples):.4f}, Std: {np.std(samples):.4f}")
print(f"P(X < 1.96) = {normal.cdf(1.96):.4f}")

logits = np.array([2.0, 1.0, 0.1])
from scipy.special import softmax, log_softmax
probs = softmax(logits)
log_probs = log_softmax(logits)
print(f"Softmax: {probs}")
print(f"Log-softmax: {log_probs}")
```

Bạn đã xây dựng chúng từ đầu. Bây giờ bạn đã biết các hàm thư viện đang thực hiện những gì.

## Exercises

1. Triển khai inverse transform sampling cho phân phối mũ (exponential distribution). Xác minh bằng cách lấy mẫu 10,000 giá trị và so sánh biểu đồ histogram với PDF thực tế.

2. Xây dựng bảng phân phối đồng thời cho hai con xúc xắc bị gắn chì (loaded dice). Tính toán các phân phối lề và kiểm tra xem các con xúc xắc có độc lập hay không.

3. Tính toán cross-entropy loss cho một bộ phân loại 5 lớp xuất ra logits `[2.0, 0.5, -1.0, 3.0, 0.1]` khi lớp đúng là chỉ số 3. Sau đó xác minh câu trả lời của bạn với `nn.CrossEntropyLoss` của PyTorch.

4. Viết một hàm nhận vào một danh sách các log xác suất và trả về chuỗi có khả năng xảy ra nhất, tổng log xác suất, và xác suất thô tương đương. Kiểm tra nó với một câu gồm 50 từ, trong đó mỗi từ có xác suất 0.01.

## Key Terms

| Thuật ngữ | Cách mọi người nói | Ý nghĩa thực sự |
|------|----------------|----------------------|
| Sample space | "Tất cả các khả năng" | Tập hợp S của mọi kết quả có thể xảy ra của một thí nghiệm |
| PMF | "Hàm xác suất" | Một hàm cung cấp xác suất chính xác của từng kết quả rời rạc, tổng bằng 1 |
| PDF | "Đường cong xác suất" | Một hàm mật độ cho các biến liên tục. Tích phân nó trên một khoảng để có xác suất |
| Conditional probability | "Xác suất khi biết điều gì đó" | P(A\|B) = P(A và B) / P(B). Nền tảng của tư duy Bayes và định lý Bayes |
| Independence | "Chúng không ảnh hưởng lẫn nhau" | P(A và B) = P(A) * P(B). Biết một biến cố không cho bạn biết gì về biến cố kia |
| Expected value | "Giá trị trung bình" | Tổng có trọng số xác suất của tất cả các kết quả. Hàm mất mát là một giá trị kỳ vọng |
| Variance | "Độ phân tán" | Độ lệch bình phương kỳ vọng so với giá trị trung bình. Phương sai cao = ước tính bị nhiễu, không ổn định |
| Normal distribution | "Đường cong hình chuông" | f(x) = (1/sqrt(2*pi*sigma^2)) * exp(-(x-mu)^2/(2*sigma^2)). Xuất hiện khắp nơi nhờ CLT |
| Central Limit Theorem | "Các giá trị trung bình trở nên chuẩn hóa" | Giá trị trung bình của nhiều mẫu độc lập hội tụ về phân phối chuẩn bất kể nguồn gốc |
| Joint distribution | "Hai biến cùng nhau" | P(X, Y) mô tả xác suất của mọi sự kết hợp giữa các kết quả của X và Y |
| Marginal distribution | "Loại bỏ biến còn lại" | P(X) = sum_y P(X, Y). Khôi phục phân phối của một biến từ phân phối đồng thời |
| Log probability | "Log của xác suất" | log P(x). Chuyển tích thành tổng, ngăn chặn tràn dưới số học trong các chuỗi dài |
| Softmax | "Chuyển điểm số thành xác suất" | softmax(z_i) = exp(z_i) / sum(exp(z_j)). Ánh xạ các logits giá trị thực thành một phân phối xác suất hợp lệ |
| Cross-entropy | "Hàm mất mát" | -sum(p_true * log(p_predicted)). Đo lường sự khác biệt giữa hai phân phối. Càng thấp càng tốt |
| Logits | "Đầu ra thô của mô hình" | Điểm số chưa chuẩn hóa trước khi qua softmax. Được đặt tên theo hàm logistic |
| Sampling | "Rút ra các giá trị ngẫu nhiên" | Tạo ra các giá trị theo một phân phối xác suất. Cách các mô hình tạo ra đầu ra |

## Further Reading

- [3Blue1Brown: But what is the Central Limit Theorem?](https://www.youtube.com/watch?v=zeJD6dqJ5lo) - minh họa trực quan tại sao các giá trị trung bình trở nên chuẩn hóa
- [Stanford CS229 Probability Review](https://cs229.stanford.edu/section/cs229-prob.pdf) - tài liệu tham khảo súc tích bao gồm mọi thứ ở đây và hơn thế nữa
- [The Log-Sum-Exp Trick](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/) - tại sao ổn định số học lại quan trọng và cách đạt được nó