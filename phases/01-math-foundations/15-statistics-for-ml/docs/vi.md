# Thống kê cho Machine Learning

> Thống kê là cách bạn biết liệu mô hình của mình thực sự hiệu quả hay chỉ là may mắn.

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 1, Lessons 06 (Xác suất và Phân phối), 07 (Định lý Bayes)
**Time:** ~120 phút

## Mục tiêu học tập

- Tính toán các thống kê mô tả, tương quan Pearson/Spearman và ma trận hiệp phương sai từ đầu (from scratch)
- Thực hiện các kiểm định giả thuyết (t-test, chi-squared) và diễn giải p-value cũng như khoảng tin cậy một cách chính xác
- Sử dụng bootstrap resampling để xây dựng khoảng tin cậy cho bất kỳ chỉ số nào mà không cần giả định về phân phối
- Phân biệt ý nghĩa thống kê (statistical significance) và ý nghĩa thực tiễn (practical significance) bằng các thước đo cỡ hiệu ứng (effect size)

## Vấn đề

Bạn đã huấn luyện hai mô hình. Mô hình A đạt điểm 0.87 trên tập kiểm tra. Mô hình B đạt 0.89. Bạn triển khai Mô hình B. Ba tuần sau, các chỉ số thực tế lại tệ hơn trước. Chuyện gì đã xảy ra?

Mô hình B thực tế không vượt trội hơn Mô hình A. Sự khác biệt 0.02 chỉ là nhiễu. Tập kiểm tra của bạn quá nhỏ, hoặc phương sai quá cao, hoặc cả hai. Bạn đã đưa sự ngẫu nhiên vào sản phẩm dưới danh nghĩa cải tiến.

Điều này xảy ra liên tục. Các bảng xếp hạng Kaggle bị xáo trộn. Các bài báo không thể tái lập kết quả. Các thử nghiệm A/B tuyên bố người chiến thắng dựa trên vài trăm mẫu. Nguyên nhân gốc rễ luôn giống nhau: ai đó đã bỏ qua thống kê.

Thống kê cung cấp cho bạn các công cụ để phân biệt tín hiệu (signal) với nhiễu (noise). Nó cho bạn biết khi nào một sự khác biệt là có thật, bạn nên tự tin đến mức nào và bạn cần bao nhiêu dữ liệu trước khi có thể tin tưởng vào một kết quả. Mọi pipeline ML, mọi so sánh mô hình, mọi thử nghiệm đều cần thống kê. Nếu không có nó, bạn chỉ đang đoán mò.

## Khái niệm

### Thống kê mô tả: Tóm tắt dữ liệu của bạn

Trước khi lập mô hình bất cứ thứ gì, bạn cần biết dữ liệu của mình trông như thế nào. Thống kê mô tả nén một tập dữ liệu thành một vài con số nắm bắt được hình dạng của nó.

**Các thước đo xu hướng trung tâm** trả lời câu hỏi "điểm giữa nằm ở đâu?"

```
Mean:   sum of all values / count
        mu = (1/n) * sum(x_i)

Median: middle value when sorted
        Robust to outliers. If you have [1, 2, 3, 4, 1000], the mean is 202
        but the median is 3.

Mode:   most frequent value
        Useful for categorical data. For continuous data, rarely informative.
```

Trung bình (mean) là điểm cân bằng. Trung vị (median) là điểm ở giữa. Khi chúng khác nhau, phân phối của bạn bị lệch. Phân phối thu nhập có mean >> median (lệch phải do các tỷ phú). Phân phối mất mát (loss) trong quá trình huấn luyện thường có mean << median (lệch trái do các mẫu dễ).

**Các thước đo độ phân tán** trả lời câu hỏi "dữ liệu phân tán như thế nào?"

```
Variance:   average squared deviation from the mean
            sigma^2 = (1/n) * sum((x_i - mu)^2)

Standard deviation:  square root of variance
                     sigma = sqrt(sigma^2)
                     Same units as the data, so more interpretable.

Range:      max - min
            Sensitive to outliers. Almost never useful alone.

IQR:        Q3 - Q1 (interquartile range)
            The range of the middle 50% of the data.
            Robust to outliers. Used for box plots and outlier detection.
```

**Phân vị (Percentiles)** chia dữ liệu đã sắp xếp thành 100 phần bằng nhau. Phân vị thứ 25 (Q1) có nghĩa là 25% giá trị nằm dưới điểm này. Phân vị thứ 50 là trung vị. Phân vị thứ 75 là Q3.

```
For latency monitoring:
  P50 = median latency        (typical user experience)
  P95 = 95th percentile       (bad but not worst case)
  P99 = 99th percentile       (tail latency, often 10x the median)
```

Trong ML, bạn quan tâm đến phân vị cho độ trễ suy luận (inference latency), phân phối độ tin cậy của dự đoán và hiểu phân phối lỗi. Một mô hình có lỗi trung bình thấp nhưng lỗi P99 khủng khiếp có thể vô dụng đối với các ứng dụng quan trọng về an toàn.

**Thống kê mẫu so với quần thể.** Khi tính phương sai từ một mẫu, hãy chia cho (n-1) thay vì n. Đây là hiệu chỉnh Bessel. Nó bù đắp cho thực tế là trung bình mẫu của bạn không phải là trung bình quần thể thực sự. Với n ở mẫu số, bạn đánh giá thấp phương sai thực tế một cách hệ thống. Với (n-1), ước tính là không chệch.

```
Population variance: sigma^2 = (1/N) * sum((x_i - mu)^2)
Sample variance:     s^2     = (1/(n-1)) * sum((x_i - x_bar)^2)
```

Trong thực tế: nếu n lớn (hàng nghìn mẫu), sự khác biệt là không đáng kể. Nếu n nhỏ (vài chục mẫu), nó rất quan trọng.

### Tương quan: Cách các biến biến thiên cùng nhau

Tương quan đo lường cường độ và hướng của mối quan hệ tuyến tính giữa hai biến.

**Hệ số tương quan Pearson** đo lường sự liên kết tuyến tính:

```
r = sum((x_i - x_bar)(y_i - y_bar)) / (n * s_x * s_y)

r = +1:  perfect positive linear relationship
r = -1:  perfect negative linear relationship
r =  0:  no linear relationship (but there might be a nonlinear one!)

Range: [-1, 1]
```

Pearson giả định mối quan hệ là tuyến tính và cả hai biến đều xấp xỉ phân phối chuẩn. Nó nhạy cảm với các giá trị ngoại lai (outliers). Một điểm cực đoan duy nhất có thể kéo r từ 0.1 lên 0.9.

**Tương quan hạng Spearman** đo lường sự liên kết đơn điệu:

```
1. Replace each value with its rank (1, 2, 3, ...)
2. Compute Pearson correlation on the ranks

Spearman catches any monotonic relationship, not just linear.
If y = x^3, Pearson gives r < 1 but Spearman gives rho = 1.
```

**Khi nào sử dụng cái nào:**

```
Pearson:    Both variables are continuous and roughly normal.
            You care about the linear relationship specifically.
            No extreme outliers.

Spearman:   Ordinal data (rankings, ratings).
            Data is not normally distributed.
            You suspect a monotonic but not linear relationship.
            Outliers are present.
```

**Quy tắc vàng:** tương quan không ngụ ý quan hệ nhân quả. Doanh số bán kem và số ca tử vong do đuối nước có tương quan vì cả hai đều tăng vào mùa hè. Độ chính xác của mô hình và số lượng tham số có tương quan, nhưng việc thêm tham số không tự động cải thiện độ chính xác (xem: overfitting).

### Ma trận hiệp phương sai

Hiệp phương sai giữa hai biến đo lường cách chúng biến thiên cùng nhau:

```
Cov(X, Y) = (1/n) * sum((x_i - x_bar)(y_i - y_bar))

Cov(X, Y) > 0:  X and Y tend to increase together
Cov(X, Y) < 0:  when X increases, Y tends to decrease
Cov(X, Y) = 0:  no linear co-movement
```

Với d đặc trưng, ma trận hiệp phương sai C là ma trận d x d trong đó C[i][j] = Cov(feature_i, feature_j). Các mục trên đường chéo C[i][i] là phương sai của từng đặc trưng.

```
C = | Var(x1)      Cov(x1,x2)  Cov(x1,x3) |
    | Cov(x2,x1)  Var(x2)      Cov(x2,x3) |
    | Cov(x3,x1)  Cov(x3,x2)  Var(x3)     |

Properties:
  - Symmetric: C[i][j] = C[j][i]
  - Positive semi-definite: all eigenvalues >= 0
  - Diagonal = variances
  - Off-diagonal = covariances
```

**Kết nối với PCA.** PCA phân rã ma trận hiệp phương sai thành các giá trị riêng (eigendecomposition). Các vectơ riêng là các thành phần chính (hướng có phương sai lớn nhất). Các giá trị riêng cho bạn biết mỗi thành phần nắm bắt được bao nhiêu phương sai. Đây chính xác là những gì Bài 10 đã đề cập, nhưng bây giờ bạn thấy tại sao ma trận hiệp phương sai là thứ đúng đắn để phân rã: nó mã hóa tất cả các mối quan hệ tuyến tính theo cặp trong dữ liệu của bạn.

**Kết nối với tương quan.** Ma trận tương quan là ma trận hiệp phương sai của các biến đã chuẩn hóa (mỗi biến chia cho độ lệch chuẩn của nó). Tương quan chuẩn hóa hiệp phương sai để tất cả các giá trị nằm trong khoảng [-1, 1].

### Kiểm định giả thuyết

Kiểm định giả thuyết là một khuôn khổ để đưa ra quyết định trong điều kiện không chắc chắn. Bạn bắt đầu với một khẳng định, thu thập dữ liệu và xác định xem dữ liệu đó có nhất quán với khẳng định hay không.

**Thiết lập:**

```
Null hypothesis (H0):        the default assumption, usually "no effect"
Alternative hypothesis (H1): what you are trying to show

Example:
  H0: Model A and Model B have the same accuracy
  H1: Model B has higher accuracy than Model A
```

**p-value** là xác suất quan sát được dữ liệu cực đoan như những gì bạn đã thấy, giả sử H0 là đúng. Nó KHÔNG phải là xác suất để H0 đúng. Đây là hiểu lầm phổ biến nhất trong thống kê.

```
p-value = P(data this extreme | H0 is true)

If p-value < alpha (typically 0.05):
    Reject H0. The result is "statistically significant."
If p-value >= alpha:
    Fail to reject H0. You do not have enough evidence.
    This does NOT mean H0 is true.
```

**Khoảng tin cậy** cung cấp một phạm vi các giá trị hợp lý cho một tham số:

```
95% confidence interval for the mean:
    x_bar +/- z * (s / sqrt(n))

where z = 1.96 for 95% confidence

Interpretation: if you repeated this experiment many times, 95% of the
computed intervals would contain the true mean. It does NOT mean there
is a 95% probability the true mean is in this specific interval.
```

Độ rộng của khoảng tin cậy cho bạn biết về độ chính xác. Khoảng rộng nghĩa là độ không chắc chắn cao. Khoảng hẹp nghĩa là ước tính của bạn chính xác (nhưng không nhất thiết là đúng, nếu dữ liệu của bạn bị chệch).

### t-test

t-test so sánh các giá trị trung bình. Có một vài biến thể.

**One-sample t-test:** trung bình quần thể có khác với một giá trị giả định không?

```
t = (x_bar - mu_0) / (s / sqrt(n))

degrees of freedom = n - 1
```

**Two-sample t-test (độc lập):** trung bình của hai nhóm có khác nhau không?

```
t = (x_bar_1 - x_bar_2) / sqrt(s1^2/n1 + s2^2/n2)

This is Welch's t-test, which does not assume equal variances.
Always use Welch's unless you have a specific reason for equal variances.
```

**Paired t-test:** khi các phép đo đi theo cặp (cùng một mô hình được đánh giá trên cùng các phân đoạn dữ liệu):

```
Compute d_i = x_i - y_i for each pair
Then run a one-sample t-test on the d_i values against mu_0 = 0
```

Trong ML, paired t-test rất phổ biến: bạn chạy cả hai mô hình trên cùng 10 fold cross-validation và so sánh điểm số của chúng theo cặp.

### Kiểm định Chi-squared

Kiểm định Chi-squared kiểm tra xem tần suất quan sát được có khớp với tần suất kỳ vọng hay không. Hữu ích cho dữ liệu phân loại.

```
chi^2 = sum((observed - expected)^2 / expected)

Example: does a language model's output distribution match the
training distribution across categories?

Category    Observed   Expected
Positive       120        100
Negative        80        100
chi^2 = (120-100)^2/100 + (80-100)^2/100 = 4 + 4 = 8

With 1 degree of freedom, chi^2 = 8 gives p < 0.005.
The difference is significant.
```

### Thử nghiệm A/B cho các mô hình ML

Thử nghiệm A/B trong ML không giống như thử nghiệm A/B trên web. So sánh mô hình có những thách thức cụ thể:

```
1. Same test set:    Both models must be evaluated on identical data.
                     Different test sets make comparison meaningless.

2. Multiple metrics: Accuracy alone is not enough. You need precision,
                     recall, F1, latency, and fairness metrics.

3. Variance:         Use cross-validation or bootstrap to estimate
                     the variance of each metric, not just point estimates.

4. Data leakage:     If the test set was used during model selection,
                     your comparison is biased. Hold out a final test set.
```

**Quy trình:**

```
1. Define your metric and significance level (alpha = 0.05)
2. Run both models on the same k-fold cross-validation splits
3. Collect paired scores: [(a1, b1), (a2, b2), ..., (ak, bk)]
4. Compute differences: d_i = b_i - a_i
5. Run a paired t-test on the differences
6. Check: is the mean difference significantly different from 0?
7. Compute a confidence interval for the mean difference
8. Compute effect size (Cohen's d) to judge practical significance
```

### Ý nghĩa thống kê vs Ý nghĩa thực tiễn

Một kết quả có thể có ý nghĩa thống kê nhưng vô nghĩa về mặt thực tiễn. Với đủ dữ liệu, ngay cả một sự khác biệt nhỏ nhặt cũng trở nên có ý nghĩa thống kê.

```
Example:
  Model A accuracy: 0.9234
  Model B accuracy: 0.9237
  n = 1,000,000 test samples
  p-value = 0.001

Statistically significant? Yes.
Practically significant? A 0.03% improvement is not worth the
engineering cost of deploying a new model.
```

**Cỡ hiệu ứng (Effect size)** định lượng sự khác biệt lớn đến mức nào, độc lập với cỡ mẫu:

```
Cohen's d = (mean_1 - mean_2) / pooled_std

d = 0.2:  small effect
d = 0.5:  medium effect
d = 0.8:  large effect
```

Luôn báo cáo cả p-value và cỡ hiệu ứng. P-value cho bạn biết sự khác biệt có thật hay không. Cỡ hiệu ứng cho bạn biết nó có quan trọng hay không.

### Vấn đề so sánh nhiều lần (Multiple Comparison Problem)

Khi bạn kiểm tra nhiều giả thuyết, một số sẽ "có ý nghĩa" do ngẫu nhiên. Nếu bạn kiểm tra 20 thứ ở mức alpha = 0.05, bạn mong đợi 1 kết quả dương tính giả ngay cả khi không có gì là thật.

```
P(at least one false positive) = 1 - (1 - alpha)^m

m = 20 tests, alpha = 0.05:
P(false positive) = 1 - 0.95^20 = 0.64

You have a 64% chance of at least one false positive.
```

**Hiệu chỉnh Bonferroni:** chia alpha cho số lượng kiểm định.

```
Adjusted alpha = alpha / m = 0.05 / 20 = 0.0025

Only reject H0 if p-value < 0.0025.
Conservative but simple. Works when tests are independent.
```

Trong ML, điều này quan trọng khi bạn so sánh một mô hình trên nhiều chỉ số, kiểm tra nhiều cấu hình siêu tham số hoặc đánh giá trên nhiều tập dữ liệu.

### Phương pháp Bootstrap

Bootstrapping ước tính phân phối lấy mẫu của một thống kê bằng cách lấy mẫu lại dữ liệu của bạn có thay thế. Không yêu cầu giả định về phân phối cơ bản.

**Thuật toán:**

```
1. You have n data points
2. Draw n samples WITH replacement (some points appear multiple times,
   some not at all)
3. Compute your statistic on this bootstrap sample
4. Repeat B times (typically B = 1000 to 10000)
5. The distribution of bootstrap statistics approximates the
   sampling distribution
```

**Khoảng tin cậy Bootstrap (phương pháp phân vị):**

```
Sort the B bootstrap statistics
95% CI = [2.5th percentile, 97.5th percentile]
```

**Tại sao bootstrap quan trọng đối với ML:**

```
- Test set accuracy is a point estimate. Bootstrap gives you
  confidence intervals.
- You cannot assume metric distributions are normal (especially
  for AUC, F1, precision at k).
- Bootstrap works for ANY statistic: median, ratio of two means,
  difference in AUC between two models.
- No closed-form formula needed.
```

**Bootstrap để so sánh mô hình:**

```
1. You have predictions from Model A and Model B on the same test set
2. For each bootstrap iteration:
   a. Resample test indices with replacement
   b. Compute metric_A and metric_B on the resampled set
   c. Store diff = metric_B - metric_A
3. 95% CI for the difference:
   [2.5th percentile of diffs, 97.5th percentile of diffs]
4. If the CI does not contain 0, the difference is significant
```

Cách này mạnh mẽ hơn paired t-test vì nó không đưa ra các giả định về phân phối.

### Kiểm định tham số vs Phi tham số

**Kiểm định tham số** giả định một phân phối cụ thể (thường là chuẩn):

```
t-test:         assumes normally distributed data (or large n by CLT)
ANOVA:          assumes normality and equal variances
Pearson r:      assumes bivariate normality
```

**Kiểm định phi tham số** không đưa ra giả định phân phối nào:

```
Mann-Whitney U:     compares two groups (replaces independent t-test)
Wilcoxon signed-rank: compares paired data (replaces paired t-test)
Spearman rho:       correlation on ranks (replaces Pearson)
Kruskal-Wallis:     compares multiple groups (replaces ANOVA)
```

**Khi nào sử dụng phi tham số:**

```
- Small sample size (n < 30) and data is clearly non-normal
- Ordinal data (ratings, rankings)
- Heavy outliers you cannot remove
- Skewed distributions
```

**Khi nào sử dụng tham số:**

```
- Large sample size (CLT makes the test statistic approximately normal)
- Data is roughly symmetric without extreme outliers
- More statistical power (better at detecting real differences)
```

Trong các thử nghiệm ML, bạn thường có n nhỏ (5 hoặc 10 fold cross-validation), vì vậy các kiểm định phi tham số như Wilcoxon signed-rank thường phù hợp hơn t-test.

### Định lý giới hạn trung tâm (CLT): Ý nghĩa thực tiễn

CLT nói rằng phân phối của các trung bình mẫu tiến tới phân phối chuẩn khi n tăng lên, bất kể phân phối quần thể cơ bản là gì.

```
If X_1, X_2, ..., X_n are iid with mean mu and variance sigma^2:

    X_bar ~ Normal(mu, sigma^2 / n)    as n -> infinity

Works for n >= 30 in most cases.
For highly skewed distributions, you might need n >= 100.
```

**Tại sao điều này quan trọng đối với ML:**

```
1. Justifies confidence intervals and t-tests on aggregated metrics
2. Explains why averaging over cross-validation folds gives stable
   estimates even when individual folds vary wildly
3. Mini-batch gradient descent works because the average gradient
   over a batch approximates the true gradient (CLT in action)
4. Ensemble methods: averaging predictions from many models gives
   more stable output than any single model
```

**Những gì CLT KHÔNG làm:**

```
- Does NOT make your data normal. It makes the MEAN of samples normal.
- Does NOT work for heavy-tailed distributions with infinite variance
  (Cauchy distribution).
- Does NOT apply to dependent data (time series without correction).
```

### Các lỗi thống kê phổ biến trong các bài báo ML

1. **Kiểm tra trên tập huấn luyện.** Đảm bảo overfitting. Luôn giữ lại dữ liệu mà mô hình chưa bao giờ thấy trong quá trình huấn luyện.

2. **Không có khoảng tin cậy.** Báo cáo một con số độ chính xác duy nhất mà không có độ không chắc chắn khiến kết quả không thể tái lập và không thể kiểm chứng.

3. **Bỏ qua so sánh nhiều lần.** Kiểm tra 50 cấu hình và báo cáo cấu hình tốt nhất mà không hiệu chỉnh sẽ làm tăng tỷ lệ dương tính giả.

4. **Nhầm lẫn giữa ý nghĩa thống kê và thực tiễn.** P-value 0.001 trên mức cải thiện độ chính xác 0.01% là không có ý nghĩa.

5. **Sử dụng độ chính xác trên dữ liệu mất cân bằng.** Độ chính xác 99% trên tập dữ liệu có 99% lớp âm tính có nghĩa là mô hình không học được gì cả. Hãy sử dụng precision, recall, F1 hoặc AUC.

6. **Chọn lọc chỉ số (Cherry-picking).** Chỉ báo cáo chỉ số mà mô hình của bạn thắng. Đánh giá trung thực phải báo cáo tất cả các chỉ số liên quan.

7. **Rò rỉ thông tin giữa các phân đoạn train/test.** Chuẩn hóa trước khi chia tách, hoặc sử dụng dữ liệu tương lai để dự đoán quá khứ.

8. **Tập kiểm tra nhỏ không có ước tính phương sai.** Đánh giá trên 100 mẫu và tuyên bố cải thiện 2% là nhiễu, không phải tín hiệu.

9. **Giả định tính độc lập khi dữ liệu không độc lập.** Hình ảnh y tế từ cùng một bệnh nhân, nhiều câu từ cùng một tài liệu. Các quan sát trong một nhóm có tương quan với nhau.

10. **P-hacking.** Thử các kiểm định, tập con hoặc tiêu chí loại trừ khác nhau cho đến khi bạn nhận được p < 0.05. Kết quả chỉ là sản phẩm của quá trình tìm kiếm.

## Xây dựng

Bạn sẽ triển khai:

1. **Thống kê mô tả từ đầu** (mean, median, mode, standard deviation, percentiles, IQR)
2. **Các hàm tương quan** (Pearson và Spearman, với ma trận hiệp phương sai)
3. **Các kiểm định giả thuyết** (one-sample t-test, two-sample t-test, chi-squared test)
4. **Khoảng tin cậy Bootstrap** (cho bất kỳ thống kê nào, không cần giả định)
5. **Trình mô phỏng thử nghiệm A/B** (tạo dữ liệu, kiểm tra, kiểm tra lỗi Loại I và Loại II)
6. **Demo ý nghĩa thống kê vs thực tiễn** (cho thấy n lớn làm cho mọi thứ trở nên "có ý nghĩa")

Tất cả từ đầu, chỉ sử dụng `math` và `random`. Không numpy, không scipy.

```figure
f3-bootstrap-resample
```

## Thuật ngữ chính

| Thuật ngữ | Định nghĩa |
|---|---|
| Mean | Tổng các giá trị chia cho số lượng. Nhạy cảm với outliers. |
| Median | Giá trị ở giữa của dữ liệu đã sắp xếp. Mạnh mẽ với outliers. |
| Standard deviation | Căn bậc hai của phương sai. Đo độ phân tán theo đơn vị gốc. |
| Percentile | Giá trị mà dưới đó một tỷ lệ phần trăm dữ liệu nhất định rơi vào. |
| IQR | Khoảng tứ phân vị. Q3 trừ Q1. Độ phân tán của 50% dữ liệu ở giữa. |
| Pearson correlation | Đo lường sự liên kết tuyến tính giữa hai biến. Phạm vi [-1, 1]. |
| Spearman correlation | Đo lường sự liên kết đơn điệu sử dụng hạng. |
| Covariance matrix | Ma trận hiệp phương sai theo cặp giữa tất cả các đặc trưng. |
| Null hypothesis | Giả định mặc định không có tác động hoặc không có sự khác biệt. |
| p-value | Xác suất dữ liệu cực đoan như vậy nếu giả thuyết không (H0) là đúng. |
| Confidence interval | Phạm vi các giá trị hợp lý cho một tham số ở mức tin cậy nhất định. |
| t-test | Kiểm tra xem các giá trị trung bình có khác biệt đáng kể không. Sử dụng phân phối t. |
| Chi-squared test | Kiểm tra xem tần suất quan sát có khác với tần suất kỳ vọng không. |
| Effect size | Độ lớn của sự khác biệt, độc lập với cỡ mẫu. Cohen's d là phổ biến. |
| Bonferroni correction | Chia ngưỡng ý nghĩa cho số lượng kiểm định để kiểm soát dương tính giả. |
| Bootstrap | Lấy mẫu lại có thay thế để ước tính phân phối lấy mẫu. |
| Type I error | Dương tính giả. Bác bỏ H0 khi nó đúng. |
| Type II error | Âm tính giả. Không bác bỏ H0 khi nó sai. |
| Statistical power | Xác suất bác bỏ đúng một H0 sai. Power = 1 trừ tỷ lệ lỗi Loại II. |
| Central limit theorem | Các trung bình mẫu hội tụ về phân phối chuẩn khi cỡ mẫu tăng. |
| Parametric test | Giả định một phân phối cụ thể cho dữ liệu (thường là chuẩn). |
| Non-parametric test | Không đưa ra giả định phân phối. Hoạt động trên hạng hoặc dấu. |