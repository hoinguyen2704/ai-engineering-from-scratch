# Các Phương pháp Lấy mẫu (Sampling Methods)

> Lấy mẫu là cách AI khám phá không gian của các khả năng.

**Type:** Build
**Languages:** Python
**Prerequisites:** Giai đoạn 1, Bài 06-07 (Xác suất, Định lý Bayes)
**Time:** ~120 phút

## Mục tiêu học tập

- Triển khai inverse CDF, rejection sampling và importance sampling từ đầu chỉ sử dụng các số ngẫu nhiên phân phối đều (uniform random numbers)
- Xây dựng các phương pháp lấy mẫu temperature, top-k và top-p (nucleus) cho việc tạo token của mô hình ngôn ngữ
- Giải thích reparameterization trick và lý do tại sao nó cho phép lan truyền ngược (backpropagation) qua quá trình lấy mẫu trong các VAE
- Chạy Metropolis-Hastings MCMC để lấy mẫu từ một phân phối mục tiêu chưa chuẩn hóa

## Vấn đề

Một mô hình ngôn ngữ hoàn tất việc xử lý prompt của bạn và tạo ra một vector gồm 50.000 logit. Mỗi logit tương ứng với một token trong từ điển. Bây giờ nó phải chọn một token. Làm thế nào để chọn?

Nếu nó luôn chọn token có xác suất cao nhất, mọi phản hồi sẽ giống hệt nhau. Có tính quyết định (deterministic). Nhàm chán. Nếu nó chọn ngẫu nhiên đồng nhất, đầu ra sẽ là những từ vô nghĩa. Câu trả lời nằm ở đâu đó giữa hai thái cực này, và "đâu đó" đó được kiểm soát bởi việc lấy mẫu.

Lấy mẫu không chỉ giới hạn ở việc tạo văn bản. Học tăng cường (Reinforcement learning) ước tính gradient chính sách bằng cách lấy mẫu các quỹ đạo. Các VAE học các biểu diễn tiềm ẩn bằng cách lấy mẫu từ các phân phối đã học và lan truyền ngược qua tính ngẫu nhiên. Các mô hình khuếch tán (Diffusion models) tạo ra hình ảnh bằng cách lấy mẫu nhiễu và khử nhiễu lặp đi lặp lại. Các phương pháp Monte Carlo ước tính các tích phân không có lời giải dạng đóng. Các thuật toán MCMC khám phá các phân phối hậu nghiệm nhiều chiều mà không thể liệt kê hết.

Mọi hệ thống AI tạo sinh đều là một hệ thống lấy mẫu. Chiến lược lấy mẫu quyết định chất lượng, tính đa dạng và khả năng kiểm soát của đầu ra. Bài học này xây dựng mọi phương pháp lấy mẫu chính từ đầu, bắt đầu từ các số ngẫu nhiên phân phối đều và kết thúc với các kỹ thuật vận hành các LLM và mô hình tạo sinh hiện đại.

## Khái niệm

### Tại sao lấy mẫu lại quan trọng

Lấy mẫu xuất hiện trong bốn vai trò cơ bản trong AI và học máy:

**Tạo sinh (Generation).** Các mô hình ngôn ngữ, mô hình khuếch tán và GAN đều tạo ra đầu ra bằng cách lấy mẫu. Thuật toán lấy mẫu kiểm soát trực tiếp tính sáng tạo, sự mạch lạc và tính đa dạng. Temperature, top-k và nucleus sampling là những nút điều chỉnh mà các kỹ sư sử dụng hàng ngày.

**Huấn luyện (Training).** Stochastic gradient descent lấy mẫu các mini-batch. Dropout lấy mẫu các neuron để vô hiệu hóa. Tăng cường dữ liệu (Data augmentation) lấy mẫu các phép biến đổi ngẫu nhiên. Importance sampling tái trọng số các mẫu để giảm phương sai gradient trong học tăng cường (PPO, TRPO).

**Ước tính (Estimation).** Nhiều đại lượng trong ML không có lời giải dạng đóng. Hàm mất mát kỳ vọng trên một phân phối dữ liệu, hàm phân vùng của mô hình dựa trên năng lượng, bằng chứng (evidence) trong suy luận Bayes. Ước tính Monte Carlo xấp xỉ tất cả những điều này bằng cách lấy trung bình trên các mẫu.

**Khám phá (Exploration).** Các thuật toán MCMC khám phá các phân phối hậu nghiệm trong suy luận Bayes. Các chiến lược tiến hóa lấy mẫu các nhiễu loạn tham số. Thompson sampling cân bằng giữa khám phá và khai thác trong các bài toán bandit.

Thách thức cốt lõi: bạn chỉ có thể lấy mẫu trực tiếp từ các phân phối đơn giản (phân phối đều, phân phối chuẩn). Đối với mọi thứ khác, bạn cần một phương pháp để chuyển đổi các mẫu đơn giản thành các mẫu từ phân phối mục tiêu của bạn.

### Lấy mẫu ngẫu nhiên phân phối đều (Uniform Random Sampling)

Mọi phương pháp lấy mẫu đều bắt đầu từ đây. Một bộ tạo số ngẫu nhiên phân phối đều tạo ra các giá trị trong [0, 1) nơi mọi khoảng con có độ dài bằng nhau đều có xác suất bằng nhau.

```
U ~ Uniform(0, 1)

P(a <= U <= b) = b - a    for 0 <= a <= b <= 1

Properties:
  E[U] = 0.5
  Var(U) = 1/12
```

Để lấy mẫu đồng nhất từ một tập hợp rời rạc gồm n phần tử, hãy tạo U và trả về floor(n * U). Để lấy mẫu từ một khoảng liên tục [a, b], hãy tính a + (b - a) * U.

Thông tin quan trọng: một số ngẫu nhiên phân phối đều duy nhất chứa chính xác lượng ngẫu nhiên cần thiết để tạo ra một mẫu từ bất kỳ phân phối nào. Bí quyết là tìm ra phép biến đổi phù hợp.

### Phương pháp Inverse CDF (Inverse Transform Sampling)

Hàm phân phối tích lũy (CDF) ánh xạ các giá trị sang xác suất:

```
F(x) = P(X <= x)

Properties:
  F is non-decreasing
  F(-inf) = 0
  F(+inf) = 1
  F maps the real line to [0, 1]
```

Inverse CDF ánh xạ xác suất ngược lại các giá trị. Nếu U ~ Uniform(0, 1), thì X = F_inverse(U) tuân theo phân phối mục tiêu.

```
Algorithm:
  1. Generate u ~ Uniform(0, 1)
  2. Return F_inverse(u)

Why it works:
  P(X <= x) = P(F_inverse(U) <= x) = P(U <= F(x)) = F(x)
```

**Ví dụ về phân phối mũ:**

```
PDF: f(x) = lambda * exp(-lambda * x),   x >= 0
CDF: F(x) = 1 - exp(-lambda * x)

Solve F(x) = u for x:
  u = 1 - exp(-lambda * x)
  exp(-lambda * x) = 1 - u
  x = -ln(1 - u) / lambda

Since (1 - U) and U have the same distribution:
  x = -ln(u) / lambda
```

Phương pháp này hoạt động hoàn hảo khi bạn có thể viết ra F_inverse ở dạng đóng. Đối với phân phối chuẩn, không có inverse CDF dạng đóng, vì vậy chúng ta sử dụng các phương pháp khác (Box-Muller, hoặc xấp xỉ số).

**Phiên bản rời rạc:** Đối với các phân phối rời rạc, hãy xây dựng CDF dưới dạng tổng tích lũy, tạo U và tìm chỉ số đầu tiên mà tại đó tổng tích lũy vượt quá U. Đây là cách `sample_categorical` hoạt động trong Bài 06.

### Rejection Sampling

Khi bạn không thể nghịch đảo CDF nhưng có thể đánh giá PDF mục tiêu đến một hằng số, rejection sampling sẽ hoạt động.

```
Target distribution: p(x)  (can evaluate, possibly unnormalized)
Proposal distribution: q(x)  (can sample from)
Bound: M such that p(x) <= M * q(x) for all x

Algorithm:
  1. Sample x ~ q(x)
  2. Sample u ~ Uniform(0, 1)
  3. If u < p(x) / (M * q(x)), accept x
  4. Otherwise, reject and go to step 1

Acceptance rate = 1/M
```

Giới hạn M càng chặt, tỷ lệ chấp nhận càng cao. Trong không gian thấp chiều (1-3), rejection sampling hoạt động tốt. Trong không gian nhiều chiều, tỷ lệ chấp nhận giảm theo cấp số nhân vì hầu hết thể tích đề xuất bị từ chối. Đây là lời nguyền chiều cao (curse of dimensionality) đối với rejection sampling.

**Ví dụ: lấy mẫu từ phân phối chuẩn bị cắt cụt (truncated normal).** Sử dụng một đề xuất phân phối đều trên phạm vi bị cắt cụt. Bao đóng M là giá trị lớn nhất của PDF phân phối chuẩn trong phạm vi đó.

**Ví dụ: lấy mẫu từ một nửa hình tròn.** Đề xuất ngẫu nhiên đồng nhất trong hình chữ nhật bao quanh. Chấp nhận nếu điểm rơi vào bên trong nửa hình tròn. Đây là cách Monte Carlo tính số pi: tỷ lệ chấp nhận bằng tỷ lệ diện tích pi/4.

### Importance Sampling

Đôi khi bạn không cần các mẫu từ phân phối mục tiêu p(x). Bạn cần ước tính một kỳ vọng theo p(x), và bạn có các mẫu từ một phân phối khác q(x).

```
Goal: estimate E_p[f(x)] = integral of f(x) * p(x) dx

Rewrite:
  E_p[f(x)] = integral of f(x) * (p(x)/q(x)) * q(x) dx
            = E_q[f(x) * w(x)]

where w(x) = p(x) / q(x)  are the importance weights.

Estimator:
  E_p[f(x)] ~ (1/N) * sum(f(x_i) * w(x_i))    where x_i ~ q(x)
```

Điều này rất quan trọng trong học tăng cường. Trong PPO (Proximal Policy Optimization), bạn thu thập các quỹ đạo theo một chính sách cũ pi_old nhưng muốn tối ưu hóa một chính sách mới pi_new. Trọng số quan trọng là pi_new(a|s) / pi_old(a|s). PPO cắt các trọng số này để ngăn chính sách mới lệch quá xa so với chính sách cũ.

Phương sai của ước tính importance sampling phụ thuộc vào mức độ tương đồng giữa q và p. Nếu q rất khác p, một vài mẫu sẽ nhận được trọng số khổng lồ và chi phối ước tính. Self-normalized importance sampling chia cho tổng các trọng số để giảm vấn đề này:

```
E_p[f(x)] ~ sum(w_i * f(x_i)) / sum(w_i)
```

### Ước tính Monte Carlo

Ước tính Monte Carlo xấp xỉ các tích phân bằng cách lấy trung bình các mẫu ngẫu nhiên. Luật số lớn đảm bảo sự hội tụ.

```
Goal: estimate I = integral of g(x) dx over domain D

Method:
  1. Sample x_1, ..., x_N uniformly from D
  2. I ~ (Volume of D / N) * sum(g(x_i))

Error: O(1 / sqrt(N))   regardless of dimension
```

Tỷ lệ lỗi không phụ thuộc vào số chiều. Đây là lý do tại sao các phương pháp Monte Carlo chiếm ưu thế trong không gian nhiều chiều nơi việc tích phân dựa trên lưới là không thể.

**Ước tính số pi:**

```
Sample (x, y) uniformly from [-1, 1] x [-1, 1]
Count how many fall inside the unit circle: x^2 + y^2 <= 1
pi ~ 4 * (count inside) / (total count)
```

**Ước tính kỳ vọng:**

```
E[f(X)] ~ (1/N) * sum(f(x_i))    where x_i ~ p(x)

The sample mean converges to the true expectation.
Variance of the estimator = Var(f(X)) / N
```

### Markov Chain Monte Carlo (MCMC): Metropolis-Hastings

MCMC xây dựng một chuỗi Markov có phân phối dừng là phân phối mục tiêu p(x). Sau đủ số bước, các mẫu từ chuỗi sẽ là (xấp xỉ) các mẫu từ p(x).

```
Target: p(x)  (known up to a normalizing constant)
Proposal: q(x'|x)  (how to propose the next state given the current state)

Metropolis-Hastings algorithm:
  1. Start at some x_0
  2. For t = 1, 2, ..., T:
     a. Propose x' ~ q(x'|x_t)
     b. Compute acceptance ratio:
        alpha = [p(x') * q(x_t|x')] / [p(x_t) * q(x'|x_t)]
     c. Accept with probability min(1, alpha):
        - If u < alpha (u ~ Uniform(0,1)): x_{t+1} = x'
        - Otherwise: x_{t+1} = x_t
  3. Discard first B samples (burn-in)
  4. Return remaining samples
```

Đối với các đề xuất đối xứng (q(x'|x) = q(x|x')), tỷ lệ đơn giản hóa thành p(x')/p(x). Đây là thuật toán Metropolis gốc.

**Tại sao nó hoạt động.** Quy tắc chấp nhận đảm bảo cân bằng chi tiết (detailed balance): xác suất ở x và di chuyển đến x' bằng xác suất ở x' và di chuyển đến x. Cân bằng chi tiết ngụ ý rằng p(x) là phân phối dừng của chuỗi.

**Các cân nhắc thực tế:**
- Burn-in: loại bỏ các mẫu ban đầu trước khi chuỗi đạt trạng thái cân bằng
- Thinning: giữ lại mỗi mẫu thứ k để giảm tự tương quan
- Quy mô đề xuất: quá nhỏ thì chuỗi di chuyển chậm (tỷ lệ chấp nhận cao, khám phá chậm); quá lớn thì hầu hết các đề xuất bị từ chối (tỷ lệ chấp nhận thấp, bị kẹt tại chỗ)
- Tỷ lệ chấp nhận tối ưu cho một đề xuất Gaussian trong không gian nhiều chiều là khoảng 0,234

### Gibbs Sampling

Gibbs sampling là một trường hợp đặc biệt của MCMC cho các phân phối đa biến. Thay vì đề xuất một bước di chuyển trong tất cả các chiều cùng một lúc, nó cập nhật từng biến một từ phân phối có điều kiện của nó.

```
Target: p(x_1, x_2, ..., x_d)

Algorithm:
  For each iteration t:
    Sample x_1^{t+1} ~ p(x_1 | x_2^t, x_3^t, ..., x_d^t)
    Sample x_2^{t+1} ~ p(x_2 | x_1^{t+1}, x_3^t, ..., x_d^t)
    ...
    Sample x_d^{t+1} ~ p(x_d | x_1^{t+1}, x_2^{t+1}, ..., x_{d-1}^{t+1})
```

Gibbs sampling yêu cầu bạn có thể lấy mẫu từ mỗi phân phối có điều kiện p(x_i | x_{-i}). Điều này khá đơn giản đối với nhiều mô hình:
- Mạng Bayes: các điều kiện tuân theo cấu trúc đồ thị
- Hỗn hợp Gaussian: các điều kiện là Gaussian
- Mô hình Ising: điều kiện của mỗi spin chỉ phụ thuộc vào các hàng xóm của nó

Tỷ lệ chấp nhận luôn là 1 (mọi đề xuất đều được chấp nhận) vì việc lấy mẫu từ điều kiện chính xác tự động thỏa mãn cân bằng chi tiết.

**Hạn chế.** Khi các biến có tương quan cao, Gibbs sampling trộn chậm vì việc cập nhật từng biến một không thể thực hiện các bước di chuyển chéo lớn qua phân phối.

### Temperature Sampling (Được sử dụng trong LLM)

Các mô hình ngôn ngữ xuất ra các logit z_1, ..., z_V cho mỗi token trong từ điển. Softmax chuyển đổi chúng thành xác suất. Temperature thay đổi quy mô các logit trước khi qua softmax:

```
p_i = exp(z_i / T) / sum(exp(z_j / T))

T = 1.0: standard softmax (original distribution)
T -> 0:  argmax (deterministic, always picks highest logit)
T -> inf: uniform (all tokens equally likely)
T < 1.0: sharpens the distribution (more confident, less diverse)
T > 1.0: flattens the distribution (less confident, more diverse)
```

**Tại sao nó hoạt động.** Chia các logit cho T < 1 sẽ khuếch đại sự khác biệt giữa các logit. Nếu z_1 = 2 và z_2 = 1, chia cho T = 0.5 sẽ cho z_1/T = 4 và z_2/T = 2, làm cho khoảng cách lớn hơn. Sau softmax, token có logit cao nhất nhận được phần lớn hơn nhiều.

**Trong thực tế:**
- T = 0.0: giải mã tham lam (greedy decoding), tốt nhất cho Q&A thực tế
- T = 0.3-0.7: sáng tạo nhẹ, tốt cho việc tạo mã
- T = 0.7-1.0: cân bằng, tốt cho hội thoại chung
- T = 1.0-1.5: viết sáng tạo, brainstorming
- T > 1.5: ngẫu nhiên tăng dần, hiếm khi hữu ích

Temperature không thay đổi những token nào có thể xảy ra. Nó thay đổi khối lượng xác suất được phân bổ cho mỗi token.

### Top-k Sampling

Top-k sampling giới hạn tập ứng viên vào k token có xác suất cao nhất, sau đó chuẩn hóa lại và lấy mẫu từ tập giới hạn đó.

```
Algorithm:
  1. Compute softmax probabilities for all V tokens
  2. Sort tokens by probability (descending)
  3. Keep only the top k tokens
  4. Renormalize: p_i' = p_i / sum(p_j for j in top-k)
  5. Sample from the renormalized distribution

k = 1:  greedy decoding
k = V:  no filtering (standard sampling)
k = 40: typical setting, removes long tail of unlikely tokens
```

Top-k ngăn mô hình chọn các token cực kỳ khó xảy ra (lỗi chính tả, vô nghĩa) tồn tại trong phần đuôi dài của phân phối từ điển. Vấn đề: k là cố định bất kể ngữ cảnh. Khi mô hình tự tin (một token có xác suất 95%), k = 40 vẫn cho phép 39 lựa chọn thay thế. Khi mô hình không chắc chắn (xác suất được phân bổ trên 1000 token), k = 40 cắt bỏ các lựa chọn hợp lý.

### Top-p (Nucleus) Sampling

Top-p sampling điều chỉnh kích thước tập ứng viên một cách linh hoạt. Thay vì giữ một số lượng token cố định, nó giữ tập hợp nhỏ nhất các token có tổng xác suất vượt quá p.

```
Algorithm:
  1. Compute softmax probabilities for all V tokens
  2. Sort tokens by probability (descending)
  3. Find smallest k such that sum of top-k probabilities >= p
  4. Keep only those k tokens
  5. Renormalize and sample

p = 0.9:  keeps tokens covering 90% of probability mass
p = 1.0:  no filtering
p = 0.1:  very restrictive, nearly greedy
```

Khi mô hình tự tin, nucleus sampling giữ ít token (có thể 2-3). Khi mô hình không chắc chắn, nó giữ nhiều (có thể 200). Hành vi thích ứng này là lý do tại sao nucleus sampling thường tạo ra văn bản tốt hơn top-k.

**Các kết hợp phổ biến:**
- Temperature 0.7 + top-p 0.9: cài đặt đa năng tốt
- Temperature 0.0 (greedy): tốt nhất cho các tác vụ có tính quyết định
- Temperature 1.0 + top-k 50: cài đặt trong bài báo gốc của Fan et al. (2018)

Top-k và top-p có thể được kết hợp. Áp dụng top-k trước, sau đó top-p trên tập còn lại.

### Reparameterization Trick (Được sử dụng trong VAE)

Variational autoencoders (VAE) học bằng cách mã hóa đầu vào thành một phân phối trong không gian tiềm ẩn, lấy mẫu từ phân phối đó và giải mã mẫu trở lại. Vấn đề: bạn không thể lan truyền ngược qua một thao tác lấy mẫu.

```
Standard sampling (not differentiable):
  z ~ N(mu, sigma^2)

  The randomness blocks gradient flow.
  d/d_mu [sample from N(mu, sigma^2)] = ???
```

Reparameterization trick tách tính ngẫu nhiên khỏi các tham số:

```
Reparameterized sampling:
  epsilon ~ N(0, 1)          (fixed random noise, no parameters)
  z = mu + sigma * epsilon   (deterministic function of parameters)

  Now z is a deterministic, differentiable function of mu and sigma.
  d(z)/d(mu) = 1
  d(z)/d(sigma) = epsilon

  Gradients flow through mu and sigma.
```

Điều này hoạt động vì N(mu, sigma^2) có cùng phân phối với mu + sigma * N(0, 1). Thông tin quan trọng: di chuyển tính ngẫu nhiên sang một nguồn không có tham số (epsilon), sau đó biểu diễn mẫu dưới dạng một phép biến đổi khả vi của các tham số.

**Trong vòng lặp huấn luyện VAE:**
1. Encoder xuất ra mu và log(sigma^2) cho mỗi đầu vào
2. Lấy mẫu epsilon ~ N(0, 1)
3. Tính z = mu + sigma * epsilon
4. Giải mã z để tái tạo đầu vào
5. Lan truyền ngược qua các bước 4, 3, 2, 1 (có thể thực hiện vì bước 3 là khả vi)

Nếu không có reparameterization trick, VAE không thể được huấn luyện bằng lan truyền ngược tiêu chuẩn. Thông tin duy nhất này đã làm cho VAE trở nên thực tế.

### Gumbel-Softmax (Lấy mẫu phân loại khả vi)

Reparameterization trick hoạt động cho các phân phối liên tục (Gaussian). Đối với các phân phối phân loại rời rạc, chúng ta cần một cách tiếp cận khác. Gumbel-Softmax cung cấp một xấp xỉ khả vi cho việc lấy mẫu phân loại.

**Gumbel-Max trick (không khả vi):**

```
To sample from a categorical distribution with log-probabilities log(p_1), ..., log(p_k):
  1. Sample g_i ~ Gumbel(0, 1) for each category
     (g = -log(-log(u)), where u ~ Uniform(0, 1))
  2. Return argmax(log(p_i) + g_i)

This produces exact categorical samples.
```

**Gumbel-Softmax (xấp xỉ khả vi):**

```
Replace the hard argmax with a soft softmax:
  y_i = exp((log(p_i) + g_i) / tau) / sum(exp((log(p_j) + g_j) / tau))

tau (temperature) controls the approximation:
  tau -> 0:  approaches a one-hot vector (hard categorical)
  tau -> inf: approaches uniform (1/k, 1/k, ..., 1/k)
  tau = 1.0: soft approximation
```

Gumbel-Softmax tạo ra một sự thư giãn liên tục của một mẫu rời rạc. Đầu ra là một vector xác suất (soft one-hot) thay vì một hard one-hot. Các gradient chảy qua softmax. Trong quá trình lan truyền xuôi (forward pass) khi huấn luyện, bạn có thể sử dụng bộ ước tính "straight-through": sử dụng hard argmax cho forward pass nhưng sử dụng các gradient Gumbel-Softmax mềm cho backward pass.

**Ứng dụng:**
- Các biến tiềm ẩn rời rạc trong VAE
- Tìm kiếm kiến trúc thần kinh (chọn các thao tác rời rạc)
- Các cơ chế attention cứng (hard attention)
- Học tăng cường với các hành động rời rạc

### Stratified Sampling

Lấy mẫu Monte Carlo tiêu chuẩn có thể để lại các khoảng trống trong không gian mẫu do ngẫu nhiên. Stratified sampling buộc phải bao phủ đều bằng cách chia không gian thành các tầng (strata) và lấy mẫu từ mỗi tầng.

```
Standard Monte Carlo:
  Sample N points uniformly from [0, 1]
  Some regions may have clusters, others gaps

Stratified sampling:
  Divide [0, 1] into N equal strata: [0, 1/N), [1/N, 2/N), ..., [(N-1)/N, 1)
  Sample one point uniformly within each stratum
  x_i = (i + u_i) / N   where u_i ~ Uniform(0, 1),  i = 0, ..., N-1
```

Stratified sampling luôn có phương sai thấp hơn hoặc bằng so với Monte Carlo tiêu chuẩn:

```
Var(stratified) <= Var(standard Monte Carlo)

The improvement is largest when f(x) varies smoothly.
For piecewise-constant functions, stratified sampling is exact.
```

**Ứng dụng:**
- Tích phân số (quasi-Monte Carlo)
- Chia dữ liệu huấn luyện (đảm bảo cân bằng lớp trong mỗi fold)
- Importance sampling với phân tầng (kết hợp cả hai kỹ thuật)
- NeRF (Neural Radiance Fields) sử dụng stratified sampling dọc theo các tia camera

### Kết nối với các mô hình khuếch tán (Diffusion Models)

Các mô hình khuếch tán tạo ra hình ảnh thông qua một quá trình lấy mẫu. Quá trình xuôi (forward process) thêm nhiễu Gaussian vào một hình ảnh qua T bước cho đến khi nó trở thành nhiễu thuần túy. Quá trình ngược (reverse process) học cách khử nhiễu, khôi phục hình ảnh gốc từng bước một.

```
Forward process (known):
  x_t = sqrt(alpha_t) * x_{t-1} + sqrt(1 - alpha_t) * epsilon
  where epsilon ~ N(0, I)

  After T steps: x_T ~ N(0, I)  (pure noise)

Reverse process (learned):
  x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (1 - alpha_t)/sqrt(1 - alpha_bar_t) * epsilon_theta(x_t, t)) + sigma_t * z
  where z ~ N(0, I)

  Each denoising step is a sampling step.
```

Kết nối với các phương pháp trong bài học này:
- Mỗi bước khử nhiễu sử dụng reparameterization trick (lấy mẫu nhiễu, áp dụng phép biến đổi xác định)
- Lịch trình nhiễu {alpha_t} kiểm soát một dạng của temperature annealing
- Huấn luyện sử dụng ước tính Monte Carlo để xấp xỉ ELBO (evidence lower bound)
- Ancestral sampling trong các mô hình khuếch tán là một chuỗi Markov (mỗi bước chỉ phụ thuộc vào trạng thái hiện tại)

Toàn bộ quá trình tạo hình ảnh là lấy mẫu lặp đi lặp lại: bắt đầu từ nhiễu, và tại mỗi bước, lấy mẫu một phiên bản ít nhiễu hơn một chút dựa trên mô hình khử nhiễu đã học.

```figure
monte-carlo-pi
```

## Xây dựng

### Bước 1: Lấy mẫu phân phối đều và inverse CDF

```python
import math
import random

def sample_uniform(a, b):
    return a + (b - a) * random.random()

def sample_exponential_inverse_cdf(lam):
    u = random.random()
    return -math.log(u) / lam
```

Tạo 10.000 mẫu phân phối mũ và xác minh giá trị trung bình là 1/lambda.

### Bước 2: Rejection sampling

```python
def rejection_sample(target_pdf, proposal_sample, proposal_pdf, M):
    while True:
        x = proposal_sample()
        u = random.random()
        if u < target_pdf(x) / (M * proposal_pdf(x)):
            return x
```

Sử dụng rejection sampling để lấy mẫu từ phân phối chuẩn bị cắt cụt. Xác minh hình dạng bằng cách vẽ biểu đồ histogram của các mẫu.

### Bước 3: Importance sampling

```python
def importance_sampling_estimate(f, target_pdf, proposal_pdf, proposal_sample, n):
    total = 0
    for _ in range(n):
        x = proposal_sample()
        w = target_pdf(x) / proposal_pdf(x)
        total += f(x) * w
    return total / n
```

Ước tính E[X^2] theo phân phối chuẩn bằng cách sử dụng đề xuất phân phối đều. So sánh với câu trả lời đã biết (mu^2 + sigma^2).

### Bước 4: Ước tính Monte Carlo số pi

```python
def monte_carlo_pi(n):
    inside = 0
    for _ in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x*x + y*y <= 1:
            inside += 1
    return 4 * inside / n
```

### Bước 5: Metropolis-Hastings MCMC

```python
def metropolis_hastings(target_log_pdf, proposal_sample, proposal_log_pdf, x0, n_samples, burn_in):
    samples = []
    x = x0
    for i in range(n_samples + burn_in):
        x_new = proposal_sample(x)
        log_alpha = (target_log_pdf(x_new) + proposal_log_pdf(x, x_new)
                     - target_log_pdf(x) - proposal_log_pdf(x_new, x))
        if math.log(random.random()) < log_alpha:
            x = x_new
        if i >= burn_in:
            samples.append(x)
    return samples
```

Lấy mẫu từ một phân phối hai đỉnh (hỗn hợp của hai phân phối Gaussian). Trực quan hóa quỹ đạo của chuỗi.

### Bước 6: Gibbs sampling

```python
def gibbs_sampling_2d(conditional_x_given_y, conditional_y_given_x, x0, y0, n_samples, burn_in):
    x, y = x0, y0
    samples = []
    for i in range(n_samples + burn_in):
        x = conditional_x_given_y(y)
        y = conditional_y_given_x(x)
        if i >= burn_in:
            samples.append((x, y))
    return samples
```

### Bước 7: Temperature sampling

```python
def softmax(logits):
    max_l = max(logits)
    exps = [math.exp(z - max_l) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def temperature_sample(logits, temperature):
    scaled = [z / temperature for z in logits]
    probs = softmax(scaled)
    return sample_from_probs(probs)
```

Cho thấy temperature thay đổi phân phối đầu ra như thế nào đối với một tập hợp các logit token.

### Bước 8: Top-k và top-p sampling

```python
def top_k_sample(logits, k):
    indexed = sorted(enumerate(logits), key=lambda x: -x[1])
    top = indexed[:k]
    top_logits = [l for _, l in top]
    probs = softmax(top_logits)
    idx = sample_from_probs(probs)
    return top[idx][0]

def top_p_sample(logits, p):
    probs = softmax(logits)
    indexed = sorted(enumerate(probs), key=lambda x: -x[1])
    cumsum = 0
    selected = []
    for token_idx, prob in indexed:
        cumsum += prob
        selected.append((token_idx, prob))
        if cumsum >= p:
            break
    sel_probs = [pr for _, pr in selected]
    total = sum(sel_probs)
    sel_probs = [pr / total for pr in sel_probs]
    idx = sample_from_probs(sel_probs)
    return selected[idx][0]
```

### Bước 9: Reparameterization trick

```python
def reparam_sample(mu, sigma):
    epsilon = random.gauss(0, 1)
    return mu + sigma * epsilon

def reparam_gradient(mu, sigma, epsilon):
    dz_dmu = 1.0
    dz_dsigma = epsilon
    return dz_dmu, dz_dsigma
```

Chứng minh rằng các gradient chảy qua mẫu đã được reparameterized nhưng không chảy qua việc lấy mẫu trực tiếp.

### Bước 10: Gumbel-Softmax

```python
def gumbel_sample():
    u = random.random()
    return -math.log(-math.log(u))

def gumbel_softmax(logits, temperature):
    gumbels = [math.log(p) + gumbel_sample() for p in logits]
    return softmax([g / temperature for g in gumbels])
```

Cho thấy việc giảm temperature làm cho đầu ra tiến gần đến một vector one-hot như thế nào.

Các triển khai đầy đủ với tất cả các trực quan hóa nằm trong `code/sampling.py`.

## Sử dụng

Với NumPy và SciPy, các phiên bản sản xuất:

```python
import numpy as np

rng = np.random.default_rng(42)

exponential_samples = rng.exponential(scale=2.0, size=10000)
print(f"Exponential mean: {exponential_samples.mean():.4f} (expected 2.0)")

from scipy import stats
normal = stats.norm(loc=0, scale=1)
print(f"CDF at 1.96: {normal.cdf(1.96):.4f}")
print(f"Inverse CDF at 0.975: {normal.ppf(0.975):.4f}")

logits = np.array([2.0, 1.0, 0.5, 0.1, -1.0])
temperature = 0.7
scaled = logits / temperature
probs = np.exp(scaled - scaled.max()) / np.exp(scaled - scaled.max()).sum()
token = rng.choice(len(logits), p=probs)
print(f"Sampled token index: {token}")
```

Đối với MCMC ở quy mô lớn, hãy sử dụng các thư viện chuyên dụng:
- PyMC: mô hình hóa Bayes đầy đủ với NUTS (adaptive HMC)
- emcee: bộ lấy mẫu MCMC tập hợp
- NumPyro/JAX: MCMC tăng tốc GPU

Bạn đã xây dựng những thứ này từ đầu. Bây giờ bạn biết các lệnh gọi thư viện đang làm gì.

## Bài tập

1. Triển khai inverse CDF sampling cho phân phối Cauchy. CDF là F(x) = 0.5 + arctan(x)/pi. Tạo 10.000 mẫu và vẽ biểu đồ histogram so với PDF thực. Lưu ý các phần đuôi nặng (các giá trị cực đoan xa trung tâm).

2. Sử dụng rejection sampling để tạo các mẫu từ phân phối Beta(2, 5) bằng cách sử dụng đề xuất Uniform(0, 1). Vẽ các mẫu được chấp nhận so với PDF Beta thực. Tỷ lệ chấp nhận lý thuyết là bao nhiêu?

3. Ước tính tích phân của sin(x) từ 0 đến pi bằng Monte Carlo với 1.000, 10.000 và 100.000 mẫu. So sánh sai số ở mỗi cấp độ. Xác minh rằng sai số tỷ lệ thuận với O(1/sqrt(N)).

4. Triển khai Metropolis-Hastings để lấy mẫu từ phân phối 2D p(x, y) tỷ lệ thuận với exp(-(x^2 * y^2 + x^2 + y^2 - 8*x - 8*y) / 2). Vẽ các mẫu và quỹ đạo chuỗi. Thử nghiệm với các độ lệch chuẩn đề xuất khác nhau.

5. Xây dựng một bản demo tạo văn bản hoàn chỉnh: với từ điển 10 từ có logit, tạo các chuỗi 20 token bằng cách sử dụng (a) greedy, (b) temperature=0.7, (c) top-k=3, (d) top-p=0.9. So sánh tính đa dạng của đầu ra qua 5 lần chạy.

## Thuật ngữ chính

| Thuật ngữ | Mọi người nói | Ý nghĩa thực sự |
|------|----------------|----------------------|
| Sampling | "Rút các giá trị ngẫu nhiên" | Tạo các giá trị theo một phân phối xác suất. Cơ chế đằng sau mọi AI tạo sinh |
| Uniform distribution | "Tất cả đều có khả năng như nhau" | Mọi giá trị trong [a, b] đều có mật độ xác suất bằng nhau 1/(b-a). Điểm khởi đầu cho mọi phương pháp lấy mẫu |
| Inverse CDF | "Biến đổi xác suất" | F_inverse(U) chuyển đổi một mẫu phân phối đều thành một mẫu từ bất kỳ phân phối nào có CDF đã biết. Chính xác và hiệu quả |
| Rejection sampling | "Đề xuất và chấp nhận/từ chối" | Tạo từ một đề xuất đơn giản, chấp nhận với xác suất tỷ lệ thuận với tỷ lệ mục tiêu/đề xuất. Chính xác nhưng lãng phí mẫu |
| Importance sampling | "Tái trọng số các mẫu" | Ước tính kỳ vọng theo p(x) bằng cách sử dụng các mẫu từ q(x) bằng cách trọng số mỗi mẫu theo p(x)/q(x). Cốt lõi của PPO trong RL |
| Monte Carlo | "Trung bình các mẫu ngẫu nhiên" | Xấp xỉ các tích phân dưới dạng trung bình mẫu. Sai số O(1/sqrt(N)) bất kể số chiều |
| MCMC | "Bước đi ngẫu nhiên hội tụ" | Xây dựng một chuỗi Markov có phân phối dừng là mục tiêu. Metropolis-Hastings là thuật toán nền tảng |
| Metropolis-Hastings | "Chấp nhận đi lên, đôi khi đi xuống" | Đề xuất di chuyển, chấp nhận dựa trên tỷ lệ mật độ. Cân bằng chi tiết đảm bảo hội tụ về phân phối mục tiêu |
| Gibbs sampling | "Từng biến một" | Cập nhật mỗi biến từ phân phối có điều kiện của nó trong khi giữ các biến khác cố định. Tỷ lệ chấp nhận 100% |
| Temperature | "Nút điều chỉnh độ tự tin" | Chia các logit cho T trước khi softmax. T<1 làm sắc nét (tự tin hơn), T>1 làm phẳng (đa dạng hơn) |
| Top-k sampling | "Giữ k cái tốt nhất" | Đặt về 0 tất cả trừ k token có xác suất cao nhất, chuẩn hóa lại, lấy mẫu. Kích thước tập ứng viên cố định |
| Nucleus sampling (top-p) | "Giữ những cái có khả năng xảy ra" | Giữ tập hợp nhỏ nhất các token có tổng xác suất vượt quá p. Kích thước tập ứng viên thích ứng |
| Reparameterization trick | "Di chuyển tính ngẫu nhiên ra ngoài" | Viết z = mu + sigma * epsilon trong đó epsilon ~ N(0,1). Làm cho việc lấy mẫu trở nên khả vi. Cần thiết cho huấn luyện VAE |
| Gumbel-Softmax | "Lấy mẫu phân loại mềm" | Xấp xỉ khả vi cho lấy mẫu phân loại bằng cách sử dụng nhiễu Gumbel + softmax với temperature |
| Stratified sampling | "Bao phủ bắt buộc" | Chia không gian mẫu thành các tầng, lấy mẫu từ mỗi tầng. Luôn có phương sai thấp hơn Monte Carlo ngây thơ |
| Burn-in | "Thời gian khởi động" | Các mẫu MCMC ban đầu bị loại bỏ trước khi chuỗi đạt đến phân phối dừng của nó |
| Detailed balance | "Điều kiện đảo ngược" | p(x) * T(x->y) = p(y) * T(y->x). Điều kiện đủ để p là phân phối dừng của một chuỗi Markov |
| Diffusion sampling | "Khử nhiễu lặp đi lặp lại" | Tạo dữ liệu bằng cách bắt đầu từ nhiễu và áp dụng các bước khử nhiễu đã học. Mỗi bước là một thao tác lấy mẫu có điều kiện |

## Đọc thêm

- [Holbrook (2023): Thuật toán Metropolis-Hastings](https://arxiv.org/abs/2304.07010) - hướng dẫn chi tiết về nền tảng MCMC
- [Jang, Gu, Poole (2017): Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/abs/1611.01144) - bài báo gốc về Gumbel-Softmax
- [Holtzman et al. (2020): The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) - bài báo về nucleus (top-p) sampling
- [Kingma & Welling (2014): Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) - bài báo VAE giới thiệu reparameterization trick
- [Ho, Jain, Abbeel (2020): Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) - DDPM kết nối lấy mẫu với tạo hình ảnh