# Độ ổn định số học (Numerical Stability)

> Số dấu phẩy động (Floating point) là một sự trừu tượng bị rò rỉ (leaky abstraction). Nó sẽ gây rắc rối cho bạn trong quá trình huấn luyện, và bạn sẽ không lường trước được điều đó.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 1, Lessons 01-04
**Time:** ~120 minutes

## Learning Objectives

- Triển khai softmax và log-sum-exp ổn định về mặt số học bằng thủ thuật trừ giá trị cực đại (max-subtraction trick)
- Nhận diện hiện tượng tràn số trên (overflow), tràn số dưới (underflow) và triệt tiêu thảm khốc (catastrophic cancellation) trong các tính toán số dấu phẩy động
- Xác minh gradient giải tích (analytical gradients) so với gradient số học (numerical gradients) bằng phương pháp sai phân hữu hạn trung tâm (centered finite differences)
- Giải thích tại sao bfloat16 được ưu tiên hơn float16 trong huấn luyện và cách loss scaling ngăn chặn hiện tượng gradient bị tràn số dưới

## Vấn đề

Mô hình của bạn huấn luyện trong ba giờ, sau đó loss trở thành NaN. Bạn thêm một câu lệnh print. Các logits vẫn ổn ở bước 9,000. Tại bước 9,001 chúng là `inf`. Đến bước 9,002, mọi gradient đều là `nan` và quá trình huấn luyện coi như kết thúc.

Hoặc: mô hình của bạn huấn luyện xong nhưng độ chính xác thấp hơn 2% so với những gì bài báo khoa học công bố. Bạn kiểm tra mọi thứ. Kiến trúc khớp. Siêu tham số khớp. Dữ liệu khớp. Vấn đề là bài báo đó sử dụng float32 còn bạn sử dụng float16 mà không có kỹ thuật scaling phù hợp. Ba mươi hai bit sai số làm tròn tích lũy đã âm thầm làm giảm độ chính xác của bạn.

Hoặc: bạn tự triển khai hàm cross-entropy loss. Nó hoạt động với các logits nhỏ. Khi logits vượt quá 100, nó trả về `inf`. Hàm softmax bị tràn số vì `exp(100)` lớn hơn mức mà float32 có thể biểu diễn. Mọi ML framework đều xử lý việc này bằng một thủ thuật hai dòng. Bạn đã không biết thủ thuật đó tồn tại.

Độ ổn định số học không phải là một mối quan tâm lý thuyết. Nó là ranh giới giữa một lần huấn luyện thành công và một lần thất bại trong im lặng. Mọi lỗi ML nghiêm trọng mà bạn sẽ phải debug cuối cùng đều quy về số dấu phẩy động.

## Khái niệm

### IEEE 754: Cách máy tính lưu trữ số thực

Máy tính lưu trữ số thực dưới dạng các giá trị dấu phẩy động theo tiêu chuẩn IEEE 754. Một số float có ba phần: một bit dấu (sign bit), phần mũ (exponent) và phần định trị (mantissa/significand).

```
Float32 layout (32 bits total):
[1 sign] [8 exponent] [23 mantissa]

Value = (-1)^sign * 2^(exponent - 127) * 1.mantissa
```

Phần định trị quyết định độ chính xác (có bao nhiêu chữ số có nghĩa). Phần mũ quyết định phạm vi (số đó có thể lớn hoặc nhỏ đến mức nào).

```
Format     Bits   Exponent  Mantissa  Decimal digits  Range (approx)
float64    64     11        52        ~15-16          +/- 1.8e308
float32    32     8         23        ~7-8            +/- 3.4e38
float16    16     5         10        ~3-4            +/- 65,504
bfloat16   16     8         7         ~2-3            +/- 3.4e38
```

float32 cung cấp cho bạn khoảng 7 chữ số thập phân chính xác. Điều đó có nghĩa là nó có thể phân biệt giữa 1.0000001 và 1.0000002, nhưng không thể phân biệt giữa 1.00000001 và 1.00000002. Sau 7 chữ số, mọi thứ chỉ là nhiễu làm tròn.

float16 cung cấp cho bạn khoảng 3 chữ số chính xác. Số lớn nhất mà nó có thể biểu diễn là 65,504. Đó là một con số nhỏ đến mức đáng lo ngại đối với ML, nơi mà logits, gradients và activations thường xuyên vượt quá giá trị này.

bfloat16 là câu trả lời của Google cho vấn đề về phạm vi của float16. Nó có cùng phần mũ 8-bit như float32 (cùng phạm vi, lên đến 3.4e38) nhưng chỉ có 7 bit định trị (độ chính xác thấp hơn float16). Đối với việc huấn luyện mạng thần kinh, phạm vi quan trọng hơn độ chính xác, vì vậy bfloat16 thường giành chiến thắng.

### Tại sao 0.1 + 0.2 != 0.3

Số 0.1 không thể được biểu diễn chính xác trong hệ nhị phân dấu phẩy động. Trong hệ cơ số 2, nó là một số thập phân vô hạn tuần hoàn:

```
0.1 in binary = 0.0001100110011001100110011... (repeating forever)
```

Float32 cắt ngắn số này còn 23 bit định trị. Giá trị được lưu trữ xấp xỉ là 0.100000001490116. Tương tự, 0.2 được lưu trữ xấp xỉ là 0.200000002980232. Tổng của chúng là 0.300000004470348, không phải 0.3.

```
In Python:
>>> 0.1 + 0.2
0.30000000000000004

>>> 0.1 + 0.2 == 0.3
False
```

Điều này quan trọng đối với ML vì:

1. Các phép so sánh loss như `if loss < threshold` có thể đưa ra kết quả sai
2. Việc tích lũy nhiều giá trị nhỏ (cập nhật gradient qua hàng ngàn bước) sẽ bị lệch khỏi tổng thực tế
3. Các bài kiểm tra checksum và tính tái lập (reproducibility) sẽ thất bại nếu bạn so sánh các số float bằng `==`

Cách khắc phục: không bao giờ so sánh các số float bằng `==`. Hãy sử dụng `abs(a - b) < epsilon` hoặc `math.isclose()`.

### Triệt tiêu thảm khốc (Catastrophic Cancellation)

Khi bạn trừ hai số dấu phẩy động gần bằng nhau, các chữ số có nghĩa sẽ triệt tiêu nhau và bạn chỉ còn lại nhiễu làm tròn được đẩy lên thành các chữ số dẫn đầu.

```
a = 1.0000001    (stored as 1.00000011920929 in float32)
b = 1.0000000    (stored as 1.00000000000000 in float32)

True difference:  0.0000001
Computed:         0.00000011920929

Relative error: 19.2%
```

Đó là sai số tương đối 19% chỉ từ một phép trừ duy nhất. Trong ML, điều này xảy ra bất cứ khi nào bạn:

- Tính phương sai của dữ liệu có giá trị trung bình lớn: `E[x^2] - E[x]^2` khi E[x] lớn
- Trừ các log-probabilities gần bằng nhau
- Tính gradient bằng sai phân hữu hạn với epsilon quá nhỏ

Cách khắc phục: sắp xếp lại các công thức để tránh trừ các số lớn và gần bằng nhau. Đối với phương sai, hãy sử dụng thuật toán Welford hoặc chuẩn hóa dữ liệu (center the data) trước. Đối với log-probabilities, hãy luôn làm việc trong không gian log.

### Tràn số trên (Overflow) và Tràn số dưới (Underflow)

Tràn số trên xảy ra khi kết quả quá lớn để biểu diễn. Tràn số dưới xảy ra khi kết quả quá nhỏ (gần bằng 0 hơn cả số dương nhỏ nhất có thể biểu diễn).

```
Float32 boundaries:
  Maximum:  3.4028235e+38
  Minimum positive (normal): 1.175e-38
  Minimum positive (denorm): 1.401e-45
  Overflow:  anything > 3.4e38 becomes inf
  Underflow: anything < 1.4e-45 becomes 0.0
```

Hàm `exp()` là nguồn gây tràn số trên chính trong ML:

```
exp(88.7)  = 3.40e+38   (barely fits in float32)
exp(89.0)  = inf         (overflow)
exp(-87.3) = 1.18e-38   (barely above underflow)
exp(-104)  = 0.0         (underflow to zero)
```

Hàm `log()` lại gặp vấn đề theo hướng ngược lại:

```
log(0.0)   = -inf
log(-1.0)  = nan
log(1e-45) = -103.3      (fine)
log(1e-46) = -inf        (input underflowed to 0, then log(0) = -inf)
```

Trong ML, `exp()` xuất hiện trong các tính toán softmax, sigmoid và xác suất. `log()` xuất hiện trong cross-entropy, log-likelihoods và KL divergence. Sự kết hợp `log(exp(x))` là một "bãi mìn" nếu không có các thủ thuật phù hợp.

### Thủ thuật Log-Sum-Exp (The Log-Sum-Exp Trick)

Tính toán `log(sum(exp(x_i)))` trực tiếp là một việc làm nguy hiểm về mặt số học. Nếu bất kỳ `x_i` nào lớn, `exp(x_i)` sẽ bị tràn số trên. Nếu tất cả `x_i` đều rất âm, mọi `exp(x_i)` sẽ bị tràn số dưới về 0 và `log(0)` sẽ là `-inf`.

Thủ thuật: trừ đi giá trị cực đại trước khi thực hiện hàm mũ.

```
log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))
```

Tại sao cách này hoạt động: sau khi trừ `max(x)`, số mũ lớn nhất là `exp(0) = 1`. Không thể xảy ra tràn số trên. Có ít nhất một số hạng trong tổng là 1, vì vậy tổng ít nhất là 1, và `log(1) = 0`. Không thể xảy ra tràn số dưới về `-inf`.

Chứng minh:

```
log(sum(exp(x_i)))
= log(sum(exp(x_i - c + c)))                    (add and subtract c)
= log(sum(exp(x_i - c) * exp(c)))               (exp(a+b) = exp(a)*exp(b))
= log(exp(c) * sum(exp(x_i - c)))               (factor out exp(c))
= c + log(sum(exp(x_i - c)))                    (log(a*b) = log(a) + log(b))
```

Đặt `c = max(x)` và hiện tượng tràn số bị loại bỏ.

Thủ thuật này xuất hiện ở khắp mọi nơi trong ML:
- Chuẩn hóa Softmax
- Tính toán Cross-entropy loss
- Tổng log-probability trong các mô hình chuỗi (sequence models)
- Mixture of Gaussians
- Suy diễn biến phân (Variational inference)

### Tại sao Softmax cần thủ thuật trừ giá trị cực đại

Softmax chuyển đổi logits thành xác suất:

```
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

Nếu không có thủ thuật này, các logits như [100, 101, 102] sẽ gây tràn số:

```
exp(100) = 2.69e43
exp(101) = 7.31e43
exp(102) = 1.99e44
sum      = 2.99e44

These overflow float32 (max ~3.4e38)? No, 2.69e43 < 3.4e38? Actually:
exp(88.7) is already at the float32 limit.
exp(100) = inf in float32.
```

Với thủ thuật này, trừ đi max(x) = 102:

```
exp(100 - 102) = exp(-2) = 0.135
exp(101 - 102) = exp(-1) = 0.368
exp(102 - 102) = exp(0)  = 1.000
sum = 1.503

softmax = [0.090, 0.245, 0.665]
```

Các xác suất là đồng nhất. Việc tính toán là an toàn. Đây không phải là một sự tối ưu hóa. Đây là một yêu cầu bắt buộc để đảm bảo tính chính xác.

### NaN và Inf: Phát hiện và Ngăn chặn

`nan` (Not a Number) và `inf` (vô cực) lan truyền như virus qua các phép tính. Một giá trị `nan` trong quá trình cập nhật gradient sẽ làm cho trọng số trở thành `nan`, khiến mọi đầu ra tiếp theo đều là `nan`. Quá trình huấn luyện sẽ hỏng chỉ trong một bước.

Cách `inf` xuất hiện:
- `exp()` của một số dương lớn
- Chia cho không: `1.0 / 0.0`
- Tràn số `float32` trong các phép tích lũy

Cách `nan` xuất hiện:
- `0.0 / 0.0`
- `inf - inf`
- `inf * 0`
- `sqrt()` của một số âm
- `log()` của một số âm
- Bất kỳ phép toán số học nào liên quan đến một `nan` đã tồn tại

Phát hiện:

```python
import math

math.isnan(x)       # True if x is nan
math.isinf(x)       # True if x is +inf or -inf
math.isfinite(x)    # True if x is neither nan nor inf
```

Chiến lược ngăn chặn:

1. Kẹp (Clamp) đầu vào trong khoảng `exp()`: `exp(clamp(x, -80, 80))`
2. Thêm một lượng epsilon nhỏ vào mẫu số: `x / (y + 1e-8)`
3. Thêm epsilon vào bên trong `log()`: `log(x + 1e-8)`
4. Sử dụng các triển khai ổn định (log-sum-exp, stable softmax)
5. Gradient clipping để ngăn chặn sự bùng nổ trọng số
6. Kiểm tra `nan`/`inf` sau mỗi lượt forward pass trong quá trình debug

### Kiểm tra Gradient số học (Numerical Gradient Checking)

Gradient giải tích (từ backpropagation) có thể có lỗi. Kiểm tra gradient số học xác minh chúng bằng cách tính toán gradient bằng sai phân hữu hạn.

Công thức sai phân trung tâm:

```
df/dx ~= (f(x + h) - f(x - h)) / (2h)
```

Công thức này có độ chính xác O(h^2), tốt hơn nhiều so với sai phân tiến `(f(x+h) - f(x)) / h` vốn chỉ có độ chính xác O(h).

Chọn h: nếu quá lớn, giá trị xấp xỉ sẽ sai. Nếu quá nhỏ, hiện tượng triệt tiêu thảm khốc sẽ phá hỏng kết quả. `h = 1e-5` đến `1e-7` là các giá trị điển hình.

Cách kiểm tra: tính sai số tương đối giữa gradient giải tích và gradient số học.

```
relative_error = |grad_analytical - grad_numerical| / max(|grad_analytical|, |grad_numerical|, 1e-8)
```

Quy tắc ngón tay cái:
- relative_error < 1e-7: hoàn hảo, gradient chính xác
- relative_error < 1e-5: chấp nhận được, có lẽ là chính xác
- relative_error > 1e-3: có gì đó sai sót
- relative_error > 1: gradient hoàn toàn sai

Luôn kiểm tra gradient khi triển khai một layer hoặc hàm loss mới. PyTorch cung cấp `torch.autograd.gradcheck()` cho việc này.

### Huấn luyện với độ chính xác hỗn hợp (Mixed Precision Training)

Các GPU hiện đại có phần cứng chuyên dụng (Tensor Cores) giúp tính toán các phép nhân ma trận float16 nhanh hơn từ 2-8 lần so với float32. Huấn luyện độ chính xác hỗn hợp tận dụng điều này:

```
1. Maintain float32 master copy of weights
2. Forward pass in float16 (fast)
3. Compute loss in float32 (prevents overflow)
4. Backward pass in float16 (fast)
5. Scale gradients to float32
6. Update float32 master weights
```

Vấn đề với việc huấn luyện thuần float16: các gradient thường rất nhỏ (1e-8 hoặc nhỏ hơn). Float16 sẽ làm tràn số dưới bất kỳ giá trị nào dưới ~6e-8 về 0. Mô hình của bạn ngừng học vì tất cả các cập nhật gradient đều bằng không.

Cách khắc phục là loss scaling:

```
1. Multiply loss by a large scale factor (e.g., 1024)
2. Backward pass computes gradients of (loss * 1024)
3. All gradients are 1024x larger (pushed above float16 underflow)
4. Divide gradients by 1024 before updating weights
5. Net effect: same update, but no underflow
```

Dynamic loss scaling tự động điều chỉnh hệ số tỷ lệ. Bắt đầu với một giá trị lớn (65536). Nếu gradient bị tràn số thành `inf`, hãy giảm một nửa giá trị đó. Nếu sau N bước không có hiện tượng tràn số, hãy gấp đôi nó.

### bfloat16 vs float16: Tại sao bfloat16 thắng thế trong huấn luyện

```
float16:   [1 sign] [5 exponent]  [10 mantissa]
bfloat16:  [1 sign] [8 exponent]  [7 mantissa]
```

float16 có độ chính xác cao hơn (10 bit định trị so với 7) nhưng phạm vi hạn chế (tối đa ~65,504). bfloat16 có độ chính xác thấp hơn nhưng có cùng phạm vi với float32 (tối đa ~3.4e38).

Đối với việc huấn luyện mạng thần kinh:

- Activations và logits thường xuyên vượt quá 65,504 trong các đợt tăng đột biến khi huấn luyện. float16 sẽ bị tràn số; bfloat16 xử lý được điều này.
- Loss scaling là bắt buộc với float16 nhưng thường không cần thiết với bfloat16 vì phạm vi của nó bao phủ được phổ độ lớn của gradient.
- bfloat16 là một sự cắt ngắn đơn giản của float32: bỏ đi 16 bit cuối của phần định trị. Việc chuyển đổi là tầm thường và không gây mất mát ở phần mũ.

float16 được ưu tiên cho suy diễn (inference) nơi các giá trị đã được giới hạn và độ chính xác quan trọng hơn. bfloat16 được ưu tiên cho huấn luyện nơi phạm vi quan trọng hơn. Đây là lý do tại sao các TPU và GPU NVIDIA hiện đại (A100, H100) hỗ trợ bfloat16 nguyên bản.

### Gradient Clipping

Bùng nổ gradient (Exploding gradients) xảy ra khi gradient tăng theo cấp số nhân qua nhiều lớp (thường gặp trong RNN, mạng sâu và transformer). Một gradient lớn duy nhất có thể làm hỏng toàn bộ trọng số trong một bước.

Hai loại clipping:

**Clip theo giá trị (Clip by value):** kẹp từng phần tử gradient một cách độc lập.

```
grad = clamp(grad, -max_val, max_val)
```

Đơn giản nhưng có thể làm thay đổi hướng của vector gradient.

**Clip theo chuẩn (Clip by norm):** tỷ lệ hóa toàn bộ vector gradient sao cho chuẩn của nó không vượt quá một ngưỡng nhất định.

```
if ||grad|| > max_norm:
    grad = grad * (max_norm / ||grad||)
```

Giữ nguyên hướng của gradient. Đây là những gì `torch.nn.utils.clip_grad_norm_()` thực hiện. Đây là lựa chọn tiêu chuẩn.

Các giá trị điển hình: `max_norm=1.0` cho transformer, `max_norm=0.5` cho RL, `max_norm=5.0` cho các mạng đơn giản hơn.

Gradient clipping không phải là một thủ thuật tạm bợ (hack). Nó là một cơ chế an toàn. Nếu không có nó, một batch dữ liệu ngoại lai duy nhất có thể tạo ra một gradient đủ lớn để phá hỏng công sức huấn luyện của nhiều tuần.

### Các lớp chuẩn hóa như bộ ổn định số học

Batch normalization, layer normalization và RMS normalization thường được giới thiệu như các bộ điều chuẩn (regularizers) giúp quá trình huấn luyện hội tụ. Chúng cũng là các bộ ổn định số học.

Nếu không có chuẩn hóa, activations có thể tăng hoặc giảm theo cấp số nhân qua các lớp:

```
Layer 1: values in [0, 1]
Layer 5: values in [0, 100]
Layer 10: values in [0, 10,000]
Layer 50: values in [0, inf]
```

Chuẩn hóa giúp tái định tâm (recenter) và tái tỷ lệ (rescale) activations ở mọi lớp:

```
LayerNorm(x) = (x - mean(x)) / (std(x) + epsilon) * gamma + beta
```

Giá trị `epsilon` (thường là 1e-5) ngăn chặn việc chia cho không khi tất cả activations giống hệt nhau. Các tham số học được `gamma` và `beta` cho phép mạng khôi phục bất kỳ tỷ lệ nào nó cần.

Điều này giữ cho các giá trị nằm trong phạm vi an toàn về mặt số học trong toàn bộ mạng, ngăn chặn cả hiện tượng tràn số trên trong lượt forward pass và bùng nổ gradient trong lượt backward pass.

### Các lỗi số học ML thường gặp

**Lỗi: Loss là NaN sau vài epoch.**
Nguyên nhân: logits tăng quá lớn, softmax bị tràn số. Hoặc learning rate quá cao và trọng số bị phân kỳ.
Khắc phục: sử dụng stable softmax (trừ giá trị cực đại), giảm learning rate, thêm gradient clipping.

**Lỗi: Loss bị kẹt ở log(num_classes).**
Nguyên nhân: đầu ra của mô hình là các xác suất gần như đồng nhất. Thường có nghĩa là gradient đang biến mất (vanishing) hoặc mô hình hoàn toàn không học được gì.
Khắc phục: kiểm tra xem nhãn dữ liệu có chính xác không, xác minh hàm loss, kiểm tra các ReLU bị "chết" (dead ReLUs).

**Lỗi: Độ chính xác trên tập validation thấp hơn dự kiến từ 1-3%.**
Nguyên nhân: sử dụng mixed precision mà không có loss scaling phù hợp. Hiện tượng tràn số dưới của gradient âm thầm triệt tiêu các cập nhật nhỏ.
Khắc phục: bật dynamic loss scaling, hoặc chuyển sang bfloat16.

**Lỗi: Chuẩn gradient (gradient norms) bằng 0.0 ở một số lớp.**
Nguyên nhân: các neuron ReLU bị chết (tất cả đầu vào đều âm), hoặc float16 bị tràn số dưới.
Khắc phục: sử dụng LeakyReLU hoặc GELU, sử dụng gradient scaling, kiểm tra khởi tạo trọng số.

**Lỗi: Mô hình hoạt động trên một GPU nhưng cho kết quả khác trên GPU khác.**
Nguyên nhân: thứ tự tích lũy số dấu phẩy động không xác định. Các phép toán song song trên GPU thực hiện tổng theo các thứ tự khác nhau trên các phần cứng khác nhau, và phép cộng số dấu phẩy động không có tính kết hợp.
Khắc phục: chấp nhận các sai số nhỏ (1e-6), hoặc thiết lập `torch.use_deterministic_algorithms(True)` và chấp nhận việc giảm tốc độ.

**Lỗi: `exp()` trả về `inf` trong tính toán loss.**
Nguyên nhân: logits thô được đưa vào `exp()` mà không có thủ thuật trừ giá trị cực đại.
Khắc phục: sử dụng `torch.nn.functional.log_softmax()`, hàm này đã triển khai log-sum-exp bên trong.

**Lỗi: Quá trình huấn luyện bị phân kỳ sau khi chuyển từ float32 sang float16.**
Nguyên nhân: float16 không thể biểu diễn độ lớn gradient dưới 6e-8 hoặc activations trên 65,504.
Khắc phục: sử dụng mixed precision với loss scaling (AMP), hoặc sử dụng bfloat16 thay thế.

```figure
logsumexp-stability
```

## Build It

### Bước 1: Minh họa giới hạn độ chính xác của số dấu phẩy động

```python
print("=== Floating Point Precision ===")
print(f"0.1 + 0.2 = {0.1 + 0.2}")
print(f"0.1 + 0.2 == 0.3? {0.1 + 0.2 == 0.3}")
print(f"Difference: {(0.1 + 0.2) - 0.3:.2e}")
```

### Bước 2: Triển khai softmax ngây thơ (naive) vs ổn định (stable)

```python
import math

def softmax_naive(logits):
    exps = [math.exp(z) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def softmax_stable(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

safe_logits = [2.0, 1.0, 0.1]
print(f"Naive:  {softmax_naive(safe_logits)}")
print(f"Stable: {softmax_stable(safe_logits)}")

dangerous_logits = [100.0, 101.0, 102.0]
print(f"Stable: {softmax_stable(dangerous_logits)}")
# softmax_naive(dangerous_logits) would return [nan, nan, nan]
```

### Bước 3: Triển khai log-sum-exp ổn định

```python
def logsumexp_naive(values):
    return math.log(sum(math.exp(v) for v in values))

def logsumexp_stable(values):
    c = max(values)
    return c + math.log(sum(math.exp(v - c) for v in values))

safe = [1.0, 2.0, 3.0]
print(f"Naive:  {logsumexp_naive(safe):.6f}")
print(f"Stable: {logsumexp_stable(safe):.6f}")

large = [500.0, 501.0, 502.0]
print(f"Stable: {logsumexp_stable(large):.6f}")
# logsumexp_naive(large) returns inf
```

### Bước 4: Triển khai cross-entropy ổn định

```python
def cross_entropy_naive(true_class, logits):
    probs = softmax_naive(logits)
    return -math.log(probs[true_class])

def cross_entropy_stable(true_class, logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = math.log(sum(math.exp(s) for s in shifted))
    log_prob = shifted[true_class] - log_sum_exp
    return -log_prob

logits = [2.0, 5.0, 1.0]
true_class = 1
print(f"Naive:  {cross_entropy_naive(true_class, logits):.6f}")
print(f"Stable: {cross_entropy_stable(true_class, logits):.6f}")
```

### Bước 5: Kiểm tra gradient (Gradient checking)

```python
def numerical_gradient(f, x, h=1e-5):
    grad = []
    for i in range(len(x)):
        x_plus = x[:]
        x_minus = x[:]
        x_plus[i] += h
        x_minus[i] -= h
        grad.append((f(x_plus) - f(x_minus)) / (2 * h))
    return grad

def check_gradient(analytical, numerical, tolerance=1e-5):
    for i, (a, n) in enumerate(zip(analytical, numerical)):
        denom = max(abs(a), abs(n), 1e-8)
        rel_error = abs(a - n) / denom
        status = "OK" if rel_error < tolerance else "FAIL"
        print(f"  param {i}: analytical={a:.8f} numerical={n:.8f} "
              f"rel_error={rel_error:.2e} [{status}]")

def f(params):
    x, y = params
    return x**2 + 3*x*y + y**3

def f_grad(params):
    x, y = params
    return [2*x + 3*y, 3*x + 3*y**2]

point = [2.0, 1.0]
analytical = f_grad(point)
numerical = numerical_gradient(f, point)
check_gradient(analytical, numerical)
```

## Use It

### Mô phỏng độ chính xác hỗn hợp (Mixed precision simulation)

```python
import struct

def float32_to_float16_round(x):
    packed = struct.pack('f', x)
    f32 = struct.unpack('f', packed)[0]
    packed16 = struct.pack('e', f32)
    return struct.unpack('e', packed16)[0]

def simulate_bfloat16(x):
    packed = struct.pack('f', x)
    as_int = int.from_bytes(packed, 'little')
    truncated = as_int & 0xFFFF0000
    repacked = truncated.to_bytes(4, 'little')
    return struct.unpack('f', repacked)[0]
```

### Gradient clipping

```python
def clip_by_norm(gradients, max_norm):
    total_norm = math.sqrt(sum(g**2 for g in gradients))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        return [g * scale for g in gradients]
    return gradients

grads = [10.0, 20.0, 30.0]
clipped = clip_by_norm(grads, max_norm=5.0)
print(f"Original norm: {math.sqrt(sum(g**2 for g in grads)):.2f}")
print(f"Clipped norm:  {math.sqrt(sum(g**2 for g in clipped)):.2f}")
print(f"Direction preserved: {[c/clipped[0] for c in clipped]} == {[g/grads[0] for g in grads]}")
```

### Phát hiện NaN/Inf

```python
def check_tensor(name, values):
    has_nan = any(math.isnan(v) for v in values)
    has_inf = any(math.isinf(v) for v in values)
    if has_nan or has_inf:
        print(f"WARNING {name}: nan={has_nan} inf={has_inf}")
        return False
    return True

check_tensor("good", [1.0, 2.0, 3.0])
check_tensor("bad",  [1.0, float('nan'), 3.0])
check_tensor("ugly", [1.0, float('inf'), 3.0])
```

Xem `code/numerical.py` để biết các triển khai hoàn chỉnh với tất cả các trường hợp biên được minh họa.

## Ship It

Bài học này tạo ra:
- `code/numerical.py` với stable softmax, log-sum-exp, cross-entropy, gradient checking và mô phỏng mixed precision
- `outputs/prompt-numerical-debugger.md` để chẩn đoán NaN/Inf và các vấn đề số học trong huấn luyện

Các triển khai ổn định này sẽ xuất hiện lại trong Phase 3 khi xây dựng vòng lặp huấn luyện và trong Phase 4 khi triển khai các cơ chế attention.

## Exercises

1. **Triệt tiêu thảm khốc.** Tính phương sai của [1000000.0, 1000001.0, 1000002.0] bằng công thức ngây thơ `E[x^2] - E[x]^2` trong float32. Sau đó tính toán bằng thuật toán online của Welford. So sánh các sai số với phương sai thực (0.6667).

2. **Săn tìm độ chính xác.** Tìm giá trị float32 dương nhỏ nhất `x` sao cho `1.0 + x == 1.0` trong Python. Đây chính là machine epsilon. Xác minh xem nó có khớp với `numpy.finfo(numpy.float32).eps` không.

3. **Các trường hợp biên của Log-sum-exp.** Kiểm tra hàm `logsumexp_stable` của bạn với: (a) tất cả các giá trị bằng nhau, (b) một giá trị lớn hơn nhiều so với các giá trị còn lại, (c) tất cả các giá trị rất âm (-1000). Xác minh rằng nó đưa ra kết quả chính xác trong khi phiên bản ngây thơ thất bại.

4. **Kiểm tra gradient của một lớp mạng thần kinh.** Triển khai một lớp tuyến tính đơn giản `y = Wx + b` và lượt backward giải tích của nó. Sử dụng `numerical_gradient` để xác minh tính chính xác cho một ma trận trọng số 3x2.

5. **Thí nghiệm Loss scaling.** Mô phỏng huấn luyện với float16: tạo các gradient ngẫu nhiên trong khoảng [1e-9, 1e-3], chuyển sang float16 và đo xem tỷ lệ bao nhiêu phần trăm trở thành không. Sau đó áp dụng loss scaling (nhân với 1024), chuyển sang float16, chia ngược lại và đo lại tỷ lệ phần trăm bằng không.

## Key Terms

| Thuật ngữ | Mọi người hay nói | Ý nghĩa thực sự |
|------|----------------|----------------------|
| IEEE 754 | "Chuẩn float" | Tiêu chuẩn quốc tế định nghĩa các định dạng nhị phân dấu phẩy động, quy tắc làm tròn và các giá trị đặc biệt (inf, nan). Mọi CPU và GPU hiện đại đều triển khai nó. |
| Machine epsilon | "Giới hạn độ chính xác" | Giá trị e nhỏ nhất sao cho 1.0 + e != 1.0 trong một định dạng float nhất định. Đối với float32, nó khoảng 1.19e-7. |
| Catastrophic cancellation | "Mất độ chính xác do phép trừ" | Khi trừ hai số dấu phẩy động gần bằng nhau, các chữ số có nghĩa triệt tiêu nhau và nhiễu làm tròn chiếm ưu thế trong kết quả. |
| Overflow | "Số quá lớn" | Kết quả vượt quá giá trị tối đa có thể biểu diễn và trở thành inf. exp(89) gây tràn số trên trong float32. |
| Underflow | "Số quá nhỏ" | Kết quả gần bằng không hơn cả số dương nhỏ nhất có thể biểu diễn và trở thành 0.0. exp(-104) gây tràn số dưới trong float32. |
| Log-sum-exp trick | "Trừ max trước" | Tính log(sum(exp(x))) bằng cách đưa exp(max(x)) ra ngoài làm nhân tử chung để ngăn chặn tràn số trên và tràn số dưới. Được dùng trong softmax, cross-entropy và toán học log-probability. |
| Stable softmax | "Softmax không bùng nổ" | Trừ max(logits) trước khi thực hiện hàm mũ. Kết quả giống hệt về mặt toán học, không thể xảy ra tràn số trên. |
| Gradient checking | "Xác minh backprop" | So sánh gradient giải tích từ backpropagation với gradient số học từ sai phân hữu hạn để phát hiện lỗi triển khai. |
| Mixed precision | "Forward float16, backward float32" | Sử dụng float độ chính xác thấp cho các hoạt động cần tốc độ và float độ chính xác cao cho các hoạt động nhạy cảm về mặt số học. Tốc độ tăng thường là 2-3 lần. |
| Loss scaling | "Ngăn chặn gradient underflow" | Nhân loss với một hằng số lớn trước khi backprop để gradient nằm trong phạm vi biểu diễn của float16, sau đó chia cho chính hằng số đó trước khi cập nhật trọng số. |
| bfloat16 | "Brain floating point" | Định dạng 16-bit của Google với 8 bit mũ (cùng phạm vi với float32) và 7 bit định trị (độ chính xác thấp hơn float16). Được ưu tiên cho huấn luyện. |
| Gradient clipping | "Giới hạn chuẩn gradient" | Tỷ lệ hóa vector gradient sao cho chuẩn của nó không vượt quá một ngưỡng. Ngăn chặn bùng nổ gradient làm hỏng trọng số. |
| NaN | "Không phải là số" | Giá trị float đặc biệt từ các phép toán không xác định (0/0, inf-inf, sqrt(-1)). Lan truyền qua tất cả các phép toán số học sau đó. |
| Inf | "Vô cực" | Giá trị float đặc biệt từ tràn số trên hoặc chia cho không. Có thể kết hợp để tạo ra NaN (inf - inf, inf * 0). |
| Numerical gradient | "Đạo hàm vét cạn" | Xấp xỉ đạo hàm bằng cách tính f(x+h) và f(x-h) rồi chia cho 2h. Chậm nhưng đáng tin cậy để xác minh. |

## Further Reading

- [What Every Computer Scientist Should Know About Floating-Point Arithmetic (Goldberg 1991)](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) -- tài liệu tham khảo kinh điển, súc tích nhưng đầy đủ
- [Mixed Precision Training (Micikevicius et al., 2018)](https://arxiv.org/abs/1710.03740) -- bài báo của NVIDIA giới thiệu loss scaling cho huấn luyện float16
- [AMP: Automatic Mixed Precision (PyTorch docs)](https://pytorch.org/docs/stable/amp.html) -- hướng dẫn thực hành về độ chính xác hỗn hợp trong PyTorch
- [bfloat16 format (Google Cloud TPU docs)](https://cloud.google.com/tpu/docs/bfloat16) -- tại sao Google chọn định dạng này cho TPU
- [Kahan Summation (Wikipedia)](https://en.wikipedia.org/wiki/Kahan_summation_algorithm) -- thuật toán giảm sai số làm tròn trong các phép tổng dấu phẩy động