# Giới thiệu về JAX

> PyTorch thay đổi các tensor (mutation). TensorFlow xây dựng các đồ thị. JAX biên dịch các hàm thuần túy (pure functions). Điều cuối cùng đó thay đổi cách bạn tư duy về deep learning.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 03 Lessons 01-10, basic NumPy
**Time:** ~90 phút

## Mục tiêu học tập

- Viết mã mạng thần kinh bằng hàm thuần túy sử dụng API chức năng của JAX (jax.numpy, jax.grad, jax.jit, jax.vmap)
- Giải thích sự khác biệt thiết kế cốt lõi giữa cơ chế thay đổi dữ liệu trực tiếp (eager mutation) của PyTorch và mô hình biên dịch chức năng của JAX
- Áp dụng biên dịch jit và vector hóa vmap để tăng tốc các vòng lặp huấn luyện so với Python thuần
- Huấn luyện một mạng đơn giản trong JAX và đối chiếu việc quản lý trạng thái tường minh với cách tiếp cận hướng đối tượng của PyTorch

## Vấn đề

Bạn đã biết cách xây dựng mạng thần kinh trong PyTorch. Bạn định nghĩa một `nn.Module`, gọi `.backward()`, thực hiện bước tối ưu hóa. Nó hoạt động tốt. Hàng triệu người đang sử dụng nó.

Nhưng PyTorch có một ràng buộc nằm trong DNA của nó: nó theo dõi các thao tác một cách trực tiếp (eagerly), từng bước một, trong Python. Mỗi `tensor + tensor` là một lần khởi chạy kernel riêng biệt. Mỗi bước huấn luyện lại diễn giải cùng một đoạn mã Python. Điều này hoạt động ổn cho đến khi bạn cần huấn luyện một mô hình 540 tỷ tham số trên 2.048 TPU. Khi đó, chi phí vận hành (overhead) sẽ làm chậm hệ thống của bạn.

Google DeepMind huấn luyện Gemini trên JAX. Anthropic huấn luyện Claude trên JAX. Đây không phải là những hoạt động nhỏ lẻ -- chúng là những đợt huấn luyện mạng thần kinh lớn nhất trên Trái đất. Họ chọn JAX vì nó coi vòng lặp huấn luyện của bạn là một chương trình có thể biên dịch được, chứ không phải là một chuỗi các lệnh gọi Python.

JAX là NumPy với ba siêu năng lực: tự động lấy đạo hàm (automatic differentiation), biên dịch JIT sang XLA, và tự động vector hóa. Bạn viết một hàm xử lý một ví dụ. JAX cung cấp cho bạn một hàm xử lý cả batch, tính toán gradient, biên dịch sang mã máy và chạy trên nhiều thiết bị. Tất cả mà không cần thay đổi hàm gốc.

## Khái niệm

### Triết lý của JAX

JAX là một framework chức năng. Không có class, không có trạng thái có thể thay đổi (mutable state), không có phương thức `.backward()`. Thay vào đó:

| PyTorch | JAX |
|---------|-----|
| Class `nn.Module` với trạng thái | Hàm thuần túy: `f(params, x) -> y` |
| `loss.backward()` | `jax.grad(loss_fn)(params, x, y)` |
| Thực thi trực tiếp (Eager) | Biên dịch JIT qua XLA |
| Vòng lặp thủ công `for x in batch:` | Tự động vector hóa `jax.vmap(f)` |
| `DataParallel` / `FSDP` | Tự động song song hóa `jax.pmap(f)` |
| `model.parameters()` có thể thay đổi | Pytree bất biến của các mảng |

Đây không phải là sở thích về phong cách. Đó là một ràng buộc của trình biên dịch. Biên dịch JIT yêu cầu các hàm thuần túy -- cùng đầu vào luôn tạo ra cùng đầu ra, không có tác dụng phụ (side effects). Sự hạn chế đó là thứ giúp đạt được tốc độ nhanh hơn 100 lần.

### jax.numpy: Bề mặt quen thuộc

JAX triển khai lại API NumPy trên các bộ tăng tốc:

```python
import jax.numpy as jnp

a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, 5.0, 6.0])
c = jnp.dot(a, b)
```

Tên hàm giống hệt. Quy tắc broadcasting giống hệt. Ngữ nghĩa cắt lát (slicing) giống hệt. Nhưng các mảng nằm trên GPU/TPU, và mọi thao tác đều có thể được trình biên dịch theo dõi.

Một sự khác biệt quan trọng: Các mảng JAX là bất biến (immutable). Không có `a[0] = 5`. Thay vào đó: `a = a.at[0].set(5)`. Điều này có vẻ lạ lẫm trong một tuần, sau đó bạn sẽ hiểu ra -- tính bất biến là thứ giúp các phép biến đổi như `grad`, `jit`, và `vmap` có thể kết hợp được với nhau.

### jax.grad: Tự động lấy đạo hàm chức năng

PyTorch gắn gradient vào các tensor (`.grad`). JAX gắn gradient vào các hàm.

```python
import jax

def f(x):
    return x ** 2

df = jax.grad(f)
df(3.0)
```

`jax.grad` nhận vào một hàm và trả về một hàm mới tính toán gradient. Không cần gọi `.backward()`. Không có đồ thị tính toán nào được lưu trữ trên các tensor. Gradient chỉ là một hàm khác mà bạn có thể gọi, kết hợp hoặc biên dịch JIT.

Điều này có thể kết hợp tùy ý:

```python
d2f = jax.grad(jax.grad(f))
d2f(3.0)
```

Đạo hàm bậc hai. Đạo hàm bậc ba. Jacobians. Hessians. Tất cả bằng cách kết hợp `grad`. PyTorch cũng có thể làm điều này (`torch.autograd.functional.hessian`), nhưng nó được thêm vào sau. Trong JAX, đó là nền tảng.

Ràng buộc: `grad` chỉ hoạt động trên các hàm thuần túy. Không có lệnh in bên trong (chúng chạy trong quá trình theo dõi, không phải thực thi). Không thay đổi trạng thái bên ngoài. Không tạo số ngẫu nhiên nếu không quản lý khóa (key) tường minh.

### jit: Biên dịch sang XLA

```python
@jax.jit
def train_step(params, x, y):
    loss = loss_fn(params, x, y)
    return loss

fast_step = jax.jit(train_step)
```

Trong lần gọi đầu tiên, JAX theo dõi hàm -- nó ghi lại các thao tác nào xảy ra mà không thực thi chúng. Sau đó, nó chuyển bản ghi đó cho XLA (Accelerated Linear Algebra), trình biên dịch của Google cho TPU và GPU. XLA hợp nhất các thao tác, loại bỏ các bản sao bộ nhớ dư thừa và tạo ra mã máy được tối ưu hóa.

Các lần gọi tiếp theo bỏ qua hoàn toàn Python. Mã đã biên dịch chạy trên bộ tăng tốc với tốc độ C++.

Khi nào JIT hữu ích:
- Các bước huấn luyện (cùng một phép tính lặp lại hàng nghìn lần)
- Suy luận (cùng một mô hình, đầu vào khác nhau)
- Bất kỳ hàm nào được gọi nhiều lần với các đầu vào có hình dạng tương tự

Khi nào JIT gây hại:
- Các hàm có luồng điều khiển Python phụ thuộc vào giá trị (`if x > 0` trong đó x là một mảng được theo dõi)
- Các phép tính chạy một lần (chi phí biên dịch vượt quá thời gian chạy)
- Gỡ lỗi (việc theo dõi che giấu quá trình thực thi thực tế)

Ràng buộc về luồng điều khiển là có thật. `jax.lax.cond` thay thế `if/else`. `jax.lax.scan` thay thế các vòng lặp `for`. Đây không phải là tùy chọn -- chúng là cái giá của việc biên dịch.

### vmap: Tự động vector hóa

Bạn viết một hàm xử lý một ví dụ:

```python
def predict(params, x):
    return jnp.dot(params['w'], x) + params['b']
```

`vmap` nâng cấp nó để xử lý cả batch:

```python
batch_predict = jax.vmap(predict, in_axes=(None, 0))
```

`in_axes=(None, 0)` có nghĩa là: không batch qua `params` (được chia sẻ), batch qua trục 0 của `x`. Không cần vòng lặp `for` thủ công. Không cần thay đổi hình dạng (reshaping). Không cần xử lý chiều batch. JAX tự tìm ra chiều batch và vector hóa toàn bộ phép tính.

Đây không phải là cú pháp đơn thuần. `vmap` tạo ra mã vector hóa đã hợp nhất chạy nhanh hơn 10-100 lần so với vòng lặp Python. Và nó kết hợp với `jit` và `grad`:

```python
per_example_grads = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))
```

Gradient cho từng ví dụ. Một dòng. Điều này gần như không thể thực hiện trong PyTorch nếu không dùng các thủ thuật.

### pmap: Song song hóa dữ liệu trên các thiết bị

```python
parallel_step = jax.pmap(train_step, axis_name='devices')
```

`pmap` sao chép hàm trên tất cả các thiết bị khả dụng (GPU/TPU) và chia nhỏ batch. Bên trong hàm, `jax.lax.pmean` và `jax.lax.psum` đồng bộ hóa gradient giữa các thiết bị.

Google huấn luyện Gemini trên hàng nghìn chip TPU v5e sử dụng `pmap` (và phiên bản kế nhiệm `shard_map`). Mô hình lập trình: viết phiên bản đơn thiết bị, bao bọc bằng `pmap`, xong.

### Pytrees: Cấu trúc dữ liệu phổ quát

JAX hoạt động trên "pytrees" -- các kết hợp lồng nhau của danh sách, tuple, dict và mảng. Các tham số mô hình của bạn là một pytree:

```python
params = {
    'layer1': {'w': jnp.zeros((784, 256)), 'b': jnp.zeros(256)},
    'layer2': {'w': jnp.zeros((256, 128)), 'b': jnp.zeros(128)},
    'layer3': {'w': jnp.zeros((128, 10)),  'b': jnp.zeros(10)},
}
```

Mọi phép biến đổi của JAX -- `grad`, `jit`, `vmap` -- đều biết cách duyệt qua các pytree. `jax.tree.map(f, tree)` áp dụng `f` cho mọi lá (leaf). Đây là cách các bộ tối ưu hóa cập nhật tất cả tham số cùng một lúc:

```python
params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
```

Không có phương thức `.parameters()`. Không đăng ký tham số. Cấu trúc cây chính là mô hình.

### Chức năng vs Hướng đối tượng

PyTorch lưu trữ trạng thái bên trong các đối tượng:

```python
class Model(nn.Module):
    def __init__(self):
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        return self.linear(x)
```

JAX sử dụng các hàm thuần túy với trạng thái tường minh:

```python
def predict(params, x):
    return jnp.dot(x, params['w']) + params['b']
```

Các tham số được truyền vào. Không có gì được lưu trữ. Không có gì bị thay đổi. Điều này làm cho mọi hàm có thể kiểm thử, kết hợp và biên dịch được. Nó cũng có nghĩa là bạn tự quản lý các tham số -- hoặc sử dụng một thư viện như Flax hoặc Equinox.

### Hệ sinh thái JAX

JAX cung cấp cho bạn các nguyên hàm. Các thư viện cung cấp cho bạn sự tiện dụng:

| Thư viện | Vai trò | Phong cách |
|---------|------|-------|
| **Flax** (Google) | Các lớp mạng thần kinh | `nn.Module` với trạng thái tường minh |
| **Equinox** (Patrick Kidger) | Các lớp mạng thần kinh | Dựa trên Pytree, Pythonic |
| **Optax** (DeepMind) | Bộ tối ưu hóa + Lịch trình LR | Các phép biến đổi gradient có thể kết hợp |
| **Orbax** (Google) | Checkpointing | Lưu/khôi phục pytrees |
| **CLU** (Google) | Chỉ số + ghi nhật ký | Tiện ích vòng lặp huấn luyện |

Optax là thư viện tối ưu hóa tiêu chuẩn. Nó tách biệt phép biến đổi gradient (Adam, SGD, clipping) khỏi việc cập nhật tham số, giúp việc kết hợp trở nên đơn giản:

```python
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-3),
)
```

### Khi nào nên dùng JAX vs PyTorch

| Yếu tố | JAX | PyTorch |
|--------|-----|---------|
| Hỗ trợ TPU | Hạng nhất (Google xây dựng cả hai) | Cộng đồng duy trì (torch_xla) |
| Hỗ trợ GPU | Tốt (CUDA qua XLA) | Tốt nhất (CUDA gốc) |
| Gỡ lỗi | Khó (theo dõi + biên dịch) | Dễ (eager, từng dòng) |
| Hệ sinh thái | Tập trung vào nghiên cứu (Flax, Equinox) | Khổng lồ (HuggingFace, torchvision, v.v.) |
| Tuyển dụng | Ngách (Google/DeepMind/Anthropic) | Phổ biến (mọi nơi) |
| Huấn luyện quy mô lớn | Vượt trội (XLA, pmap, mesh) | Tốt (FSDP, DeepSpeed) |
| Tốc độ tạo mẫu | Chậm hơn (chi phí chức năng) | Nhanh hơn (thay đổi và chạy) |
| Suy luận sản xuất | TensorFlow Serving, Vertex AI | TorchServe, Triton, ONNX |
| Ai sử dụng | DeepMind (Gemini), Anthropic (Claude) | Meta (Llama), OpenAI (GPT), Stability AI |

Câu trả lời trung thực: hãy sử dụng PyTorch trừ khi bạn có lý do cụ thể để dùng JAX. Những lý do đó là -- truy cập TPU, cần gradient cho từng ví dụ, huấn luyện đa thiết bị ở quy mô lớn, hoặc làm việc tại Google/DeepMind/Anthropic.

### Số ngẫu nhiên trong JAX

JAX không có trạng thái ngẫu nhiên toàn cục. Mọi thao tác ngẫu nhiên đều yêu cầu một khóa PRNG tường minh:

```python
key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)
w = jax.random.normal(key1, shape=(784, 256))
```

Điều này gây khó chịu lúc đầu. Nhưng nó đảm bảo tính tái lập trên các thiết bị và các lần biên dịch -- một thuộc tính mà `torch.manual_seed` của PyTorch không thể đảm bảo trong môi trường đa GPU.

```figure
batchnorm-effect
```

## Xây dựng

### Bước 1: Thiết lập và Dữ liệu

Chúng ta sẽ huấn luyện một MLP 3 lớp trên MNIST sử dụng JAX và Optax. 784 đầu vào, hai lớp ẩn 256 và 128 neuron, 10 lớp đầu ra.

```python
import jax
import jax.numpy as jnp
from jax import random
import optax

def get_mnist_data():
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype('float32') / 255.0
    y = mnist.target.astype('int')
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    return X_train, y_train, X_test, y_test
```

### Bước 2: Khởi tạo tham số

Không có class. Chỉ là một hàm trả về một pytree:

```python
def init_params(key):
    k1, k2, k3 = random.split(key, 3)
    scale1 = jnp.sqrt(2.0 / 784)
    scale2 = jnp.sqrt(2.0 / 256)
    scale3 = jnp.sqrt(2.0 / 128)
    params = {
        'layer1': {
            'w': scale1 * random.normal(k1, (784, 256)),
            'b': jnp.zeros(256),
        },
        'layer2': {
            'w': scale2 * random.normal(k2, (256, 128)),
            'b': jnp.zeros(128),
        },
        'layer3': {
            'w': scale3 * random.normal(k3, (128, 10)),
            'b': jnp.zeros(10),
        },
    }
    return params
```

He-initialization, thực hiện thủ công. Ba khóa PRNG được tách từ một hạt giống (seed). Mỗi trọng số là một mảng bất biến trong một dict lồng nhau.

### Bước 3: Forward Pass

```python
def forward(params, x):
    x = jnp.dot(x, params['layer1']['w']) + params['layer1']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer2']['w']) + params['layer2']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer3']['w']) + params['layer3']['b']
    return x

def loss_fn(params, x, y):
    logits = forward(params, x)
    one_hot = jax.nn.one_hot(y, 10)
    return -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1))
```

Các hàm thuần túy. Tham số vào, dự đoán ra. Không có `self`, không có trạng thái lưu trữ. `loss_fn` tính toán cross-entropy từ đầu -- softmax, log, trung bình âm.

### Bước 4: Bước huấn luyện biên dịch JIT

```python
@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

@jax.jit
def accuracy(params, x, y):
    logits = forward(params, x)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == y)
```

`jax.value_and_grad` trả về cả giá trị loss và gradient trong một lần chạy. Decorator `@jax.jit` biên dịch cả hai hàm sang XLA. Sau lần gọi đầu tiên, mỗi bước huấn luyện chạy mà không cần chạm vào Python.

### Bước 5: Vòng lặp huấn luyện

```python
optimizer = optax.adam(learning_rate=1e-3)

X_train, y_train, X_test, y_test = get_mnist_data()
X_train, X_test = jnp.array(X_train), jnp.array(X_test)
y_train, y_test = jnp.array(y_train), jnp.array(y_test)

key = random.PRNGKey(0)
params = init_params(key)
opt_state = optimizer.init(params)

batch_size = 128
n_epochs = 10

for epoch in range(n_epochs):
    key, subkey = random.split(key)
    perm = random.permutation(subkey, len(X_train))
    X_shuffled = X_train[perm]
    y_shuffled = y_train[perm]

    epoch_loss = 0.0
    n_batches = len(X_train) // batch_size
    for i in range(n_batches):
        start = i * batch_size
        xb = X_shuffled[start:start + batch_size]
        yb = y_shuffled[start:start + batch_size]
        params, opt_state, loss = train_step(params, opt_state, xb, yb)
        epoch_loss += loss

    train_acc = accuracy(params, X_train[:5000], y_train[:5000])
    test_acc = accuracy(params, X_test, y_test)
    print(f"Epoch {epoch + 1:2d} | Loss: {epoch_loss / n_batches:.4f} | "
          f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
```

10 epoch. Độ chính xác kiểm tra ~97%. Epoch đầu tiên chậm (biên dịch JIT). Các epoch 2-10 nhanh.

Để ý những gì còn thiếu: không có `.zero_grad()`, không có `.backward()`, không có `.step()`. Toàn bộ quá trình cập nhật là một lệnh gọi hàm kết hợp. Gradient được tính toán, biến đổi bởi Adam và áp dụng vào các tham số -- tất cả bên trong `train_step`.

## Sử dụng

### Flax: Tiêu chuẩn của Google

Flax là thư viện mạng thần kinh JAX phổ biến nhất. Nó thêm `nn.Module` trở lại, nhưng với quản lý trạng thái tường minh:

```python
import flax.linen as nn

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

model = MLP()
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 784)))
logits = model.apply(params, x_batch)
```

Cấu trúc giống PyTorch, nhưng `params` tách biệt với mô hình. `model.init()` tạo tham số. `model.apply(params, x)` chạy forward pass. Đối tượng mô hình không có trạng thái.

### Equinox: Giải pháp thay thế Pythonic

Equinox (bởi Patrick Kidger) biểu diễn các mô hình dưới dạng pytrees:

```python
import equinox as eqx

model = eqx.nn.MLP(
    in_size=784, out_size=10, width_size=256, depth=2,
    activation=jax.nn.relu, key=jax.random.PRNGKey(0)
)
logits = model(x)
```

Bản thân mô hình là một pytree. Không cần `.apply()`. Các tham số chỉ là các lá của mô hình. Điều này gần hơn với cách JAX tư duy.

### Optax: Bộ tối ưu hóa có thể kết hợp

Optax tách biệt phép biến đổi gradient khỏi việc cập nhật:

```python
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=1e-3,
    warmup_steps=1000, decay_steps=50000
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=schedule, weight_decay=0.01),
)
```

Gradient clipping, learning rate warmup, weight decay -- tất cả được kết hợp như một chuỗi các phép biến đổi. Mỗi phép biến đổi nhìn thấy gradient, sửa đổi chúng và chuyển cho bước tiếp theo. Không có class tối ưu hóa nguyên khối.

## Triển khai

**Cài đặt:**

```bash
pip install jax jaxlib optax flax
```

Để hỗ trợ GPU:

```bash
pip install jax[cuda12]
```

Cho TPU (Google Cloud):

```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

**Các lưu ý về hiệu năng:**

- Lần gọi JIT đầu tiên chậm (biên dịch). Hãy khởi động (warm up) trước khi đo điểm chuẩn.
- Tránh các vòng lặp Python qua các mảng JAX bên trong JIT. Sử dụng `jax.lax.scan` hoặc `jax.lax.fori_loop`.
- `jax.debug.print()` hoạt động bên trong JIT. `print()` thông thường thì không.
- Profile với `jax.profiler` hoặc TensorBoard. Biên dịch XLA có thể che giấu các điểm nghẽn.
- JAX mặc định cấp phát trước 75% bộ nhớ GPU. Thiết lập `XLA_PYTHON_CLIENT_PREALLOCATE=false` để vô hiệu hóa.

**Checkpointing:**

```python
import orbax.checkpoint as ocp
checkpointer = ocp.PyTreeCheckpointer()
checkpointer.save('/tmp/model', params)
restored = checkpointer.restore('/tmp/model')
```

**Bài học này tạo ra:**
- `outputs/prompt-jax-optimizer.md` -- một gợi ý để chọn cấu hình bộ tối ưu hóa JAX phù hợp
- `outputs/skill-jax-patterns.md` -- một kỹ năng bao gồm các mẫu chức năng trong JAX

## Bài tập

1. Thêm dropout vào MLP. Trong JAX, dropout yêu cầu một khóa PRNG -- truyền khóa qua forward pass và tách nó cho mỗi lớp dropout. So sánh độ chính xác kiểm tra có và không có dropout.

2. Sử dụng `jax.vmap` để tính gradient cho từng ví dụ cho một batch 32 ảnh MNIST. Tính chuẩn gradient cho mỗi ví dụ. Ví dụ nào có gradient lớn nhất, và tại sao?

3. Thay thế hàm forward thủ công bằng một `mlp_forward(params, x)` chung hoạt động cho bất kỳ số lượng lớp nào. Sử dụng `jax.tree.leaves` để xác định độ sâu tự động.

4. Đo điểm chuẩn bước huấn luyện có và không có `@jax.jit`. Đo thời gian 100 bước của mỗi loại. Tốc độ tăng bao nhiêu trên phần cứng của bạn? Chi phí biên dịch trong lần gọi đầu tiên là bao nhiêu?

5. Triển khai gradient clipping bằng cách kết hợp `optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))`. Huấn luyện có và không có clipping. Vẽ biểu đồ chuẩn gradient trong quá trình huấn luyện để thấy hiệu quả.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|----------------------|
| XLA | "Thứ làm cho JAX nhanh" | Accelerated Linear Algebra -- trình biên dịch hợp nhất các thao tác và tạo ra các kernel GPU/TPU tối ưu từ đồ thị tính toán |
| JIT | "Biên dịch tức thời" | JAX theo dõi hàm trong lần gọi đầu tiên, biên dịch sang XLA, sau đó chạy phiên bản đã biên dịch trong các lần gọi tiếp theo |
| Hàm thuần túy | "Không có tác dụng phụ" | Hàm mà đầu ra chỉ phụ thuộc vào đầu vào -- không có trạng thái toàn cục, không thay đổi dữ liệu, không ngẫu nhiên nếu không có khóa tường minh |
| vmap | "Tự động batching" | Biến đổi một hàm xử lý một ví dụ thành hàm xử lý cả batch mà không cần viết lại |
| pmap | "Tự động song song hóa" | Sao chép một hàm trên nhiều thiết bị và chia nhỏ batch đầu vào |
| Pytree | "Dict lồng nhau của các mảng" | Bất kỳ cấu trúc lồng nhau nào của danh sách, tuple, dict và mảng mà JAX có thể duyệt và biến đổi |
| Tracing | "Ghi lại phép tính" | JAX thực thi hàm với các giá trị trừu tượng để xây dựng đồ thị tính toán mà không cần tính toán kết quả thực |
| Functional autodiff | "đạo hàm của một hàm" | Tính đạo hàm bằng cách biến đổi các hàm, không phải bằng cách gắn bộ lưu trữ gradient vào các tensor |
| Optax | "Thư viện tối ưu hóa của JAX" | Thư viện các phép biến đổi gradient có thể kết hợp -- Adam, SGD, clipping, lập lịch -- được liên kết với nhau |
| Flax | "nn.Module của JAX" | Thư viện mạng thần kinh của Google cho JAX, thêm các trừu tượng lớp trong khi vẫn giữ trạng thái tường minh |

## Đọc thêm

- Tài liệu JAX: https://jax.readthedocs.io/ -- tài liệu chính thức, với các hướng dẫn tuyệt vời về grad, jit và vmap
- "JAX: composable transformations of Python+NumPy programs" (Bradbury et al., 2018) -- bài báo gốc giải thích triết lý thiết kế
- Tài liệu Flax: https://flax.readthedocs.io/ -- thư viện mạng thần kinh của Google cho JAX
- Patrick Kidger, "Equinox: neural networks in JAX via callable PyTrees and filtered transformations" (2021) -- giải pháp thay thế Pythonic cho Flax
- DeepMind, "Optax: composable gradient transformation and optimisation" -- thư viện tối ưu hóa tiêu chuẩn
- "You Don't Know JAX" (Colin Raffel, 2020) -- hướng dẫn thực tế về các lỗi và mẫu trong JAX, từ một trong những tác giả của T5