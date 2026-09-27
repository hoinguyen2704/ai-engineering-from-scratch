# Vectors, Matrices & Operations

> Mọi neural network thực chất chỉ là matrix multiplication với một vài bước bổ sung.

**Type:** Build
**Languages:** Python, Julia
**Prerequisites:** Phase 1, Lesson 01 (Linear Algebra Intuition)
**Time:** ~60 minutes

## Learning Objectives

- Xây dựng một Matrix class với các phép toán element-wise, matrix multiplication, transpose, determinant, và inverse
- Phân biệt element-wise multiplication với matrix multiplication và giải thích khi nào áp dụng mỗi loại
- Triển khai một lớp neural network dense đơn lẻ (`relu(W @ x + b)`) chỉ sử dụng Matrix class tự viết
- Giải thích các quy tắc broadcasting và cách bias addition hoạt động trong các neural network frameworks

## The Problem

Bạn muốn xây dựng một neural network. Bạn đọc code và thấy thế này:

```
output = activation(weights @ input + bias)
```

`@` đó chính là matrix multiplication. `weights` là một ma trận. `input` là một vector. Nếu bạn không biết những phép toán đó làm gì, dòng code này giống như phép thuật. Nếu bạn đã biết, đó là toàn bộ forward pass của một layer chỉ trong ba phép toán.

Mọi hình ảnh mà model của bạn xử lý là một ma trận các giá trị pixel. Mỗi word embedding là một vector. Mỗi layer của mọi neural network là một matrix transformation. Bạn không thể xây dựng các hệ thống AI mà không thành thạo các phép toán ma trận, cũng giống như bạn không thể viết code mà không hiểu về variables.

Bài học này giúp bạn xây dựng sự thành thạo đó từ con số không.

## The Concept

### Vectors: danh sách các con số có thứ tự

Một vector là một danh sách các con số có hướng và độ lớn. Trong AI, các vectors đại diện cho các data points, features, hoặc parameters.

```
v = [3, 4]        -- a 2D vector
w = [1, 0, -2]    -- a 3D vector
```

Một 2D vector `[3, 4]` trỏ đến tọa độ (3, 4) trên một mặt phẳng. Chiều dài (độ lớn) của nó là 5 (theo tam giác 3-4-5).

### Matrices: lưới các con số

Một ma trận là một lưới 2D gồm các hàng (rows) và cột (columns). Một ma trận m x n có m hàng và n cột.

```
A = | 1  2  3 |     -- 2x3 matrix (2 rows, 3 columns)
    | 4  5  6 |
```

Trong neural networks, các weight matrices biến đổi các input vectors thành các output vectors. Một layer với 784 inputs và 128 outputs sử dụng một weight matrix kích thước 128x784.

### Tại sao shapes lại quan trọng

Matrix multiplication có một quy tắc nghiêm ngặt: `(m x n) @ (n x p) = (m x p)`. Các inner dimensions phải khớp nhau.

```
(128 x 784) @ (784 x 1) = (128 x 1)
  weights       input       output

Inner dimensions: 784 = 784  -- valid
```

Nếu bạn gặp lỗi shape mismatch trong PyTorch, đây chính là lý do.

### Bản đồ các phép toán

| Operation | What it does | Neural network use |
|-----------|-------------|-------------------|
| Addition | Kết hợp element-wise | Cộng bias vào output |
| Scalar multiply | Thay đổi tỷ lệ mọi phần tử | Learning rate * gradients |
| Matrix multiply | Biến đổi vectors | Forward pass của layer |
| Transpose | Đảo ngược hàng và cột | Backpropagation |
| Determinant | Tóm tắt bằng một con số duy nhất | Kiểm tra tính khả nghịch (invertibility) |
| Inverse | Hoàn tác một phép biến đổi | Giải các hệ phương trình tuyến tính |
| Identity | Ma trận "không làm gì" | Khởi tạo, residual connections |

### Element-wise vs matrix multiplication

Sự khác biệt này thường xuyên làm khó những người mới bắt đầu.

Element-wise: nhân các vị trí tương ứng. Cả hai ma trận phải có cùng shape.

```
| 1  2 |   | 5  6 |   | 5  12 |
| 3  4 | * | 7  8 | = | 21 32 |
```

Matrix multiplication: tích vô hướng (dot products) của các hàng và cột. Các inner dimensions phải khớp nhau.

```
| 1  2 |   | 5  6 |   | 1*5+2*7  1*6+2*8 |   | 19  22 |
| 3  4 | @ | 7  8 | = | 3*5+4*7  3*6+4*8 | = | 43  50 |
```

Các phép toán khác nhau, kết quả khác nhau, quy tắc khác nhau.

### Broadcasting

Khi bạn cộng một bias vector vào một ma trận các outputs, các shapes không khớp nhau. Broadcasting sẽ "kéo giãn" mảng nhỏ hơn để vừa khớp.

```
| 1  2  3 |   +   [10, 20, 30]
| 4  5  6 |

Broadcasting stretches the vector across rows:

| 1  2  3 |   | 10  20  30 |   | 11  22  33 |
| 4  5  6 | + | 10  20  30 | = | 14  25  36 |
```

Mọi framework hiện đại đều thực hiện việc này một cách tự động. Hiểu rõ nó giúp tránh nhầm lẫn khi các shapes có vẻ sai nhưng code vẫn chạy.

```figure
vector-projection
```

## Build It

### Bước 1: Vector class

```python
class Vector:
    def __init__(self, data):
        self.data = list(data)
        self.size = len(self.data)

    def __repr__(self):
        return f"Vector({self.data})"

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.data, other.data)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.data, other.data)])

    def __mul__(self, scalar):
        return Vector([x * scalar for x in self.data])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.data, other.data))

    def magnitude(self):
        return sum(x ** 2 for x in self.data) ** 0.5
```

### Bước 2: Matrix class với các phép toán cốt lõi

```python
class Matrix:
    def __init__(self, data):
        self.data = [list(row) for row in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0])
        self.shape = (self.rows, self.cols)

    def __repr__(self):
        rows_str = "\n  ".join(str(row) for row in self.data)
        return f"Matrix({self.shape}):\n  {rows_str}"

    def __add__(self, other):
        return Matrix([
            [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def __sub__(self, other):
        return Matrix([
            [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def scalar_multiply(self, scalar):
        return Matrix([
            [self.data[i][j] * scalar for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def element_wise_multiply(self, other):
        return Matrix([
            [self.data[i][j] * other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def matmul(self, other):
        return Matrix([
            [
                sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                for j in range(other.cols)
            ]
            for i in range(self.rows)
        ])

    def transpose(self):
        return Matrix([
            [self.data[j][i] for j in range(self.rows)]
            for i in range(self.cols)
        ])

    def determinant(self):
        if self.shape == (1, 1):
            return self.data[0][0]
        if self.shape == (2, 2):
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        det = 0
        for j in range(self.cols):
            minor = Matrix([
                [self.data[i][k] for k in range(self.cols) if k != j]
                for i in range(1, self.rows)
            ])
            det += ((-1) ** j) * self.data[0][j] * minor.determinant()
        return det

    def inverse_2x2(self):
        det = self.determinant()
        if det == 0:
            raise ValueError("Matrix is singular, no inverse exists")
        return Matrix([
            [self.data[1][1] / det, -self.data[0][1] / det],
            [-self.data[1][0] / det, self.data[0][0] / det]
        ])

    @staticmethod
    def identity(n):
        return Matrix([
            [1 if i == j else 0 for j in range(n)]
            for i in range(n)
        ])
```

### Bước 3: Xem cách nó hoạt động

```python
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])

print("A + B =", (A + B).data)
print("A @ B =", A.matmul(B).data)
print("A^T =", A.transpose().data)
print("det(A) =", A.determinant())
print("A^-1 =", A.inverse_2x2().data)

I = Matrix.identity(2)
print("A @ A^-1 =", A.matmul(A.inverse_2x2()).data)
```

### Bước 4: Kết nối với neural networks

```python
import random

inputs = Matrix([[0.5], [0.8], [0.2]])
weights = Matrix([
    [random.uniform(-1, 1) for _ in range(3)]
    for _ in range(2)
])
bias = Matrix([[0.1], [0.1]])

def relu_matrix(m):
    return Matrix([[max(0, val) for val in row] for row in m.data])

pre_activation = weights.matmul(inputs) + bias
output = relu_matrix(pre_activation)

print(f"Input shape: {inputs.shape}")
print(f"Weight shape: {weights.shape}")
print(f"Output shape: {output.shape}")
print(f"Output: {output.data}")
```

Đây là một lớp dense đơn lẻ: `output = relu(W @ x + b)`. Mọi dense layer trong mọi neural network đều thực hiện chính xác điều này.

## Use It

NumPy thực hiện tất cả những điều trên với ít dòng code hơn và nhanh hơn gấp nhiều lần.

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A + B =\n", A + B)
print("A * B (element-wise) =\n", A * B)
print("A @ B (matrix multiply) =\n", A @ B)
print("A^T =\n", A.T)
print("det(A) =", np.linalg.det(A))
print("A^-1 =\n", np.linalg.inv(A))
print("I =\n", np.eye(2))

inputs = np.random.randn(3, 1)
weights = np.random.randn(2, 3)
bias = np.array([[0.1], [0.1]])
output = np.maximum(0, weights @ inputs + bias)

print(f"\nNeural network layer: {weights.shape} @ {inputs.shape} = {output.shape}")
print(f"Output:\n{output}")
```

Toán tử `@` trong Python gọi `__matmul__`. NumPy triển khai nó với các optimized BLAS routines được viết bằng C và Fortran. Cùng một phép toán, nhưng nhanh hơn 100 lần.

Broadcasting trong NumPy:

```python
matrix = np.array([[1, 2, 3], [4, 5, 6]])
bias = np.array([10, 20, 30])
print(matrix + bias)
```

NumPy tự động broadcasts bias 1D qua cả hai hàng. Đây là cách bias addition hoạt động trong mọi neural network framework.

## Ship It

Bài học này tạo ra một prompt để dạy các phép toán ma trận thông qua trực giác hình học. Xem `outputs/prompt-matrix-operations.md`.

Matrix class được xây dựng ở đây là nền tảng cho mini neural network framework mà chúng ta sẽ xây dựng trong Phase 3, Lesson 10.

## Exercises

1. **Xác minh inverse.** Nhân `A @ A.inverse_2x2()` và xác nhận bạn nhận được identity matrix. Thử với ba ma trận 2x2 khác nhau. Điều gì xảy ra khi determinant bằng không?

2. **Triển khai 3x3 inverse.** Mở rộng Matrix class để tính inverses cho các ma trận 3x3 bằng phương pháp adjugate. Kiểm tra lại với `np.linalg.inv` của NumPy.

3. **Xây dựng mạng hai lớp.** Chỉ sử dụng Matrix class của bạn (không dùng NumPy), hãy tạo một neural network hai lớp: input (3) -> hidden (4) -> output (2). Khởi tạo random weights, chạy một lượt forward pass, và xác minh tất cả các shapes đều chính xác.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Vector | "Một mũi tên" | Một danh sách các con số có thứ tự. Trong AI: một điểm trong không gian nhiều chiều. |
| Matrix | "Một bảng số" | Một linear transformation. Nó ánh xạ các vectors từ không gian này sang không gian khác. |
| Matrix multiply | "Chỉ cần nhân các con số" | Dot products giữa mỗi hàng của ma trận thứ nhất và mỗi cột của ma trận thứ hai. Thứ tự rất quan trọng. |
| Transpose | "Lật ngược nó" | Hoán đổi hàng và cột. Biến một ma trận m x n thành n x m. Cực kỳ quan trọng trong backpropagation. |
| Determinant | "Một con số từ ma trận" | Đo lường mức độ ma trận làm thay đổi tỷ lệ diện tích (2D) hoặc thể tích (3D). Bằng không nghĩa là phép biến đổi làm triệt tiêu một chiều không gian. |
| Inverse | "Hoàn tác ma trận" | Ma trận đảo ngược lại phép biến đổi. Chỉ tồn tại khi determinant khác không. |
| Identity matrix | "Ma trận nhàm chán" | Ma trận tương đương với việc nhân với 1. Được sử dụng trong residual connections (ResNets). |
| Broadcasting | "Sửa shape thần kỳ" | Kéo giãn một mảng nhỏ hơn để khớp với mảng lớn hơn bằng cách lặp lại dọc theo các chiều còn thiếu. |
| Element-wise | "Phép nhân thông thường" | Nhân các vị trí tương ứng. Cả hai mảng phải có cùng shape (hoặc có thể broadcast). |

## Further Reading

- [3Blue1Brown: Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra) - trực giác hình học cho mọi phép toán được đề cập ở đây
- [NumPy documentation on broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) - các quy tắc chính xác mà NumPy tuân theo
- [Stanford CS229 Linear Algebra Review](http://cs229.stanford.edu/section/cs229-linalg.pdf) - tài liệu tham khảo súc tích về đại số tuyến tính dành riêng cho ML