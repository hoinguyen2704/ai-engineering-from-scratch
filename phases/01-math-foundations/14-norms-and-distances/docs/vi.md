# Các chuẩn (Norms) và Khoảng cách (Distances)

> Hàm khoảng cách của bạn định nghĩa thế nào là "tương đồng". Nếu chọn sai, mọi thứ ở các bước sau đều sẽ hỏng.

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 1, Lessons 01 (Linear Algebra Intuition), 02 (Vectors, Matrices & Operations)
**Time:** ~90 phút

## Mục tiêu học tập

- Triển khai các hàm khoảng cách L1, L2, cosine, Mahalanobis, Jaccard và edit distance từ đầu (from scratch)
- Lựa chọn metric khoảng cách phù hợp cho một tác vụ ML cụ thể và giải thích lý do tại sao các phương án khác không hiệu quả
- Kết nối các chuẩn L1 và L2 với kỹ thuật điều chuẩn (regularization) LASSO và Ridge cùng các vùng ràng buộc hình học của chúng
- Chứng minh rằng cùng một tập dữ liệu sẽ tạo ra các điểm lân cận gần nhất (nearest neighbors) khác nhau dưới các metric khác nhau

## Vấn đề

Bạn có hai vector. Có thể chúng là các word embedding. Có thể là hồ sơ người dùng. Hoặc là các mảng pixel. Bạn cần biết: chúng gần nhau đến mức nào?

Câu trả lời phụ thuộc hoàn toàn vào hàm khoảng cách mà bạn chọn. Hai điểm dữ liệu có thể là lân cận gần nhất theo metric này nhưng lại cách xa nhau theo metric khác. Bộ phân loại KNN, công cụ gợi ý, cơ sở dữ liệu vector, thuật toán phân cụm, hàm mất mát (loss function) của bạn — tất cả đều phụ thuộc vào lựa chọn này. Nếu chọn sai, mô hình của bạn sẽ tối ưu hóa cho mục tiêu sai lệch.

Không có khoảng cách nào là tốt nhất cho mọi trường hợp. L2 hiệu quả với dữ liệu không gian. Cosine similarity thống trị trong NLP. Jaccard xử lý các tập hợp. Edit distance xử lý chuỗi. Mahalanobis tính đến các tương quan. Wasserstein di chuyển khối lượng xác suất. Mỗi loại mã hóa một giả định khác nhau về ý nghĩa của từ "tương đồng".

Bài học này sẽ xây dựng mọi hàm khoảng cách quan trọng từ đầu, chỉ cho bạn khi nào mỗi loại là công cụ phù hợp, và chứng minh cách dữ liệu giống nhau tạo ra các lân cận gần nhất hoàn toàn khác biệt tùy thuộc vào metric bạn sử dụng.

## Khái niệm

### Các chuẩn (Norms): đo lường độ lớn của vector

Một chuẩn đo lường "kích thước" của một vector. Mọi hàm khoảng cách giữa hai vector đều có thể được viết dưới dạng chuẩn của hiệu số giữa chúng: d(a, b) = ||a - b||. Vì vậy, hiểu về các chuẩn chính là hiểu về các khoảng cách.

### Chuẩn L1 (Khoảng cách Manhattan)

Chuẩn L1 tính tổng các giá trị tuyệt đối của tất cả các thành phần.

```
||x||_1 = |x_1| + |x_2| + ... + |x_n|
```

Nó được gọi là khoảng cách Manhattan vì nó đo quãng đường bạn đi trên lưới ô bàn cờ của thành phố, nơi bạn chỉ có thể di chuyển dọc theo các trục. Không có đường chéo.

```
Point A = (1, 1)
Point B = (4, 5)

L1 distance = |4-1| + |5-1| = 3 + 4 = 7

On a grid, you walk 3 blocks east and 4 blocks north.
```

Khi nào nên dùng L1:
- Dữ liệu thưa (sparse) có số chiều cao (đặc trưng văn bản, one-hot encoding)
- Khi bạn muốn sự bền vững trước các giá trị ngoại lai (outliers) (một sự khác biệt cực lớn đơn lẻ sẽ không chiếm ưu thế)
- Các bài toán chọn lọc đặc trưng (L1 regularization thúc đẩy tính thưa)

Kết nối với L1 regularization (Lasso): thêm ||w||_1 vào hàm mất mát sẽ phạt tổng các giá trị trọng số tuyệt đối. Điều này đẩy các trọng số nhỏ về đúng bằng 0, thực hiện chọn lọc đặc trưng tự động. Hình phạt L1 tạo ra các vùng ràng buộc hình kim cương trong không gian trọng số, và các góc của kim cương nằm trên các trục nơi một số trọng số bằng 0.

Kết nối với các hàm mất mát: Mean Absolute Error (MAE) là khoảng cách L1 trung bình giữa dự đoán và mục tiêu. Nó phạt mọi sai số một cách tuyến tính, giúp nó bền vững với các giá trị ngoại lai hơn so với MSE.

### Chuẩn L2 (Khoảng cách Euclidean)

Chuẩn L2 là khoảng cách đường thẳng. Căn bậc hai của tổng các bình phương thành phần.

```
||x||_2 = sqrt(x_1^2 + x_2^2 + ... + x_n^2)
```

Đây là khoảng cách bạn đã học trong lớp hình học. Định lý Pythagoras trong n chiều.

```
Point A = (1, 1)
Point B = (4, 5)

L2 distance = sqrt((4-1)^2 + (5-1)^2) = sqrt(9 + 16) = sqrt(25) = 5.0

The straight line, cutting diagonally through the grid.
```

Khi nào nên dùng L2:
- Dữ liệu liên tục có số chiều từ thấp đến trung bình
- Khi các thang đo đặc trưng có thể so sánh được
- Khoảng cách vật lý (dữ liệu không gian, số liệu cảm biến)
- Độ tương đồng hình ảnh ở cấp độ pixel

Kết nối với L2 regularization (Ridge): thêm ||w||_2^2 vào hàm mất mát sẽ phạt các trọng số lớn. Không giống như L1, nó không đẩy trọng số về 0. Nó thu nhỏ tất cả các trọng số về phía 0 một cách tỷ lệ. Hình phạt L2 tạo ra các vùng ràng buộc hình tròn, vì vậy không có các góc trên các trục. Các trọng số trở nên nhỏ nhưng hiếm khi bằng đúng 0.

Kết nối với các hàm mất mát: Mean Squared Error (MSE) là trung bình của các khoảng cách L2 bình phương. Việc bình phương làm cho các sai số lớn bị phạt nặng hơn nhiều so với các sai số nhỏ.

```
MAE (L1 loss):  |y - y_hat|         Linear penalty. Robust to outliers.
MSE (L2 loss):  (y - y_hat)^2       Quadratic penalty. Sensitive to outliers.
```

### Các chuẩn Lp: họ tổng quát

L1 và L2 là các trường hợp đặc biệt của chuẩn Lp:

```
||x||_p = (|x_1|^p + |x_2|^p + ... + |x_n|^p)^(1/p)
```

Các giá trị p khác nhau tạo ra các "hình cầu đơn vị" (tập hợp tất cả các điểm có khoảng cách 1 từ gốc tọa độ) với hình dạng khác nhau:

```
p=1:    Diamond shape      (corners on axes)
p=2:    Circle/sphere      (the usual round ball)
p=3:    Superellipse       (rounded square)
p=inf:  Square/hypercube   (flat sides along axes)
```

### Chuẩn L-vô cùng (Khoảng cách Chebyshev)

Khi p tiến tới vô cùng, chuẩn Lp hội tụ về thành phần tuyệt đối lớn nhất.

```
||x||_inf = max(|x_1|, |x_2|, ..., |x_n|)
```

Khoảng cách giữa hai điểm được xác định bởi chiều duy nhất mà chúng khác biệt nhiều nhất. Tất cả các chiều khác đều bị bỏ qua.

```
Point A = (1, 1)
Point B = (4, 5)

L-inf distance = max(|4-1|, |5-1|) = max(3, 4) = 4
```

Khi nào nên dùng L-vô cùng:
- Khi độ lệch xấu nhất trong bất kỳ chiều đơn lẻ nào là quan trọng
- Bàn cờ (vua trong cờ vua di chuyển theo L-vô cùng: một bước theo bất kỳ hướng nào tốn 1 đơn vị)
- Dung sai sản xuất (mọi chiều phải nằm trong thông số kỹ thuật)

### Cosine Similarity và Cosine Distance

Cosine similarity đo góc giữa hai vector, bỏ qua độ lớn của chúng.

```
cos_sim(a, b) = (a . b) / (||a||_2 * ||b||_2)
```

Nó dao động từ -1 (hướng ngược nhau) đến +1 (cùng hướng). Các vector vuông góc có cosine similarity bằng 0.

Cosine distance chuyển đổi nó thành một khoảng cách: cosine_distance = 1 - cosine_similarity. Giá trị này dao động từ 0 (cùng hướng) đến 2 (ngược hướng).

```
a = (1, 0)    b = (1, 1)

cos_sim = (1*1 + 0*1) / (1 * sqrt(2)) = 1/sqrt(2) = 0.707
cos_dist = 1 - 0.707 = 0.293
```

Tại sao cosine thống trị NLP và các embedding: trong văn bản, độ dài tài liệu không nên ảnh hưởng đến độ tương đồng. Một tài liệu về mèo dài gấp đôi một tài liệu khác về mèo vẫn nên được coi là "tương đồng". Cosine similarity bỏ qua độ lớn (độ dài) và chỉ quan tâm đến hướng. Hai tài liệu có cùng phân phối từ ngữ nhưng độ dài khác nhau sẽ chỉ cùng một hướng và có cosine similarity là 1.0.

Khi nào nên dùng cosine similarity:
- Độ tương đồng văn bản (vector TF-IDF, word embedding, sentence embedding)
- Bất kỳ lĩnh vực nào mà độ lớn là nhiễu và hướng là tín hiệu
- Hệ thống gợi ý (vector sở thích người dùng)
- Tìm kiếm embedding (cơ sở dữ liệu vector hầu như luôn sử dụng cosine hoặc dot product)

### Dot Product Similarity vs Cosine Similarity

Tích vô hướng (dot product) của hai vector là:

```
a . b = a_1*b_1 + a_2*b_2 + ... + a_n*b_n
      = ||a|| * ||b|| * cos(angle)
```

Cosine similarity là tích vô hướng được chuẩn hóa bởi cả hai độ lớn. Khi cả hai vector đã được chuẩn hóa đơn vị (độ lớn = 1), dot product và cosine similarity là giống hệt nhau.

```
If ||a|| = 1 and ||b|| = 1:
    a . b = cos(angle between a and b)
```

Khi nào chúng khác nhau: dot product bao gồm thông tin về độ lớn. Một vector có độ lớn lớn hơn sẽ nhận được điểm dot product cao hơn. Điều này quan trọng trong một số hệ thống truy xuất nơi bạn muốn các mục "phổ biến" được xếp hạng cao hơn. Độ lớn đóng vai trò như một tín hiệu ngầm về chất lượng hoặc tầm quan trọng.

```
a = (3, 0)    b = (1, 0)    c = (0, 1)

dot(a, b) = 3     dot(a, c) = 0
cos(a, b) = 1.0   cos(a, c) = 0.0

Both agree on direction, but dot product also reflects magnitude.
```

Trong thực tế:
- Sử dụng cosine similarity khi bạn muốn độ tương đồng thuần túy về hướng
- Sử dụng dot product khi độ lớn mang thông tin có ý nghĩa
- Nhiều cơ sở dữ liệu vector (Pinecone, Weaviate, Qdrant) cho phép bạn chọn giữa chúng
- Nếu các embedding của bạn đã được chuẩn hóa L2, lựa chọn này không quan trọng

### Khoảng cách Mahalanobis

Khoảng cách Euclidean coi mọi chiều là bình đẳng. Nhưng nếu các đặc trưng của bạn có tương quan hoặc có thang đo khác nhau, L2 sẽ đưa ra kết quả sai lệch.

Khoảng cách Mahalanobis tính đến cấu trúc hiệp phương sai (covariance) của dữ liệu.

```
d_M(x, y) = sqrt((x - y)^T * S^(-1) * (x - y))
```

trong đó S là ma trận hiệp phương sai của dữ liệu.

Trực giác: Khoảng cách Mahalanobis trước tiên khử tương quan và chuẩn hóa dữ liệu (whitening), sau đó tính khoảng cách L2 trong không gian đã biến đổi đó. Nếu S là ma trận đơn vị (các đặc trưng không tương quan, phương sai đơn vị), khoảng cách Mahalanobis sẽ rút gọn về khoảng cách Euclidean.

```
Example: height and weight are correlated.
Someone 6'2" and 180 lbs is not unusual.
Someone 5'0" and 180 lbs is unusual.

Euclidean distance might say they are equally far from the mean.
Mahalanobis distance correctly identifies the second as an outlier
because it accounts for the height-weight correlation.
```

Khi nào nên dùng khoảng cách Mahalanobis:
- Phát hiện ngoại lai (các điểm có khoảng cách Mahalanobis lớn từ giá trị trung bình là các điểm ngoại lai)
- Phân loại khi các đặc trưng có thang đo và tương quan khác nhau
- Khi bạn có đủ dữ liệu để ước tính một ma trận hiệp phương sai đáng tin cậy
- Kiểm soát chất lượng trong sản xuất (giám sát quy trình đa biến)

### Jaccard Similarity (cho các tập hợp)

Jaccard similarity đo lường sự chồng lấp giữa hai tập hợp.

```
J(A, B) = |A intersect B| / |A union B|
```

Nó dao động từ 0 (không chồng lấp) đến 1 (các tập hợp giống hệt nhau). Jaccard distance = 1 - Jaccard similarity.

```
A = {cat, dog, fish}
B = {cat, bird, fish, snake}

Intersection = {cat, fish}         size = 2
Union = {cat, dog, fish, bird, snake}  size = 5

Jaccard similarity = 2/5 = 0.4
Jaccard distance = 0.6
```

Khi nào nên dùng Jaccard:
- So sánh các tập hợp thẻ (tags), danh mục hoặc đặc trưng
- Độ tương đồng tài liệu dựa trên sự hiện diện của từ (không phải tần suất)
- Phát hiện các tài liệu gần giống nhau (xấp xỉ MinHash của Jaccard)
- So sánh các vector đặc trưng nhị phân (dữ liệu hiện diện/vắng mặt)
- Đánh giá các mô hình phân đoạn (Intersection over Union = Jaccard)

### Edit Distance (Khoảng cách Levenshtein)

Edit distance đếm số lượng tối thiểu các thao tác trên từng ký tự cần thiết để biến đổi chuỗi này thành chuỗi kia. Các thao tác là: chèn, xóa hoặc thay thế.

```
"kitten" -> "sitting"

kitten -> sitten  (substitute k -> s)
sitten -> sittin  (substitute e -> i)
sittin -> sitting (insert g)

Edit distance = 3
```

Được tính toán bằng quy hoạch động. Điền vào một ma trận nơi mục (i, j) là edit distance giữa i ký tự đầu tiên của chuỗi A và j ký tự đầu tiên của chuỗi B.

```
        ""  s  i  t  t  i  n  g
    ""   0  1  2  3  4  5  6  7
    k    1  1  2  3  4  5  6  7
    i    2  2  1  2  3  4  5  6
    t    3  3  2  1  2  3  4  5
    t    4  4  3  2  1  2  3  4
    e    5  5  4  3  2  2  3  4
    n    6  6  5  4  3  3  2  3
```

Khi nào nên dùng edit distance:
- Kiểm tra và sửa lỗi chính tả
- Căn chỉnh trình tự DNA (với các thao tác có trọng số)
- Khớp chuỗi mờ (fuzzy string matching)
- Khử trùng lặp dữ liệu văn bản nhiễu

### KL Divergence (không phải khoảng cách, nhưng được dùng như vậy)

KL divergence đo lường cách một phân phối xác suất khác biệt với phân phối khác. Được đề cập trong Bài 09, nhưng nó thuộc về thảo luận này vì mọi người sử dụng nó như một "khoảng cách" mặc dù nó không phải vậy.

```
D_KL(P || Q) = sum(p(x) * log(p(x) / q(x)))
```

Tính chất quan trọng: KL divergence KHÔNG đối xứng.

```
D_KL(P || Q) != D_KL(Q || P)
```

Điều này có nghĩa là nó không thỏa mãn yêu cầu cơ bản của một metric khoảng cách. Nó cũng không thỏa mãn bất đẳng thức tam giác. Nó là một sự phân kỳ (divergence), không phải khoảng cách.

Forward KL (D_KL(P || Q)) là "tìm kiếm trung bình": Q cố gắng bao phủ tất cả các mode của P.
Reverse KL (D_KL(Q || P)) là "tìm kiếm mode": Q tập trung vào một mode duy nhất của P.

Khi bạn thấy KL divergence:
- VAEs (thành phần KL trong ELBO đẩy phân phối tiềm ẩn về phía một phân phối tiên nghiệm)
- Chưng cất tri thức (student cố gắng khớp với phân phối của teacher)
- RLHF (hình phạt KL giữ cho mô hình đã tinh chỉnh gần với mô hình cơ sở)
- Các phương pháp policy gradient (ràng buộc cập nhật chính sách)

### Khoảng cách Wasserstein (Earth Mover's Distance)

Khoảng cách Wasserstein đo lường "công" tối thiểu cần thiết để biến đổi một phân phối xác suất này thành phân phối khác. Hãy nghĩ về nó như sau: nếu một phân phối là một đống đất và phân phối kia là một cái hố, bạn phải di chuyển bao nhiêu đất và di chuyển bao xa?

```
W(P, Q) = inf over all transport plans gamma of E[d(x, y)]
```

Đối với các phân phối 1D, nó đơn giản hóa thành tích phân của sự khác biệt tuyệt đối giữa các hàm phân phối tích lũy (CDF):

```
W_1(P, Q) = integral |CDF_P(x) - CDF_Q(x)| dx
```

Tại sao Wasserstein quan trọng:
- Nó là một metric thực sự (đối xứng, thỏa mãn bất đẳng thức tam giác)
- Nó cung cấp gradient ngay cả khi các phân phối không chồng lấp (KL divergence tiến tới vô cùng)
- Tính chất này làm cho nó trở thành trung tâm của Wasserstein GANs (WGANs), giải quyết sự mất ổn định khi huấn luyện của các GAN gốc.

```
Distributions with no overlap:

P: [1, 0, 0, 0, 0]    Q: [0, 0, 0, 0, 1]

KL divergence: infinity (log of zero)
Wasserstein: 4 (move all mass 4 bins)

Wasserstein gives a meaningful gradient. KL does not.
```

Khi nào nên dùng Wasserstein:
- Huấn luyện GAN (WGAN, WGAN-GP)
- So sánh các phân phối có thể không chồng lấp
- Các bài toán vận tải tối ưu (optimal transport)
- Truy xuất hình ảnh (so sánh biểu đồ màu)

### Tại sao các tác vụ khác nhau cần các khoảng cách khác nhau

| Tác vụ | Khoảng cách tốt nhất | Tại sao |
|------|--------------|-----|
| Độ tương đồng văn bản | Cosine | Độ lớn là nhiễu, hướng là ý nghĩa |
| So sánh pixel hình ảnh | L2 | Quan hệ không gian quan trọng, các đặc trưng có thang đo tương đương |
| Đặc trưng thưa, số chiều cao | L1 | Bền vững, không khuếch đại các khác biệt lớn hiếm gặp |
| Chồng lấp tập hợp (thẻ, danh mục) | Jaccard | Dữ liệu tự nhiên là tập hợp, không phải vector |
| Khớp chuỗi | Edit distance | Các thao tác ánh xạ tới trực giác chỉnh sửa của con người |
| Phát hiện ngoại lai | Mahalanobis | Tính đến tương quan và thang đo của đặc trưng |
| So sánh các phân phối | KL divergence | Đo lường thông tin bị mất khi sử dụng Q thay vì P |
| Huấn luyện GAN | Wasserstein | Cung cấp gradient ngay cả khi các phân phối không chồng lấp |
| Embeddings (vector DB) | Cosine hoặc dot product | Embeddings được huấn luyện để mã hóa ý nghĩa vào hướng |
| Gợi ý | Dot product | Độ lớn có thể mã hóa độ phổ biến hoặc độ tin cậy |
| Trình tự DNA | Weighted edit distance | Chi phí thay thế thay đổi theo cặp nucleotide |
| QC sản xuất | L-vô cùng | Độ lệch xấu nhất trong bất kỳ chiều nào là quan trọng |

### Kết nối với các hàm mất mát

Các hàm mất mát là các hàm khoảng cách được áp dụng cho dự đoán so với mục tiêu.

```
Loss function       Distance it uses       Behavior
MSE                 L2 squared             Penalizes large errors heavily
MAE                 L1                     Penalizes all errors equally
Huber loss          L1 for large errors,   Best of both: robust to outliers,
                    L2 for small errors    smooth gradient near zero
Cross-entropy       KL divergence          Measures distribution mismatch
Hinge loss          max(0, margin - d)     Only penalizes below margin
Triplet loss        L2 (typically)         Pulls positives close, pushes
                                           negatives away
Contrastive loss    L2                     Similar pairs close, dissimilar
                                           pairs beyond margin
```

### Kết nối với Regularization

Regularization thêm một hình phạt chuẩn trên các trọng số vào hàm mất mát.

```
L1 regularization (Lasso):   loss + lambda * ||w||_1
  -> Sparse weights. Some weights become exactly zero.
  -> Automatic feature selection.
  -> Solution has corners (non-differentiable at zero).

L2 regularization (Ridge):   loss + lambda * ||w||_2^2
  -> Small weights. All weights shrink toward zero.
  -> No feature selection (nothing goes to exactly zero).
  -> Smooth solution everywhere.

Elastic Net:                  loss + lambda_1 * ||w||_1 + lambda_2 * ||w||_2^2
  -> Combines sparsity of L1 with stability of L2.
  -> Groups of correlated features are kept or dropped together.
```

Tại sao L1 tạo ra tính thưa còn L2 thì không: hãy hình dung vùng ràng buộc trong không gian trọng số 2D. L1 là hình kim cương, L2 là hình tròn. Các đường đồng mức của hàm mất mát (hình elip) có khả năng cao nhất chạm vào kim cương tại một góc, nơi một trọng số bằng 0. Chúng chạm vào hình tròn tại một điểm trơn, nơi cả hai trọng số đều khác 0.

### Tìm kiếm lân cận gần nhất (Nearest Neighbor Search)

Mỗi hàm khoảng cách đều ngụ ý một bài toán tìm kiếm lân cận gần nhất: cho một điểm truy vấn, tìm các điểm gần nhất trong tập dữ liệu.

Tìm kiếm lân cận gần nhất chính xác là O(n * d) cho mỗi truy vấn trong tập dữ liệu n điểm với d chiều. Đối với các tập dữ liệu lớn, điều này quá chậm.

Các thuật toán Approximate Nearest Neighbor (ANN) đánh đổi một lượng nhỏ độ chính xác để đạt được tốc độ cực nhanh:

```
Algorithm         Approach                      Used by
KD-trees          Axis-aligned space partition   scikit-learn (low-dim)
Ball trees        Nested hyperspheres            scikit-learn (medium-dim)
LSH               Random hash projections        Near-duplicate detection
HNSW              Hierarchical navigable         FAISS, Qdrant, Weaviate
                  small-world graph
IVF               Inverted file index with       FAISS (billion-scale)
                  cluster-based search
Product quant.    Compress vectors, search       FAISS (memory-constrained)
                  in compressed space
```

HNSW (Hierarchical Navigable Small World) là thuật toán thống trị trong các cơ sở dữ liệu vector hiện đại. Nó xây dựng một đồ thị đa tầng nơi mỗi nút kết nối với các lân cận gần nhất xấp xỉ của nó. Tìm kiếm bắt đầu ở tầng trên cùng (thưa, bước nhảy dài) và đi xuống tầng dưới cùng (dày, bước nhảy ngắn).

```figure
norm-unit-balls
```

## Xây dựng (Build It)

### Bước 1: Tất cả các hàm chuẩn và khoảng cách

Xem `code/distances.py` để biết triển khai đầy đủ. Mỗi hàm được xây dựng từ đầu chỉ sử dụng toán học Python cơ bản.

### Bước 2: Dữ liệu giống nhau, khoảng cách khác nhau, lân cận khác nhau

Bản demo trong `distances.py` tạo ra một tập dữ liệu, chọn một điểm truy vấn và cho thấy cách lân cận gần nhất thay đổi tùy thuộc vào metric khoảng cách. Điểm "gần nhất" theo L1 có thể không phải là gần nhất theo L2 hoặc cosine.

### Bước 3: Tìm kiếm độ tương đồng embedding

Mã nguồn bao gồm một tìm kiếm độ tương đồng embedding giả lập, tìm các "tài liệu" tương tự nhất với truy vấn bằng cách sử dụng cosine similarity so với khoảng cách L2, cho thấy rằng thứ hạng có thể khác nhau.

## Sử dụng (Use It)

Ứng dụng thực tế phổ biến nhất: tìm các mục tương tự trong cơ sở dữ liệu vector.

```python
import numpy as np

def cosine_similarity_matrix(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    X_normalized = X / norms
    return X_normalized @ X_normalized.T

embeddings = np.random.randn(1000, 768)

sim_matrix = cosine_similarity_matrix(embeddings)

query_idx = 0
similarities = sim_matrix[query_idx]
top_k = np.argsort(similarities)[::-1][1:6]
print(f"Top 5 most similar to item 0: {top_k}")
print(f"Similarities: {similarities[top_k]}")
```

Khi bạn gọi `model.encode(text)` và sau đó tìm kiếm trong cơ sở dữ liệu vector, đây là những gì xảy ra bên dưới. Mô hình embedding ánh xạ văn bản thành các vector. Cơ sở dữ liệu vector tính toán cosine similarity (hoặc dot product) giữa vector truy vấn của bạn và mọi vector được lưu trữ, sử dụng các thuật toán ANN để tránh việc kiểm tra tất cả chúng.

## Bài tập

1. Tính khoảng cách L1, L2 và L-vô cùng giữa (1, 2, 3) và (4, 0, 6). Xác minh rằng L-inf <= L2 <= L1 luôn đúng cho bất kỳ cặp điểm nào. Chứng minh tại sao thứ tự này được đảm bảo.

2. Tạo hai vector có cosine similarity cao (> 0.9) nhưng khoảng cách L2 lớn (> 10). Giải thích về mặt hình học điều gì đang xảy ra. Sau đó tạo hai vector có cosine similarity thấp (< 0.3) nhưng khoảng cách L2 nhỏ (< 0.5).

3. Triển khai một hàm nhận vào một tập dữ liệu và một điểm truy vấn, trả về lân cận gần nhất theo khoảng cách L1, L2, cosine và Mahalanobis. Tìm một tập dữ liệu mà cả bốn metric đều không đồng ý về việc điểm nào là gần nhất.

4. Tính khoảng cách Wasserstein giữa [0.5, 0.5, 0, 0] và [0, 0, 0.5, 0.5] bằng tay sử dụng phương pháp CDF. Sau đó tính giữa [0.25, 0.25, 0.25, 0.25] và [0, 0, 0.5, 0.5]. Cái nào lớn hơn và tại sao?

5. Triển khai MinHash cho xấp xỉ Jaccard similarity. Tạo 100 tập hợp ngẫu nhiên, tính Jaccard chính xác cho tất cả các cặp, và so sánh với xấp xỉ MinHash sử dụng 50, 100 và 200 hàm băm. Vẽ biểu đồ sai số xấp xỉ.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|----------------------|
| Norm | "Kích thước của vector" | Một hàm ánh xạ vector sang vô hướng không âm, thỏa mãn bất đẳng thức tam giác, tính đồng nhất tuyệt đối và bằng 0 chỉ khi là vector không |
| L1 norm | "Khoảng cách Manhattan" | Tổng các giá trị tuyệt đối của thành phần. Tạo ra tính thưa trong tối ưu hóa. Bền vững với ngoại lai |
| L2 norm | "Khoảng cách Euclidean" | Căn bậc hai của tổng các bình phương thành phần. Khoảng cách đường thẳng trong không gian Euclidean |
| Lp norm | "Chuẩn tổng quát" | Căn bậc p của tổng các lũy thừa bậc p của các thành phần tuyệt đối. L1 và L2 là các trường hợp đặc biệt |
| L-infinity norm | "Max norm" hoặc "Khoảng cách Chebyshev" | Giá trị tuyệt đối lớn nhất của thành phần. Giới hạn của Lp khi p tiến tới vô cùng |
| Cosine similarity | "Góc giữa các vector" | Tích vô hướng được chuẩn hóa bởi cả hai độ lớn. Dao động từ -1 đến +1. Bỏ qua độ dài vector |
| Cosine distance | "1 trừ cosine similarity" | Chuyển đổi cosine similarity thành khoảng cách. Dao động từ 0 đến 2 |
| Dot product | "Cosine chưa chuẩn hóa" | Tổng các tích của từng thành phần. Bằng cosine similarity nhân với cả hai độ lớn |
| Mahalanobis distance | "Khoảng cách nhận biết tương quan" | Khoảng cách L2 trong không gian đã được làm trắng (khử tương quan và chuẩn hóa) bằng ma trận hiệp phương sai dữ liệu |
| Jaccard similarity | "Chồng lấp tập hợp" | Kích thước phần giao chia cho kích thước phần hợp. Dành cho tập hợp, không phải vector |
| Edit distance | "Khoảng cách Levenshtein" | Số lượng tối thiểu các thao tác chèn, xóa, thay thế để biến đổi chuỗi này thành chuỗi kia |
| KL divergence | "Khoảng cách giữa các phân phối" | Không phải khoảng cách thực sự (không đối xứng). Đo lường số bit dư thừa khi dùng Q để mã hóa P |
| Wasserstein distance | "Earth mover's distance" | Công tối thiểu để vận chuyển khối lượng từ phân phối này sang phân phối khác. Một metric thực sự |
| Approximate nearest neighbor | "Tìm kiếm ANN" | Các thuật toán (HNSW, LSH, IVF) tìm các điểm gần nhất xấp xỉ nhanh hơn nhiều so với tìm kiếm chính xác |
| HNSW | "Thuật toán vector DB" | Đồ thị Hierarchical Navigable Small World. Đồ thị đa tầng cho tìm kiếm lân cận gần nhất xấp xỉ nhanh |
| L1 regularization | "Lasso" | Thêm chuẩn L1 của trọng số vào hàm mất mát. Đẩy trọng số về 0 (tính thưa) |
| L2 regularization | "Ridge" hoặc "weight decay" | Thêm chuẩn L2 bình phương của trọng số vào hàm mất mát. Thu nhỏ trọng số về 0 mà không tạo tính thưa |
| Elastic Net | "L1 + L2" | Kết hợp regularization L1 và L2. Xử lý các nhóm đặc trưng tương quan tốt hơn so với dùng riêng lẻ |

## Đọc thêm

- [FAISS: A Library for Efficient Similarity Search](https://github.com/facebookresearch/faiss) - Thư viện của Meta cho tìm kiếm ANN quy mô hàng tỷ
- [Wasserstein GAN (Arjovsky et al., 2017)](https://arxiv.org/abs/1701.07875) - bài báo giới thiệu Earth Mover's distance vào GANs
- [Locality-Sensitive Hashing (Indyk & Motwani, 1998)](https://dl.acm.org/doi/10.1145/276698.276876) - thuật toán ANN nền tảng
- [Efficient Estimation of Word Representations (Mikolov et al., 2013)](https://arxiv.org/abs/1301.3781) - Word2Vec, nơi cosine similarity trở thành mặc định cho các embedding
- [sklearn.neighbors documentation](https://scikit-learn.org/stable/modules/neighbors.html) - hướng dẫn thực tế về các metric khoảng cách và thuật toán lân cận trong scikit-learn