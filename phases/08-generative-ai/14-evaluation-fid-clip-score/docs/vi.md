# Đánh giá — FID, CLIP Score, Human Preference

> Mọi bảng xếp hạng mô hình tạo sinh đều trích dẫn FID, CLIP score và tỷ lệ thắng từ các đấu trường ưu tiên của con người (human-preference arena). Mỗi con số đều có các chế độ lỗi (failure mode) mà một nhà nghiên cứu quyết tâm có thể thao túng. Nếu bạn không biết các chế độ lỗi này, bạn không thể phân biệt được đâu là cải tiến thực sự và đâu là kết quả từ việc "chơi chiêu" trên các chỉ số.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 01 (Taxonomy), Phase 2 · 04 (Evaluation Metrics)
**Time:** ~45 phút

## Vấn đề

Một mô hình tạo sinh được đánh giá dựa trên *chất lượng mẫu* (sample quality) và *độ bám sát điều kiện* (conditioning adherence). Không có thước đo dạng đóng (closed-form) cho cả hai yếu tố này. Mô hình của bạn phải render 10.000 hình ảnh; cần có thứ gì đó gán cho chúng các con số; và bạn phải tin tưởng vào các con số đó trên các dòng mô hình, độ phân giải và kiến trúc khác nhau. Ba chỉ số đã vượt qua thử thách từ 2014-2026:

- **FID (Fréchet Inception Distance).** Khoảng cách giữa hai phân phối — thực và tạo sinh — trong không gian đặc trưng của mạng Inception. Chỉ số càng thấp càng tốt.
- **CLIP score.** Độ tương đồng cosine giữa embedding CLIP-image của ảnh được tạo và embedding CLIP-text của prompt. Chỉ số càng cao càng tốt. Đo lường độ bám sát prompt.
- **Human preference.** Đặt hai mô hình đối đầu trực tiếp trên cùng một prompt, để con người (hoặc mô hình lớp GPT-4) chọn mô hình tốt hơn, sau đó tổng hợp thành điểm Elo.

Bạn cũng sẽ thấy: IS (Inception Score, phần lớn đã lỗi thời), KID, CMMD, ImageReward, PickScore, HPSv2, MJHQ-30k. Mỗi chỉ số đều khắc phục một lỗi của chỉ số trước đó.

## Khái niệm

![FID, CLIP, and preference: three axes, different failure modes](../assets/evaluation.svg)

### FID — chất lượng mẫu

Heusel và cộng sự (2017). Các bước:

1. Trích xuất đặc trưng Inception-v3 (2048-D) cho N ảnh thực và N ảnh tạo sinh.
2. Khớp một phân phối Gaussian cho mỗi tập: tính trung bình `μ_r, μ_g` và hiệp phương sai `Σ_r, Σ_g`.
3. FID = `||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2 · (Σ_r · Σ_g)^0.5)`.

Giải thích: Khoảng cách Fréchet giữa hai phân phối Gaussian đa biến trong không gian đặc trưng. Thấp hơn = phân phối tương đồng hơn.

Các chế độ lỗi:
- **Thiên kiến trên N nhỏ.** FID là bình phương trung bình trên phân phối đặc trưng — N nhỏ sẽ đánh giá thấp hiệp phương sai, dẫn đến FID thấp giả tạo. Luôn sử dụng N ≥ 10.000.
- **Phụ thuộc vào Inception.** Inception-v3 được huấn luyện trên ImageNet. Các miền dữ liệu xa lạ với ImageNet (khuôn mặt, nghệ thuật, ảnh văn bản) sẽ tạo ra FID vô nghĩa. Hãy sử dụng bộ trích xuất đặc trưng chuyên biệt cho miền dữ liệu đó.
- **Thao túng (Gaming).** Overfitting vào prior của Inception sẽ cho FID thấp mà không cải thiện chất lượng hình ảnh. Hãy khắc phục bằng CMMD (bên dưới).

### CLIP score — độ bám sát prompt

Radford và cộng sự (2021). Đối với một cặp ảnh tạo sinh + prompt:

```
clip_score = cos_sim( CLIP_image(x_gen), CLIP_text(prompt) )
```

Lấy trung bình trên 30k ảnh tạo sinh → một giá trị vô hướng có thể so sánh giữa các mô hình.

Các chế độ lỗi:
- **Điểm mù của chính CLIP.** CLIP có khả năng suy luận thành phần yếu ("một khối đỏ trên một hình cầu xanh" thường thất bại). Các mô hình có thể xếp hạng cao trên CLIP score mà không thực sự tuân theo các prompt phức tạp.
- **Thiên kiến prompt ngắn.** Các prompt ngắn có nhiều kết quả khớp CLIP-image hơn trong tự nhiên. Các prompt dài hơn thường có CLIP score thấp hơn về mặt cơ học.
- **Thao túng prompt.** Việc thêm "high quality, 4k, masterpiece" vào prompt sẽ làm tăng CLIP score mà không cải thiện sự gắn kết giữa ảnh và văn bản.

CMMD (Jayasumana và cộng sự, 2024) khắc phục một số vấn đề này: sử dụng đặc trưng CLIP thay vì Inception, sử dụng maximum-mean discrepancy thay vì Fréchet. Tốt hơn trong việc phát hiện các khác biệt chất lượng tinh tế.

### Human preference — sự thật khách quan

Chọn một tập hợp các prompt. Tạo ảnh với mô hình A và mô hình B. Hiển thị các cặp cho con người (hoặc một LLM judge mạnh). Tổng hợp các lượt thắng thành điểm Elo hoặc Bradley-Terry. Các benchmark:

- **PartiPrompts (Google)**: 1.600 prompt đa dạng, 12 danh mục.
- **HPSv2**: 107k chú thích từ con người, được sử dụng rộng rãi như một proxy tự động.
- **ImageReward**: 137k cặp ưu tiên prompt-ảnh, giấy phép MIT.
- **PickScore**: huấn luyện trên 2,6 triệu ưu tiên từ Pick-a-Pic.
- **Chatbot-Arena-style image arenas**: https://imagearena.ai/ và các loại khác.

Các chế độ lỗi:
- **Sự biến thiên của giám khảo.** Người không chuyên có sở thích khác với chuyên gia. Hãy sử dụng cả hai.
- **Phân phối prompt.** Các prompt được chọn lọc kỹ (cherry-picked) sẽ ưu tiên một dòng mô hình nhất định. Luôn ghi chép lại.
- **Hack phần thưởng LLM-judge.** GPT-4-judge dễ bị đánh lừa bởi các kết quả đầu ra "đẹp nhưng sai". Hãy đối chiếu với con người.

## Sử dụng kết hợp

Một báo cáo đánh giá sản phẩm nên bao gồm:

1. FID trên 10-30k mẫu so với phân phối thực (chất lượng mẫu).
2. CLIP score / CMMD trên cùng các mẫu đó so với prompt của chúng (độ bám sát).
3. Tỷ lệ thắng trong đấu trường mù (blinded arena) so với mô hình trước đó (ưu tiên tổng thể).
4. Phân tích chế độ lỗi: 50 kết quả được lấy mẫu ngẫu nhiên, gắn cờ cho các vấn đề đã biết (giải phẫu bàn tay, render văn bản, số lượng vật thể nhất quán).

Bất kỳ chỉ số đơn lẻ nào cũng là một lời nói dối. Ba chỉ số xác thực + đánh giá định tính mới là một khẳng định.

```figure
gx-fid-distributions
```

## Xây dựng

`code/main.py` triển khai FID, CLIP-score và tổng hợp Elo trên các "vector đặc trưng" tổng hợp (chúng ta sử dụng vector 4-D làm đại diện cho đặc trưng Inception). Bạn sẽ thấy:

- Tính toán FID trên N nhỏ và N lớn — sự thiên kiến.
- "CLIP score" dưới dạng độ tương đồng cosine giữa các nhóm đặc trưng.
- Quy tắc cập nhật Elo từ luồng ưu tiên tổng hợp.

### Bước 1: FID trong bốn dòng

```python
def fid(real_features, gen_features):
    mu_r, cov_r = mean_and_cov(real_features)
    mu_g, cov_g = mean_and_cov(gen_features)
    mean_diff = sum((a - b) ** 2 for a, b in zip(mu_r, mu_g))
    trace_term = trace(cov_r) + trace(cov_g) - 2 * sqrt_cov_product(cov_r, cov_g)
    return mean_diff + trace_term
```

### Bước 2: Độ tương đồng cosine kiểu CLIP

```python
def clip_like(image_feat, text_feat):
    dot = sum(a * b for a, b in zip(image_feat, text_feat))
    norm = math.sqrt(dot_self(image_feat) * dot_self(text_feat))
    return dot / max(norm, 1e-8)
```

### Bước 3: Tổng hợp Elo

```python
def elo_update(r_a, r_b, winner, k=32):
    expected_a = 1 / (1 + 10 ** ((r_b - r_a) / 400))
    actual_a = 1.0 if winner == "a" else 0.0
    r_a_new = r_a + k * (actual_a - expected_a)
    r_b_new = r_b - k * (actual_a - expected_a)
    return r_a_new, r_b_new
```

## Các cạm bẫy

- **FID tại N=1000.** Heuristic không đáng tin cậy dưới N=10k. Các bài báo báo cáo FID ở N thấp là đang thao túng.
- **So sánh FID qua các độ phân giải.** Việc resize 299×299 của Inception làm thay đổi phân phối đặc trưng. Chỉ so sánh ở cùng độ phân giải.
- **Báo cáo một seed.** Chạy tối thiểu 3 seed. Báo cáo độ lệch chuẩn (std).
- **Lạm phát CLIP score qua negative prompt.** Một số pipeline tăng CLIP bằng cách overfitting vào prompt. Kiểm tra độ bão hòa hình ảnh.
- **Thiên kiến Elo từ sự trùng lặp prompt.** Nếu cả hai mô hình đều thấy một prompt benchmark trong quá trình huấn luyện, Elo là vô nghĩa. Sử dụng các tập prompt giữ lại (held-out).
- **Sự lệch lạc của đám đông đánh giá.** Người chú thích trên Prolific, MTurk thường trẻ hơn / am hiểu công nghệ hơn. Hãy kết hợp với các chuyên gia nghệ thuật/thiết kế.

## Sử dụng

Giao thức đánh giá sản phẩm năm 2026:

| Trụ cột | Tối thiểu | Khuyến nghị |
|--------|---------|-------------|
| Chất lượng mẫu | FID trên 10k vs thực tế giữ lại | + CMMD trên 5k + FID trên tập con theo danh mục |
| Độ bám sát prompt | CLIP score trên 30k | + HPSv2 + ImageReward + VQA-style question answering |
| Ưu tiên | 200 cặp mù vs baseline | + 2000 cặp con người + LLM-judge + Chatbot Arena |
| Phân tích lỗi | 50 gắn cờ thủ công | 500 gắn cờ thủ công + phân loại an toàn tự động |

Cả bốn trụ cột trong một báo cáo = khẳng định. Chỉ một trụ cột = marketing.

## Triển khai

Lưu `outputs/skill-eval-report.md`. Skill lấy một checkpoint mô hình mới + baseline và xuất ra một kế hoạch đánh giá đầy đủ: kích thước mẫu, chỉ số, các probe kiểm tra chế độ lỗi, tiêu chí phê duyệt.

## Bài tập

1. **Dễ.** Chạy `code/main.py`. So sánh FID tại N=100 vs N=1000 trên cùng các phân phối tổng hợp. Báo cáo độ lớn của thiên kiến.
2. **Trung bình.** Triển khai CMMD từ các đặc trưng kiểu CLIP tổng hợp (xem Jayasumana và cộng sự, 2024 cho công thức). So sánh độ nhạy với các khác biệt chất lượng so với FID.
3. **Khó.** Tái tạo thiết lập HPSv2: lấy 1000 cặp ảnh-prompt từ một tập con của Pick-a-Pic, fine-tune một bộ chấm điểm dựa trên CLIP nhỏ trên các ưu tiên, và đo lường sự đồng thuận của nó với một tập giữ lại.

## Thuật ngữ chính

| Thuật ngữ | Mọi người nói | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| FID | "Fréchet Inception Distance" | Khoảng cách Fréchet của các khớp Gaussian với đặc trưng Inception thực vs tạo sinh. |
| CLIP score | "Độ tương đồng văn bản-ảnh" | Độ tương đồng cosine giữa embedding CLIP ảnh và văn bản. |
| CMMD | "Thay thế cho FID" | MMD đặc trưng CLIP; ít thiên kiến hơn, không giả định Gaussian. |
| IS | "Inception score" | Exp KL(p(y|x) || p(y)); tương quan kém trên các mô hình hiện đại, đã lỗi thời. |
| HPSv2 / ImageReward / PickScore | "Proxy ưu tiên đã học" | Các mô hình nhỏ được huấn luyện trên ưu tiên của con người; dùng làm giám khảo tự động. |
| Elo | "Xếp hạng cờ vua" | Tổng hợp Bradley-Terry của các lượt thắng theo cặp. |
| PartiPrompts | "Tập prompt benchmark" | 1.600 prompt do Google tuyển chọn qua 12 danh mục. |
| FD-DINO | "Thay thế tự giám sát" | FD sử dụng đặc trưng DINOv2; tốt hơn cho các miền ngoài ImageNet. |

## Lưu ý sản xuất: đánh giá cũng là một workload inference

Chạy FID trên 10k mẫu nghĩa là tạo ra 10k hình ảnh. Đối với SDXL base 50 bước ở 1024² trên một card L4, đó là ~11 giờ inference đơn lẻ. Ngân sách đánh giá là có thật, và khung làm việc chính xác là kịch bản offline-inference (tối đa hóa thông lượng, bỏ qua TTFT):

- **Batch lớn, quên độ trễ đi.** Đánh giá offline = static batching ở kích thước lớn nhất vừa bộ nhớ. `pipe(...).images` với `num_images_per_prompt=8` trên một H100 80GB chạy nhanh hơn 4-6 lần so với request đơn lẻ.
- **Cache các đặc trưng thực.** Việc trích xuất đặc trưng Inception (FID) hoặc CLIP (CLIP-score, CMMD) trên tập tham chiếu thực được chạy *một lần*, lưu trữ dưới dạng `.npz`. Không tính toán lại mỗi lần đánh giá.

Đối với CI / regression gates: chạy FID + CLIP score trên tập con 500 mẫu mỗi PR (~30 phút); chạy full 10k FID + HPSv2 + Elo hàng đêm.

## Đọc thêm

- [Heusel và cộng sự (2017). GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (FID)](https://arxiv.org/abs/1706.08500) — Bài báo FID.
- [Jayasumana và cộng sự (2024). Rethinking FID: Towards a Better Evaluation Metric for Image Generation (CMMD)](https://arxiv.org/abs/2401.09603) — CMMD.
- [Radford và cộng sự (2021). Learning Transferable Visual Models from Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020) — CLIP.
- [Wu và cộng sự (2023). HPSv2: A Comprehensive Human Preference Score](https://arxiv.org/abs/2306.09341) — HPSv2.
- [Xu và cộng sự (2023). ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation](https://arxiv.org/abs/2304.05977) — ImageReward.
- [Yu và cộng sự (2023). Scaling Autoregressive Models for Content-Rich Text-to-Image Generation (Parti + PartiPrompts)](https://arxiv.org/abs/2206.10789) — PartiPrompts.
- [Stein và cộng sự (2023). Exposing flaws of generative model evaluation metrics](https://arxiv.org/abs/2306.04675) — Khảo sát về chế độ lỗi.