# Speaker Recognition & Verification

> ASR hỏi "họ đã nói gì?" Speaker recognition hỏi "ai là người nói?" Toán học của cả hai khá tương đồng — embeddings cộng với cosine — nhưng mọi quyết định trong sản xuất đều xoay quanh một con số EER duy nhất.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms & Mel), Phase 5 · 22 (Embedding Models)
**Time:** ~45 phút

## Vấn đề

Một người dùng nói một cụm từ xác thực. Bạn muốn biết: đây có phải là người mà họ tự nhận (*verification*, 1:1), hay đây là người đầu tiên trong ngân hàng đăng ký của bạn (*identification*, 1:N)? Hoặc không phải cả hai — đây có phải là một người nói lạ (*open-set*)?

Trước 2018: GMM-UBM + i-vectors. EER ở mức chấp nhận được nhưng dễ bị ảnh hưởng bởi sự thay đổi kênh (điện thoại so với laptop) và cảm xúc. 2018–2022: x-vectors (kiến trúc TDNN được huấn luyện với angular margin). 2022+: ECAPA-TDNN và WavLM-large embeddings. Đến năm 2026, lĩnh vực này bị thống trị bởi ba mô hình và một chỉ số.

Chỉ số đó là **EER** — Equal Error Rate. Thiết lập ngưỡng quyết định sao cho False Accept Rate = False Reject Rate. Điểm giao nhau đó chính là EER. Nó được sử dụng trong mọi bài báo, mọi bảng xếp hạng và mọi yêu cầu mua sắm công nghệ.

## Khái niệm

![Enrollment + verification pipeline with embedding + cosine + EER](../assets/speaker-verification.svg)

**Pipeline.** Đăng ký (Enrollment): ghi âm 5–30 giây của người nói mục tiêu; tính toán một embedding có kích thước cố định (192-d cho ECAPA-TDNN, 256-d cho WavLM-large). Xác thực (Verification): lấy embedding của đoạn âm thanh kiểm tra; tính toán độ tương đồng cosine; so sánh với một ngưỡng.

**ECAPA-TDNN (2020, vẫn thống trị năm 2026).** Emphasized Channel Attention, Propagation and Aggregation - Time-Delay Neural Network. Các khối 1D conv với squeeze-excitation, multi-head attention pooling, theo sau là một lớp tuyến tính để ra 192-d. Được huấn luyện trên VoxCeleb 1+2 (2.700 người nói, 1,1 triệu đoạn âm thanh) với hàm mất mát Additive Angular Margin (AAM-softmax).

**WavLM-SV (2022+).** Fine-tune một backbone SSL WavLM-large đã được huấn luyện trước với hàm mất mát AAM. Chất lượng cao hơn nhưng chậm hơn — 300+ MB so với 15 MB.

**x-vector (baseline).** TDNN + statistics pooling. Cổ điển; vẫn hữu ích trên CPU / edge.

**AAM-softmax.** Softmax tiêu chuẩn với margin bổ sung `m` trong không gian góc: `cos(θ + m)` cho lớp đúng. Ép buộc sự phân tách góc giữa các lớp. Thường là `m=0.2`, scale `s=30`.

### Chấm điểm (Scoring)

- **Cosine** giữa embedding đăng ký và embedding kiểm tra. Quyết định dựa trên ngưỡng.
- **PLDA (Probabilistic LDA).** Chiếu các embedding vào một không gian tiềm ẩn nơi tỷ lệ khả năng (likelihood ratio) giữa cùng người nói và khác người nói có dạng đóng. Được thêm vào trên nền tảng cosine để giảm thêm 10–20% EER. Tiêu chuẩn trước 2020; hiện chỉ dùng trong các thiết lập closed-set.
- **Chuẩn hóa điểm số (Score normalization).** `S-norm` hoặc `AS-norm`: chuẩn hóa từng điểm số dựa trên một nhóm các giá trị trung bình và độ lệch chuẩn của kẻ giả mạo. Rất cần thiết cho đánh giá đa miền (cross-domain).

### Các con số bạn cần biết (2026)

| Mô hình | VoxCeleb1-O EER | Tham số | Thông lượng (A100) |
|-------|-----------------|--------|-------------------|
| x-vector (cổ điển) | 3.10% | 5 M | 400× RT |
| ECAPA-TDNN | 0.87% | 15 M | 200× RT |
| WavLM-SV large | 0.42% | 316 M | 20× RT |
| Pyannote 3.1 segmentation + embedding | 0.65% | 6 M | 100× RT |
| ReDimNet (2024) | 0.39% | 24 M | 100× RT |

### Diarization

"Ai nói khi nào" trong một đoạn clip có nhiều người nói. Pipeline: VAD → phân đoạn → nhúng (embed) từng phân đoạn → phân cụm (agglomerative hoặc spectral) → làm mịn ranh giới. Stack hiện đại: `pyannote.audio` 3.1, tích hợp phân đoạn người nói + nhúng + phân cụm trong một lệnh gọi. SOTA DER năm 2026 trên AMI là ~15% (giảm từ 23% năm 2022).

```figure
sp-eer-crossover
```

## Xây dựng (Build It)

### Bước 1: toy embedding từ thống kê MFCC

```python
def embed_mfcc_stats(signal, sr):
    frames = featurize_mfcc(signal, sr, n_mfcc=13)
    mean = [sum(f[i] for f in frames) / len(frames) for i in range(13)]
    std = [
        math.sqrt(sum((f[i] - mean[i]) ** 2 for f in frames) / len(frames))
        for i in range(13)
    ]
    return mean + std  # 26-d
```

Chưa phải là SOTA — chỉ dùng để giảng dạy. `code/main.py` sử dụng cách này như một bằng chứng khái niệm trên dữ liệu người nói tổng hợp.

### Bước 2: độ tương đồng cosine + ngưỡng

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0

def verify(enroll, test, threshold=0.75):
    return cosine(enroll, test) >= threshold
```

### Bước 3: EER từ các cặp tương đồng

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 1.0, 0.0)  # (fa, fr, threshold)
    for t in thresholds:
        fr = sum(1 for s in same_scores if s < t) / len(same_scores)
        fa = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        if abs(fa - fr) < abs(best[0] - best[1]):
            best = (fa, fr, t)
    return (best[0] + best[1]) / 2, best[2]
```

Trả về (eer, threshold_at_eer). Hãy báo cáo cả hai.

### Bước 4: sản xuất với SpeechBrain

```python
from speechbrain.pretrained import EncoderClassifier

clf = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

# enroll: average the embeddings of 3-5 clean samples
enroll = torch.stack([clf.encode_batch(load(x)) for x in enrollment_clips]).mean(0)
# verify
score = clf.similarity(enroll, clf.encode_batch(load("test.wav"))).item()
verdict = score > 0.25   # ECAPA typical threshold; tune on your data
```

### Bước 5: diarize với pyannote

```python
from pyannote.audio import Pipeline

pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
diarization = pipe("meeting.wav", num_speakers=None)
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{turn.start:.1f}–{turn.end:.1f}  {speaker}")
```

## Sử dụng (Use It)

Stack năm 2026:

| Tình huống | Lựa chọn |
|-----------|------|
| Xác thực 1:1 closed-set, edge | ECAPA-TDNN + ngưỡng cosine |
| Xác thực open-set, cloud | WavLM-SV + AS-norm |
| Diarization (họp, podcast) | `pyannote/speaker-diarization-3.1` |
| Chống giả mạo (phát lại / phát hiện deepfake) | AASIST hoặc RawNet2 |
| Tiny embedded (KWS + đăng ký) | Titanet-Small (NeMo) |

## Những cạm bẫy

- **Sai lệch kênh (Channel mismatch).** Mô hình huấn luyện trên VoxCeleb (video web) ≠ âm thanh cuộc gọi điện thoại. Luôn đánh giá trên kênh mục tiêu.
- **Đoạn âm thanh ngắn.** EER suy giảm mạnh dưới 3 giây âm thanh kiểm tra.
- **Đăng ký với tiếng ồn.** Một lần đăng ký bị nhiễu sẽ làm hỏng dữ liệu gốc. Hãy sử dụng ≥3 mẫu sạch và lấy trung bình.
- **Ngưỡng cố định trên các điều kiện.** Luôn điều chỉnh ngưỡng trên tập dev được giữ lại từ miền mục tiêu.
- **Cosine trên các embedding chưa chuẩn hóa.** Hãy L2-normalize trước; nếu không, độ lớn sẽ chiếm ưu thế.

## Triển khai (Ship It)

Lưu dưới dạng `outputs/skill-speaker-verifier.md`. Chọn mô hình, giao thức đăng ký, kế hoạch điều chỉnh ngưỡng và các biện pháp bảo vệ chống gian lận.

## Bài tập

1. **Dễ.** Chạy `code/main.py`. Xây dựng các "người nói" tổng hợp (các cấu hình âm sắc khác nhau), đăng ký, tính EER trên danh sách thử nghiệm 100 cặp.
2. **Trung bình.** Sử dụng SpeechBrain ECAPA trên 30 đoạn âm thanh VoxCeleb1 (5 người nói × 6 đoạn mỗi người). Tính EER với cosine so với PLDA.
3. **Khó.** Xây dựng pipeline đầy đủ từ đăng ký → diarize → xác thực với `pyannote.audio`. Đánh giá DER trên tập dev AMI.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| EER | Chỉ số tiêu đề | Ngưỡng nơi False Accept = False Reject. |
| Verification | 1:1 | "Đây có phải là Alice không?" |
| Identification | 1:N | "Ai đang nói?" |
| Open-set | Có thể là người lạ | Tập kiểm tra có thể chứa những người nói chưa đăng ký. |
| Enrollment | Đăng ký | Tính toán embedding tham chiếu của người nói. |
| AAM-softmax | Hàm mất mát | Softmax với additive angular margin; ép buộc phân tách cụm. |
| PLDA | Chấm điểm cổ điển | Probabilistic LDA; chấm điểm tỷ lệ khả năng trên các embedding. |
| DER | Chỉ số diarization | Diarization Error Rate — miss + false alarm + confusion. |

## Đọc thêm

- [Snyder et al. (2018). X-Vectors: Robust DNN Embeddings for Speaker Recognition](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf) — bài báo kinh điển về deep-embedding.
- [Desplanques et al. (2020). ECAPA-TDNN](https://arxiv.org/abs/2005.07143) — kiến trúc thống trị 2020–2026.
- [Chen et al. (2022). WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing](https://arxiv.org/abs/2110.13900) — backbone SSL cho SV và diarization.
- [Bredin et al. (2023). pyannote.audio 3.1](https://github.com/pyannote/pyannote-audio) — stack diarization + embedding cho sản xuất.
- [VoxCeleb leaderboard (cập nhật 2026)](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) — bảng xếp hạng EER hiện tại giữa các mô hình.