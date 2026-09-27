# Phân loại âm thanh — Từ k-NN trên MFCC đến AST và BEATs

> Mọi thứ từ "chó sủa vs còi báo động" cho đến "đây là ngôn ngữ gì" đều là phân loại âm thanh. Các đặc trưng là mels. Kiến trúc thay đổi theo từng thập kỷ. Việc đánh giá vẫn dựa trên AUC, F1 và recall trên mỗi lớp.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms & Mel), Phase 3 · 06 (CNNs), Phase 5 · 08 (CNNs & RNNs for Text)
**Time:** ~75 phút

## Vấn đề

Bạn có một đoạn clip dài 10 giây. Bạn muốn biết: "đây là gì?" Âm thanh đô thị (còi báo động, máy khoan, chó), lệnh thoại (có/không/dừng), ID ngôn ngữ (en/es/ar), cảm xúc người nói (giận dữ/trung tính), hoặc âm thanh môi trường (trong nhà/ngoài trời, tiếng ồn ào). Tất cả những thứ này đều là *phân loại âm thanh*, và vào năm 2026, kiến trúc cơ sở đã rất hoàn thiện: log-mel → CNN hoặc Transformer → softmax.

Khó khăn cốt lõi không nằm ở mạng thần kinh. Nó nằm ở dữ liệu. Các tập dữ liệu âm thanh có sự mất cân bằng lớp nghiêm trọng, sự dịch chuyển miền mạnh (sạch vs ồn) và nhiễu nhãn (ai là người quyết định "tiếng ồn đô thị" vs "tiếng ồn nhà hàng"?). 80% vấn đề nằm ở việc quản lý dữ liệu, tăng cường dữ liệu và đánh giá, chứ không phải việc thay thế CNN bằng Transformer.

## Khái niệm

![Audio classification ladder: k-NN on MFCCs to AST to BEATs](../assets/audio-classification.svg)

**k-NN trên MFCC (cơ sở từ những năm 1990).** Làm phẳng MFCC cho mỗi clip, tính độ tương đồng cosine với một ngân hàng dữ liệu đã dán nhãn, trả về kết quả bình chọn đa số của K hàng xóm gần nhất. Đáng ngạc nhiên là nó hoạt động rất mạnh trên các tập dữ liệu nhỏ, sạch (Speech Commands, ESC-50). Chạy mà không cần GPU.

**2D CNN trên log-mels (2015-2019).** Coi log-mel `(T, n_mels)` như một hình ảnh. Áp dụng ResNet-18 hoặc kiến trúc kiểu VGG. Thực hiện global mean pool trên trục thời gian. Softmax qua các lớp. Vẫn là cơ sở trong hầu hết các cuộc thi kaggle năm 2026.

**Audio Spectrogram Transformer, AST (2021-2024).** Chia log-mel thành các bản vá (ví dụ: 16×16 patches), thêm position embeddings, đưa vào ViT. Đạt trạng thái SOTA trên AudioSet (mAP 0.485) cho học có giám sát.

**BEATs và WavLM-base (2024-2026).** Tiền huấn luyện tự giám sát trên hàng triệu giờ dữ liệu. Tinh chỉnh (fine-tune) trên tác vụ của bạn với 1-10% dữ liệu có giám sát mà bạn cần. Vào năm 2026, đây là điểm khởi đầu mặc định cho âm thanh không phải giọng nói. BEATs-iter3 vượt AST từ 1-2 mAP trên AudioSet trong khi chỉ sử dụng 1/4 tài nguyên tính toán.

**Whisper-encoder làm backbone đóng băng (2024).** Lấy encoder của Whisper, bỏ decoder, gắn thêm một bộ phân loại tuyến tính. Gần đạt SOTA về ID ngôn ngữ và phân loại sự kiện đơn giản mà không cần tăng cường âm thanh. Đây là cơ sở "miễn phí" (free lunch).

### Mất cân bằng lớp là thách thức thực sự

ESC-50: 50 lớp, mỗi lớp 40 clip — cân bằng, dễ. UrbanSound8K: 10 lớp, mất cân bằng 10:1. AudioSet: 632 lớp với đuôi dài 100.000:1. Các kỹ thuật hiệu quả:

- Lấy mẫu cân bằng trong quá trình huấn luyện (không phải trong đánh giá).
- Mixup: nội suy tuyến tính hai clip (và nhãn của chúng) như một cách tăng cường dữ liệu.
- SpecAugment: che các dải thời gian và tần số ngẫu nhiên. Đơn giản; nhưng rất quan trọng.

### Đánh giá

- Đa lớp độc quyền (Speech Commands): độ chính xác top-1, độ chính xác top-5.
- Đa lớp đa nhãn (AudioSet, kiểu UrbanSound): mean average precision (mAP).
- Mất cân bằng nghiêm trọng: recall trên mỗi lớp + macro F1.

Các con số năm 2026 bạn nên biết:

| Benchmark | Cơ sở | SOTA 2026 | Nguồn |
|-----------|----------|-----------|--------|
| ESC-50 | 82% (AST) | 97.0% (BEATs-iter3) | BEATs paper (2024) |
| AudioSet mAP | 0.485 (AST) | 0.548 (BEATs-iter3) | HEAR leaderboard 2026 |
| Speech Commands v2 | 98% (CNN) | 99.0% (Audio-MAE) | HEAR v2 results |

```figure
mfcc-pipeline
```

## Xây dựng

### Bước 1: trích xuất đặc trưng

```python
def featurize_mfcc(signal, sr, n_mfcc=13, n_mels=40, frame_len=400, hop=160):
    mag = stft_magnitude(signal, frame_len, hop)
    fb = mel_filterbank(n_mels, frame_len, sr)
    mels = apply_filterbank(mag, fb)
    log = log_transform(mels)
    return [dct_ii(frame, n_mfcc) for frame in log]
```

### Bước 2: tóm tắt độ dài cố định

```python
def summarize(mfcc_frames):
    n = len(mfcc_frames[0])
    mean = [sum(f[i] for f in mfcc_frames) / len(mfcc_frames) for i in range(n)]
    var = [
        sum((f[i] - mean[i]) ** 2 for f in mfcc_frames) / len(mfcc_frames) for i in range(n)
    ]
    return mean + var
```

Đơn giản nhưng mạnh mẽ: trung bình + phương sai theo thời gian tạo ra một embedding cố định 26 chiều cho 13 hệ số MFCC. Chạy tức thì. Đã từng đánh bại các cơ sở NN hiện đại nhất trên ESC-50 vào năm 2017.

### Bước 3: k-NN

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(x * x for x in b)) or 1e-12
    return dot / (na * nb)

def knn_classify(q, bank, labels, k=5):
    sims = sorted(range(len(bank)), key=lambda i: -cosine(q, bank[i]))[:k]
    votes = Counter(labels[i] for i in sims)
    return votes.most_common(1)[0][0]
```

### Bước 4: nâng cấp lên CNN trên log-mels

Trong PyTorch:

```python
import torch.nn as nn

class AudioCNN(nn.Module):
    def __init__(self, n_mels=80, n_classes=50):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):  # x: (B, 1, T, n_mels)
        return self.head(self.body(x).flatten(1))
```

3 triệu tham số. Huấn luyện trong ~10 phút trên ESC-50 với một card RTX 4090. Độ chính xác 80%+.

### Bước 5: mặc định năm 2026 — tinh chỉnh BEATs

```python
from transformers import ASTFeatureExtractor, ASTForAudioClassification

ext = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=50,
    ignore_mismatched_sizes=True,
)

inputs = ext(audio, sampling_rate=16000, return_tensors="pt")
logits = model(**inputs).logits
```

Đối với BEATs, hãy sử dụng `microsoft/BEATs-base` thông qua thư viện `beats`; API transformers có cấu trúc tương tự.

## Sử dụng

Stack công nghệ năm 2026:

| Tình huống | Bắt đầu với |
|-----------|-----------|
| Tập dữ liệu nhỏ (<1000 clip) | k-NN trên trung bình MFCC (cơ sở của bạn) + tăng cường âm thanh |
| Tập dữ liệu trung bình (1K–100K) | Tinh chỉnh BEATs hoặc AST |
| Tập dữ liệu lớn (>100K) | Huấn luyện từ đầu hoặc tinh chỉnh Whisper-encoder |
| Thời gian thực, edge | 40-MFCC CNN, lượng tử hóa sang int8 (kiểu KWS) |
| Đa nhãn (AudioSet) | BEATs-iter3 với hàm mất mát BCE + mixup + SpecAugment |
| ID ngôn ngữ | MMS-LID, cơ sở SpeechBrain VoxLingua107 |

Quy tắc quyết định: **bắt đầu với một backbone đóng băng, không phải một mô hình mới hoàn toàn**. Tinh chỉnh phần đầu (head) của BEATs giúp bạn đạt 95% SOTA trong vài giờ, thay vì vài tuần.

## Triển khai

Lưu dưới dạng `outputs/skill-classifier-designer.md`. Chọn kiến trúc, các phương pháp tăng cường, chiến lược cân bằng lớp và chỉ số đánh giá cho một tác vụ phân loại âm thanh cụ thể.

## Bài tập

1. **Dễ.** Chạy `code/main.py`. Nó huấn luyện cơ sở k-NN MFCC trên tập dữ liệu tổng hợp 4 lớp (các âm thuần ở các cao độ khác nhau). Báo cáo ma trận nhầm lẫn (confusion matrix).
2. **Trung bình.** Thay thế `summarize` bằng [mean, var, skew, kurtosis]. Liệu pooling 4-moment có đánh bại trung bình+phương sai trên cùng tập dữ liệu tổng hợp đó không?
3. **Khó.** Sử dụng `torchaudio`, huấn luyện một CNN 2D trên ESC-50 fold 1. Báo cáo độ chính xác kiểm chứng chéo 5-fold. Thêm SpecAugment (time mask = 20, freq mask = 10) và báo cáo sự thay đổi (delta).

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| AudioSet | ImageNet của âm thanh | Tập dữ liệu YouTube 2 triệu clip, 632 lớp được dán nhãn yếu của Google. |
| ESC-50 | Benchmark phân loại nhỏ | 50 lớp × 40 clip âm thanh môi trường. |
| AST | Audio Spectrogram Transformer | ViT trên các bản vá log-mel; SOTA năm 2021. |
| BEATs | Âm thanh tự giám sát | Mô hình của Microsoft, iter3 dẫn đầu AudioSet tính đến năm 2026. |
| Mixup | Tăng cường theo cặp | `x = λ·x1 + (1-λ)·x2; y = λ·y1 + (1-λ)·y2`. |
| SpecAugment | Tăng cường dựa trên che | Loại bỏ các dải thời gian và tần số ngẫu nhiên của phổ âm. |
| mAP | Chỉ số đa nhãn chính | Mean average precision trên các lớp và ngưỡng. |

## Đọc thêm

- [Gong, Chung, Glass (2021). AST: Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778) — kiến trúc tiêu chuẩn từ 2021–2024.
- [Chen et al. (2022, rev. 2024). BEATs: Audio Pre-Training with Acoustic Tokenizers](https://arxiv.org/abs/2212.09058) — mặc định từ 2024+.
- [Park et al. (2019). SpecAugment](https://arxiv.org/abs/1904.08779) — phương pháp tăng cường âm thanh thống trị.
- [Piczak (2015). ESC-50 dataset](https://github.com/karolpiczak/ESC-50) — benchmark 50 lớp vẫn còn phổ biến.
- [Gemmeke et al. (2017). AudioSet](https://research.google.com/audioset/) — phân loại YouTube 632 lớp; vẫn là tiêu chuẩn vàng.