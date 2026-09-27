# Chống giả mạo giọng nói & Đóng dấu âm thanh (Audio Watermarking) — ASVspoof 5, AudioSeal, WaveVerify

> Công nghệ nhân bản giọng nói (voice cloning) đã phát triển nhanh hơn các biện pháp phòng thủ. Các hệ thống giọng nói thương mại năm 2026 cần hai yếu tố: một bộ phát hiện (AASIST, RawNet2) để phân loại giọng nói thật/giả, và một dấu chìm (AudioSeal) có khả năng tồn tại sau khi nén và chỉnh sửa. Hãy triển khai cả hai hoặc đừng triển khai công nghệ nhân bản giọng nói.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 06 (Speaker Recognition), Phase 6 · 08 (Voice Cloning)
**Time:** ~75 phút

## Vấn đề

Ba biện pháp phòng thủ liên quan:

1. **Chống giả mạo / Phát hiện deepfake.** Với một đoạn âm thanh, đó là giọng tổng hợp hay giọng thật? Các bộ benchmark ASVspoof (ASVspoof 2019 → 2021 → 5) là tiêu chuẩn vàng.
2. **Đóng dấu âm thanh (Audio watermarking).** Nhúng một tín hiệu không thể nhận biết vào âm thanh được tạo ra để bộ phát hiện có thể trích xuất sau đó. AudioSeal (Meta) và WavMark là các lựa chọn mã nguồn mở.
3. **Xác thực nguồn gốc (Authenticated provenance).** Ký mã hóa các tệp âm thanh + metadata. C2PA / Content Authenticity Initiative.

Phát hiện xử lý các đối thủ không hợp tác. Đóng dấu xử lý vấn đề tuân thủ — âm thanh do AI tạo ra phải được nhận diện là do AI. Cả hai đều là yêu cầu bắt buộc trong năm 2026.

## Khái niệm

![Anti-spoofing vs watermarking vs provenance — three defense layers](../assets/spoofing-watermark.svg)

### ASVspoof 5 — benchmark giai đoạn 2024-2025

Thay đổi lớn nhất so với các phiên bản trước:

- **Dữ liệu crowdsourced** (không phải dữ liệu sạch trong studio) — điều kiện thực tế.
- **~2000 người nói** (so với ~100 trước đây).
- **32 thuật toán tấn công.** TTS + chuyển đổi giọng nói + nhiễu đối nghịch (adversarial perturbation).
- **Hai track.** Countermeasure (CM) phát hiện độc lập; Spoofing-robust ASV (SASV) cho các hệ thống sinh trắc học.

Trạng thái SOTA trên ASVspoof 5: ~7.23% EER. Trên ASVspoof 2019 LA cũ: 0.42% EER. Triển khai thực tế: kỳ vọng 5-10% EER trên các đoạn âm thanh ngoài tự nhiên.

### AASIST và RawNet2 — các họ mô hình phát hiện

**AASIST** (2021, cập nhật đến 2026). Graph-attention trên các đặc trưng phổ. SOTA hiện tại cho tác vụ countermeasure của ASVspoof 5.

**RawNet2.** Convolutional front-end trên dạng sóng thô + backbone TDNN. Baseline đơn giản hơn; vẫn cạnh tranh tốt khi fine-tuning.

**NeXt-TDNN + SSL features.** Biến thể 2025: kiểu ECAPA + đặc trưng WavLM + focal loss. Đạt 0.42% EER trên ASVspoof 2019 LA.

### AudioSeal — tiêu chuẩn watermark năm 2024

**AudioSeal** của Meta (tháng 1/2024, v0.2 tháng 12/2024). Thiết kế chính:

- **Cục bộ (Localized).** Phát hiện watermark theo từng khung hình ở độ phân giải mẫu 16 kHz (1/16000 giây).
- **Generator + detector được huấn luyện chung.** Generator học cách nhúng tín hiệu không nghe thấy được; detector học cách tìm ra nó thông qua các phép tăng cường (augmentations).
- **Bền vững.** Tồn tại sau khi nén MP3 / AAC, EQ, thay đổi tốc độ ±10%, trộn nhiễu +10 dB SNR.
- **Nhanh.** Detector chạy nhanh gấp 485 lần thời gian thực; nhanh hơn 1000 lần so với WavMark.
- **Dung lượng.** Payload 16-bit (có thể mã hóa ID mô hình, dấu thời gian tạo, ID người dùng) có thể nhúng vào mỗi câu nói.

### WavMark

Baseline mã nguồn mở trước AudioSeal. Mạng thần kinh nghịch đảo, 32 bit/giây. Các vấn đề:

- Brute-force đồng bộ hóa chậm.
- Có thể bị loại bỏ bởi nhiễu Gaussian hoặc nén MP3.
- Không thân thiện với thời gian thực.

### WaveVerify (tháng 7/2025)

Giải quyết các điểm yếu của AudioSeal — cụ thể là các thao tác theo thời gian (đảo ngược, tốc độ). Sử dụng generator dựa trên FiLM + detector Mixture-of-Experts. Cạnh tranh với AudioSeal trên các cuộc tấn công tiêu chuẩn; xử lý tốt các chỉnh sửa thời gian.

### Khoảng trống mà đối thủ khai thác

Từ AudioMarkBench: "dưới tác động của pitch shift, tất cả các watermark đều cho thấy Bit Recovery Accuracy dưới 0.6, cho thấy việc loại bỏ gần như hoàn toàn." **Pitch-shift là cuộc tấn công phổ quát.** Không có watermark nào năm 2026 hoàn toàn bền vững trước việc thay đổi cao độ mạnh. Đây là lý do tại sao bạn cần phát hiện (AASIST) song song với đóng dấu.

### C2PA / Content Authenticity Initiative

Không phải kỹ thuật ML — mà là định dạng manifest. Các tệp âm thanh mang metadata được ký mã hóa về công cụ tạo, tác giả, ngày tháng. Audobox / Seamless sử dụng nó. Tốt cho nguồn gốc; không có tác dụng nếu kẻ xấu mã hóa lại và loại bỏ metadata.

```figure
v4-audio-watermark
```

## Xây dựng

### Bước 1: bộ phát hiện đặc trưng phổ đơn giản (toy)

```python
def spectral_rolloff(spec, percentile=0.85):
    cum = 0
    total = sum(spec)
    if total == 0:
        return 0
    threshold = total * percentile
    for k, v in enumerate(spec):
        cum += v
        if cum >= threshold:
            return k
    return len(spec) - 1

def is_suspicious(audio):
    spec = magnitude_spectrum(audio)
    rolloff = spectral_rolloff(spec)
    return rolloff / len(spec) > 0.92
```

Giọng nói tổng hợp thường có năng lượng tần số cao phẳng bất thường. Các bộ phát hiện thương mại sử dụng AASIST, không phải cái này. Nhưng trực giác vẫn đúng.

### Bước 2: Nhúng + phát hiện AudioSeal

```python
from audioseal import AudioSeal
import torch

generator = AudioSeal.load_generator("audioseal_wm_16bits")
detector = AudioSeal.load_detector("audioseal_detector_16bits")

audio = load_wav("generated.wav", sr=16000)[None, None, :]
payload = torch.tensor([[1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0]])
watermark = generator.get_watermark(audio, sample_rate=16000, message=payload)
watermarked = audio + watermark

result, decoded_payload = detector.detect_watermark(watermarked, sample_rate=16000)
# result: float in [0, 1] — probability of watermark presence
# decoded_payload: 16 bits; match against embedded payload
```

### Bước 3: đánh giá — EER

```python
def eer(real_scores, fake_scores):
    thresholds = sorted(set(real_scores + fake_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in fake_scores if s >= t) / len(fake_scores)
        frr = sum(1 for s in real_scores if s < t) / len(real_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

### Bước 4: tích hợp thương mại

```python
def safe_tts(text, voice, clone_reference=None):
    if clone_reference is not None:
        verify_consent(user_id, clone_reference)
    audio = tts_model.synthesize(text, voice)
    audio_with_wm = audioseal_embed(audio, payload=build_payload(user_id, model_id))
    manifest = c2pa_sign(audio_with_wm, user_id, timestamp=now())
    return audio_with_wm, manifest
```

Mỗi lần tạo đều xuất ra: (1) watermark, (2) manifest đã ký, (3) nhật ký kiểm toán tuân thủ chính sách lưu trữ.

## Sử dụng

| Trường hợp sử dụng | Phòng thủ |
|----------|---------|
| Triển khai TTS / voice cloning | Nhúng AudioSeal trên mọi đầu ra (bắt buộc) |
| Mở khóa giọng nói sinh trắc học | AASIST + ECAPA ensemble; thử thách liveness |
| Phát hiện gian lận tổng đài | AASIST trên 20% mẫu cuộc gọi đến |
| Tính xác thực của Podcast | Ký C2PA khi tải lên, AudioSeal nếu do AI tạo |
| Nghiên cứu / huấn luyện detector | Các tập train/dev/eval của ASVspoof 5 |

## Cạm bẫy

- **Watermark mà không bao giờ chạy detector.** Vô nghĩa. Hãy tích hợp detector vào CI của bạn.
- **Phát hiện mà không hiệu chuẩn (calibration).** AASIST huấn luyện trên ASVspoof LA bị overfitting; độ chính xác thực tế giảm. Hãy hiệu chuẩn trên domain của bạn.
- **Khoảng trống pitch-shift.** Pitch shift mạnh loại bỏ hầu hết watermark. Hãy có phương án dự phòng phát hiện.
- **Loại bỏ và đăng lại metadata.** C2PA dễ dàng bị vượt qua bằng cách mã hóa lại. Luôn thêm cả phòng thủ mã hóa + phòng thủ nhận thức (watermark).
- **Liveness là phát hiện.** Yêu cầu người dùng nói một cụm từ ngẫu nhiên. Ngăn chặn các cuộc tấn công phát lại (replay) nhưng không ngăn được cloning thời gian thực.

## Triển khai

Lưu dưới dạng `outputs/skill-spoof-defender.md`. Chọn mô hình phát hiện, watermark, manifest nguồn gốc và playbook vận hành cho việc triển khai tạo giọng nói.

## Bài tập

1. **Dễ.** Chạy `code/main.py`. Bộ phát hiện toy + nhúng/phát hiện watermark toy trên âm thanh tổng hợp.
2. **Trung bình.** Cài đặt `audioseal`, nhúng payload 16-bit vào đầu ra TTS, giải mã lại. Làm hỏng âm thanh bằng nhiễu và đo Bit Recovery Accuracy.
3. **Khó.** Fine-tune RawNet2 hoặc AASIST trên ASVspoof 2019 LA. Đo EER. Kiểm tra trên tập các đoạn clip do F5-TTS tạo ra — xem độ chính xác phát hiện OOD suy giảm như thế nào.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| ASVspoof | Benchmark | Thử thách hai năm một lần; 2024 = ASVspoof 5. |
| CM (countermeasure) | Detector | Phân loại: giọng thật vs tổng hợp / chuyển đổi. |
| SASV | Speaker verif + CM | Xác thực người nói + phát hiện giả mạo tích hợp. |
| AudioSeal | Meta watermark | Cục bộ, payload 16-bit, nhanh gấp 485 lần WavMark. |
| Bit Recovery Accuracy | Độ bền watermark | Tỷ lệ các bit payload khôi phục được sau tấn công. |
| C2PA | Provenance manifest | Metadata mã hóa về việc tạo / quyền tác giả. |
| AASIST | Detector family | SOTA chống giả mạo dựa trên graph-attention. |

## Đọc thêm

- [Todisco et al. (2024). ASVspoof 5](https://dl.acm.org/doi/10.1016/j.csl.2025.101825) — benchmark hiện tại.
- [Defossez et al. (2024). AudioSeal](https://arxiv.org/abs/2401.17264) — watermark mặc định.
- [Chen et al. (2025). WaveVerify](https://arxiv.org/abs/2507.21150) — detector MoE cho các tấn công thời gian.
- [Jung et al. (2022). AASIST](https://arxiv.org/abs/2110.01200) — backbone phát hiện SOTA.
- [AudioMarkBench (2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/5d9b7775296a641a1913ab6b4425d5e8-Paper-Datasets_and_Benchmarks_Track.pdf) — đánh giá độ bền vững.
- [C2PA specification](https://c2pa.org/specifications/specifications/) — định dạng manifest nguồn gốc.