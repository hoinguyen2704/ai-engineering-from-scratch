# Spectrograms, Mel Scale & Audio Features

> Các mạng thần kinh không xử lý tốt dạng sóng thô (raw waveforms). Chúng xử lý spectrogram. Chúng xử lý mel spectrogram thậm chí còn tốt hơn. Mọi hệ thống ASR, TTS và phân loại âm thanh vào năm 2026 đều thành bại dựa trên lựa chọn tiền xử lý đơn lẻ này.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 01 (Audio Fundamentals)
**Time:** ~45 phút

## Vấn đề

Hãy lấy một đoạn clip 10 giây ở tần số 16 kHz. Đó là 160.000 giá trị float, tất cả đều nằm trong `[-1, 1]`, hầu như không có mối tương quan nào với nhãn "chó sủa" hay "từ mèo". Dạng sóng thô chứa thông tin nhưng ở dạng mà mô hình không thể dễ dàng trích xuất. Hai âm vị giống hệt nhau được phát âm cách nhau 100 ms có các mẫu thô hoàn toàn khác biệt.

Một spectrogram giải quyết vấn đề này. Nó nén các chi tiết thời gian mà nhận thức con người bỏ qua (độ nhiễu micro giây) và bảo toàn cấu trúc mà nhận thức tập trung vào (tần số nào có năng lượng, trong các cửa sổ thời gian khoảng 10–25 ms).

Mel spectrogram tiến xa hơn. Con người cảm nhận cao độ theo thang logarit: 100 Hz so với 200 Hz nghe "cách xa nhau" giống như 1000 Hz so với 2000 Hz. Thang đo mel làm biến dạng trục tần số để khớp với điều này. Một spectrogram theo thang mel là đặc trưng quan trọng nhất trong ML về giọng nói từ năm 2010 đến 2026.

## Khái niệm

![Waveform to STFT to mel spectrogram to MFCC ladder](../assets/mel-features.svg)

**STFT (Short-Time Fourier Transform).** Chia dạng sóng thành các khung chồng lấp (thông thường: cửa sổ 25 ms, bước nhảy 10 ms = 400 mẫu / 160 mẫu ở 16 kHz). Nhân mỗi khung với một hàm cửa sổ (Hann là mặc định; Hamming có sự đánh đổi khác biệt đôi chút). Thực hiện FFT trên mỗi khung. Xếp chồng các phổ biên độ thành một ma trận có hình dạng `(n_frames, n_freq_bins)`. Đó chính là spectrogram của bạn.

**Log-magnitude.** Các biên độ thô trải dài 5-6 bậc độ lớn. Sử dụng `log(|X| + 1e-6)` hoặc `20 * log10(|X|)` để nén dải động. Mọi pipeline sản xuất đều sử dụng log-magnitude, không phải biên độ thô.

**Thang đo Mel.** Tần số `f` tính bằng Hz ánh xạ sang mel `m` theo công thức `m = 2595 * log10(1 + f / 700)`. Phép ánh xạ này gần như tuyến tính dưới 1 kHz và gần như logarit ở trên. 80 mel bin bao phủ 0–8 kHz là đầu vào tiêu chuẩn cho ASR.

**Mel filterbank.** Một tập hợp các bộ lọc hình tam giác được sắp xếp cách đều nhau trên thang mel. Mỗi bộ lọc là tổng có trọng số của các bin FFT liền kề. Nhân biên độ STFT với ma trận filterbank sẽ cho ra mel spectrogram trong một phép matmul.

**Log-mel spectrogram.** `log(mel_spec + 1e-10)`. Đầu vào của Whisper. Đầu vào của Parakeet. Đầu vào của SeamlessM4T. Frontend âm thanh phổ quát năm 2026.

**MFCCs.** Lấy log-mel spectrogram, áp dụng DCT (loại II), giữ lại 13 hệ số đầu tiên. Nó khử tương quan các đặc trưng và nén thêm nữa. Đây là đặc trưng thống trị cho đến khoảng năm 2015 khi các CNN/Transformer trên log-mel thô bắt đầu bắt kịp. Vẫn được sử dụng trong nhận dạng người nói (x-vectors, ECAPA).

**Đánh đổi độ phân giải.** FFT lớn hơn = độ phân giải tần số tốt hơn nhưng độ phân giải thời gian kém hơn. 25 ms / 10 ms là mặc định cho audio-ML; 50 ms / 12.5 ms cho âm nhạc; 5 ms / 2 ms cho phát hiện tín hiệu thoáng qua (tiếng trống, âm tắc).

```figure
spectrogram-window
```

## Xây dựng

### Bước 1: tạo khung cho dạng sóng

```python
def frame(signal, frame_len, hop):
    n = 1 + (len(signal) - frame_len) // hop
    return [signal[i * hop : i * hop + frame_len] for i in range(n)]
```

Một clip 10 giây 16 kHz với `frame_len=400, hop=160` tạo ra 998 khung.

### Bước 2: Cửa sổ Hann

```python
import math

def hann(N):
    return [0.5 * (1 - math.cos(2 * math.pi * n / (N - 1))) for n in range(N)]
```

Nhân theo từng phần tử trước khi thực hiện FFT. Loại bỏ hiện tượng rò rỉ phổ (spectral leakage) gây ra do việc cắt cụt tại các điểm cuối không bằng không.

### Bước 3: Biên độ STFT

```python
def stft_magnitude(signal, frame_len=400, hop=160):
    win = hann(frame_len)
    frames = frame(signal, frame_len, hop)
    return [magnitudes(dft([w * s for w, s in zip(win, f)])) for f in frames]
```

Sản xuất sử dụng `torch.stft` hoặc `librosa.stft` (hỗ trợ FFT, được vector hóa). Vòng lặp ở đây mang tính sư phạm; nó chạy trên các clip ngắn trong `code/main.py`.

### Bước 4: Mel filterbank

```python
def hz_to_mel(f):
    return 2595.0 * math.log10(1.0 + f / 700.0)

def mel_to_hz(m):
    return 700.0 * (10 ** (m / 2595.0) - 1)

def mel_filterbank(n_mels, n_fft, sr, fmin=0, fmax=None):
    fmax = fmax or sr / 2
    mels = [hz_to_mel(fmin) + (hz_to_mel(fmax) - hz_to_mel(fmin)) * i / (n_mels + 1)
            for i in range(n_mels + 2)]
    hzs = [mel_to_hz(m) for m in mels]
    bins = [int(h * n_fft / sr) for h in hzs]
    fb = [[0.0] * (n_fft // 2 + 1) for _ in range(n_mels)]
    for m in range(n_mels):
        for k in range(bins[m], bins[m + 1]):
            fb[m][k] = (k - bins[m]) / max(1, bins[m + 1] - bins[m])
        for k in range(bins[m + 1], bins[m + 2]):
            fb[m][k] = (bins[m + 2] - k) / max(1, bins[m + 2] - bins[m + 1])
    return fb
```

80 mel bao phủ 0–8 kHz với `n_fft=400` tạo ra ma trận `(80, 201)`. Nhân biên độ STFT `(n_frames, 201)` với ma trận chuyển vị để có mel spectrogram `(n_frames, 80)`.

### Bước 5: Log-mel

```python
def log_mel(mel_spec, eps=1e-10):
    return [[math.log(max(v, eps)) for v in frame] for frame in mel_spec]
```

Các lựa chọn thay thế phổ biến: `librosa.power_to_db` (dB chuẩn hóa theo tham chiếu), `10 * log10(power + eps)`. Whisper sử dụng quy trình clip + chuẩn hóa phức tạp hơn (xem `log_mel_spectrogram` của Whisper).

### Bước 6: MFCCs

```python
def dct_ii(x, n_coeffs):
    N = len(x)
    return [
        sum(x[n] * math.cos(math.pi * k * (2 * n + 1) / (2 * N)) for n in range(N))
        for k in range(n_coeffs)
    ]
```

Áp dụng DCT cho mỗi khung log-mel, giữ lại 13 hệ số đầu tiên. Đó là ma trận MFCC của bạn. Hệ số đầu tiên thường bị loại bỏ (nó mã hóa năng lượng tổng thể).

## Sử dụng

Stack công nghệ năm 2026:

| Tác vụ | Đặc trưng |
|------|----------|
| ASR (Whisper, Parakeet, SeamlessM4T) | 80 log-mels, bước nhảy 10 ms, cửa sổ 25 ms |
| Mô hình âm thanh TTS (VITS, F5-TTS, Kokoro) | 80 mels, bước nhảy 5–12 ms để kiểm soát thời gian tinh vi |
| Phân loại âm thanh (AST, PANNs, BEATs) | 128 log-mels, bước nhảy 10 ms |
| Nhúng người nói (ECAPA-TDNN, WavLM) | 80 log-mels hoặc SSL dạng sóng thô |
| Âm nhạc (MusicGen, Stable Audio 2) | EnCodec discrete tokens (không phải mels) |
| Phát hiện từ khóa | 40 MFCCs cho các thiết bị nhỏ |

Quy tắc ngón tay cái: **nếu bạn không làm việc với âm nhạc, hãy bắt đầu với 80 log-mels.** Gánh nặng chứng minh thuộc về bất kỳ sự sai lệch nào.

## Các cạm bẫy vẫn tồn tại vào năm 2026

- **Sai lệch số lượng mel.** Huấn luyện với 80 mel, suy luận với 128 mel. Lỗi im lặng. Hãy ghi log hình dạng đặc trưng ở cả hai đầu.
- **Sai lệch tốc độ lấy mẫu (sample-rate) ở thượng nguồn.** Mels tính toán ở 22.05 kHz trông khác với 16 kHz. Hãy sửa SR *trước khi* trích xuất đặc trưng.
- **dB so với log.** Whisper mong đợi log-mel, không phải dB-mel. Một số pipeline HF tự động phát hiện; mã tùy chỉnh của bạn thì không.
- **Trôi chuẩn hóa (Normalization drift).** Chuẩn hóa theo từng đoạn trong khi huấn luyện, chuẩn hóa toàn cục trong khi suy luận. Lỗi sản xuất làm tăng gấp đôi WER.
- **Rò rỉ từ padding.** Padding bằng 0 ở cuối clip tạo ra một phổ phẳng ở các khung cuối. Hãy pad đối xứng hoặc sao chép.

## Triển khai

Lưu dưới dạng `outputs/skill-feature-extractor.md`. Kỹ năng này chọn loại đặc trưng, số lượng mel, khung/bước nhảy và chuẩn hóa cho một mục tiêu mô hình nhất định.

## Bài tập

1. **Dễ.** Chạy `code/main.py`. Nó tổng hợp một tiếng chirp (tần số quét 200 → 4000 Hz) và in ra argmax mel bin cho mỗi khung. Vẽ biểu đồ (tùy chọn) và xác nhận nó khớp với quá trình quét.
2. **Trung bình.** Chạy lại với `n_mels` trong `{40, 80, 128}` và `frame_len` trong `{200, 400, 800}`. Đo băng thông đỉnh nhọn trên trục thời gian. Sự kết hợp nào giải quyết tiếng chirp tốt nhất?
3. **Khó.** Triển khai `power_to_db` và so sánh độ chính xác ASR của một bộ phân loại CNN nhỏ trên AudioMNIST sử dụng (a) log-mel thô, (b) dB-mel với `ref=max`, (c) MFCC-13 + delta + delta-delta. Báo cáo độ chính xác top-1.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| Frame | Một lát cắt | Đoạn dạng sóng 25 ms được đưa vào một FFT. |
| Hop | Bước nhảy | Số mẫu giữa các khung liên tiếp; 10 ms là mặc định cho ASR. |
| Window | Thứ Hann/Hamming | Bộ nhân theo điểm làm giảm dần các cạnh khung về 0. |
| STFT | Bộ tạo Spectrogram | FFT đã đóng khung + cửa sổ; tạo ra ma trận thời gian × tần số. |
| Mel | Tần số biến dạng | Thang đo nhận thức logarit; `m = 2595·log10(1 + f/700)`. |
| Filterbank | Ma trận | Các bộ lọc tam giác chiếu STFT lên các mel bin. |
| Log-mel | Đầu vào của Whisper | `log(mel_spec + eps)`; tiêu chuẩn hóa vào năm 2026. |
| MFCC | Đặc trưng kiểu cũ | DCT của log-mel; 13 hệ số, đã khử tương quan. |

## Đọc thêm

- [Davis, Mermelstein (1980). Comparison of parametric representations for monosyllabic word recognition](https://ieeexplore.ieee.org/document/1163420) — bài báo về MFCC.
- [Stevens, Volkmann, Newman (1937). A Scale for the Measurement of the Psychological Magnitude Pitch](https://pubs.aip.org/asa/jasa/article-abstract/8/3/185/735757/) — thang đo mel gốc.
- [OpenAI — Whisper source, log_mel_spectrogram](https://github.com/openai/whisper/blob/main/whisper/audio.py) — đọc triển khai tham chiếu.
- [librosa feature extraction docs](https://librosa.org/doc/main/feature.html) — tham chiếu cho `mfcc`, `melspectrogram`, và hop/window.
- [NVIDIA NeMo — audio preprocessing](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/asr_all.html#featurizers) — pipeline quy mô sản xuất cho các mô hình Parakeet + Canary.