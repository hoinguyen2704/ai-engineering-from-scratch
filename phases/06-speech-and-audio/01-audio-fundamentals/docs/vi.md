# Audio Fundamentals — Waveforms, Sampling, Fourier Transform

> Waveforms là tín hiệu thô. Spectrograms là biểu diễn. Mel features là dạng dữ liệu thân thiện với ML. Mọi pipeline ASR và TTS hiện đại đều đi theo lộ trình này, và bước đầu tiên là hiểu về sampling và Fourier.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 1 · 06 (Vectors & Matrices), Phase 1 · 14 (Probability Distributions)
**Time:** ~45 phút

## Vấn đề

Microphone tạo ra tín hiệu áp suất theo thời gian. Neural net của bạn tiêu thụ các tensor. Giữa chúng là một tập hợp các quy ước mà nếu vi phạm, sẽ tạo ra các lỗi "im lặng": mô hình huấn luyện bình thường nhưng WER tăng gấp đôi, hoặc TTS tạo ra tiếng rít, hoặc hệ thống voice cloning ghi nhớ đặc điểm của microphone thay vì người nói.

Mọi lỗi trong hệ thống giọng nói đều bắt nguồn từ một trong ba câu hỏi:

1. Dữ liệu được ghi ở sample rate nào, và mô hình mong đợi gì?
2. Tín hiệu có bị aliased không?
3. Bạn đang thao tác trên các mẫu thô (raw samples) hay trên biểu diễn tần số?

Giải quyết đúng những điều này thì phần còn lại của Phase 6 sẽ trở nên dễ dàng. Nếu sai, ngay cả Whisper-Large-v4 cũng sẽ tạo ra kết quả rác.

## Khái niệm

![Waveform, sampling, DFT, and frequency bins visualized](../assets/audio-fundamentals.svg)

**Waveform.** Một mảng một chiều các số thực (floats) trong `[-1.0, 1.0]`. Được đánh chỉ số theo số thứ tự mẫu. Để chuyển đổi sang giây, chia cho sample rate: `t = n / sr`. Một đoạn clip 10 giây ở 16 kHz là một mảng gồm 160.000 số thực.

**Sampling rate (sr).** Số lượng mẫu trên mỗi giây. Các tốc độ phổ biến trong năm 2026:

| Tốc độ | Sử dụng |
|------|-----|
| 8 kHz | Điện thoại, VOIP cũ. Nyquist ở 4 kHz làm mất các phụ âm. Tránh dùng cho ASR. |
| 16 kHz | Tiêu chuẩn ASR. Whisper, Parakeet, SeamlessM4T v2 đều tiêu thụ 16 kHz. |
| 22.05 kHz | Huấn luyện TTS vocoder cho các mô hình cũ. |
| 24 kHz | TTS hiện đại (Kokoro, F5-TTS, xTTS v2). |
| 44.1 kHz | CD audio, âm nhạc. |
| 48 kHz | Phim, âm thanh chuyên nghiệp, TTS độ trung thực cao (VALL-E 2, NaturalSpeech 3). |

**Nyquist-Shannon.** Một sample rate là `sr` có thể biểu diễn chính xác các tần số lên đến `sr/2`. Ranh giới `sr/2` chính là *tần số Nyquist*. Năng lượng vượt quá Nyquist sẽ bị *aliased* — gập xuống các tần số thấp hơn — và làm hỏng tín hiệu. Luôn sử dụng bộ lọc low-pass trước khi downsampling.

**Bit depth.** 16-bit PCM (signed int16, phạm vi ±32,767) là định dạng trao đổi phổ quát. 24-bit cho âm nhạc, 32-bit float cho DSP nội bộ. Các thư viện như `soundfile` đọc int16 nhưng xuất ra các mảng float32 trong `[-1, 1]`.

**Fourier Transform.** Bất kỳ tín hiệu hữu hạn nào cũng là tổng của các sóng hình sin ở các tần số khác nhau. Discrete Fourier Transform (DFT) tính toán, cho `N` mẫu, `N` hệ số phức — mỗi hệ số cho một bin tần số. `bin k` ánh xạ tới tần số `k · sr / N` Hz. Độ lớn (magnitude) là biên độ tại tần số đó, góc (angle) là pha.

**FFT.** Fast Fourier Transform: một thuật toán `O(N log N)` cho DFT khi `N` là lũy thừa của 2. Mọi thư viện âm thanh đều sử dụng FFT bên dưới. Một FFT 1024 mẫu ở 16 kHz cho ra 512 bin tần số hữu dụng trải dài từ 0–8 kHz với độ phân giải 15.6 Hz.

**Framing + window.** Chúng ta không FFT toàn bộ clip. Chúng ta chia nó thành các *khung* (frames) chồng lấp (thường là 25 ms với bước nhảy 10 ms), nhân mỗi khung với một hàm cửa sổ (Hann, Hamming) để loại bỏ sự gián đoạn ở biên, sau đó FFT từng khung. Đây là Short-Time Fourier Transform (STFT). Bài 02 sẽ tiếp tục từ đây.

```figure
mel-scale
```

## Xây dựng

### Bước 1: đọc một clip và vẽ waveform

`code/main.py` chỉ sử dụng module `wave` tiêu chuẩn để giữ cho bản demo không phụ thuộc vào thư viện ngoài. Trong môi trường production, bạn sẽ sử dụng `soundfile` hoặc `torchaudio.load` (cả hai đều trả về các tuple `(waveform, sr)`):

```python
import soundfile as sf
waveform, sr = sf.read("clip.wav", dtype="float32")  # shape (T,), sr=int
```

### Bước 2: tổng hợp một sóng sin từ nguyên lý cơ bản

```python
import math

def sine(freq_hz, sr, seconds, amp=0.5):
    n = int(sr * seconds)
    return [amp * math.sin(2 * math.pi * freq_hz * i / sr) for i in range(n)]
```

Một sóng sin 440 Hz (nốt La chuẩn) ở 16 kHz trong 1 giây là 16,000 số thực. Ghi bằng `wave.open(..., "wb")` sử dụng mã hóa 16-bit PCM.

### Bước 3: tính toán DFT thủ công

```python
def dft(x):
    N = len(x)
    out = []
    for k in range(N):
        re = sum(x[n] * math.cos(-2 * math.pi * k * n / N) for n in range(N))
        im = sum(x[n] * math.sin(-2 * math.pi * k * n / N) for n in range(N))
        out.append((re, im))
    return out
```

`O(N²)` — ổn để `N=256` xác nhận tính đúng đắn, nhưng vô dụng cho âm thanh thực tế. Mã nguồn thực tế sẽ gọi `numpy.fft.rfft` hoặc `torch.fft.rfft`.

### Bước 4: tìm tần số chiếm ưu thế

Chỉ số đỉnh của độ lớn `k_star` ánh xạ tới tần số `k_star * sr / N`. Chạy mã này trên sóng sin 440 Hz sẽ trả về đỉnh tại bin `440 * N / sr`.

### Bước 5: minh họa aliasing

Lấy mẫu một sóng sin 7 kHz ở 10 kHz (Nyquist = 5 kHz). Tông 7 kHz nằm trên Nyquist và gập xuống `10 − 7 = 3 kHz`. Đỉnh FFT xuất hiện tại 3 kHz. Đây là ví dụ kinh điển về aliasing và là lý do tại sao mọi DAC/ADC đều đi kèm với bộ lọc low-pass "brick-wall".

## Sử dụng

Stack bạn sẽ thực sự triển khai trong năm 2026:

| Tác vụ | Thư viện | Tại sao |
|------|---------|-----|
| Đọc/ghi WAV/FLAC/OGG | `soundfile` (wrapper của libsndfile) | Nhanh nhất, ổn định, trả về float32. |
| Resample | `torchaudio.transforms.Resample` hoặc `librosa.resample` | Tích hợp sẵn chống aliasing chính xác. |
| STFT / Mel | `torchaudio` hoặc `librosa` | Thân thiện với GPU; hệ sinh thái PyTorch. |
| Streaming thời gian thực | `sounddevice` hoặc `pyaudio` | Các binding PortAudio đa nền tảng. |
| Kiểm tra file | `ffprobe` hoặc `soxi` | CLI, nhanh, báo cáo sr/channels/codec. |

Quy tắc quyết định: **khớp sample rate trước khi khớp bất cứ thứ gì khác**. Whisper mong đợi 16 kHz mono float32. Nếu đưa vào 44.1 kHz stereo, bạn sẽ nhận được kết quả rác trông giống như lỗi mô hình.

## Triển khai

Lưu dưới dạng `outputs/skill-audio-loader.md`. Kỹ năng này giúp bạn kiểm tra xem đầu vào âm thanh có khớp với mong đợi của mô hình hạ nguồn hay không và thực hiện resample chính xác khi không khớp.

## Bài tập

1. **Dễ.** Tổng hợp một bản mix 1 giây gồm 220 Hz + 440 Hz + 880 Hz ở 16 kHz. Chạy DFT. Xác nhận ba đỉnh tại các bin dự kiến.
2. **Trung bình.** Ghi một file WAV 3 giây giọng nói của bạn ở 48 kHz. Downsample xuống 16 kHz bằng `torchaudio.transforms.Resample` (có chống aliasing), sau đó xuống 16 kHz bằng cách decimation ngây thơ (lấy mỗi mẫu thứ ba). FFT cả hai. Aliasing xuất hiện ở đâu?
3. **Khó.** Xây dựng STFT từ đầu chỉ sử dụng `math` và DFT từ Bước 3. Kích thước khung 400, bước nhảy 160, cửa sổ Hann. Vẽ các độ lớn bằng `matplotlib.pyplot.imshow`. Đây chính là spectrogram của Bài 02.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| Sample rate | Số lượng mẫu mỗi giây | Tần số (Hz) mà ADC đo tín hiệu. |
| Nyquist | Tần số tối đa có thể biểu diễn | `sr/2`; năng lượng vượt quá nó sẽ bị alias ngược xuống. |
| Bit depth | Độ phân giải của mỗi mẫu | `int16` = 65,536 mức; `float32` = độ chính xác 24-bit trong `[-1, 1]`. |
| DFT | Fourier transform cho chuỗi | `N` mẫu → `N` hệ số tần số phức. |
| FFT | DFT nhanh | `O(N log N)` thuật toán yêu cầu `N` = lũy thừa của 2. |
| Bin | Cột tần số | `k · sr / N` Hz; độ phân giải = `sr / N`. |
| STFT | Spectrogram bên dưới | FFT có khung + cửa sổ theo thời gian. |
| Aliasing | Bóng ma tần số kỳ lạ | Năng lượng trên Nyquist phản chiếu xuống các bin thấp hơn. |

## Đọc thêm

- [Shannon (1949). Communication in the Presence of Noise](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) — bài báo nền tảng của định lý lấy mẫu.
- [Smith — The Scientist and Engineer's Guide to Digital Signal Processing](https://www.dspguide.com/ch8.htm) — sách giáo khoa DSP kinh điển, miễn phí.
- [librosa docs — audio primer](https://librosa.org/doc/latest/tutorial.html) — hướng dẫn thực hành với mã nguồn.
- [Heinrich Kuttruff — Room Acoustics (6th ed.)](https://www.routledge.com/Room-Acoustics/Kuttruff/p/book/9781482260434) — tài liệu tham khảo về lý do tại sao âm thanh thực tế không phải là sóng sin sạch.
- [Steve Eddins — FFT Interpretation notebook](https://blogs.mathworks.com/steve/2020/03/30/fft-spectrum-and-spectral-densities/) — trực quan hóa về bin tần số trong 10 phút.