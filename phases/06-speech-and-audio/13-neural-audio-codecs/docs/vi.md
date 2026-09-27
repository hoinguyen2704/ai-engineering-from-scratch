# Neural Audio Codecs — EnCodec, SNAC, Mimi, DAC và sự phân tách Semantic-Acoustic

> Năm 2026, việc tạo âm thanh gần như hoàn toàn dựa trên các token. EnCodec, SNAC, Mimi và DAC chuyển đổi dạng sóng liên tục thành các chuỗi rời rạc mà một Transformer có thể dự đoán được. Sự phân tách token semantic-vs-acoustic — codebook đầu tiên là semantic, các codebook còn lại là acoustic — là bước chuyển dịch kiến trúc quan trọng nhất kể từ khi Transformer xuất hiện trong lĩnh vực âm thanh.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms), Phase 10 · 11 (Quantization), Phase 5 · 19 (Subword Tokenization)
**Time:** ~60 phút

## Vấn đề

Các mô hình ngôn ngữ hoạt động trên các token rời rạc. Âm thanh thì liên tục. Nếu bạn muốn xây dựng một mô hình kiểu LLM cho giọng nói / âm nhạc — MusicGen, Moshi, Sesame CSM, VibeVoice, Orpheus — trước tiên bạn cần một **neural audio codec**: một bộ mã hóa (encoder) đã qua huấn luyện giúp rời rạc hóa âm thanh thành một bộ từ vựng nhỏ gồm các token, và một bộ giải mã (decoder) tương ứng để tái tạo lại dạng sóng.

Hai nhóm codec đã xuất hiện:

1. **Codec ưu tiên tái tạo (Reconstruction-first)** — EnCodec, DAC. Tối ưu hóa chất lượng âm thanh cảm nhận. Các token mang tính "acoustic" — chúng nắm bắt mọi thứ bao gồm danh tính người nói, âm sắc, tiếng ồn nền.
2. **Codec ưu tiên ngữ nghĩa (Semantic-first)** — Mimi (Kyutai), SpeechTokenizer. Ép buộc codebook đầu tiên phải mã hóa nội dung ngôn ngữ / ngữ âm (thường bằng cách chưng cất từ WavLM). Các codebook tiếp theo chứa chi tiết về âm thanh (acoustic).

Góc nhìn giai đoạn 2024-2026: **một codec thuần tái tạo sẽ tạo ra giọng nói bị nhòe khi bạn cố gắng tạo từ văn bản.** LLM hoạt động trên các token của codec phải học cả cấu trúc ngôn ngữ VÀ cấu trúc âm thanh trong cùng một codebook, điều này không thể mở rộng (scale). Việc tách biệt chúng — codebook semantic ở vị trí 0, các codebook acoustic ở vị trí 1-N — chính là chìa khóa giúp Moshi và Sesame CSM hoạt động hiệu quả.

## Khái niệm

![Four codec landscape: EnCodec, DAC, SNAC (multi-scale), Mimi (semantic+acoustic)](../assets/codec-comparison.svg)

### Thủ thuật cốt lõi: Residual Vector Quantization (RVQ)

Thay vì dùng một codebook lớn (cần hàng triệu mã để đạt chất lượng tốt), tất cả các codec âm thanh hiện đại đều sử dụng **RVQ**: một chuỗi các codebook nhỏ. Codebook đầu tiên lượng tử hóa đầu ra của encoder; codebook thứ hai lượng tử hóa phần dư (residual); và cứ tiếp tục như vậy. Mỗi codebook có 1024 mã. 8 codebook = từ vựng hiệu dụng là 1024^8 = 10^24.

Tại thời điểm suy luận (inference), bộ giải mã cộng tất cả các mã đã chọn trên mỗi khung hình (frame) để tái tạo lại âm thanh.

### Bốn codec quan trọng trong năm 2026

**EnCodec (Meta, 2022).** Tiêu chuẩn cơ sở. Encoder-decoder trên dạng sóng, nút thắt cổ chai RVQ. 24 kHz, hỗ trợ tối đa 32 codebook, mặc định 4 codebook @ 1.5 kbps. Sử dụng kiến trúc `1D conv + transformer + 1D conv`. Được sử dụng bởi MusicGen.

**DAC (Descript, 2023).** RVQ với các codebook được chuẩn hóa L2, các hàm kích hoạt tuần hoàn, cải thiện hàm mất mát (loss). Độ trung thực tái tạo cao nhất trong số các codec mã nguồn mở — đôi khi không thể phân biệt được với giọng nói gốc với 12 codebook. 44.1 kHz full-band.

**SNAC (Hubert Siuzdak, 2024).** RVQ đa quy mô (multi-scale) — các codebook thô (coarse) hoạt động ở tốc độ khung hình thấp hơn các codebook chi tiết (fine). Mô hình hóa âm thanh theo phân cấp một cách hiệu quả: một "bản phác thảo" thô ở ~12 Hz cộng với chi tiết ở 50 Hz. Được sử dụng bởi Orpheus-3B vì cấu trúc phân cấp ánh xạ tốt với quá trình tạo dựa trên LM.

**Mimi (Kyutai, 2024).** Bước ngoặt của năm 2026. Tốc độ khung hình 12.5 Hz (cực thấp), 8 codebook @ 4.4 kbps. Codebook 0 được **chưng cất từ WavLM** — được huấn luyện để dự đoán các đặc trưng nội dung giọng nói của WavLM. Các codebook 1-7 là các phần dư âm thanh. Sự phân tách này là nền tảng cho Moshi (Bài 15) và Sesame CSM.

### Tốc độ khung hình quan trọng đối với mô hình ngôn ngữ

Tốc độ khung hình thấp hơn = chuỗi ngắn hơn = LM nhanh hơn.

| Codec | Tốc độ khung hình | 1 s = N khung hình | Phù hợp cho |
|-------|-----------|----------------|---------|
| EnCodec-24k | 75 Hz | 75 | âm nhạc, âm thanh chung |
| DAC-44.1k | 86 Hz | 86 | âm nhạc độ trung thực cao |
| SNAC-24k (coarse) | ~12 Hz | 12 | AR-LM hiệu quả |
| Mimi | 12.5 Hz | 12.5 | truyền phát giọng nói (streaming) |

Ở mức 12.5 Hz, một đoạn âm thanh 10 giây chỉ có 125 khung hình codec — một Transformer có thể dự đoán chúng một cách dễ dàng.

### Token Semantic vs Acoustic

```
frame_t → [semantic_token_t, acoustic_token_0_t, acoustic_token_1_t, ..., acoustic_token_6_t]
```

- **Semantic token (codebook 0 trong Mimi).** Mã hóa những gì được nói — âm vị, từ ngữ, nội dung. Được chưng cất từ WavLM thông qua một hàm mất mát dự đoán phụ trợ.
- **Acoustic tokens (codebook 1-7).** Mã hóa âm sắc, danh tính người nói, ngữ điệu, tiếng ồn nền, chi tiết tinh vi.

Một AR LM dự đoán token semantic trước (có điều kiện từ văn bản), sau đó dự đoán các token acoustic (có điều kiện từ semantic + tham chiếu người nói). Sự phân tách này là lý do tại sao TTS hiện đại có thể sao chép giọng nói zero-shot: mô hình semantic xử lý nội dung; mô hình acoustic xử lý âm sắc.

### Chất lượng tái tạo năm 2026 (bit trên giây, bitrate thấp hơn là tốt hơn)

| Codec | Bitrate | PESQ | ViSQOL |
|-------|---------|------|--------|
| Opus-20kbps | 20 kbps | 4.0 | 4.3 |
| EnCodec-6kbps | 6 kbps | 3.2 | 3.8 |
| DAC-6kbps | 6 kbps | 3.5 | 4.0 |
| SNAC-3kbps | 3 kbps | 3.3 | 3.8 |
| Mimi-4.4kbps | 4.4 kbps | 3.1 | 3.7 |

Các codec truyền thống như Opus vẫn thắng về chất lượng cảm nhận trên mỗi bit. Các neural codec thắng về **token rời rạc** (điều mà Opus không tạo ra) và **chất lượng mô hình tạo sinh** (những gì LM có thể làm với các token đó).

```figure
rvq-codec-cascade
```

## Xây dựng

### Bước 1: mã hóa với EnCodec

```python
from encodec import EncodecModel
import torch

model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)  # kbps

wav = torch.randn(1, 1, 24000)
with torch.no_grad():
    encoded = model.encode(wav)
codes, scale = encoded[0]
# codes: (1, n_codebooks, n_frames), dtype=int64
```

`n_codebooks=8` ở mức 6 kbps. Mỗi mã là 0-1023 (10-bit).

### Bước 2: giải mã và đo lường sự tái tạo

```python
with torch.no_grad():
    wav_recon = model.decode([(codes, scale)])

from torchaudio.functional import compute_deltas
import torch.nn.functional as F

mse = F.mse_loss(wav_recon[:, :, :wav.shape[-1]], wav).item()
```

### Bước 3: sự phân tách semantic-acoustic (kiểu Mimi)

```python
from moshi.models import loaders
mimi = loaders.get_mimi()

with torch.no_grad():
    codes = mimi.encode(wav)  # shape (1, 8, frames@12.5Hz)

semantic = codes[:, 0]
acoustic = codes[:, 1:]
```

Codebook semantic 0 được căn chỉnh theo WavLM. Bạn có thể huấn luyện một Transformer text-to-semantic — từ vựng nhỏ hơn nhiều so với việc đi thẳng đến âm thanh. Sau đó, một bộ giải mã acoustic-to-waveform riêng biệt sẽ điều kiện hóa trên một tham chiếu người nói.

### Bước 4: tại sao AR LM trên các token codec lại hiệu quả

Đối với một đoạn giọng nói 10 giây ở tốc độ 12.5 Hz × 8 codebook của Mimi:

```
N_tokens = 10 * 12.5 * 8 = 1000 tokens
```

1000 token là một ngữ cảnh tầm thường đối với một Transformer. Một Transformer 256M tham số có thể tạo ra 10 giây giọng nói trong vài mili giây trên một GPU hiện đại.

## Sử dụng

Ánh xạ bài toán → codec:

| Tác vụ | Codec |
|------|-------|
| Tạo âm nhạc chung | EnCodec-24k |
| Tái tạo độ trung thực cao nhất | DAC-44.1k |
| AR LM trên giọng nói (TTS) | SNAC hoặc Mimi |
| Truyền phát giọng nói full-duplex | Mimi (12.5 Hz) |
| Thư viện hiệu ứng âm thanh với văn bản | EnCodec + điều kiện T5 |
| Chỉnh sửa âm thanh chi tiết | DAC + inpainting |

Quy tắc ngón tay cái: **nếu bạn đang xây dựng một mô hình tạo sinh, hãy bắt đầu với Mimi hoặc SNAC. Nếu bạn đang xây dựng một pipeline nén, hãy sử dụng Opus.**

## Các cạm bẫy

- **Quá nhiều codebook.** Thêm codebook làm tăng độ trung thực một cách tuyến tính nhưng cũng làm tăng độ dài chuỗi LM một cách tuyến tính. Hãy dừng lại ở 8-12.
- **Sai lệch tốc độ khung hình.** Huấn luyện LM trên Mimi 12.5 Hz sau đó tinh chỉnh (fine-tune) trên EnCodec 50 Hz sẽ thất bại mà không có thông báo lỗi rõ ràng.
- **Giả định tất cả codebook đều như nhau.** Trong Mimi, codebook 0 mang nội dung; mất nó sẽ phá hủy khả năng hiểu. Mất codebook 7 hầu như không đáng kể.
- **Chỉ sử dụng chất lượng tái tạo làm thước đo duy nhất.** Một codec có thể tái tạo tuyệt vời nhưng vô dụng cho việc tạo sinh dựa trên LM nếu cấu trúc ngữ nghĩa kém.

## Triển khai

Lưu dưới dạng `outputs/skill-codec-picker.md`. Chọn một codec cho một tác vụ tạo sinh hoặc nén cụ thể.

## Bài tập

1. **Dễ.** Chạy `code/main.py`. Nó triển khai một bộ lượng tử hóa vô hướng + phần dư đơn giản và đo lường lỗi tái tạo khi bạn thêm các codebook.
2. **Trung bình.** Cài đặt `encodec` và so sánh 1, 4, 8, 32 codebook trên một đoạn giọng nói kiểm thử. Vẽ biểu đồ PESQ hoặc MSE so với bitrate.
3. **Khó.** Tải Mimi. Mã hóa một đoạn clip. Thay thế codebook 0 bằng các số nguyên ngẫu nhiên; giải mã. Sau đó thay thế codebook 7 tương tự. So sánh hai sự hư hỏng này — sự hư hỏng ở codebook 0 sẽ phá hủy khả năng hiểu; sự hư hỏng ở codebook 7 hầu như không làm thay đổi gì.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| RVQ | Lượng tử hóa phần dư | Chuỗi các codebook nhỏ; mỗi cái lượng tử hóa phần dư của cái trước. |
| Tốc độ khung hình | Tốc độ codec | Số lượng token-frame mỗi giây. Thấp hơn = LM nhanh hơn. |
| Semantic codebook | Codebook 0 (Mimi) | Codebook được chưng cất từ các đặc trưng SSL; mã hóa nội dung. |
| Acoustic codebooks | Các codebook còn lại | Âm sắc, ngữ điệu, tiếng ồn, chi tiết tinh vi. |
| PESQ / ViSQOL | Chất lượng cảm nhận | Các chỉ số khách quan tương quan với MOS. |
| EnCodec | Codec của Meta | Tiêu chuẩn cơ sở RVQ; được MusicGen sử dụng. |
| Mimi | Codec của Kyutai | Tốc độ khung hình 12.5 Hz; phân tách semantic-acoustic; nền tảng của Moshi. |

## Đọc thêm

- [Défossez et al. (2023). EnCodec](https://arxiv.org/abs/2210.13438) — tiêu chuẩn cơ sở RVQ.
- [Kumar et al. (2023). Descript Audio Codec (DAC)](https://arxiv.org/abs/2306.06546) — codec mã nguồn mở có độ trung thực cao nhất.
- [Siuzdak (2024). SNAC](https://arxiv.org/abs/2410.14411) — RVQ đa quy mô.
- [Kyutai (2024). Mimi codec](https://kyutai.org/codec-explainer) — phân tách semantic-acoustic, chưng cất WavLM.
- [Borsos et al. (2023). AudioLM](https://arxiv.org/abs/2209.03143) — mô hình hai giai đoạn semantic/acoustic.
- [Zeghidour et al. (2021). SoundStream](https://arxiv.org/abs/2107.03312) — codec RVQ có thể truyền phát (streamable) nguyên bản.