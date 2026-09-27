# Music Generation — MusicGen, Stable Audio, Suno và Cú sốc bản quyền

> Music generation năm 2026: Suno v5 và Udio v4 thống trị phân khúc thương mại; MusicGen, Stable Audio Open và ACE-Step dẫn đầu mảng mã nguồn mở. Vấn đề kỹ thuật về cơ bản đã được giải quyết. Vấn đề pháp lý (thỏa thuận 500 triệu USD của Warner Music, thỏa thuận của UMG) đã định hình lại lĩnh vực này trong giai đoạn 2025-2026.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms), Phase 4 · 10 (Diffusion Models)
**Time:** ~75 phút

## Vấn đề

Text → một đoạn nhạc từ 30 giây đến 4 phút, bao gồm lời bài hát, giọng hát và cấu trúc. Có ba bài toán con:

1. **Tạo nhạc không lời (Instrumental).** Text như "lo-fi hip-hop drums with warm keys" → audio. MusicGen, Stable Audio, AudioLDM.
2. **Tạo bài hát (có giọng hát + lời).** "Country song about rainy Texas nights" → bài hát hoàn chỉnh. Suno, Udio, YuE, ACE-Step.
3. **Có điều kiện / Có thể kiểm soát.** Kéo dài một đoạn clip có sẵn, tạo lại đoạn bridge, đổi thể loại, tách stem, hoặc inpaint. Tính năng inpainting + tách stem của Udio là tiêu chuẩn cần đạt tới vào năm 2026.

## Khái niệm

![Music generation: token-LM vs diffusion, the 2026 model map](../assets/music-generation.svg)

### Token LM trên các neural-codec tokens

**MusicGen** của Meta (2023, MIT) và nhiều biến thể: điều kiện hóa trên các text/melody embeddings, dự đoán tự hồi quy (autoregressively) các EnCodec tokens (32 kHz, 4 codebooks), sau đó giải mã bằng EnCodec. Tham số từ 300M - 3.3B. Đây là baseline mạnh mẽ; tuy nhiên gặp khó khăn với các đoạn dài hơn 30 giây.

**ACE-Step** (mã nguồn mở, bản 4B XL phát hành tháng 4/2026) mở rộng mô hình này để tạo bài hát hoàn chỉnh có điều kiện về lời. Đây là giải pháp gần nhất với Suno trong cộng đồng mã nguồn mở.

### Diffusion trên mels hoặc latents

**Stable Audio (2023)** và **Stable Audio Open (2024)**: latent diffusion trên âm thanh nén. Rất mạnh trong việc tạo các vòng lặp (loops), thiết kế âm thanh, và các kết cấu ambient. Không thực sự tốt với các bài hát có cấu trúc phức tạp.

**AudioLDM / AudioLDM2**: text-to-audio thông qua latent diffusion kiểu T2I, được tổng quát hóa cho âm nhạc, hiệu ứng âm thanh và giọng nói.

### Hybrid (sản xuất) — Suno, Udio, Lyria

Trọng số đóng. Nhiều khả năng là sự kết hợp giữa AR codec LM + diffusion-based vocoder với các head chuyên biệt cho giọng hát / trống / giai điệu. Suno v5 (2026) là người dẫn đầu về chất lượng với ELO 1293. Udio v4 bổ sung tính năng inpainting + tách stem (tải xuống riêng biệt bass, trống, giọng hát).

### Đánh giá

- **FAD (Fréchet Audio Distance).** Khoảng cách ở mức embedding giữa phân phối âm thanh được tạo ra và âm thanh thực tế, sử dụng các đặc trưng VGGish hoặc PANNs. Chỉ số càng thấp càng tốt. MusicGen bản nhỏ: 4.5 FAD trên MusicCaps; SOTA khoảng ~3.0.
- **Tính nhạc (Musicality - chủ quan).** Sự ưu tiên của con người. Suno v5 với ELO 1293 đang dẫn đầu.
- **Độ khớp Text-audio.** Điểm CLAP giữa prompt và output.
- **Các lỗi về tính nhạc.** Chuyển đoạn lệch nhịp, trôi cụm từ giọng hát, mất cấu trúc sau 30 giây.

## Bản đồ mô hình 2026

| Model | Params | Độ dài | Giọng hát | License |
|-------|--------|--------|--------|---------|
| MusicGen-large | 3.3B | 30 s | không | MIT |
| Stable Audio Open | 1.2B | 47 s | không | Stability non-commercial |
| ACE-Step XL (Tháng 4/2026) | 4B | &gt; 2 phút | có | Apache-2.0 |
| YuE | 7B | &gt; 2 phút | có, đa ngôn ngữ | Apache-2.0 |
| Suno v5 (đóng) | ? | 4 phút | có, ELO 1293 | thương mại |
| Udio v4 (đóng) | ? | 4 phút | có + stems | thương mại |
| Google Lyria 3 (đóng) | ? | thời gian thực | có | thương mại |
| MiniMax Music 2.5 | ? | 4 phút | có | API thương mại |

## Bối cảnh pháp lý (2025-2026)

- **Thỏa thuận Warner Music vs Suno.** 500 triệu USD. WMG hiện có quyền giám sát các nội dung AI mô phỏng, bản quyền âm nhạc và các bản nhạc do người dùng tạo trên Suno. Thỏa thuận tương tự với UMG trên Udio.
- **EU AI Act** + **California SB 942**: Âm nhạc do AI tạo ra phải được công khai/gắn nhãn.
- **Riffusion / MusicGen** theo giấy phép MIT không gặp rào cản tuân thủ nhưng cũng không có giọng hát thương mại.

Các mô hình triển khai an toàn:

1. Chỉ tạo nhạc không lời (MusicGen, Stable Audio Open, các output MIT/CC0).
2. Sử dụng các API thương mại (Suno, Udio, ElevenLabs Music) với giấy phép theo lượt tạo.
3. Huấn luyện trên danh mục sở hữu hoặc đã được cấp phép (hầu hết các doanh nghiệp đều chọn cách này).
4. Gắn watermark + metadata vào các bản nhạc được tạo.

```figure
sp-codec-tokens
```

## Xây dựng

### Bước 1: tạo nhạc với MusicGen

```python
from audiocraft.models import MusicGen
import torchaudio

model = MusicGen.get_pretrained("facebook/musicgen-small")
model.set_generation_params(duration=10)
wav = model.generate(["upbeat synthwave with driving drums, 128 BPM"])
torchaudio.save("out.wav", wav[0].cpu(), 32000)
```

Ba kích thước: `small` (300M, nhanh), `medium` (1.5B), `large` (3.3B). Bản nhỏ là đủ để kiểm tra xem ý tưởng có khả thi hay không.

### Bước 2: điều kiện hóa giai điệu (melody conditioning)

```python
melody, sr = torchaudio.load("humming.wav")
wav = model.generate_with_chroma(
    ["jazz piano cover"],
    melody.squeeze(),
    sr,
)
```

MusicGen-melody nhận đầu vào là chromagram và giữ nguyên giai điệu trong khi thay đổi âm sắc (timbre). Hữu ích cho việc "chuyển giai điệu này thành bản phối tứ tấu đàn dây".

### Bước 3: đánh giá FAD

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()

fad.get_fad_score("generated_folder/", "reference_folder/")
```

Tính toán khoảng cách VGGish-embedding. Hữu ích cho các bài kiểm tra hồi quy ở cấp độ thể loại; không thay thế được việc nghe bằng tai người.

### Bước 4: tích hợp vào quy trình LLM-music

Kết hợp với các ý tưởng từ Bài 7-8:

```python
prompt = "Write a 30-second jazz loop. Describe the drums, bass, and piano voicing."
description = llm.complete(prompt)
music = musicgen.generate([description], duration=30)
```

## Sử dụng

| Mục tiêu | Stack |
|------|-------|
| Thiết kế âm thanh không lời | Stable Audio Open |
| Game / nhạc thích ứng | Google Lyria RealTime (đóng) |
| Bài hát hoàn chỉnh có lời (thương mại) | Suno v5 hoặc Udio v4 với giấy phép rõ ràng |
| Bài hát hoàn chỉnh có lời (mở) | ACE-Step XL hoặc YuE |
| Nhạc quảng cáo ngắn | MusicGen điều kiện hóa giai điệu từ bản ngâm nga |
| Nhạc nền video | MusicGen + Stable Video Diffusion |

## Những cạm bẫy vẫn tồn tại trong năm 2026

- **Prompt "rửa" bản quyền.** "Song in the style of Taylor Swift" — các nền tảng thương mại như Suno/Udio hiện đã lọc các từ khóa này, nhưng các mô hình mở thì không. Hãy tự xây dựng danh sách lọc của riêng bạn.
- **Lặp lại / trôi cấu trúc sau 30 giây.** Các mô hình AR thường bị lặp. Hãy crossfade nhiều đoạn tạo ra, hoặc sử dụng ACE-Step để có cấu trúc mạch lạc hơn.
- **Trôi nhịp (Tempo drift).** Các mô hình thường bị lệch BPM. Hãy sử dụng thẻ BPM trong prompt và lọc hậu kỳ bằng `beat_track` của librosa.
- **Độ rõ của lời hát.** Suno làm rất tốt; các mô hình mở thường bị nhòe chữ. Nếu lời bài hát quan trọng, hãy dùng API thương mại hoặc fine-tune.
- **Output Mono.** Các mô hình mở tạo ra âm thanh mono hoặc stereo giả. Hãy nâng cấp bằng cách tái tạo stereo chuẩn (ezst, Cartesia's stereo diffusion).

## Triển khai

Lưu dưới dạng `outputs/skill-music-designer.md`. Chọn mô hình, chiến lược giấy phép, kế hoạch độ dài/cấu trúc và metadata công khai cho việc triển khai music-gen.

## Bài tập

1. **Dễ.** Chạy `code/main.py`. Nó tạo ra một chuỗi hợp âm "generative" + mẫu trống dưới dạng ký tự ASCII — một dạng hoạt hình của music-gen. Phát lại qua bất kỳ trình render MIDI nào nếu bạn muốn.
2. **Trung bình.** Cài đặt `audiocraft`, tạo các clip 10 giây qua 4 prompt thể loại với MusicGen-small, đo FAD so với tập hợp thể loại tham chiếu.
3. **Khó.** Sử dụng ACE-Step (hoặc MusicGen-melody), tạo ba biến thể của cùng một giai điệu với các prompt âm sắc khác nhau. Tính toán độ tương đồng CLAP với prompt để xác minh độ khớp.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| FAD | Audio FID | Khoảng cách Fréchet giữa các phân phối embedding của âm thanh thực và âm thanh tạo ra. |
| Chromagram | Giai điệu dưới dạng cao độ | Vector 12 chiều mỗi khung hình; đầu vào cho điều kiện hóa giai điệu. |
| Stems | Các track nhạc cụ | Bass / trống / giọng hát / giai điệu được tách riêng dưới dạng WAV. |
| Inpainting | Tạo lại một phần | Che một khoảng thời gian; mô hình chỉ tạo lại phần đó. |
| CLAP | Text-audio CLIP | Contrastive audio-text embedding; đánh giá độ khớp text-audio. |
| EnCodec | Music codec | Neural codec của Meta được MusicGen sử dụng; 32 kHz, 4 codebooks. |

## Đọc thêm

- [Copet et al. (2023). MusicGen](https://arxiv.org/abs/2306.05284) — benchmark tự hồi quy mã nguồn mở.
- [Evans et al. (2024). Stable Audio Open](https://arxiv.org/abs/2407.14358) — tiêu chuẩn cho thiết kế âm thanh.
- [ACE-Step](https://github.com/ace-step/ACE-Step) — mô hình tạo bài hát hoàn chỉnh 4B mã nguồn mở, tháng 4/2026.
- [Tài liệu nền tảng Suno v5](https://suno.com) — dẫn đầu về chất lượng thương mại.
- [AudioLDM2](https://arxiv.org/abs/2308.05734) — latent diffusion cho âm nhạc + hiệu ứng âm thanh.
- [Thông tin về thỏa thuận WMG-Suno](https://www.musicbusinessworldwide.com/suno-warner-music-settlement/) — tiền lệ tháng 11/2025.