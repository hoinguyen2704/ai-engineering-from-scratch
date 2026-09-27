# Audio Generation

> Audio là một tín hiệu 1-D ở tần số 16-48 kHz. Một đoạn clip dài 5 giây chứa 80-240k mẫu. Không có Transformer nào xử lý trực tiếp chuỗi đó. Giải pháp cho mọi mô hình âm thanh thương mại vào năm 2026 đều giống nhau: một neural codec (Encodec, SoundStream, DAC) nén âm thanh thành các token rời rạc ở tần số 50-75 Hz, và một mô hình Transformer hoặc Diffusion sẽ tạo ra các token đó.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Audio Features), Phase 6 · 04 (ASR), Phase 8 · 06 (DDPM)
**Time:** ~45 phút

## Vấn đề

Ba tác vụ tạo âm thanh:

1. **Text-to-speech.** Với văn bản đầu vào, tạo ra giọng nói. Giọng nói sạch là dạng băng thông hẹp và có cấu trúc ngữ âm mạnh mẽ — được giải quyết tốt bởi Transformer-over-tokens. VALL-E (Microsoft), NaturalSpeech 3, ElevenLabs, OpenAI TTS.
2. **Music generation.** Với prompt đầu vào (văn bản, giai điệu, tiến trình hợp âm, thể loại), tạo ra âm nhạc. Phân phối dữ liệu rộng hơn nhiều. MusicGen (Meta), Stable Audio 2.5, Suno v4, Udio, Riffusion.
3. **Audio effects / sound design.** Với prompt đầu vào, tạo ra âm thanh môi trường hoặc Foley. AudioGen, AudioLDM 2, Stable Audio Open.

Cả ba đều chạy trên cùng một nền tảng: neural audio codec + token-AR hoặc trình tạo diffusion.

## Khái niệm

![Audio generation: codec tokens + transformer or diffusion](../assets/audio-generation.svg)

### Neural audio codecs

Encodec (Meta, 2022), SoundStream (Google, 2021), Descript Audio Codec (DAC, 2023). Một bộ mã hóa tích chập (convolutional encoder) nén dạng sóng thành một vector cho mỗi bước thời gian; residual vector quantization (RVQ) chuyển đổi mỗi vector thành một chuỗi các chỉ số codebook K. Bộ giải mã (decoder) thực hiện ngược lại. Âm thanh 24 kHz ở tốc độ 2 kbps sử dụng 8 codebook RVQ ở tần số 75 Hz = 600 token/giây.

```
waveform (16000 samples/sec)
    └─ encoder conv ─┐
                     ├─ RVQ layer 1 → indices at 75 Hz
                     ├─ RVQ layer 2 → indices at 75 Hz
                     ├─ ...
                     └─ RVQ layer 8
```

### Hai mô hình tạo sinh (generative paradigms) phổ biến

**Token-autoregressive.** Làm phẳng các token RVQ thành một chuỗi, chạy một Transformer chỉ có bộ giải mã (decoder-only). MusicGen sử dụng "delayed parallel" để phát ra các luồng codebook K song song với độ lệch (offset) cho mỗi luồng. VALL-E tạo ra các token giọng nói từ một prompt văn bản + mẫu giọng nói 3 giây.

**Latent diffusion.** Đóng gói các token codec dưới dạng các latent liên tục hoặc mô hình hóa chúng bằng diffusion phân loại. Stable Audio 2.5 sử dụng flow matching trên các latent âm thanh liên tục. AudioLDM 2 sử dụng diffusion từ văn bản sang mel-spectrogram rồi sang âm thanh.

Xu hướng 2024-2026: flow matching đang chiếm ưu thế cho âm nhạc (suy luận nhanh hơn, mẫu sạch hơn) trong khi token-AR vẫn thống trị mảng giọng nói vì tính nhân quả tự nhiên và khả năng streaming tốt.

## Bối cảnh sản xuất

| Hệ thống | Tác vụ | Backbone | Độ trễ |
|--------|------|----------|---------|
| ElevenLabs V3 | TTS | Token-AR + neural vocoder | ~300ms token đầu tiên |
| OpenAI GPT-4o audio | Full-duplex speech | End-to-end multimodal AR | ~200ms |
| NaturalSpeech 3 | TTS | Latent flow matching | Không streaming |
| Stable Audio 2.5 | Music / SFX | DiT + flow matching trên audio latents | ~10s cho clip 1 phút |
| Suno v4 | Full songs | Không công bố; nghi là token-AR | ~30s mỗi bài |
| Udio v1.5 | Full songs | Không công bố | ~30s mỗi bài |
| MusicGen 3.3B | Music | Token-AR trên Encodec 32kHz | Thời gian thực |
| AudioCraft 2 | Music + SFX | Flow matching | ~5s cho clip 5s |
| Riffusion v2 | Music | Spectrogram diffusion | ~10s |

```figure
score-matching
```

## Xây dựng

`code/main.py` mô phỏng ý tưởng cốt lõi: huấn luyện một Transformer dự đoán token tiếp theo cực nhỏ trên các chuỗi "audio token" tổng hợp được tạo từ hai "phong cách" riêng biệt (xen kẽ các token thấp và cao cho phong cách A, tăng dần đơn điệu cho phong cách B). Điều kiện hóa theo phong cách và lấy mẫu.

### Bước 1: audio tokens tổng hợp

```python
def make_tokens(style, length, vocab_size, rng):
    if style == 0:  # "speech-like": alternating
        return [i % vocab_size for i in range(length)]
    # "music-like": ramp
    return [(i * 3) % vocab_size for i in range(length)]
```

### Bước 2: huấn luyện trình dự đoán token nhỏ

Một trình dự đoán kiểu bigram được điều kiện hóa theo phong cách. Điểm mấu chốt là mô hình: codec tokens → huấn luyện cross-entropy → lấy mẫu tự hồi quy (autoregressive sampling).

### Bước 3: lấy mẫu có điều kiện

Với token phong cách và một token bắt đầu, lấy mẫu token tiếp theo từ phân phối đã dự đoán. Tiếp tục trong 20-40 token.

## Các cạm bẫy

- **Chất lượng codec giới hạn chất lượng đầu ra.** Nếu codec không thể biểu diễn âm thanh một cách trung thực, không có chất lượng trình tạo nào có thể bù đắp. DAC hiện là codec mã nguồn mở tốt nhất.
- **Tích lũy lỗi RVQ.** Mỗi lớp RVQ mô hình hóa phần dư (residual) của lớp trước. Lỗi ở lớp 1 sẽ lan truyền. Lấy mẫu với nhiệt độ (temperature) bằng 0 ở các lớp cao hơn sẽ giúp ích.
- **Cấu trúc âm nhạc.** 30 giây âm thanh tương đương hơn 20k token ở tần số 75 Hz. Rất khó cho các Transformer. MusicGen sử dụng cửa sổ trượt (sliding window) + tiếp nối prompt; Stable Audio sử dụng các clip ngắn hơn + crossfading.
- **Artifacts tại các ranh giới.** Việc crossfading giữa các clip được tạo cần kỹ thuật overlap-add cẩn thận.
- **Nhu cầu dữ liệu sạch.** Các trình tạo nhạc cần hàng chục nghìn giờ nhạc có bản quyền. Vụ kiện RIAA đối với Suno / Udio (2024) đã làm nổi bật vấn đề này.
- **Đạo đức nhân bản giọng nói.** Một mẫu 3 giây cộng với một prompt văn bản là đủ để VALL-E / XTTS / ElevenLabs nhân bản giọng nói. Mọi mô hình sản xuất đều cần tính năng phát hiện lạm dụng + danh sách từ chối (opt-out).

## Sử dụng

| Tác vụ | Stack 2026 |
|------|------------|
| TTS thương mại | ElevenLabs, OpenAI TTS, hoặc Azure Neural |
| Nhân bản giọng nói (đã xác minh sự đồng ý) | XTTS v2 (mở) hoặc ElevenLabs Pro |
| Nhạc nền, nhanh | Stable Audio 2.5 API, Suno, hoặc Udio |
| Nhạc có lời | Suno v4 hoặc Udio v1.5 |
| Hiệu ứng âm thanh / Foley | AudioCraft 2, ElevenLabs SFX, hoặc Stable Audio Open |
| Trợ lý giọng nói thời gian thực | GPT-4o realtime hoặc Gemini Live |
| Nghiên cứu âm nhạc mã nguồn mở | MusicGen 3.3B, Stable Audio Open 1.0, AudioLDM 2 |
| Lồng tiếng / dịch thuật | HeyGen, ElevenLabs Dubbing |

## Triển khai

Lưu `outputs/skill-audio-brief.md`. Kỹ năng này bao gồm tóm tắt âm thanh (tác vụ, thời lượng, phong cách, giọng nói, giấy phép) và xuất ra: mô hình + hosting, định dạng prompt (thẻ thể loại, mô tả phong cách, đánh dấu cấu trúc), chuỗi codec + generator + vocoder, giao thức seed, và kế hoạch đánh giá (MOS / CLAP score / CER cho TTS / A/B testing).

## Bài tập

1. **Dễ.** Chạy `code/main.py` và thiết lập phong cách một cách rõ ràng. Xác minh các chuỗi được tạo khớp với mô hình của phong cách đó.
2. **Trung bình.** Thêm giải mã song song trễ (delayed parallel decoding): mô phỏng 2 luồng token phải giữ độ lệch 1 bước. Huấn luyện một trình dự đoán chung.
3. **Khó.** Sử dụng HuggingFace transformers để chạy MusicGen-small cục bộ. Tạo một clip 10 giây với ba prompt khác nhau; thực hiện A/B test để kiểm tra độ bám sát phong cách.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| Codec | "Nén thần kinh" | Bộ mã hóa/giải mã âm thanh; đầu ra điển hình là token 50-75 Hz. |
| RVQ | "Residual VQ" | Chuỗi các bộ lượng tử hóa; mỗi bộ mô hình hóa phần dư của bộ trước. |
| Token | "Một ký hiệu codec" | Chỉ số rời rạc trong codebook; thường là 1024 hoặc 2048. |
| Delayed parallel | "Codebook lệch" | Phát ra các luồng token K với độ lệch so le để giảm độ dài chuỗi. |
| Flow matching | "Chiến thắng 2024 cho âm thanh" | Giải pháp thay thế diffusion với đường dẫn thẳng hơn; lấy mẫu nhanh hơn. |
| Voice prompt | "Mẫu 3 giây" | Speaker embedding hoặc tiền tố token điều hướng giọng nói được nhân bản. |
| Mel spectrogram | "Hình ảnh trực quan" | Spectrogram cảm nhận log-magnitude; được nhiều hệ thống TTS sử dụng. |
| Vocoder | "Mel sang sóng" | Thành phần thần kinh chuyển đổi mel spectrogram trở lại âm thanh. |

## Lưu ý sản xuất: âm thanh là vấn đề streaming

Âm thanh là phương thức đầu ra duy nhất mà người dùng mong đợi nhận được *ngay khi nó được tạo ra*, chứ không phải đợi tất cả cùng lúc. Trong sản xuất, điều này có nghĩa là TPOT (Time Per Output Token) rất quan trọng vì tốc độ nghe của người dùng là thông lượng mục tiêu — không phải tốc độ đọc. Đối với âm thanh 16kHz được token hóa ở mức ~75 token/giây (Encodec), máy chủ phải tạo ra ≥75 token/giây cho mỗi người dùng để giữ cho việc phát lại mượt mà.

Hai hệ quả kiến trúc:

- **Các mô hình âm thanh flow-matching không thể streaming một cách đơn giản.** Stable Audio 2.5 và AudioCraft 2 kết xuất một độ dài clip cố định trong một lần truyền. Để streaming, bạn phải chia nhỏ clip và chồng lấp các ranh giới — hãy nghĩ đến diffusion cửa sổ trượt — làm tăng thêm 100-300ms độ trễ so với mô hình codec AR.

Nếu sản phẩm là "trò chuyện giọng nói trực tiếp" hoặc "tiếp nối âm nhạc thời gian thực", hãy chọn con đường codec AR. Nếu là "kết xuất clip 30 giây khi gửi yêu cầu", flow-matching thắng về chất lượng và tổng độ trễ.

## Đọc thêm

- [Défossez et al. (2022). Encodec: High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) — tiêu chuẩn codec.
- [Zeghidour et al. (2021). SoundStream](https://arxiv.org/abs/2107.03312) — neural audio codec đầu tiên được sử dụng rộng rãi.
- [Kumar et al. (2023). High-Fidelity Audio Compression with Improved RVQGAN (DAC)](https://arxiv.org/abs/2306.06546) — DAC.
- [Wang et al. (2023). Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers (VALL-E)](https://arxiv.org/abs/2301.02111) — VALL-E.
- [Copet et al. (2023). Simple and Controllable Music Generation (MusicGen)](https://arxiv.org/abs/2306.05284) — MusicGen.
- [Liu et al. (2023). AudioLDM 2: Learning Holistic Audio Generation with Self-supervised Pretraining](https://arxiv.org/abs/2308.05734) — AudioLDM 2.
- [Stability AI (2024). Stable Audio 2.5](https://stability.ai/news/introducing-stable-audio-2-5) — text-to-music 2025 với flow matching.