# Text-to-Speech (TTS) — Từ Tacotron đến F5 và Kokoro

> ASR chuyển đổi giọng nói thành văn bản; TTS chuyển đổi văn bản thành giọng nói. Stack công nghệ năm 2026 gồm ba phần: văn bản → tokens, tokens → mel, mel → dạng sóng (waveform). Mỗi phần đều có một mô hình mặc định chạy được trên laptop.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms & Mel), Phase 5 · 09 (Seq2Seq), Phase 7 · 05 (Full Transformer)
**Time:** ~75 phút

## Vấn đề

Bạn có một chuỗi văn bản: "Please remind me to water the plants at 6 pm." Bạn cần một đoạn âm thanh dài 3 giây nghe tự nhiên, có ngữ điệu chính xác (ngắt nghỉ, nhấn âm), phát âm từ "plants" với nguyên âm đúng, và chạy trong dưới 300 ms trên CPU cho một trợ lý giọng nói thời gian thực. Bạn cũng cần khả năng thay đổi giọng nói, xử lý đầu vào đa ngôn ngữ ("remind me at 6 pm, daijoubu?"), và không bị lỗi khi đọc tên riêng.

Các pipeline TTS hiện đại thường có cấu trúc như sau:

1. **Text frontend.** Chuẩn hóa văn bản (ngày tháng, số, email), chuyển đổi thành phonemes hoặc subword tokens, dự đoán các đặc trưng ngữ điệu.
2. **Acoustic model.** Văn bản → mel spectrogram. Tacotron 2 (2017), FastSpeech 2 (2020), VITS (2021), F5-TTS (2024), Kokoro (2024).
3. **Vocoder.** Mel → dạng sóng. WaveNet (2016), WaveRNN, HiFi-GAN (2020), BigVGAN (2022), các neural codec vocoder từ 2024 trở đi.

Vào năm 2026, ranh giới giữa acoustic model và vocoder dần mờ nhạt với các mô hình diffusion và flow-matching end-to-end. Tuy nhiên, mô hình tư duy ba phần vẫn rất hữu ích cho việc debug.

## Khái niệm

![Tacotron, FastSpeech, VITS, F5/Kokoro side-by-side](../assets/tts.svg)

**Tacotron 2 (2017).** Seq2seq: char-embedding → BiLSTM encoder → location-sensitive attention → autoregressive LSTM decoder tạo ra các mel frames. Chậm (do tính chất autoregressive), dễ bị lỗi trên văn bản dài. Vẫn được trích dẫn như một baseline.

**FastSpeech 2 (2020).** Non-autoregressive. Duration predictor xuất ra số lượng mel frames cho mỗi phoneme. Chạy 1-pass, nhanh gấp 10 lần Tacotron. Mất đi một chút độ tự nhiên (monotonic alignment) nhưng được sử dụng rộng rãi.

**VITS (2021).** Huấn luyện đồng thời encoder + flow-based duration + HiFi-GAN vocoder end-to-end với variational inference. Chất lượng cao, mô hình đơn nhất. Là TTS mã nguồn mở thống trị giai đoạn 2022–2024. Các biến thể: YourTTS (multi-speaker zero-shot), XTTS v2 (2024, Coqui).

**F5-TTS (2024).** Diffusion transformer dựa trên flow matching. Ngữ điệu tự nhiên, zero-shot voice cloning với 5 giây âm thanh tham chiếu. Đứng đầu các bảng xếp hạng TTS mã nguồn mở năm 2026. 335M tham số.

**Kokoro (2024).** Nhỏ gọn (82M), chạy được trên CPU, là TTS tiếng Anh tốt nhất cho mục đích thời gian thực. Closed-vocabulary, chỉ hỗ trợ tiếng Anh, giấy phép apache-2.0.

**OpenAI TTS-1-HD, ElevenLabs v2.5, Google Chirp-3.** Các giải pháp thương mại hiện đại nhất. ElevenLabs v2.5 với các thẻ cảm xúc ("[whispered]", "[laughing]") và giọng nhân vật đang thống trị mảng sản xuất sách nói vào năm 2026.

### Sự tiến hóa của Vocoder

| Kỷ nguyên | Vocoder | Độ trễ | Chất lượng |
|-----|---------|---------|---------|
| 2016 | WaveNet | chỉ offline | SOTA tại thời điểm ra mắt |
| 2018 | WaveRNN | ~thời gian thực | tốt |
| 2020 | HiFi-GAN | 100× thời gian thực | gần như con người |
| 2022 | BigVGAN | 50× thời gian thực | tổng quát hóa tốt qua các giọng/ngôn ngữ |
| 2024 | SNAC, DAC (neural codecs) | tích hợp với các mô hình AR | discrete tokens, hiệu quả về bit |

Đến năm 2026, hầu hết các mô hình "TTS" đều là end-to-end từ văn bản sang dạng sóng; mel spectrogram chỉ còn là một biểu diễn nội bộ.

### Đánh giá

- **MOS (Mean Opinion Score).** Thang điểm 1–5, dựa trên đánh giá cộng đồng. Vẫn là tiêu chuẩn vàng; nhưng rất chậm.
- **CMOS (Comparative MOS).** So sánh A-vs-B. Khoảng tin cậy chặt chẽ hơn trên mỗi lượt đánh giá.
- **UTMOS, DNSMOS.** Các bộ dự đoán MOS dựa trên neural không cần tham chiếu. Được dùng cho các bảng xếp hạng.
- **CER (Character Error Rate) qua ASR.** Chạy đầu ra TTS qua Whisper, tính CER so với văn bản gốc. Chỉ số đại diện cho độ rõ ràng.
- **SECS (Speaker Embedding Cosine Similarity).** Chất lượng sao chép giọng nói (voice-cloning).

Số liệu năm 2026 trên tập LibriTTS test-clean:

| Mô hình | UTMOS | CER (qua Whisper) | Kích thước |
|-------|-------|-------------------|------|
| Ground truth | 4.08 | 1.2% | — |
| F5-TTS | 3.95 | 2.1% | 335M |
| XTTS v2 | 3.81 | 3.5% | 470M |
| VITS | 3.62 | 3.1% | 25M |
| Kokoro v0.19 | 3.87 | 1.8% | 82M |
| Parler-TTS Large | 3.76 | 2.8% | 2.3B |

```figure
sp-tts-stack
```

## Xây dựng

### Bước 1: phonemize đầu vào

```python
from phonemizer import phonemize
ph = phonemize("Hello world", language="en-us", backend="espeak")
# 'həloʊ wɜːld'
```

Phonemes là cầu nối phổ quát. Hãy tránh đưa văn bản thô vào bất kỳ mô hình nào dưới cấp độ chất lượng của VITS.

### Bước 2: chạy Kokoro (mặc định cho CPU năm 2026)

```python
from kokoro import KPipeline
tts = KPipeline(lang_code="a")  # "a" = American English
audio, sr = tts("Please remind me to water the plants at 6 pm.", voice="af_bella")
# audio: float32 tensor, sr=24000
```

Chạy offline, file đơn, 82M tham số.

### Bước 3: chạy F5-TTS với voice cloning

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="my_voice_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please remind me to water the plants.",
)
```

Truyền vào một đoạn clip tham chiếu 5 giây + bản ghi của nó; F5 sẽ sao chép ngữ điệu và âm sắc.

### Bước 4: HiFi-GAN vocoder từ đầu

Quá lớn để đưa vào script hướng dẫn, nhưng cấu trúc là:

```python
class HiFiGAN(nn.Module):
    def __init__(self, mel_channels=80, upsample_rates=[8, 8, 2, 2]):
        super().__init__()
        # 4 upsample blocks, total 256x to go from mel-rate to audio-rate
        ...
    def forward(self, mel):
        return self.blocks(mel)  # -> waveform
```

Huấn luyện: adversarial (discriminator trên các cửa sổ ngắn) + mel-spectrogram reconstruction loss + feature-matching loss. Đã được thương mại hóa — hãy sử dụng các checkpoint tiền huấn luyện từ repo `hifi-gan` hoặc nvidia-NeMo.

### Bước 5: toàn bộ pipeline (pseudocode)

```python
text = "Please remind me at 6 pm."
phones = phonemize(text)
mel = acoustic_model(phones, speaker=alice)      # [T, 80]
wav = vocoder(mel)                                # [T * 256]
soundfile.write("out.wav", wav, 24000)
```

## Sử dụng

Stack công nghệ năm 2026:

| Tình huống | Lựa chọn |
|-----------|------|
| Trợ lý giọng nói tiếng Anh thời gian thực | Kokoro (CPU) hoặc XTTS v2 (GPU) |
| Sao chép giọng nói từ 5 giây tham chiếu | F5-TTS |
| Giọng nhân vật thương mại | ElevenLabs v2.5 |
| Đọc sách nói | ElevenLabs v2.5 hoặc XTTS v2 + fine-tune |
| Ngôn ngữ ít tài nguyên | Huấn luyện VITS trên 5–20 giờ dữ liệu ngôn ngữ mục tiêu |
| Thẻ cảm xúc / biểu cảm | ElevenLabs v2.5 hoặc StyleTTS 2 fine-tune |

Dẫn đầu mã nguồn mở năm 2026: **F5-TTS về chất lượng, Kokoro về hiệu năng**. Đừng dùng Tacotron trừ khi bạn là một nhà sử học.

## Các cạm bẫy

- **Thiếu bộ chuẩn hóa văn bản.** "Dr. Smith" đọc là "Doctor" hay "Drive"? "2026" là "twenty twenty six" hay "two zero two six"? Hãy chuẩn hóa TRƯỚC khi qua phonemizer.
- **Danh từ riêng OOV (Out-of-vocabulary).** "Ghumare" → "ghyu-mair"? Hãy trang bị một mô hình grapheme-to-phoneme dự phòng cho các token lạ.
- **Clipping.** Đầu ra vocoder hiếm khi bị clipping, nhưng sự không khớp về mel scaling khi inference có thể vượt quá ±1.0. Luôn luôn `np.clip(wav, -1, 1)`.
- **Không khớp sample-rate.** Kokoro xuất ra 24 kHz; pipeline hạ nguồn của bạn mong đợi 16 kHz → hãy resample hoặc sẽ bị aliasing.

## Triển khai

Lưu dưới dạng `outputs/skill-tts-designer.md`. Thiết kế một pipeline TTS cho một giọng nói, độ trễ và ngôn ngữ mục tiêu cụ thể.

## Bài tập

1. **Dễ.** Chạy `code/main.py`. Xây dựng một từ điển phoneme từ một bộ từ vựng nhỏ, ước tính thời lượng cho mỗi phoneme, và in ra một lịch trình "mel" giả lập.
2. **Trung bình.** Cài đặt Kokoro, tổng hợp cùng một câu với giọng `af_bella` và `am_adam`. So sánh thời lượng âm thanh và chất lượng chủ quan.
3. **Khó.** Ghi âm một đoạn tham chiếu 5 giây của chính bạn. Sử dụng F5-TTS để sao chép nó. Báo cáo chỉ số SECS giữa đoạn tham chiếu và đầu ra đã sao chép.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| Phoneme | Đơn vị âm thanh | Lớp âm thanh trừu tượng; 39 trong tiếng Anh (ARPABet). |
| Duration predictor | Thời lượng mỗi phoneme | Đầu ra mô hình non-AR; số lượng frame nguyên cho mỗi phoneme. |
| Vocoder | Mel → dạng sóng | Mạng neural ánh xạ mel-spec sang các mẫu thô. |
| HiFi-GAN | Vocoder tiêu chuẩn | Dựa trên GAN; thống trị giai đoạn 2020–2024. |
| MOS | Chất lượng chủ quan | Điểm đánh giá trung bình 1–5 từ người nghe. |
| SECS | Chỉ số sao chép giọng | Độ tương đồng cosine giữa embedding của giọng mục tiêu và đầu ra. |
| F5-TTS | SOTA mã nguồn mở 2024 | Flow-matching diffusion; zero-shot cloning. |
| Kokoro | Dẫn đầu CPU tiếng Anh | Mô hình 82M tham số, Apache 2.0. |

## Đọc thêm

- [Shen et al. (2017). Tacotron 2](https://arxiv.org/abs/1712.05884) — baseline seq2seq.
- [Kim, Kong, Son (2021). VITS](https://arxiv.org/abs/2106.06103) — end-to-end dựa trên flow.
- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885) — SOTA mã nguồn mở hiện tại.
- [Kong, Kim, Bae (2020). HiFi-GAN](https://arxiv.org/abs/2010.05646) — vocoder vẫn được sử dụng trong năm 2026.
- [Kokoro-82M trên HuggingFace](https://huggingface.co/hexgrad/Kokoro-82M) — TTS tiếng Anh thân thiện với CPU năm 2024.