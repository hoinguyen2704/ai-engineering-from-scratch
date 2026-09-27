# Audio Transformers — Kiến trúc Whisper

> Âm thanh là một hình ảnh của tần số theo thời gian. Whisper là một ViT "ăn" các mel spectrogram và phản hồi bằng lời nói.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 7 · 08 (Encoder-Decoder), Phase 7 · 09 (ViT)
**Time:** ~45 phút

## Vấn đề

Trước khi Whisper (OpenAI, Radford và cộng sự 2022) ra đời, các hệ thống nhận dạng giọng nói tự động (ASR) hiện đại nhất thường dựa vào wav2vec 2.0 và HuBERT — các bộ trích xuất đặc trưng tự giám sát (self-supervised) kết hợp với một lớp phân loại (head) được tinh chỉnh. Chúng có chất lượng cao nhưng đòi hỏi các pipeline dữ liệu đắt đỏ và kém linh hoạt với các miền dữ liệu khác nhau. Việc nhận dạng giọng nói đa ngôn ngữ đòi hỏi các mô hình riêng biệt cho từng ngữ hệ.

Whisper đã đặt cược vào ba yếu tố:

1. **Huấn luyện trên mọi thứ.** 680.000 giờ âm thanh được gán nhãn yếu (weakly-labeled) thu thập từ internet trên 97 ngôn ngữ. Không có tập dữ liệu học thuật sạch. Không có nhãn âm vị (phoneme).
2. **Mô hình đơn cho đa tác vụ.** Một bộ giải mã (decoder) duy nhất được huấn luyện đồng thời cho các tác vụ phiên âm, dịch thuật, phát hiện giọng nói (VAD), xác định ngôn ngữ và gắn nhãn thời gian thông qua các task token.
3. **Transformer encoder-decoder tiêu chuẩn.** Encoder tiêu thụ các log-mel spectrogram. Decoder tạo ra các token văn bản theo cơ chế tự hồi quy (autoregressive). Không vocoder, không CTC, không HMM.

Kết quả: Whisper large-v3 hoạt động mạnh mẽ trên nhiều chất giọng, môi trường nhiễu và các ngôn ngữ không có dữ liệu sạch được gán nhãn. Đây là giao diện giọng nói mặc định cho mọi trợ lý ảo mã nguồn mở và hầu hết các trợ lý thương mại vào năm 2026.

## Khái niệm

![Whisper pipeline: audio → mel → encoder → decoder → text](../assets/whisper.svg)

### Bước 1 — resample + window

Âm thanh ở tần số 16 kHz. Cắt/đệm (clip/pad) thành 30 giây. Tính toán log-mel spectrogram: 80 mel bin, bước nhảy (stride) 10 ms → ~3.000 khung hình × 80 đặc trưng. Đây là "hình ảnh đầu vào" mà Whisper nhìn thấy.

### Bước 2 — convolutional stem

Hai lớp Conv1D với kernel 3 và stride 2 giúp giảm 3.000 khung hình xuống còn 1.500. Việc này làm giảm một nửa độ dài chuỗi mà không làm tăng đáng kể số lượng tham số.

### Bước 3 — encoder

Một bộ transformer encoder 24 lớp (đối với bản large) xử lý trên 1.500 mốc thời gian. Sử dụng mã hóa vị trí hình sin (sinusoidal positional encoding), self-attention, và GELU FFN. Tạo ra 1.500 × 1.280 trạng thái ẩn (hidden states).

### Bước 4 — decoder

Một bộ transformer decoder 24 lớp. Nó tạo ra các token một cách tự hồi quy từ bộ từ vựng BPE, vốn là một tập siêu của GPT-2 với một vài token đặc biệt dành riêng cho âm thanh.

### Bước 5 — task tokens

Lời nhắc (prompt) của decoder bắt đầu bằng các token điều khiển cho mô hình biết cần làm gì:

```
<|startoftranscript|>  <|en|>  <|transcribe|>  <|0.00|>
```

hoặc

```
<|startoftranscript|>  <|fr|>  <|translate|>   <|0.00|>
```

Mô hình được huấn luyện dựa trên quy ước này. Bạn điều khiển tác vụ bằng tiền tố (prefix). Đây là phiên bản 2026 của instruction-tuning, nhưng được áp dụng cho giọng nói.

### Bước 6 — output

Sử dụng beam search (độ rộng 5) với ngưỡng log-prob. Các nhãn thời gian (timestamps) được dự đoán mỗi 0,02 giây âm thanh khi token `<|notimestamps|>` vắng mặt.

### Các kích thước của Whisper

| Model | Tham số | Lớp | d_model | Heads | VRAM (fp16) |
|-------|--------|--------|---------|-------|-------------|
| Tiny | 39M | 4 | 384 | 6 | ~1 GB |
| Base | 74M | 6 | 512 | 8 | ~1 GB |
| Small | 244M | 12 | 768 | 12 | ~2 GB |
| Medium | 769M | 24 | 1024 | 16 | ~5 GB |
| Large | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3 | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3-turbo | 809M | 32 | 1280 | 20 | ~6 GB (decoder 4 lớp) |

Large-v3-turbo (2024) đã cắt giảm decoder từ 32 lớp xuống còn 4. Tốc độ giải mã nhanh gấp 8 lần với độ suy giảm WER dưới 1 điểm. Việc mở khóa tốc độ giải mã đó là lý do tại sao Whisper-turbo là lựa chọn mặc định cho các tác nhân giọng nói thời gian thực vào năm 2026.

### Những gì Whisper không làm được

- Không thực hiện diarization (xác định ai đang nói). Hãy kết hợp với pyannote cho việc này.
- Không hỗ trợ streaming thời gian thực nguyên bản — cửa sổ 30 giây là cố định. Các wrapper hiện đại (`faster-whisper`, `WhisperX`) bổ sung tính năng streaming thông qua VAD + overlap.
- Không có ngữ cảnh dài hơn 30 giây nếu không có sự phân đoạn (chunking) bên ngoài. Trong thực tế, điều này hoạt động tốt vì giọng nói con người hiếm khi cần ngữ cảnh dài để phiên âm.

### Bối cảnh năm 2026

| Tác vụ | Model | Ghi chú |
|------|-------|-------|
| English ASR | Whisper-turbo, Moonshine | Moonshine nhanh gấp 4 lần trên thiết bị edge |
| Multilingual ASR | Whisper-large-v3 | 97 ngôn ngữ |
| Streaming ASR | faster-whisper + VAD | Đạt mục tiêu độ trễ 150 ms |
| TTS | Piper, XTTS-v2, Kokoro | Mô hình encoder-decoder, nhưng theo kiến trúc Whisper |
| Audio + language | AudioLM, SeamlessM4T | Token văn bản + token âm thanh trong cùng một transformer |

```figure
n5-mel-decode
```

## Xây dựng

Xem `code/main.py`. Chúng ta không huấn luyện lại Whisper — chúng ta xây dựng pipeline log-mel spectrogram + bộ định dạng prompt task-token. Đó là những phần bạn thực sự thao tác trong môi trường production.

### Bước 1: tổng hợp âm thanh

Tạo một sóng sin 1 giây ở tần số 440 Hz được lấy mẫu ở 16 kHz. 16.000 mẫu.

### Bước 2: log-mel spectrogram (đơn giản hóa)

Mel spectrogram đầy đủ cần FFT. Chúng ta thực hiện phiên bản đóng khung (framing) + năng lượng trên mỗi khung hình để minh họa pipeline mà không cần `librosa`:

```python
def frame_signal(x, frame_size=400, hop=160):
    frames = []
    for start in range(0, len(x) - frame_size + 1, hop):
        frames.append(x[start:start + frame_size])
    return frames
```

Khung = 25 ms, bước nhảy = 10 ms. Khớp với cửa sổ của Whisper. Năng lượng trên mỗi khung hình đóng vai trò thay thế cho các mel bin để phục vụ mục đích sư phạm.

### Bước 3: đệm thành 30 giây

Whisper luôn xử lý các đoạn 30 giây. Đệm (hoặc cắt) spectrogram thành 3.000 khung hình.

### Bước 4: xây dựng các token prompt

```python
def whisper_prompt(lang="en", task="transcribe", timestamps=True):
    tokens = ["<|startoftranscript|>", f"<|{lang}|>", f"<|{task}|>"]
    if not timestamps:
        tokens.append("<|notimestamps|>")
    return tokens
```

Đó là toàn bộ bề mặt điều khiển tác vụ. Một tiền tố 4 token.

## Sử dụng

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("meeting.wav", language="en", task="transcribe")
print(result["text"])
print(result["segments"][0]["start"], result["segments"][0]["end"])
```

Nhanh hơn, tương thích với OpenAI:

```python
from faster_whisper import WhisperModel
model = WhisperModel("large-v3-turbo", compute_type="int8_float16")
segments, info = model.transcribe("meeting.wav", vad_filter=True)
for s in segments:
    print(f"{s.start:.2f} - {s.end:.2f}: {s.text}")
```

**Khi nào nên chọn Whisper vào năm 2026:**

- ASR đa ngôn ngữ với một mô hình duy nhất.
- Phiên âm mạnh mẽ các âm thanh nhiễu, đa dạng.
- Nghiên cứu / tạo mẫu ASR — điểm khởi đầu nhanh nhất.

**Khi nào nên chọn giải pháp khác:**

- Streaming độ trễ cực thấp trên thiết bị edge — Moonshine vượt trội hơn Whisper ở chất lượng tương đương.
- AI hội thoại thời gian thực cần <200 ms — sử dụng ASR streaming chuyên dụng.
- Speaker diarization — Whisper không làm được việc này; hãy kết hợp với pyannote.

## Triển khai

Xem `outputs/skill-asr-configurator.md`. Kỹ năng này bao gồm việc chọn mô hình ASR, các tham số giải mã và pipeline tiền xử lý cho một ứng dụng giọng nói mới.

## Bài tập

1. **Dễ.** Chạy `code/main.py`. Xác nhận số lượng khung hình cho tín hiệu 1 giây ở 16 kHz với bước nhảy 10 ms là ~100 khung hình. Đối với 30 giây: ~3.000 khung hình.
2. **Trung bình.** Xây dựng log-mel spectrogram đầy đủ bằng `numpy.fft`. Xác minh 80 mel bin khớp với `librosa.feature.melspectrogram(n_mels=80)` trong sai số cho phép.
3. **Khó.** Triển khai suy luận streaming: chia âm thanh thành các cửa sổ 10 giây với 2 giây chồng lấp, chạy Whisper trên mỗi đoạn, hợp nhất các bản phiên âm. Đo lường tỷ lệ lỗi từ (WER) so với xử lý đơn lẻ trên một mẫu podcast 5 phút.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| Mel spectrogram | "Hình ảnh âm thanh" | Biểu diễn 2D: các bin tần số trên một trục, các khung thời gian trên trục kia; năng lượng log-scaled trên mỗi ô. |
| Log-mel | "Thứ Whisper nhìn thấy" | Mel spectrogram đi qua hàm log; xấp xỉ nhận thức về độ lớn âm thanh của con người. |
| Frame | "Một lát cắt thời gian" | Cửa sổ 25 ms của các mẫu; chồng lấp ở bước nhảy 10 ms. |
| Task token | "Tiền tố prompt cho giọng nói" | Các token đặc biệt như `<\|transcribe\|>` / `<\|translate\|>` trong prompt của decoder. |
| Voice activity detection (VAD) | "Tìm giọng nói" | Cổng loại bỏ khoảng lặng trước khi ASR; cắt giảm chi phí đáng kể. |
| CTC | "Connectionist Temporal Classification" | Hàm mất mát ASR cổ điển cho huấn luyện không cần căn chỉnh; Whisper KHÔNG sử dụng nó. |
| Whisper-turbo | "Decoder nhỏ, encoder đầy đủ" | Encoder large-v3 + decoder 4 lớp; giải mã nhanh gấp 8 lần. |
| Faster-whisper | "Wrapper cho production" | Tái triển khai CTranslate2; lượng tử hóa int8; nhanh gấp 4 lần bản tham chiếu của OpenAI. |

## Đọc thêm

- [Radford và cộng sự (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — Bài báo về Whisper.
- [OpenAI Whisper repo](https://github.com/openai/whisper) — mã nguồn tham chiếu + trọng số mô hình. Đọc `whisper/model.py` để thấy Conv1D stem + encoder + decoder từ trên xuống dưới trong ~400 dòng.
- [OpenAI Whisper — `whisper/decoding.py`](https://github.com/openai/whisper/blob/main/whisper/decoding.py) — logic beam-search + task-token được mô tả trong Bước 5–6 nằm ở đây; 500 dòng, hoàn toàn dễ đọc.
- [Baevski và cộng sự (2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477) — tiền thân; vẫn là các đặc trưng SOTA trong một số bối cảnh.
- [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) — wrapper cho production, nhanh gấp 4 lần bản tham chiếu.
- [Jia và cộng sự (2024). Moonshine: Speech Recognition for Live Transcription and Voice Commands](https://arxiv.org/abs/2410.15608) — ASR thân thiện với thiết bị edge năm 2024, hình dáng giống Whisper nhưng nhỏ hơn.
- [HuggingFace blog — "Fine-Tune Whisper For Multilingual ASR with 🤗 Transformers"](https://huggingface.co/blog/fine-tune-whisper) — công thức tinh chỉnh chuẩn bao gồm bộ tiền xử lý mel spectrogram và xử lý token-timestamp.
- [HuggingFace `modeling_whisper.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py) — triển khai đầy đủ (encoder, decoder, cross-attention, generation) phản ánh sơ đồ kiến trúc của bài học.