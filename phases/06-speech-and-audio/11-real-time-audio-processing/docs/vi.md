# Xử lý âm thanh thời gian thực (Real-Time Audio Processing)

> Các pipeline xử lý theo lô (batch) xử lý toàn bộ tệp tin. Các pipeline thời gian thực xử lý 20 mili giây tiếp theo trước khi 20 mili giây kế tiếp ập đến. Mọi hệ thống AI hội thoại, phòng thu phát sóng và bot điện thoại đều sống và chết dựa trên ngân sách độ trễ (latency budget) này.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms), Phase 6 · 04 (ASR), Phase 6 · 07 (TTS)
**Time:** ~75 phút

## Vấn đề

Bạn muốn một trợ lý giọng nói có cảm giác sống động. Độ trễ lượt hội thoại của con người là khoảng 230 ms (từ lúc im lặng đến khi phản hồi). Bất cứ thứ gì trên 500 ms đều tạo cảm giác máy móc; trên 1500 ms thì cảm giác như bị hỏng. Ngân sách cho một vòng lặp **nghe → hiểu → phản hồi → nói** hoàn chỉnh vào năm 2026 là:

| Giai đoạn | Ngân sách |
|-------|--------|
| Mic → buffer | 20 ms |
| VAD | 10 ms |
| ASR (streaming) | 150 ms |
| LLM (first token) | 100 ms |
| TTS (first chunk) | 100 ms |
| Render → loa | 20 ms |
| **Tổng** | **~400 ms** |

Moshi (Kyutai, 2024) đạt mức 200 ms full-duplex. GPT-4o-realtime (2024) đạt khoảng 320 ms. Các pipeline phân tầng (cascaded) vào năm 2022 thường có độ trễ 2500 ms. Sự cải thiện gấp 10 lần đến từ ba kỹ thuật: (1) streaming mọi nơi, (2) pipelining bất đồng bộ với kết quả từng phần, (3) tạo nội dung có thể ngắt quãng.

## Khái niệm

![Streaming audio pipeline with ring buffer, VAD gate, interruption](../assets/real-time.svg)

**Frame / chunk / window.** Âm thanh thời gian thực truyền đi dưới dạng các khối có kích thước cố định. Lựa chọn phổ biến: 20 ms (320 mẫu tại 16 kHz). Mọi thành phần phía sau phải theo kịp nhịp độ này.

**Ring buffer.** Bộ đệm vòng có kích thước cố định. Luồng producer ghi các frame mới, luồng consumer đọc chúng. Ngăn chặn việc cấp phát bộ nhớ trong đường dẫn nóng (hot path). Kích thước ≈ độ trễ tối đa × tốc độ lấy mẫu; một ring buffer 2 giây tại 16 kHz = 32.000 mẫu.

**VAD (Voice Activity Detection).** Cổng chặn các tác vụ phía sau khi không có ai nói. Silero VAD 4.0 (2024) chạy dưới 1 ms cho mỗi frame 30 ms trên CPU. `webrtcvad` là giải pháp thay thế cũ hơn.

**Streaming ASR.** Các mô hình phát ra bản ghi từng phần khi âm thanh truyền đến. Parakeet-CTC-0.6B ở chế độ streaming (NeMo, 2024) đạt WER 2–5% với độ trễ 320 ms. Whisper-Streaming (Macháček và cộng sự, 2023) chia nhỏ Whisper để đạt trạng thái gần như streaming với độ trễ khoảng 2 giây.

**Interruption (Ngắt quãng).** Khi người dùng nói trong lúc trợ lý đang nói, bạn phải (a) phát hiện sự chen ngang (barge-in), (b) dừng TTS, (c) loại bỏ phần đầu ra LLM còn lại. Tất cả phải diễn ra trong vòng 100 ms, nếu không người dùng sẽ cảm thấy trợ lý bị "điếc".

**WebRTC Opus transport.** Frame 20 ms, 48 kHz, bitrate thích ứng 8–128 kbps. Tiêu chuẩn cho trình duyệt và di động. LiveKit, Daily.co, Pion là các stack năm 2026 để xây dựng ứng dụng giọng nói.

**Jitter buffer.** Các gói tin mạng đến không đúng thứ tự hoặc bị trễ. Jitter buffer sắp xếp lại và làm mượt; quá nhỏ → gây ngắt quãng âm thanh, quá lớn → gây độ trễ. Thông thường là 60–80 ms.

### Các vấn đề thường gặp

- **Tranh chấp luồng (Thread contention).** GIL của Python cộng với các mô hình nặng có thể làm nghẽn luồng âm thanh. Hãy sử dụng thư viện âm thanh C-callback (sounddevice, PortAudio) và giữ Python tránh xa đường dẫn nóng.
- **Độ trễ chuyển đổi tốc độ lấy mẫu.** Resampling bên trong pipeline làm tăng thêm 5–20 ms. Hãy resample ngay từ đầu hoặc sử dụng bộ resampler độ trễ bằng 0 (PolyPhase, `soxr_hq`).
- **TTS priming.** Ngay cả các TTS nhanh như Kokoro cũng có thời gian khởi động 100–200 ms cho yêu cầu đầu tiên. Hãy cache mô hình và làm nóng nó bằng một lượt chạy thử trước khi bắt đầu phiên thực tế.
- **Echo cancellation (Khử tiếng vọng).** Nếu không có AEC, đầu ra của TTS sẽ đi ngược vào mic và kích hoạt ASR trên chính giọng nói của bot. WebRTC AEC3 là tiêu chuẩn mã nguồn mở mặc định.

```figure
nyquist-aliasing
```

## Xây dựng

### Bước 1: ring buffer

```python
import collections

class RingBuffer:
    def __init__(self, capacity):
        self.buf = collections.deque(maxlen=capacity)
    def write(self, frame):
        self.buf.extend(frame)
    def read(self, n):
        return [self.buf.popleft() for _ in range(min(n, len(self.buf)))]
    def level(self):
        return len(self.buf)
```

Dung lượng quyết định độ trễ đệm tối đa. 32.000 mẫu tại 16 kHz = 2 giây.

### Bước 2: VAD gate

```python
def simple_energy_vad(frame, threshold=0.01):
    return sum(x * x for x in frame) / len(frame) > threshold ** 2
```

Thay thế bằng Silero VAD trong môi trường production:

```python
import torch
vad, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
is_speech = vad(torch.tensor(frame), 16000).item() > 0.5
```

### Bước 3: streaming ASR

```python
# Parakeet-CTC-0.6B streaming via NeMo
from nemo.collections.asr.models import EncDecCTCModelBPE
asr = EncDecCTCModelBPE.from_pretrained("nvidia/parakeet-ctc-0.6b")
# chunk_ms=320 ms, look_ahead_ms=80 ms
for chunk in audio_stream():
    partial_text = asr.transcribe_streaming(chunk)
    print(partial_text, end="\r")
```

### Bước 4: interruption handler

```python
class Dialog:
    def __init__(self):
        self.tts_task = None

    def on_user_speech(self, frame):
        if self.tts_task and not self.tts_task.done():
            self.tts_task.cancel()   # barge-in
        # then feed to streaming ASR

    def on_final_user_utterance(self, text):
        self.tts_task = asyncio.create_task(self.reply(text))

    async def reply(self, text):
        async for tts_chunk in llm_then_tts(text):
            speaker.write(tts_chunk)
```

Phụ thuộc vào I/O bất đồng bộ và khả năng hủy streaming TTS. WebRTC peerconnection.stop() trên track âm thanh là cách làm chuẩn mực.

## Sử dụng

Stack năm 2026:

| Lớp | Lựa chọn |
|-------|------|
| Transport | LiveKit (WebRTC) hoặc Pion (Go) |
| VAD | Silero VAD 4.0 |
| Streaming ASR | Parakeet-CTC-0.6B hoặc Whisper-Streaming |
| LLM first-token | Groq, Cerebras, vLLM-streaming |
| Streaming TTS | Kokoro hoặc ElevenLabs Turbo v2.5 |
| Echo cancel | WebRTC AEC3 |
| End-to-end native | OpenAI Realtime API hoặc Moshi |

## Cạm bẫy

- **Đệm 500 ms để an toàn.** Bộ đệm *chính là* mức độ trễ tối thiểu của bạn. Hãy thu nhỏ nó lại.
- **Không ghim luồng (pinning threads).** Callback âm thanh chạy trên luồng có độ ưu tiên thấp hơn luồng UI = gây giật lag khi tải cao.
- **Chunk TTS quá nhỏ.** Các chunk dưới 200 ms làm cho các artifact của vocoder trở nên rõ rệt. Chunk 320 ms là điểm tối ưu.
- **Không có jitter buffer.** Mạng thực tế luôn có jitter; nếu không làm mượt, bạn sẽ nghe thấy tiếng lách tách.
- **Xử lý lỗi đơn lẻ.** Pipeline âm thanh phải chống treo. Một ngoại lệ có thể làm hỏng cả phiên làm việc.

## Triển khai

Lưu dưới dạng `outputs/skill-realtime-designer.md`. Thiết kế một pipeline âm thanh thời gian thực với ngân sách độ trễ cụ thể cho từng giai đoạn.

## Bài tập

1. **Dễ.** Chạy `code/main.py`. Mô phỏng một ring buffer + VAD năng lượng; in ra độ trễ từng giai đoạn cho một luồng giả lập 10 giây.
2. **Trung bình.** Sử dụng `sounddevice`, xây dựng một vòng lặp passthrough xử lý mic của bạn theo các frame 20 ms và in trạng thái VAD tại mỗi frame.
3. **Khó.** Xây dựng một bài kiểm tra echo full duplex với `aiortc`: trình duyệt → WebRTC → Python → WebRTC → trình duyệt. Đo độ trễ glass-to-glass bằng xung 1 kHz.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| Ring buffer | Hàng đợi vòng | FIFO kích thước cố định, không khóa (hoặc khóa SPSC) cho các frame âm thanh. |
| VAD | Cổng im lặng | Mô hình hoặc heuristic đánh dấu đoạn có tiếng nói vs không có tiếng nói. |
| Streaming ASR | STT thời gian thực | Phát ra văn bản từng phần khi âm thanh đến; có giới hạn lookahead. |
| Jitter buffer | Bộ làm mượt mạng | Hàng đợi sắp xếp lại các gói tin đến sai thứ tự; thông thường 60–80 ms. |
| AEC | Khử tiếng vọng | Loại bỏ đường dẫn phản hồi từ loa vào mic. |
| Barge-in | Ngắt quãng người dùng | Hệ thống phát hiện người dùng nói giữa chừng khi TTS đang chạy; phải hủy phát lại. |
| Full duplex | Song công | Người dùng và bot có thể nói cùng lúc; Moshi là full duplex. |

## Đọc thêm

- [Macháček và cộng sự (2023). Whisper-Streaming](https://arxiv.org/abs/2307.14743) — Whisper streaming dạng chunk.
- [Kyutai (2024). Moshi](https://kyutai.org/Moshi.pdf) — độ trễ 200 ms full-duplex.
- [LiveKit Agents framework (2024)](https://docs.livekit.io/agents/) — điều phối tác nhân âm thanh trong production.
- [Silero VAD repo](https://github.com/snakers4/silero-vad) — VAD dưới 1 ms, Apache 2.0.
- [WebRTC AEC3 paper](https://webrtc.googlesource.com/src/+/main/modules/audio_processing/aec3/) — khử tiếng vọng mã nguồn mở.