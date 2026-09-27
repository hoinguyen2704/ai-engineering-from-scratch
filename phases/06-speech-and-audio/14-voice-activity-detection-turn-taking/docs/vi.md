# Voice Activity Detection & Turn-Taking — Silero, Cobra, và Flush Trick

> Mọi voice agent đều sống hoặc chết dựa trên hai quyết định: người dùng có đang nói không, và họ đã nói xong chưa? VAD trả lời câu hỏi đầu tiên. Turn-detection (VAD + silence-hangover + semantic endpoint model) trả lời câu hỏi thứ hai. Nếu sai một trong hai, trợ lý của bạn sẽ hoặc là ngắt lời người dùng, hoặc là không bao giờ im lặng.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 11 (Real-Time Audio), Phase 6 · 12 (Voice Assistant)
**Time:** ~45 phút

## Vấn đề

Ba quyết định riêng biệt mà một voice agent thực hiện trên mỗi chunk 20 ms:

1. **Đây có phải là giọng nói không?** — VAD. Nhị phân, theo từng frame.
2. **Người dùng đã bắt đầu một lượt nói mới chưa?** — onset detection.
3. **Người dùng đã nói xong chưa?** — end-pointing (kết thúc lượt nói).

Cách tiếp cận ngây thơ (ngưỡng năng lượng - energy threshold) sẽ thất bại trước bất kỳ tiếng ồn nào — giao thông, bàn phím, tiếng ồn đám đông. Giải pháp năm 2026: Silero VAD (mã nguồn mở, deep-learned) + một mô hình turn-detection (semantic endpointing) + silence hangover được hiệu chỉnh theo VAD.

## Khái niệm

![VAD cascade: energy → Silero → turn-detector → flush trick](../assets/vad-turn-taking.svg)

### Cấu trúc VAD ba tầng

**Tầng 1: energy gate.** Rẻ nhất. Ngưỡng RMS tại -40 dBFS. Lọc bỏ sự im lặng rõ ràng nhưng sẽ kích hoạt với bất kỳ tiếng ồn nào trên ngưỡng.

**Tầng 2: Silero VAD** (2020-2026, MIT). 1 triệu tham số. Được huấn luyện trên hơn 6000 ngôn ngữ. Chạy trong ~1 ms cho mỗi chunk 30 ms trên một luồng CPU đơn. TPR 87.7% tại 5% FPR. Lựa chọn mặc định mã nguồn mở.

**Tầng 3: semantic turn detector.** Mô hình turn-detection của LiveKit (2024-2026) hoặc bộ phân loại nhỏ của riêng bạn. Phân biệt giữa "ngừng giữa câu" và "nói xong". Sử dụng ngữ cảnh ngôn ngữ (ngữ điệu + các từ gần đây), không chỉ dựa vào sự im lặng.

### Các tham số chính và giá trị mặc định

- **Threshold.** Silero xuất ra xác suất; phân loại là giọng nói tại > 0.5 (mặc định) hoặc > 0.3 (nhạy). Ngưỡng thấp hơn = ít bị cắt mất từ đầu hơn, nhưng nhiều dương tính giả (false positives) hơn.
- **Minimum speech duration.** Loại bỏ giọng nói ngắn hơn 250 ms — thường là tiếng ho hoặc tiếng ghế.
- **Silence hangover (end-pointing).** Sau khi VAD trả về 0, đợi 500-800 ms trước khi tuyên bố kết thúc lượt nói. Quá ngắn → ngắt lời người dùng. Quá dài → cảm giác phản hồi chậm chạp.
- **Pre-roll buffer.** Giữ lại 300-500 ms âm thanh trước khi VAD kích hoạt. Ngăn chặn việc từ "hey" bị cắt mất.

### Flush trick (Kyutai 2025)

Các mô hình STT streaming có độ trễ look-ahead (500 ms cho Kyutai STT-1B, 2.5 s cho STT-2.6B). Thông thường, bạn phải đợi khoảng thời gian đó sau khi kết thúc giọng nói để có bản ghi. Flush trick: khi VAD kích hoạt kết thúc giọng nói, **gửi tín hiệu flush đến STT** để buộc xuất kết quả ngay lập tức. STT xử lý ở tốc độ ~4× thời gian thực, vì vậy buffer 500 ms sẽ hoàn tất trong ~125 ms.

End-to-end: 125 ms VAD + flush STT = độ trễ hội thoại tối ưu.

### So sánh VAD năm 2026

| VAD | TPR @ 5% FPR | Độ trễ | Giấy phép |
|-----|--------------|---------|---------|
| WebRTC VAD (Google, 2013) | 50.0% | 30 ms | BSD |
| Silero VAD (2020-2026) | 87.7% | ~1 ms | MIT |
| Cobra VAD (Picovoice) | 98.9% | ~1 ms | thương mại |
| pyannote segmentation | 95% | ~10 ms | MIT-ish |

Silero là lựa chọn mặc định phù hợp. Cobra là bản nâng cấp về độ chính xác/tuân thủ. VAD chỉ dựa trên năng lượng không còn chỗ đứng trong môi trường production năm 2026.

```figure
sp-vad-cascade
```

## Xây dựng

### Bước 1: energy gate

```python
def energy_vad(chunk, threshold_dbfs=-40.0):
    rms = (sum(x * x for x in chunk) / len(chunk)) ** 0.5
    dbfs = 20.0 * math.log10(max(rms, 1e-10))
    return dbfs > threshold_dbfs
```

### Bước 2: Silero VAD trong Python

```python
from silero_vad import load_silero_vad, get_speech_timestamps

vad = load_silero_vad()
audio = torch.tensor(waveform_16k, dtype=torch.float32)
segments = get_speech_timestamps(
    audio, vad, sampling_rate=16000,
    threshold=0.5,
    min_speech_duration_ms=250,
    min_silence_duration_ms=500,
    speech_pad_ms=300,
)
for s in segments:
    print(f"{s['start']/16000:.2f}s - {s['end']/16000:.2f}s")
```

### Bước 3: máy trạng thái (state machine) cho kết thúc lượt nói

```python
class TurnDetector:
    def __init__(self, silence_hangover_ms=500, min_speech_ms=250):
        self.state = "idle"
        self.speech_ms = 0
        self.silence_ms = 0
        self.silence_hangover_ms = silence_hangover_ms
        self.min_speech_ms = min_speech_ms

    def update(self, is_speech, chunk_ms=20):
        if is_speech:
            self.speech_ms += chunk_ms
            self.silence_ms = 0
            if self.state == "idle" and self.speech_ms >= self.min_speech_ms:
                self.state = "speaking"
                return "START"
        else:
            self.silence_ms += chunk_ms
            if self.state == "speaking" and self.silence_ms >= self.silence_hangover_ms:
                self.state = "idle"
                self.speech_ms = 0
                return "END"
        return None
```

### Bước 4: khung sườn cho flush trick

```python
def flush_on_end(stt_client, audio_buffer):
    stt_client.send_audio(audio_buffer)
    stt_client.send_flush()
    return stt_client.recv_transcript(timeout_ms=150)
```

STT (Kyutai, Deepgram, AssemblyAI) phải hỗ trợ flush để cách này hoạt động. Whisper streaming không hỗ trợ — nó dựa trên block và luôn đợi đủ chunk.

## Sử dụng

| Tình huống | Lựa chọn VAD |
|-----------|-----------|
| Mở, nhanh, tổng quát | Silero VAD |
| Tổng đài thương mại | Cobra VAD |
| Trên thiết bị (điện thoại) | Silero VAD ONNX |
| Nghiên cứu / diarization | pyannote segmentation |
| Dự phòng không phụ thuộc | WebRTC VAD (cũ) |
| Cần chất lượng kết thúc lượt nói | Silero + LiveKit turn-detector kết hợp |

Quy tắc chung: không bao giờ triển khai VAD chỉ dựa trên năng lượng trừ khi bạn thực sự không còn lựa chọn nào khác.

## Các cạm bẫy

- **Ngưỡng cố định.** Hoạt động tốt trong môi trường yên tĩnh, thất bại trong môi trường ồn ào. Hãy hiệu chỉnh trên thiết bị hoặc chuyển sang Silero.
- **Silence hangover quá ngắn.** Agent ngắt lời giữa câu. 500-800 ms là điểm ngọt cho hội thoại.
- **Hangover quá dài.** Cảm giác phản hồi chậm. Hãy A/B test với người dùng mục tiêu.
- **Không có pre-roll buffer.** 200-300 ms đầu của người dùng bị mất. Luôn giữ một pre-roll cuộn.
- **Bỏ qua semantic endpointing.** "Hmm, để tôi nghĩ..." chứa những khoảng lặng dài. Người dùng ghét bị ngắt lời khi đang suy nghĩ. Hãy sử dụng turn-detector của LiveKit hoặc tương tự.

## Triển khai

Lưu dưới dạng `outputs/skill-vad-tuner.md`. Chọn mô hình VAD, ngưỡng, hangover, pre-roll và chiến lược turn-detection cho workload của bạn.

## Bài tập

1. **Dễ.** Chạy `code/main.py`. Nó mô phỏng chuỗi giọng nói + im lặng + giọng nói + tiếng ho và kiểm tra ba tầng VAD.
2. **Trung bình.** Cài đặt `silero-vad`, xử lý bản ghi 5 phút, tinh chỉnh ngưỡng để giảm thiểu cả việc cắt mất từ đầu và kích hoạt giả. Báo cáo độ chính xác (precision/recall).
3. **Khó.** Xây dựng một mini turn-detector: Silero VAD + MLP 3 lớp trên embedding của 10 từ cuối (sử dụng sentence-transformers). Huấn luyện trên tập dữ liệu kết thúc lượt nói được gán nhãn thủ công. Đánh bại Silero-only với 10% F1.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| VAD | Voice detector | Nhị phân theo frame: đây có phải giọng nói không? |
| Turn detection | End-pointing | VAD + silence-hangover + semantic endpoint. |
| Silence hangover | Wait-after-speech | Thời gian chờ trước khi tuyên bố kết thúc lượt nói; 500-800 ms. |
| Pre-roll | Pre-speech buffer | Giữ 300-500 ms âm thanh trước khi VAD kích hoạt. |
| Flush trick | Kyutai hack | VAD → flush-STT → độ trễ 125 ms thay vì 500 ms. |
| Semantic endpoint | "Họ có định dừng không?" | Bộ phân loại ML nhìn vào từ ngữ, không chỉ sự im lặng. |
| TPR @ FPR 5% | ROC point | Điểm chuẩn VAD tiêu chuẩn; 87.7% cho Silero, 50% WebRTC. |

## Đọc thêm

- [Silero VAD](https://github.com/snakers4/silero-vad) — VAD mã nguồn mở tham chiếu.
- [Picovoice Cobra VAD](https://picovoice.ai/products/cobra/) — dẫn đầu về độ chính xác thương mại.
- [Kyutai — Unmute + flush trick](https://kyutai.org/stt) — kỹ thuật dưới 200 ms.
- [LiveKit — turn detection](https://docs.livekit.io/agents/logic/turns/) — semantic endpointing trong production.
- [WebRTC VAD](https://webrtc.googlesource.com/src/) — baseline cũ.
- [pyannote segmentation](https://github.com/pyannote/pyannote-audio) — phân đoạn cấp độ diarization.