# Xây dựng Pipeline Trợ lý Giọng nói — Đồ án cuối khóa Giai đoạn 6

> Mọi thứ từ bài 01-11, được kết nối lại với nhau. Xây dựng một trợ lý giọng nói có khả năng lắng nghe, suy luận và phản hồi. Vào năm 2026, đây là một bài toán kỹ thuật đã được giải quyết, không còn là bài toán nghiên cứu — nhưng các chi tiết tích hợp mới là yếu tố quyết định liệu sản phẩm có được triển khai hay không.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 04, 05, 06, 07, 11; Phase 11 · 09 (Function Calling); Phase 14 · 01 (Agent Loop)
**Time:** ~120 phút

## Bài toán

Xây dựng một trợ lý end-to-end:

1. Thu âm từ micro (16 kHz mono).
2. Phát hiện điểm bắt đầu/kết thúc lời nói của người dùng.
3. Chuyển đổi giọng nói thành văn bản (STT) theo thời gian thực (streaming).
4. Chuyển văn bản sang LLM có khả năng gọi công cụ (hẹn giờ, thời tiết, lịch).
5. Truyền phát (stream) văn bản từ LLM sang TTS.
6. Phát âm thanh lại cho người dùng.
7. Dừng lại nếu người dùng ngắt lời giữa chừng.

Mục tiêu độ trễ: byte âm thanh TTS đầu tiên xuất hiện trong vòng 800 ms kể từ khi người dùng kết thúc câu nói trên CPU laptop. Mục tiêu chất lượng: không bỏ sót từ, không tạo phụ đề ảo khi im lặng, không rò rỉ voice cloning, không bị tấn công prompt injection.

## Khái niệm

![Voice assistant pipeline: mic → VAD → STT → LLM+tools → TTS → speaker](../assets/voice-assistant.svg)

### Bảy thành phần chính

1. **Thu âm (Audio capture).** Mic → 16 kHz mono → các chunk 20 ms. Thường dùng `sounddevice` trong Python hoặc AudioUnit/ALSA/WASAPI gốc trong môi trường production.
2. **VAD (Bài 11).** Silero VAD với ngưỡng 0.5, thời gian nói tối thiểu 250 ms, thời gian chờ im lặng 500 ms. Phát tín hiệu "bắt đầu" và "kết thúc".
3. **Streaming STT (Bài 4-5).** Whisper-streaming, Parakeet-TDT, hoặc Deepgram Nova-3 (API). Trả về kết quả từng phần (partial) và kết quả cuối cùng (final).
4. **LLM với tool calling.** GPT-4o / Claude 3.5 / Gemini 2.5 Flash. Sử dụng JSON schema cho các công cụ. Stream các token.
5. **Streaming TTS (Bài 7).** Kokoro-82M (mô hình mở nhanh nhất) hoặc Cartesia Sonic (thương mại). Bắt đầu TTS sau khi có 20 token từ LLM.
6. **Phát âm thanh (Playback).** Xuất ra loa; mã hóa opus cho các mạng băng thông thấp.
7. **Xử lý ngắt lời (Interruption handler).** Nếu VAD kích hoạt trong khi TTS đang phát, dừng phát, hủy LLM, khởi động lại STT.

### Ba dạng lỗi bạn sẽ gặp phải

1. **Mất từ đầu câu (First-word clip).** VAD bắt đầu chậm một nhịp. Từ "hey" của người dùng bị mất. Hãy đặt ngưỡng bắt đầu ở mức 0.3 thay vì 0.5.
2. **Nhầm lẫn khi ngắt lời.** LLM tiếp tục tạo văn bản sau khi người dùng ngắt lời; trợ lý nói đè lên người dùng. Hãy kết nối VAD → hủy LLM.
3. **Ảo giác im lặng (Silence hallucination).** Whisper xuất ra "Cảm ơn đã theo dõi" trong các khung hình im lặng lúc khởi động. Luôn sử dụng VAD-gate.

### Các stack tham khảo cho production năm 2026

| Stack | Độ trễ | Giấy phép | Ghi chú |
|-------|---------|---------|-------|
| LiveKit + Deepgram + GPT-4o + Cartesia | 350-500 ms | API thương mại | Tiêu chuẩn ngành 2026 |
| Pipecat + Whisper-streaming + GPT-4o + Kokoro | 500-800 ms | chủ yếu là mã nguồn mở | Thân thiện với DIY |
| Moshi (full-duplex) | 200-300 ms | CC-BY 4.0 | Mô hình đơn lẻ; kiến trúc khác, bài 15 |
| Vapi / Retell (managed) | 300-500 ms | thương mại | Triển khai nhanh nhất; tùy biến hạn chế |
| Whisper.cpp + llama.cpp + Kokoro-ONNX | offline | mở | Quyền riêng tư / edge |

```figure
v4-voice-latency
```

## Xây dựng

### Bước 1: thu âm với chunking (pseudocode)

```python
import sounddevice as sd

def mic_stream(chunk_ms=20, sr=16000):
    q = queue.Queue()
    def cb(indata, frames, time, status):
        q.put(indata.copy().flatten())
    with sd.InputStream(channels=1, samplerate=sr, blocksize=int(sr * chunk_ms/1000), callback=cb):
        while True:
            yield q.get()
```

### Bước 2: thu âm theo lượt (turn) được kiểm soát bởi VAD

```python
def capture_turn(stream, vad, pre_roll_ms=300, silence_ms=500):
    buf, pre, triggered = [], collections.deque(maxlen=pre_roll_ms // 20), False
    silent = 0
    for chunk in stream:
        pre.append(chunk)
        if vad(chunk):
            if not triggered:
                buf = list(pre)
                triggered = True
            buf.append(chunk)
            silent = 0
        elif triggered:
            silent += 20
            buf.append(chunk)
            if silent >= silence_ms:
                return b"".join(buf)
```

### Bước 3: streaming STT → LLM → TTS

```python
async def turn(audio_bytes):
    transcript = await stt.transcribe(audio_bytes)
    async for token in llm.stream(transcript):
        async for audio in tts.stream(token):
            await speaker.play(audio)
```

### Bước 4: gọi công cụ trong vòng lặp LLM

```python
tools = [
    {"name": "get_weather", "parameters": {"location": "string"}},
    {"name": "set_timer", "parameters": {"seconds": "int"}},
]

async for chunk in llm.stream(user_text, tools=tools):
    if chunk.type == "tool_call":
        result = dispatch(chunk.name, chunk.args)
        continue_streaming(result)
    if chunk.type == "text":
        await tts.stream(chunk.text)
```

### Bước 5: xử lý ngắt lời

```python
tts_task = asyncio.create_task(tts_loop())
while True:
    chunk = await mic.get()
    if vad(chunk):
        tts_task.cancel()
        await speaker.stop()
        await new_turn()
        break
```

## Sử dụng

Xem `code/main.py` để có một bản mô phỏng có thể chạy được, kết nối tất cả bảy thành phần với các mô hình giả lập (stub), giúp bạn thấy được hình dạng của pipeline ngay cả khi không có phần cứng. Đối với triển khai thực tế, hãy thay thế các stub bằng:

- `silero-vad` (`pip install silero-vad`)
- `deepgram-sdk` hoặc `openai-whisper`
- `openai` (`gpt-4o`) hoặc `anthropic`
- `kokoro` hoặc `cartesia`
- `sounddevice` cho I/O

## Các cạm bẫy

- **Lưu trữ PII vĩnh viễn.** Âm thanh toàn bộ lượt nói là PII (thông tin nhận dạng cá nhân) ở hầu hết các khu vực pháp lý. Lưu trữ 30 ngày, mã hóa khi không hoạt động.
- **Không có tính năng barge-in.** Người dùng sẽ ngắt lời. Trợ lý của bạn phải dừng nói.
- **TTS chặn luồng (blocking).** TTS đồng bộ sẽ chặn vòng lặp sự kiện. Hãy sử dụng async hoặc một luồng riêng biệt.
- **Không xử lý lỗi tool-call.** Các công cụ có thể thất bại. LLM phải nhận lại lỗi + thử lại một lần, sau đó suy giảm chức năng một cách nhẹ nhàng.
- **Bộ lọc ảo giác quá mức.** Lọc quá mức sẽ khiến trợ lý lặp lại "Tôi không thể giúp việc đó". Lọc quá ít sẽ khiến nó nói bất cứ điều gì. Hãy hiệu chỉnh trên tập dữ liệu kiểm thử (held-out set).
- **Không có tùy chọn wake-word.** Luôn lắng nghe là một rủi ro về quyền riêng tư. Hãy thêm một cổng wake-word (Porcupine hoặc openWakeWord).

## Triển khai

Lưu dưới dạng `outputs/skill-voice-assistant-architect.md`. Dựa trên ngân sách + quy mô + ngôn ngữ + các ràng buộc tuân thủ, hãy tạo một đặc tả stack đầy đủ.

## Bài tập

1. **Dễ.** Chạy `code/main.py`. Nó mô phỏng một lượt nói end-to-end với các module stub và in ra độ trễ từng giai đoạn.
2. **Trung bình.** Thay thế stub STT bằng mô hình Whisper thực tế trên một tệp `.wav` đã ghi âm trước. Đo lường WER và độ trễ end-to-end.
3. **Khó.** Thêm gọi công cụ: triển khai `get_weather` (bất kỳ API nào) và `set_timer`. Định tuyến LLM qua các công cụ và xác minh rằng khi người dùng nói "đặt hẹn giờ 5 phút", hàm phù hợp sẽ được kích hoạt và phản hồi bằng giọng nói xác nhận điều đó.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| Turn | Một vòng lặp người dùng + trợ lý | Một lượt nói của người dùng được giới hạn bởi VAD + một phản hồi LLM-TTS. |
| Barge-in | Ngắt lời | Người dùng nói trong khi trợ lý đang nói; trợ lý dừng lại. |
| Wake word | "Hey assistant" | Bộ phát hiện từ khóa ngắn; Porcupine, Snowboy, openWakeWord. |
| End-pointing | Kết thúc lượt nói | Quyết định dựa trên VAD + thời gian im lặng tối thiểu để xác định người dùng đã nói xong. |
| Pre-roll | Bộ đệm trước khi nói | Giữ lại 200-400 ms âm thanh trước khi VAD kích hoạt để tránh mất từ đầu câu. |
| Tool call | Gọi hàm | LLM phát ra JSON; runtime điều phối; kết quả được đưa ngược lại vào vòng lặp. |

## Đọc thêm

- [LiveKit — voice agent quickstart](https://docs.livekit.io/agents/) — tài liệu tham khảo chuẩn production.
- [Pipecat — voice agent examples](https://github.com/pipecat-ai/pipecat) — framework thân thiện với DIY.
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — con đường quản lý tập trung cho voice-native.
- [Kyutai Moshi](https://github.com/kyutai-labs/moshi) — tài liệu tham khảo full-duplex (Bài 15).
- [Porcupine wake-word](https://picovoice.ai/products/porcupine/) — cổng wake-word.
- [Anthropic — tool use guide](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) — gọi hàm LLM.