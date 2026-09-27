# Voice Agents: Pipecat và LiveKit

> Voice agents là một danh mục sản xuất hàng đầu trong năm 2026. Pipecat cung cấp cho bạn một pipeline dựa trên frame bằng Python (VAD → STT → LLM → TTS → transport). LiveKit Agents kết nối các mô hình AI với người dùng qua WebRTC. Mục tiêu độ trễ sản xuất đạt mức 450–600ms end-to-end cho các stack cao cấp.

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 12 (Workflow Patterns)
**Time:** ~60 phút

## Mục tiêu học tập

- Mô tả pipeline dựa trên frame của Pipecat: DOWNSTREAM (nguồn→đích) và UPSTREAM (điều khiển).
- Kể tên các giai đoạn pipeline giọng nói chuẩn và các transport mà Pipecat hỗ trợ.
- Giải thích hai lớp voice agent của LiveKit Agents (MultimodalAgent, VoicePipelineAgent) và trường hợp sử dụng của từng loại.
- Tóm tắt kỳ vọng về độ trễ sản xuất năm 2026 và cách chúng thúc đẩy các lựa chọn kiến trúc.

## Vấn đề

Voice agents không chỉ là một vòng lặp văn bản với TTS được gắn thêm vào. Ngân sách độ trễ rất khắt khe (~600ms), âm thanh một phần (partial audio) là mặc định, phát hiện lượt nói (turn detection) là một mô hình, và các transport trải dài từ điện thoại SIP đến WebRTC. Bạn hoặc là xây dựng một pipeline dựa trên frame (Pipecat), hoặc dựa vào một nền tảng (LiveKit).

## Khái niệm

### Pipecat (pipecat-ai/pipecat)

- Framework pipeline dựa trên frame bằng Python.
- `Frame` → `FrameProcessor` chain.
- Hai hướng luồng:
  - **DOWNSTREAM** — nguồn → đích (audio vào, TTS ra).
  - **UPSTREAM** — phản hồi và điều khiển (hủy, số liệu, barge-in).
- `PipelineTask` quản lý vòng đời với các sự kiện (`on_pipeline_started`, `on_pipeline_finished`, `on_idle_timeout`) và các observer cho số liệu/tracing/RTVI.

Pipeline điển hình:

```
VAD (Silero) → STT → LLM (context alternates user/assistant) → TTS → transport
```

Transports: Daily, LiveKit, SmallWebRTCTransport, FastAPI WebSocket, WhatsApp.

Pipecat Flows bổ sung các cuộc hội thoại có cấu trúc (máy trạng thái). Pipecat Cloud là runtime được quản lý.

### LiveKit Agents (livekit/agents)

- Kết nối các mô hình AI với người dùng qua WebRTC.
- Các khái niệm chính: `Agent`, `AgentSession`, `entrypoint`, `AgentServer`.
- Hai lớp voice agent:
  - **MultimodalAgent** — âm thanh trực tiếp qua OpenAI Realtime hoặc tương đương.
  - **VoicePipelineAgent** — chuỗi STT → LLM → TTS; cung cấp khả năng kiểm soát ở cấp độ văn bản.
- Phát hiện lượt nói ngữ nghĩa (semantic turn detection) thông qua mô hình Transformer.
- Tích hợp MCP nguyên bản.
- Điện thoại qua SIP.
- 50+ mô hình không cần API key thông qua LiveKit Inference; 200+ mô hình khác thông qua các plugin.

### Các nền tảng thương mại

Vapi (~450–600ms trên stack cao cấp được tối ưu hóa) và Retell (~600ms end-to-end qua 180 cuộc gọi thử nghiệm) được xây dựng dựa trên những công nghệ này. Hãy chọn một nền tảng khi bạn muốn một stack giọng nói được quản lý mà không cần đội ngũ WebRTC.

### Những sai lầm thường gặp

- **Không xử lý barge-in.** Người dùng ngắt lời; agent vẫn tiếp tục nói. Yêu cầu các frame hủy UPSTREAM trong Pipecat, hoặc tương đương trong LiveKit.
- **Bỏ qua độ tin cậy của STT.** Các bản ghi có độ tin cậy thấp được đưa vào LLM như thể đó là sự thật. Hãy chặn dựa trên độ tin cậy hoặc yêu cầu xác nhận.
- **Ngắt quãng TTS giữa câu.** Khi pipeline hủy giữa chừng, TTS cần biết hoặc phải cắt âm thanh.
- **Bỏ qua ngân sách độ trễ.** Mỗi thành phần thêm vào 50–200ms. Hãy tính tổng chuỗi của bạn trước khi triển khai.

### Độ trễ điển hình năm 2026

- VAD: 20–60ms
- STT partial: 100–250ms
- LLM first token: 150–400ms
- TTS first audio: 100–200ms
- Transport RTT: 30–80ms

End-to-end 450–600ms là mức cao cấp. 800–1200ms là phổ biến. Bất cứ thứ gì > 1500ms đều tạo cảm giác bị lỗi.

```figure
voice-pipeline
```

## Xây dựng

`code/main.py` là một pipeline đồ chơi dựa trên frame với:

- `Frame` các loại (audio, transcript, text, tts_audio, control).
- `Processor` giao diện với `process(frame)`.
- Một pipeline năm giai đoạn (VAD → STT → LLM → TTS → transport) dưới dạng các bộ xử lý (processors) được lập trình.
- Một frame hủy UPSTREAM để minh họa barge-in.

Chạy nó:

```
python3 code/main.py
```

Dấu vết (trace) cho thấy luồng bình thường và một lệnh hủy barge-in giúp dừng TTS ngay giữa câu.

## Sử dụng

- **Pipecat** để kiểm soát toàn diện — các bộ xử lý tùy chỉnh, ưu tiên Python, các nhà cung cấp có thể cắm vào.
- **LiveKit Agents** cho các triển khai ưu tiên WebRTC và điện thoại.
- **Vapi / Retell** cho các voice agent được lưu trữ mà không cần đội ngũ WebRTC.
- **OpenAI Realtime / Gemini Live** cho âm thanh vào/ra trực tiếp (MultimodalAgent).

## Triển khai

`outputs/skill-voice-pipeline.md` tạo khung cho một pipeline giọng nói theo kiểu Pipecat với VAD + STT + LLM + TTS + transport cộng với xử lý barge-in.

## Bài tập

1. Thêm một observer số liệu vào pipeline đồ chơi của bạn: đếm số frame mỗi giai đoạn mỗi giây. Độ trễ tích tụ ở đâu?
2. Triển khai STT có chặn theo độ tin cậy: dưới ngưỡng, yêu cầu "bạn có thể nhắc lại không?"
3. Thêm phát hiện lượt nói ngữ nghĩa: quy tắc đơn giản — nếu bản ghi kết thúc bằng "?", đó là kết thúc lượt nói.
4. Đọc tài liệu về transport của Pipecat. Thay thế transport stdlib bằng cấu hình SmallWebRTCTransport (stub).
5. Đo lường OpenAI Realtime so với chuỗi STT+LLM+TTS trên cùng một truy vấn. Chi phí độ trễ của việc kiểm soát cấp độ văn bản là bao nhiêu?

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|------------------------|
| Frame | "Sự kiện" | Đơn vị dữ liệu được định kiểu trong pipeline (audio, transcript, text, control) |
| Processor | "Giai đoạn pipeline" | Trình xử lý với process(frame) |
| DOWNSTREAM | "Luồng xuôi" | Nguồn đến đích: audio vào, giọng nói ra |
| UPSTREAM | "Luồng phản hồi" | Điều khiển: hủy, số liệu, barge-in |
| VAD | "Phát hiện giọng nói" | Phát hiện khi người dùng đang nói |
| Semantic turn detection | "Kết thúc lượt nói thông minh" | Quyết định dựa trên mô hình rằng người dùng đã nói xong |
| MultimodalAgent | "Agent âm thanh trực tiếp" | Audio vào, audio ra; không có văn bản ở giữa |
| VoicePipelineAgent | "Agent chuỗi" | STT + LLM + TTS; kiểm soát ở cấp độ văn bản |

## Đọc thêm

- [Tài liệu Pipecat](https://docs.pipecat.ai/getting-started/introduction) — pipeline dựa trên frame, processors, transports
- [Tài liệu LiveKit Agents](https://docs.livekit.io/agents/) — WebRTC + các nguyên hàm giọng nói
- [Vapi](https://vapi.ai/) — nền tảng giọng nói được quản lý
- [Retell AI](https://www.retellai.com/) — giọng nói được quản lý, đã đo lường độ trễ