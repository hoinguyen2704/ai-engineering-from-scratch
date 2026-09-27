# Capstone 03 — Trợ lý giọng nói thời gian thực (ASR đến LLM đến TTS)

> Một tác nhân giọng nói (voice agent) mang lại trải nghiệm tốt cần có độ trễ end-to-end dưới 800ms, biết khi nào người dùng đã ngừng nói, xử lý được tình huống ngắt lời (barge-in) và có thể gọi công cụ mà không bị khựng. Retell, Vapi, LiveKit Agents và Pipecat đều đạt được tiêu chuẩn này vào năm 2026. Chúng thực hiện điều đó với cùng một cấu trúc: ASR streaming, bộ phát hiện lượt nói (turn-detector), LLM streaming và TTS streaming, tất cả được kết nối qua WebRTC với ngân sách độ trễ khắt khe tại mỗi chặng. Hãy xây dựng một hệ thống, đo lường WER, MOS và tỷ lệ ngắt lời sai (false-cutoff rate), đồng thời chạy nó trong điều kiện mất gói tin.

**Type:** Capstone
**Languages:** Python (agent + pipeline), TypeScript (web client)
**Prerequisites:** Phase 6 (speech and audio), Phase 7 (transformers), Phase 11 (LLM engineering), Phase 13 (tools), Phase 14 (agents), Phase 17 (infrastructure)
**Phases exercised:** P6 · P7 · P11 · P13 · P14 · P17
**Time:** 30 giờ

## Vấn đề

Giọng nói là danh mục UX AI phát triển nhanh nhất trong giai đoạn 2025-2026. Rào cản kỹ thuật đã giảm xuống qua từng quý. OpenAI Realtime API, Gemini 2.5 Live, Cartesia Sonic-2, ElevenLabs Flash v3, LiveKit Agents 1.0 và Pipecat 0.0.70 đều giúp việc đạt được âm thanh đầu ra đầu tiên (first-audio-out) dưới 800ms trở nên khả thi. Tiêu chuẩn không chỉ nằm ở độ trễ. Đó là cảm giác tương tác: không cắt ngang người dùng, không bị cắt ngang, phục hồi sau khi bị ngắt quãng giữa câu, gọi công cụ giữa cuộc hội thoại mà không làm gián đoạn âm thanh, và duy trì ổn định trên các mạng di động chập chờn.

Bạn không thể đạt được điều này bằng cách ghép nối ba lệnh gọi REST. Kiến trúc phải là streaming pipeline end-to-end. Hãy xây dựng nó và các chế độ lỗi sẽ lộ diện: một VAD được tinh chỉnh cho âm thanh điện thoại nhưng lại kích hoạt bởi tiếng TV nền, một bộ phát hiện lượt nói chờ đợi dấu câu không bao giờ đến, một TTS đệm 400ms trước khi phát âm thanh. Capstone này yêu cầu bạn khắc phục từng vấn đề một dưới tải trọng và xuất bản báo cáo về độ trễ và chất lượng.

## Khái niệm

Pipeline có năm giai đoạn streaming: **audio in** (WebRTC từ trình duyệt hoặc PSTN), **ASR** (streaming các bản ghi một phần từ Deepgram Nova-3 hoặc faster-whisper), **turn detection** (VAD cộng với một mô hình phát hiện lượt nói nhỏ đọc các bản ghi một phần để tìm tín hiệu kết thúc), **LLM** (streaming token ngay khi lượt nói được đánh giá là hoàn tất), **TTS** (streaming âm thanh đầu ra trong vòng ~200ms kể từ token LLM đầu tiên).

Ba mối quan tâm xuyên suốt: **Barge-in**: khi người dùng bắt đầu nói trong khi tác nhân đang nói, TTS sẽ hủy và ASR sẽ tiếp nhận ngay lập tức. **Tool use**: các lệnh gọi hàm giữa cuộc hội thoại (thời tiết, lịch) phải chạy trên kênh phụ mà không làm khựng âm thanh; tác nhân sẽ phát trước một token xác nhận ("đợi một chút...") nếu độ trễ vượt quá 300ms. **Backpressure**: trong điều kiện mất gói tin, các bản ghi một phần sẽ được giữ lại, VAD nâng ngưỡng cổng giọng nói (speech-gate threshold) và tác nhân tránh nói đè lên một thông điệp chưa được xác nhận.

Tiêu chuẩn đo lường mang tính định lượng. WER dưới 8% trên benchmark Hamming VAD ở mức SNR 15 dB. First-audio-out p50 dưới 800ms trên 100 cuộc gọi được đo lường. Tỷ lệ ngắt lời sai dưới 3%. MOS trên 4.2 đối với TTS. 50 cuộc gọi đồng thời trên một máy g5.xlarge. Những con số này là kết quả cần bàn giao.

## Kiến trúc

```
browser / Twilio PSTN
        |
        v
   WebRTC / SIP edge
        |
        v
  LiveKit Agents 1.0  (or Pipecat 0.0.70)
        |
   +----+--------------+--------------+-----------------+
   |                   |              |                 |
   v                   v              v                 v
  ASR              VAD v5         turn-detector     side-channel
(Deepgram         (Silero)          (LiveKit)        tools
 Nova-3 /         speech-gate    completion score    (weather,
 Whisper-v3)      per 20ms        on partials        calendar)
   |                   |              |
   +--------+----------+--------------+
            v
        LLM (streaming)
     GPT-4o-realtime / Gemini 2.5 Flash /
     cascaded Claude Haiku 4.5
            |
            v
        TTS streaming
     Cartesia Sonic-2 / ElevenLabs Flash v3
            |
            v
     audio back to caller
            |
            v
   OpenTelemetry voice traces -> Langfuse
```

## Stack

- Transport: LiveKit Agents 1.0 (WebRTC) cộng với Twilio PSTN gateway; Pipecat 0.0.70 làm framework thay thế
- ASR: Deepgram Nova-3 (streaming, first partial dưới 300ms) hoặc faster-whisper Whisper-v3-turbo tự lưu trữ
- VAD: Silero VAD v5 cộng với bộ phát hiện lượt nói của LiveKit (transformer nhỏ đọc các bản ghi một phần)
- LLM: OpenAI GPT-4o-realtime để tích hợp chặt chẽ, Gemini 2.5 Flash Live, hoặc Claude Haiku 4.5 (streaming completions, đường dẫn âm thanh riêng biệt)
- TTS: Cartesia Sonic-2 (first-byte thấp nhất), ElevenLabs Flash v3, hoặc Orpheus mã nguồn mở để tự lưu trữ
- Tools: FastMCP side-channel cho thời tiết/lịch/đặt chỗ; tác nhân phát trước nội dung đệm nếu công cụ mất >300ms
- Observability: OpenTelemetry voice spans, Langfuse voice traces với khả năng phát lại âm thanh
- Deployment: một máy g5.xlarge (24GB VRAM) cho Whisper + Orpheus tự lưu trữ; các API được lưu trữ để có độ trễ thấp nhất

```figure
ce-voice-latency
```

## Xây dựng

1. **WebRTC session.** Thiết lập một phòng LiveKit và một web client truyền phát âm thanh micro. Trên server, gắn một agent worker tham gia vào phòng đó.

2. **ASR streaming.** Cung cấp các khung PCM 20ms cho Deepgram Nova-3 (hoặc faster-whisper trên GPU). Đăng ký nhận các bản ghi một phần và bản ghi cuối cùng. Ghi lại độ trễ trên mỗi phần.

3. **VAD và turn detector.** Chạy Silero VAD v5 trên luồng khung hình. Khi có sự kiện kết thúc giọng nói, kích hoạt bộ phát hiện lượt nói của LiveKit dựa trên bản ghi một phần mới nhất. Chỉ xác nhận "lượt nói hoàn tất" khi VAD báo im lặng trong 500ms và bộ phát hiện lượt nói chấm điểm hoàn tất > 0.6.

4. **LLM stream.** Khi lượt nói hoàn tất, bắt đầu gọi LLM với cuộc hội thoại đang diễn ra cộng với bản ghi cuối cùng. Stream các token ra ngoài. Tại token đầu tiên, chuyển giao cho TTS.

5. **TTS stream.** Cartesia Sonic-2 stream các đoạn âm thanh trở lại. Đoạn đầu tiên phải rời khỏi server trong vòng 200ms kể từ token LLM đầu tiên. Phát các đoạn âm thanh vào phòng LiveKit; client phát qua bộ đệm jitter của WebRTC.

6. **Barge-in.** Khi VAD phát hiện giọng nói mới của người dùng trong khi TTS đang phát, hủy ngay lập tức luồng TTS, loại bỏ đầu ra LLM còn lại và kích hoạt lại ASR. Xuất bản một span `tts_canceled`.

7. **Tool side channel.** Đăng ký thời tiết và lịch làm các công cụ gọi hàm. Khi được gọi, thực hiện lệnh gọi đồng thời; nếu không phản hồi trong vòng 300ms, để LLM phát "đợi một chút, để tôi kiểm tra" như một nội dung đệm; tiếp tục khi công cụ trả về kết quả.

8. **Eval harness.** Ghi lại 100 cuộc gọi. Tính toán WER (so với bản ghi đối chứng), tỷ lệ ngắt lời sai (TTS bị hủy khi người dùng đang nói giữa chừng), first-audio-out p50, TTS MOS (con người hoặc NISQA), và kiểm tra mất gói tin (loại bỏ 3% gói tin).

9. **Load test.** Chạy 50 cuộc gọi đồng thời trên một máy g5.xlarge với trình gọi giả lập. Đo lường first-audio-out p95 duy trì.

## Sử dụng

```
caller: "what is the weather in tokyo tomorrow"
[asr  ] partial @280ms: "what is the"
[asr  ] partial @540ms: "what is the weather"
[turn ] completion score 0.82 at @820ms; commit
[llm  ] first token @960ms
[tool ] weather.tokyo tomorrow -> 68/52 partly cloudy @1140ms
[tts  ] first audio-out @1040ms: "Tokyo tomorrow will be partly cloudy..."
turn latency: 1040ms user-stop -> audio-out
```

## Xuất bản

`outputs/skill-voice-agent.md` là kết quả bàn giao. Với một lĩnh vực cụ thể (chăm sóc khách hàng, đặt lịch hoặc kiosk), nó thiết lập một tác nhân LiveKit với pipeline ASR/VAD/LLM/TTS được tinh chỉnh theo tiêu chuẩn đo lường. Bảng đánh giá:

| Trọng số | Tiêu chí | Cách đo lường |
|:-:|---|---|
| 25 | Độ trễ end-to-end | p50 first-audio-out dưới 800ms trên 100 cuộc gọi được ghi lại |
| 20 | Chất lượng lượt nói | Tỷ lệ ngắt lời sai dưới 3% trên benchmark Hamming VAD |
| 20 | Độ chính xác khi dùng công cụ | Các lệnh gọi công cụ giữa cuộc hội thoại trả về đúng dữ liệu mà không làm khựng âm thanh |
| 20 | Độ tin cậy khi mất gói tin | WER và sự ổn định của lượt nói với 3% gói tin bị mất |
| 15 | Độ hoàn thiện của Eval harness | Các phép đo có thể tái lập với cấu hình công khai |
| **100** | | |

## Bài tập

1. Thay thế Deepgram Nova-3 bằng faster-whisper v3 turbo trên g5.xlarge. Đo lường khoảng cách về độ trễ và WER. Xác định nơi các quyết định CPU-vs-GPU trở nên quan trọng.

2. Thêm chính sách trọng tài ngắt quãng: tác nhân sẽ làm gì khi người dùng ngắt lời trong khi đang gọi công cụ? So sánh ba chính sách (hủy cứng, hoàn tất công cụ rồi dừng, xếp hàng lượt nói tiếp theo).

3. Chạy kiểm tra bộ phát hiện lượt nói đối kháng: cho người dùng tạm dừng lâu giữa câu. Tinh chỉnh ngưỡng im lặng của VAD và ngưỡng điểm của bộ phát hiện lượt nói để có tỷ lệ ngắt lời sai thấp nhất mà không vượt quá 900ms.

4. Triển khai cùng tác nhân đó trên PSTN qua Twilio. So sánh first-audio-out của PSTN với WebRTC. Giải thích sự khác biệt về bộ đệm jitter và codec.

5. Thêm tính năng phát hiện hoạt động giọng nói cho các ngôn ngữ không phải tiếng Anh (tiếng Nhật, tiếng Tây Ban Nha). Đo lường tỷ lệ kích hoạt sai của Silero VAD v5 so với các bản tinh chỉnh theo ngôn ngữ cụ thể.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|------------------------|
| Turn detection | "Kết thúc câu nói" | Bộ phân loại quyết định người dùng đã nói xong dựa trên sự im lặng của VAD và bản ghi một phần |
| Barge-in | "Xử lý ngắt quãng" | Hủy TTS giữa chừng khi VAD phát hiện giọng nói mới của người dùng |
| First-audio-out | "Độ trễ" | Thời gian từ khi người dùng ngừng nói đến gói âm thanh đầu tiên rời khỏi server |
| VAD | "Cổng giọng nói" | Mô hình phân loại khung âm thanh là giọng nói hay im lặng; Silero VAD v5 là mặc định năm 2026 |
| Jitter buffer | "Làm mượt âm thanh" | Bộ đệm phía client giữ các gói tin trong thời gian ngắn để hấp thụ sự biến thiên của mạng |
| Filler | "Token xác nhận" | Cụm từ ngắn tác nhân phát ra để tránh im lặng khi công cụ phản hồi chậm |
| MOS | "Điểm trung bình" | Đánh giá chất lượng giọng nói cảm nhận; NISQA là proxy tự động |

## Đọc thêm

- [LiveKit Agents 1.0](https://github.com/livekit/agents) — framework tác nhân WebRTC tham chiếu
- [Pipecat](https://github.com/pipecat-ai/pipecat) — framework tác nhân streaming ưu tiên Python thay thế
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — tham chiếu cho các mô hình giọng nói tích hợp
- [Deepgram Nova-3 documentation](https://developers.deepgram.com/docs) — tham chiếu ASR streaming
- [Silero VAD v5](https://github.com/snakers4/silero-vad) — mô hình VAD tham chiếu
- [Cartesia Sonic-2](https://docs.cartesia.ai) — tham chiếu TTS độ trễ thấp
- [Retell AI architecture](https://docs.retellai.com) — kiến trúc tác nhân giọng nói sản xuất
- [Vapi.ai production stack](https://docs.vapi.ai) — tham chiếu sản xuất thay thế