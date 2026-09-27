# Streaming Speech-to-Speech — Moshi, Hibiki và Full-Duplex Dialogue

> Giai đoạn 2024-2026 đã định nghĩa lại AI giọng nói. Moshi là một mô hình đơn nhất có khả năng nghe và nói đồng thời với độ trễ 200 ms. Hibiki thực hiện dịch speech-to-speech theo từng đoạn (chunk-by-chunk). Cả hai đều từ bỏ pipeline ASR → LLM → TTS truyền thống để chuyển sang kiến trúc full-duplex thống nhất dựa trên các token của codec Mimi. Đây chính là thiết kế tham chiếu mới.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 6 · 13 (Neural Audio Codecs), Phase 6 · 11 (Real-Time Audio), Phase 7 · 05 (Full Transformer)
**Time:** ~75 phút

## Vấn đề

Mọi voice agent được xây dựng từ Bài 11 + 12 đều có mức độ trễ cơ bản khoảng 300-500 ms: VAD kích hoạt, STT xử lý, LLM suy luận, TTS tạo âm thanh. Mỗi giai đoạn đều có độ trễ tối thiểu riêng. Bạn có thể tinh chỉnh và song song hóa, nhưng hình thái của pipeline vẫn giới hạn hiệu suất của bạn.

Moshi (Kyutai, 2024-2026) đặt ra một câu hỏi khác: điều gì sẽ xảy ra nếu không có pipeline nào cả? Điều gì sẽ xảy ra nếu một mô hình nhận âm thanh đầu vào và phát âm thanh đầu ra trực tiếp, liên tục, với văn bản đóng vai trò là "độc thoại nội tâm" (inner monologue) thay vì là một giai đoạn bắt buộc?

Câu trả lời là **full-duplex speech-to-speech**. Độ trễ lý thuyết là 160 ms (80 ms khung Mimi + 80 ms độ trễ âm học). Độ trễ thực tế là 200 ms trên một GPU L4 đơn lẻ. Con số này chỉ bằng một nửa so với những voice agent sử dụng pipeline tốt nhất hiện nay.

## Khái niệm

![Moshi architecture: two parallel Mimi streams + inner-monologue text](../assets/moshi-hibiki.svg)

### Kiến trúc Moshi

**Đầu vào.** Hai luồng codec Mimi, cả hai đều ở mức 12.5 Hz × 8 codebook:

- Luồng 1: âm thanh người dùng (được mã hóa Mimi, liên tục truyền đến)
- Luồng 2: âm thanh của chính Moshi (do Moshi tạo ra)

**Transformer.** Một Temporal Transformer với 7 tỷ tham số xử lý cả hai luồng và một luồng văn bản "độc thoại nội tâm". Tại mỗi bước 80 ms, nó:

1. Tiêu thụ các token Mimi mới nhất của người dùng (8 codebook).
2. Tiêu thụ các token Mimi gần nhất của Moshi (8 codebook, ngay khi được tạo ra).
3. Tạo ra token văn bản tiếp theo của Moshi (độc thoại nội tâm).
4. Tạo ra các token Mimi tiếp theo của Moshi (8 codebook thông qua một Depth Transformer nhỏ).

Cả ba luồng — âm thanh người dùng, âm thanh Moshi, văn bản Moshi — chạy song song. Moshi có thể nghe người dùng trong khi đang nói; có thể tự ngắt lời khi người dùng ngắt lời; có thể phản hồi phụ ("mhm") mà không làm gián đoạn câu nói chính.

**Depth transformer.** Trong một khung hình, 8 codebook không được dự đoán song song — chúng có sự phụ thuộc lẫn nhau. Một "depth transformer" 2 lớp nhỏ sẽ dự đoán chúng theo trình tự trong vòng 80 ms. Đây là phương pháp phân tách tiêu chuẩn cho các AR codec LM (cũng được sử dụng bởi VALL-E, VibeVoice).

### Tại sao văn bản "độc thoại nội tâm" lại hữu ích?

Nếu không có văn bản rõ ràng, mô hình phải tự mô hình hóa ngôn ngữ một cách ẩn trong luồng âm thanh. Điểm sáng của Moshi: ép mô hình phát ra các token văn bản cùng với âm thanh. Luồng văn bản về cơ bản là bản ghi chép những gì Moshi đang nói. Điều này cải thiện tính mạch lạc về ngữ nghĩa, giúp việc thay thế đầu (head) mô hình ngôn ngữ trở nên dễ dàng hơn và cung cấp bản ghi chép miễn phí.

### Hibiki: streaming speech-to-speech translation

Cùng một kiến trúc, được huấn luyện trên các cặp dịch thuật. Âm thanh nguồn vào, âm thanh ngôn ngữ đích ra, liên tục. Hibiki-Zero (tháng 2/2026) loại bỏ nhu cầu về dữ liệu huấn luyện căn chỉnh theo cấp độ từ — sử dụng dữ liệu cấp độ câu + học tăng cường GRPO để tối ưu hóa độ trễ.

Ban đầu hỗ trợ bốn cặp ngôn ngữ; có thể thích nghi với ngôn ngữ mới với khoảng 1000 giờ dữ liệu.

### Hệ sinh thái Kyutai (2026)

- **Moshi** — đối thoại full-duplex (tiếng Pháp trước, tiếng Anh được hỗ trợ tốt)
- **Hibiki / Hibiki-Zero** — dịch thuật giọng nói đồng thời
- **Kyutai STT** — streaming ASR (500 ms hoặc 2.5 s look-ahead)
- **Kyutai Pocket TTS** — TTS 100 triệu tham số chạy trên CPU (tháng 1/2026)
- **Unmute** — pipeline đầy đủ kết hợp các thành phần trên trên máy chủ công cộng

Thông lượng trên GPU L40S: 64 phiên đồng thời ở tốc độ 3× thời gian thực.

### Sesame CSM — người anh em họ

Sesame CSM (2025) sử dụng ý tưởng tương tự — xương sống Llama-3 với đầu codec Mimi. Nhưng CSM là đơn hướng (nhận ngữ cảnh + văn bản, tạo ra giọng nói) thay vì full-duplex. Đây là TTS có "sự hiện diện giọng nói" tốt nhất trên thị trường; không hoàn toàn giống với khả năng full-duplex của Moshi.

### Chỉ số hiệu năng năm 2026

| Mô hình | Độ trễ | Trường hợp sử dụng | Giấy phép |
|-------|---------|----------|---------|
| Moshi | 200 ms (L4) | Đối thoại full-duplex Anh / Pháp | CC-BY 4.0 |
| Hibiki | 12.5 Hz framerate | Dịch thuật streaming Pháp ↔ Anh | CC-BY 4.0 |
| Hibiki-Zero | tương tự | 5 cặp ngôn ngữ, không cần dữ liệu căn chỉnh | CC-BY 4.0 |
| Sesame CSM-1B | 200 ms TTFA | TTS có điều kiện ngữ cảnh | Apache-2.0 |
| GPT-4o Realtime | ~300 ms | đóng, OpenAI API | thương mại |
| Gemini 2.5 Live | ~350 ms | đóng, Google API | thương mại |

```figure
sp-fullduplex
```

## Xây dựng

### Bước 1: giao diện

Moshi cung cấp một máy chủ WebSocket nhận các đoạn âm thanh mã hóa Mimi 80 ms và trả về các đoạn âm thanh mã hóa Mimi 80 ms. Cả hai chiều. Liên tục.

```python
import asyncio
import websockets
from moshi.client_utils import encode_audio_mimi, decode_audio_mimi

async def moshi_chat():
    async with websockets.connect("ws://localhost:8998/api/chat") as ws:
        mic_task = asyncio.create_task(stream_mic_to(ws))
        spk_task = asyncio.create_task(stream_from_to_speaker(ws))
        await asyncio.gather(mic_task, spk_task)
```

### Bước 2: vòng lặp full-duplex

```python
async def stream_mic_to(ws):
    async for chunk_80ms in mic_stream_at_12_5_hz():
        mimi_tokens = encode_audio_mimi(chunk_80ms)
        await ws.send(serialize(mimi_tokens))

async def stream_from_to_speaker(ws):
    async for msg in ws:
        mimi_tokens, text_token = deserialize(msg)
        audio = decode_audio_mimi(mimi_tokens)
        await play(audio)
```

Cả hai chiều chạy đồng thời. Python asyncio hoặc Rust futures là phương thức truyền tải tiêu chuẩn.

### Bước 3: mục tiêu huấn luyện (khái niệm)

Đối với mỗi khung 80 ms `t`:

- Đầu vào: `user_mimi[0..t]`, `moshi_mimi[0..t-1]`, `moshi_text[0..t-1]`
- Dự đoán: `moshi_text[t]`, sau đó là `moshi_mimi[t, codebook_0..7]`

Văn bản được dự đoán trước âm thanh (độc thoại nội tâm); âm thanh được dự đoán theo trình tự codebook bên trong depth transformer.

### Bước 4: Moshi thắng ở đâu và không thắng ở đâu

Moshi thắng ở:

- Độ trễ end-to-end dưới 250 ms trên phần cứng giá rẻ.
- Phản hồi phụ và ngắt lời tự nhiên.
- Không cần mã kết nối (glue code) cho pipeline.

Moshi không thắng ở:

- Gọi công cụ (tool calling) (không được huấn luyện cho việc này; bạn cần một đường dẫn LLM riêng).
- Suy luận dài (Moshi là mô hình đối thoại khoảng 8B, không phải Claude/GPT-4).
- Độ chính xác thực tế trên các chủ đề chuyên biệt.
- Hầu hết các trường hợp sử dụng doanh nghiệp (vẫn sử dụng pipeline vào năm 2026).

## Sử dụng

| Tình huống | Lựa chọn |
|-----------|------|
| Trợ lý giọng nói độ trễ thấp nhất | Moshi |
| Cuộc gọi dịch thuật trực tiếp | Hibiki |
| Demo giọng nói / nghiên cứu | Moshi, CSM |
| Agent doanh nghiệp có công cụ | Pipeline (Bài 12), không phải Moshi |
| TTS giọng tùy chỉnh trong ngữ cảnh | Sesame CSM |
| Speech-to-speech, bất kỳ ngôn ngữ nào | GPT-4o Realtime hoặc Gemini 2.5 Live (thương mại) |

## Các cạm bẫy

- **Khả năng gọi công cụ hạn chế.** Moshi là mô hình đối thoại, không phải khung làm việc cho agent. Hãy kết hợp với pipeline nếu cần công cụ.
- **Điều kiện giọng nói cụ thể.** Moshi sử dụng một nhân vật được huấn luyện duy nhất; việc nhân bản giọng nói là một quá trình huấn luyện riêng biệt.
- **Độ phủ ngôn ngữ.** Tiếng Pháp + Anh rất xuất sắc; các ngôn ngữ khác còn hạn chế. Hibiki-Zero giúp ích, nhưng bạn vẫn cần dữ liệu huấn luyện.
- **Chi phí tài nguyên.** Một phiên Moshi đầy đủ chiếm một slot GPU; không phải là mô hình triển khai chia sẻ tài nguyên giá rẻ.

## Triển khai

Lưu dưới dạng `outputs/skill-duplex-pipeline.md`. Chọn giữa kiến trúc pipeline hoặc full-duplex cho workload voice-agent, kèm theo lý do.

## Bài tập

1. **Dễ.** Chạy `code/main.py`. Nó mô phỏng kiến trúc hai luồng + độc thoại nội tâm một cách biểu tượng.
2. **Trung bình.** Lấy Moshi từ HuggingFace, chạy máy chủ, kiểm tra một cuộc hội thoại. Đo độ trễ thực tế từ khi người dùng kết thúc câu nói đến khi Moshi bắt đầu phản hồi.
3. **Khó.** Lấy agent pipeline từ Bài 12 của bạn và so sánh độ trễ P50 với Moshi trên 20 câu kiểm tra khớp nhau. Viết báo cáo về trường hợp mà pipeline vẫn thắng về mặt kiến trúc.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| Full-duplex | Nghe và nói cùng lúc | Hai luồng âm thanh hoạt động đồng thời trên cùng một mô hình. |
| Inner monologue | Luồng văn bản của mô hình | Moshi phát ra các token văn bản cùng với đầu ra âm thanh. |
| Depth transformer | Bộ dự đoán liên codebook | Transformer nhỏ dự đoán 8 codebook trong một khung 80 ms. |
| Mimi | Codec của Kyutai | 12.5 Hz × 8 codebook; ngữ nghĩa + âm học; vận hành Moshi. |
| Streaming S2S | Audio → audio trực tiếp | Dịch thuật/đối thoại theo từng đoạn, không có giai đoạn pipeline. |
| Back-channeling | Phản ứng "Mhm" | Moshi có thể phát ra các xác nhận nhỏ mà không làm gián đoạn lượt nói. |

## Đọc thêm

- [Défossez et al. (2024). Moshi — speech-text foundation model](https://arxiv.org/html/2410.00037v2) — bài báo gốc.
- [Kyutai Labs (2026). Hibiki-Zero](https://arxiv.org/abs/2602.12345) — dịch thuật streaming không cần dữ liệu căn chỉnh.
- [Sesame (2025). Crossing the uncanny valley of voice](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice) — thông số kỹ thuật CSM.
- [Kyutai — Moshi repo](https://github.com/kyutai-labs/moshi) — cài đặt + máy chủ.
- [OpenAI — Realtime API](https://platform.openai.com/docs/guides/realtime) — đối thủ thương mại đóng.
- [Kyutai — Delayed Streams Modeling](https://github.com/kyutai-labs/delayed-streams-modeling) — khung STT/TTS bên dưới.