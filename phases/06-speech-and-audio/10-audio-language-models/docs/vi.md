# Audio-Language Models — Qwen2.5-Omni, Audio Flamingo, GPT-4o Audio

> Các mô hình audio-language năm 2026 có khả năng suy luận trên dữ liệu giọng nói + âm thanh môi trường + âm nhạc. Qwen2.5-Omni-7B đạt hiệu suất ngang ngửa GPT-4o Audio trên MMAU-Pro. Audio Flamingo Next vượt qua Gemini 2.5 Pro trên LongAudioBench. Khoảng cách giữa các mô hình mã nguồn mở và đóng về cơ bản đã được thu hẹp — ngoại trừ các tác vụ đa âm thanh (multi-audio), nơi tất cả các mô hình đều chỉ đạt kết quả gần như ngẫu nhiên.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 6 · 04 (ASR), Phase 12 · 03 (Vision-Language Models), Phase 7 · 10 (Audio Transformers)
**Time:** ~45 phút

## Vấn đề

Bạn có 5 giây âm thanh: tiếng chó sủa, ai đó hét lên "dừng lại!", sau đó là sự im lặng. Các câu hỏi hữu ích trải dài trên nhiều trục:

- **Chuyển văn bản (Transcription).** "Người đó đã nói gì?" — thuộc phạm vi của ASR.
- **Suy luận ngữ nghĩa (Semantic reasoning).** "Người đó có đang gặp nguy hiểm không?" — đòi hỏi sự hiểu biết kết hợp giữa tiếng chó sủa + tiếng hét + sự im lặng.
- **Suy luận âm nhạc (Music reasoning).** "Những nhạc cụ nào đang chơi giai điệu này?"
- **Truy xuất âm thanh dài (Long-audio retrieval).** "Trong bài giảng dài 90 phút này, giảng viên đã giải thích về gradient descent ở đoạn nào?"

Một mô hình duy nhất trả lời tất cả những câu hỏi này với một prompt được gọi là **audio-language model** (LALM / ALM). Khác với ASR thuần túy: LALM tạo ra các câu trả lời bằng ngôn ngữ tự nhiên tự do, không chỉ là bản ghi chép.

## Khái niệm

![Audio-language model: audio encoder + projector + LLM decoder](../assets/alm-architecture.svg)

### Cấu trúc ba thành phần

Mọi LALM năm 2026 đều có cùng một khung xương:

1. **Audio encoder.** Whisper encoder · BEATs · CLAP · WavLM · hoặc một encoder tùy chỉnh cho từng mô hình.
2. **Projector.** Lớp Linear hoặc MLP giúp kết nối các đặc trưng từ audio-encoder vào không gian nhúng (embedding space) của LLM.
3. **LLM.** Decoder dựa trên Llama / Qwen / Gemma. Tiếp nhận các token văn bản + âm thanh xen kẽ; tạo ra văn bản.

Quy trình huấn luyện:

- **Giai đoạn 1.** Đóng băng encoder + LLM; chỉ huấn luyện projector trên dữ liệu ASR / captioning.
- **Giai đoạn 2.** Fine-tune toàn bộ hoặc dùng LoRA trên các tác vụ âm thanh tuân thủ chỉ dẫn (QA, suy luận, hiểu âm nhạc).
- **Giai đoạn 3 (tùy chọn).** Voice-in / voice-out bổ sung thêm một speech decoder. Qwen2.5-Omni và AF3-Chat thực hiện điều này.

### Bản đồ các mô hình năm 2026

| Mô hình | Backbone | Audio encoder | Output modality | Truy cập |
|-------|----------|---------------|-----------------|--------|
| Qwen2.5-Omni-7B | Qwen2.5-7B | Tùy chỉnh + Whisper | văn bản + giọng nói | Apache-2.0 |
| Qwen3-Omni | Qwen3 | Tùy chỉnh | văn bản + giọng nói | Apache-2.0 |
| Audio Flamingo 3 | Qwen2 | AF-CLAP | văn bản | NVIDIA phi thương mại |
| Audio Flamingo Next | Qwen2 | AF-CLAP v2 | văn bản | NVIDIA phi thương mại |
| SALMONN | Vicuna | Whisper + BEATs | văn bản | Apache-2.0 |
| LTU / LTU-AS | Llama | CAV-MAE | văn bản | Apache-2.0 |
| GAMA | Llama | AST + Q-Former | văn bản | Apache-2.0 |
| Gemini 2.5 Flash/Pro (đóng) | Gemini | độc quyền | văn bản + giọng nói | API |
| GPT-4o Audio (đóng) | GPT-4o | độc quyền | văn bản + giọng nói | API |

### Kiểm chứng thực tế qua Benchmark (2026)

**MMAU-Pro.** 1800 cặp QA bao gồm giọng nói / âm thanh / âm nhạc / hỗn hợp. Đã bao gồm tập con đa âm thanh (multi-audio).

| Mô hình | Tổng thể | Giọng nói | Âm thanh | Âm nhạc | Đa âm thanh |
|-------|---------|--------|-------|-------|-------------|
| Gemini 2.5 Pro | ~60% | 73.4% | 51.9% | 64.9% | ~22% |
| Gemini 2.5 Flash | ~57% | 73.4% | 50.5% | 64.9% | 21.2% |
| GPT-4o Audio | 52.5% | — | — | — | 26.5% |
| Qwen2.5-Omni-7B | 52.2% | 57.4% | 47.6% | 61.5% | ~20% |
| Audio Flamingo 3 | ~54% | — | — | — | — |
| Audio Flamingo Next | SOTA trên LongAudioBench | — | — | — | — |

**Cột đa âm thanh là một thất bại đối với tất cả các mô hình.** Xác suất ngẫu nhiên cho câu hỏi trắc nghiệm 4 lựa chọn là 25%; hầu hết các mô hình đều đạt điểm quanh mức đó. LALM vẫn gặp khó khăn trong việc so sánh hai đoạn âm thanh.

### LALM hữu ích ở đâu trong năm 2026

- **Kiểm toán tuân thủ các bản ghi âm tổng đài.** "Nhân viên có đề cập đến thông tin bắt buộc phải tiết lộ không?"
- **Hỗ trợ tiếp cận.** Mô tả các sự kiện âm thanh cho người khiếm thính (không chỉ là chuyển văn bản).
- **Kiểm duyệt nội dung.** Phát hiện ngôn ngữ bạo lực + tông giọng đe dọa + bối cảnh nền.
- **Chia chương podcast / cuộc họp.** Tóm tắt ngữ nghĩa, không chỉ là lượt nói của người tham gia.
- **Phân tích danh mục âm nhạc.** "Tìm tất cả các bản nhạc có sự thay đổi khóa ở đoạn B-section."

### LALM CHƯA hữu ích ở đâu

- Lý thuyết âm nhạc chi tiết (dưới mức hợp âm).
- Suy luận dựa trên người nói trong các cuộc hội thoại dài (hiệu suất giảm sau 10 phút).
- So sánh đa âm thanh (22-26% chỉ nhỉnh hơn ngẫu nhiên một chút).
- Suy luận luồng thời gian thực (hầu hết là suy luận batch ngoại tuyến).

```figure
v4-alm-tokens
```

## Xây dựng

### Bước 1: truy vấn Qwen2.5-Omni

```python
from transformers import AutoModelForCausalLM, AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Omni-7B", torch_dtype="auto")

audio, sr = load_wav("clip.wav", sr=16000)
messages = [{
    "role": "user",
    "content": [
        {"type": "audio", "audio": audio},
        {"type": "text", "text": "What sounds do you hear, and what's happening?"},
    ],
}]
inputs = processor.apply_chat_template(messages, tokenize=True, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(output[0], skip_special_tokens=True))
```

### Bước 2: mô hình projector

```python
import torch.nn as nn

class AudioProjector(nn.Module):
    def __init__(self, audio_dim=1280, llm_dim=4096):
        super().__init__()
        self.down = nn.Linear(audio_dim, llm_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(llm_dim, llm_dim)

    def forward(self, audio_features):
        return self.up(self.act(self.down(audio_features)))
```

Chỉ vậy thôi. Projector thường là 1-3 lớp linear. Huấn luyện nó trên các cặp ASR (âm thanh → bản ghi) là tác vụ tiền đề của Giai đoạn 1.

### Bước 3: benchmark MMAU / LongAudioBench

```python
from datasets import load_dataset
mmau = load_dataset("MMAU/MMAU-Pro")

correct = 0
for item in mmau["test"]:
    answer = call_model(item["audio"], item["question"], item["choices"])
    if answer == item["correct_choice"]:
        correct += 1
print(f"Accuracy: {correct / len(mmau['test']):.3f}")
```

Báo cáo theo từng danh mục (giọng nói / âm thanh / âm nhạc / đa âm thanh) riêng biệt. Các con số tổng hợp thường che giấu những điểm yếu của mô hình.

## Sử dụng

| Tác vụ | Lựa chọn 2026 |
|------|-----------|
| Audio QA tự do (mở) | Qwen2.5-Omni-7B |
| Tốt nhất cho âm thanh dài (mở) | Audio Flamingo Next |
| Tốt nhất (đóng) | Gemini 2.5 Pro |
| Tác nhân voice-in / voice-out | Qwen2.5-Omni hoặc GPT-4o Audio |
| Suy luận âm nhạc | Audio Flamingo 3 hoặc 2 (AF-CLAP chuyên dụng âm nhạc) |
| Kiểm toán tổng đài | Gemini 2.5 Pro qua API, kết hợp RAG với tài liệu chính sách của bạn |

## Các cạm bẫy

- **Quá tin tưởng vào khả năng đa âm thanh.** Nếu tác vụ của bạn cần "đoạn clip nào có X", hiệu suất ở mức ngẫu nhiên là thực tế.
- **Suy giảm trên âm thanh dài.** Sau 10 phút, khả năng xác định người nói của hầu hết các mô hình đều bị suy giảm. Hãy thực hiện diarization trước (Bài 6), sau đó mới tóm tắt.
- **Ảo giác trên sự im lặng.** Vấn đề kiểu Whisper này vẫn tồn tại ở các LALM sử dụng Whisper encoder. Hãy dùng VAD-gate.
- **Chọn lọc benchmark.** Các bài đăng blog của nhà cung cấp thường làm nổi bật các danh mục tốt nhất. Hãy tự chạy tập con đa âm thanh của MMAU-Pro.

## Triển khai

Lưu dưới dạng `outputs/skill-alm-picker.md`. Chọn LALM + tập con benchmark + output-modality (văn bản so với giọng nói) cho một tác vụ hiểu âm thanh cụ thể.

## Bài tập

1. **Dễ.** Chạy `code/main.py` để xem mô hình projector đơn giản + cách định tuyến giả lập của LALM (audio-embedding, text-tokens) → output tokens.
2. **Trung bình.** Chấm điểm Qwen2.5-Omni-7B trên 100 mục giọng nói của MMAU-Pro. So sánh với con số được báo cáo trong bài báo.
3. **Khó.** Xây dựng một baseline captioning âm thanh tối giản: BEATs encoder + projector 2 lớp + Llama-3.2-1B đóng băng. Chỉ fine-tune projector trên AudioCaps. So sánh với SALMONN trên Clotho-AQA.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| LALM | Audio ChatGPT | Audio encoder + projector + LLM decoder. |
| Projector | Adapter | MLP nhỏ ánh xạ đặc trưng âm thanh vào không gian nhúng của LLM. |
| MMAU | Benchmark | 10k cặp audio-QA trên giọng nói, âm thanh, âm nhạc. |
| MMAU-Pro | MMAU khó hơn | 1800 câu hỏi đa âm thanh / suy luận chuyên sâu. |
| LongAudioBench | Đánh giá dài | Các đoạn clip dài nhiều phút với truy vấn ngữ nghĩa. |
| Voice-in / voice-out | Speech-native | Mô hình tiếp nhận giọng nói và phát ra giọng nói mà không cần chuyển qua văn bản. |

## Đọc thêm

- [Chu et al. (2024). Qwen2-Audio](https://arxiv.org/abs/2407.10759) — kiến trúc tham chiếu.
- [Alibaba (2025). Qwen2.5-Omni](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) — speech-in-speech-out.
- [NVIDIA (2025). Audio Flamingo 3](https://arxiv.org/abs/2507.08128) — dẫn đầu về âm thanh dài mã nguồn mở.
- [NVIDIA (2026). Audio Flamingo Next](https://arxiv.org/abs/2604.10905) — SOTA trên LongAudioBench.
- [Tang et al. (2023). SALMONN](https://arxiv.org/abs/2310.13289) — tiên phong về dual-encoder.
- [MMAU-Pro leaderboard](https://mmaubenchmark.github.io/) — bảng xếp hạng trực tiếp năm 2026.