# Whisper — Kiến trúc & Fine-Tuning

> Whisper là một mô hình transformer encoder-decoder với cửa sổ 30 giây, được huấn luyện trên 680 nghìn giờ dữ liệu âm thanh-văn bản đa ngôn ngữ theo phương pháp weakly-supervised. Một kiến trúc duy nhất, đa nhiệm, hoạt động mạnh mẽ trên 99 ngôn ngữ. Đây là tiêu chuẩn ASR tham chiếu cho năm 2026.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 04 (ASR), Phase 5 · 10 (Attention), Phase 7 · 05 (Full Transformer)
**Time:** ~75 phút

## Vấn đề

Whisper, được OpenAI ra mắt vào tháng 9 năm 2022, là mô hình ASR đầu tiên trở thành một sản phẩm thương mại phổ biến: dán âm thanh vào, nhận văn bản, hỗ trợ 99 ngôn ngữ, chống nhiễu tốt và chạy được trên laptop. Đến năm 2024, OpenAI đã phát hành các biến thể Large-v3 và Turbo; đến năm 2026, Whisper là baseline mặc định cho mọi thứ, từ chuyển đổi podcast thành văn bản đến trợ lý giọng nói và phụ đề YouTube.

Tuy nhiên, Whisper không phải là một pipeline mà bạn có thể coi là "hộp đen" mãi mãi. Domain shift (sự thay đổi miền dữ liệu) sẽ làm giảm hiệu năng của nó — thuật ngữ chuyên ngành, giọng địa phương, danh từ riêng, các đoạn clip ngắn, khoảng lặng. Bạn cần biết:

1. Bản chất bên trong của nó là gì.
2. Cách cung cấp âm thanh dạng chunked, streaming hoặc long-form một cách chính xác.
3. Khi nào cần fine-tune và thực hiện như thế nào.

## Khái niệm

![Whisper encoder-decoder, tasks, chunked inference, fine-tune](../assets/whisper.svg)

**Kiến trúc.** Transformer encoder-decoder tiêu chuẩn.

- Input: Log-mel spectrogram 30 giây, 80 mels, hop 10 ms → 3000 frames. Các clip ngắn hơn sẽ được zero-padded, clip dài hơn sẽ được chia nhỏ (chunked).
- Encoder: conv-downsample (stride 2) + `N` khối transformer. Với Large-v3: 32 lớp, 1280-dim, 20 heads.
- Decoder: `N` khối transformer với causal self-attn + cross-attn tới đầu ra của encoder. Kích thước tương đương encoder.
- Output: BPE tokens trên tập từ vựng 51.865 token.

Large-v3 có 1,55 tỷ tham số. Turbo sử dụng decoder 4 lớp (thay vì 32), giúp giảm độ trễ 8 lần với mức giảm WER <1%.

**Định dạng prompt.** Whisper là mô hình đa nhiệm được điều khiển bởi các token đặc biệt trong decoder prompt:

```
<|startoftranscript|><|en|><|transcribe|><|notimestamps|> Hello world.<|endoftext|>
```

- `<|en|>` — thẻ ngôn ngữ; ép buộc hành vi dịch thuật hoặc phiên âm.
- `<|transcribe|>` hoặc `<|translate|>` — dịch đầu ra sang tiếng Anh từ bất kỳ ngôn ngữ đầu vào nào, hoặc giữ nguyên văn.
- `<|notimestamps|>` — bỏ qua dấu thời gian cấp độ từ (nhanh hơn).

Prompt là thứ cho phép một mô hình thực hiện nhiều tác vụ. Thay đổi `<|en|>` thành `<|fr|>` và nó sẽ phiên âm tiếng Pháp.

**Cửa sổ 30 giây.** Mọi thứ đều được cố định ở 30 giây. Các clip dài hơn cần được chia nhỏ; clip ngắn hơn được đệm (padded). Các cửa sổ không được stream một cách tự nhiên — đây là lý do tại sao WhisperX, Whisper-Streaming và faster-whisper ra đời.

**Log-mel normalization.** `(log_mel - mean) / std` nơi các thông số thống kê được lấy từ chính tập dữ liệu huấn luyện của Whisper. Bạn *phải* sử dụng tiền xử lý của Whisper (`whisper.audio.log_mel_spectrogram`), không phải `librosa.feature.melspectrogram`.

### Các biến thể năm 2026

| Biến thể | Tham số | Độ trễ (A100) | WER (LibriSpeech-clean) |
|---------|--------|----------------|------------------------|
| Tiny | 39M | 1× realtime | 5.4% |
| Base | 74M | 1× | 4.1% |
| Small | 244M | 1× | 3.0% |
| Medium | 769M | 1× | 2.7% |
| Large-v3 | 1.55B | 2× | 1.8% |
| Large-v3-turbo | 809M | 8× | 1.58% |
| Whisper-Streaming (2024) | 1.55B | streaming | 2.0% |

### Fine-tuning

Quy trình chuẩn năm 2026:

1. Thu thập 10–100 giờ âm thanh thuộc miền mục tiêu cùng với bản ghi chép đã căn chỉnh.
2. Chạy `transformers.Seq2SeqTrainer` với callback `generate_with_loss`.
3. Hiệu quả tham số: LoRA trên `q_proj`, `k_proj`, `v_proj` của các lớp attention giúp giảm bộ nhớ GPU 4 lần với chi phí WER <0.3.
4. Đóng băng (freeze) encoder nếu bạn có dưới 10 giờ dữ liệu. Chỉ tune decoder.
5. Sử dụng tokenizer và định dạng prompt của chính Whisper; không bao giờ thay đổi tokenizer.

Kết quả từ cộng đồng: fine-tune Medium trên 20 giờ dữ liệu đọc y tế giúp giảm WER từ 12% xuống 4.5% đối với từ vựng y khoa. Fine-tune Turbo trên 4 giờ tiếng Iceland giúp giảm WER từ 18% xuống 6%.

```figure
sp-asr-attention
```

## Xây dựng

### Bước 1: Chạy Whisper cơ bản

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe(
    "clip.wav",
    language="en",
    task="transcribe",
    temperature=0.0,
    condition_on_previous_text=False,  # prevents runaway repetition
)
print(result["text"])
for seg in result["segments"]:
    print(f"[{seg['start']:.2f}–{seg['end']:.2f}] {seg['text']}")
```

Các mặc định quan trọng bạn luôn nên ghi đè: `temperature=0.0` (lấy mẫu mặc định là 0.0 → 0.2 → 0.4 … chuỗi dự phòng), `condition_on_previous_text=False` (ngăn chặn vấn đề ảo giác dây chuyền), và `no_speech_threshold=0.6` (phát hiện khoảng lặng).

### Bước 2: Xử lý long-form theo chunk

```python
# whisperx is the 2026 reference for long-form with word-level timestamps
import whisperx
model = whisperx.load_model("large-v3-turbo", device="cuda", compute_type="float16")
segments = model.transcribe("1hour.mp3", batch_size=16, chunk_size=30)
```

WhisperX bổ sung (1) Silero VAD gating, (2) căn chỉnh cấp độ từ thông qua wav2vec 2.0, (3) diarization thông qua `pyannote.audio`. Đây là công cụ chủ lực cho việc phiên âm sản xuất năm 2026.

### Bước 3: Fine-tune với LoRA

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import LoraConfig, get_peft_model

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo")
lora = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1, bias="none", task_type="SEQ_2_SEQ_LM",
)
model = get_peft_model(model, lora)
# model.print_trainable_parameters()  -> ~3M trainable / 809M total
```

Sau đó sử dụng vòng lặp Trainer tiêu chuẩn. Lưu checkpoint mỗi 1000 bước. Đánh giá bằng WER trên tập dữ liệu giữ lại (held-out).

### Bước 4: Kiểm tra những gì mỗi lớp học được

```python
# Grab cross-attention weights during decode to see what the decoder attends to.
with torch.inference_mode():
    out = model.generate(
        input_features=features,
        return_dict_in_generate=True,
        output_attentions=True,
    )
# out.cross_attentions: layer × head × step × src_len
```

Trực quan hóa bằng heatmap — bạn sẽ thấy sự căn chỉnh đường chéo khi các bước decoder quét qua các khung encoder. Đường chéo đó chính là khái niệm về dấu thời gian từ của Whisper.

## Sử dụng

Stack công nghệ năm 2026:

| Tình huống | Lựa chọn |
|-----------|------|
| Tiếng Anh phổ thông, offline | Large-v3-turbo qua `whisperx` |
| Mobile / edge | Whisper-Tiny lượng tử hóa (int8) hoặc Moonshine |
| Đa ngôn ngữ long-form | Large-v3 qua `whisperx` + diarization |
| Ngôn ngữ ít tài nguyên | Fine-tune Medium hoặc Turbo với LoRA |
| Streaming (độ trễ 2 giây) | Whisper-Streaming hoặc Parakeet-TDT |
| Dấu thời gian cấp độ từ | WhisperX (căn chỉnh cưỡng bức qua wav2vec 2.0) |

`faster-whisper` (backend CTranslate2) là runtime suy luận CPU+GPU nhanh nhất năm 2026 — nhanh hơn 4 lần so với bản gốc với đầu ra giống hệt.

## Những cạm bẫy vẫn tồn tại trong năm 2026

- **Văn bản ảo giác trên khoảng lặng.** Whisper được huấn luyện trên các phụ đề bao gồm "Cảm ơn đã xem!", "Đăng ký!", lời bài hát. Luôn sử dụng VAD-gate trước khi gọi mô hình.
- **Chuỗi `condition_on_previous_text`.** Một ảo giác sẽ làm ô nhiễm các cửa sổ tiếp theo. Thiết lập `False` trừ khi bạn cần sự trôi chảy giữa các đoạn.
- **Đệm clip ngắn.** Một clip 2 giây được đệm lên 30 giây có thể tạo ra ảo giác trong khoảng lặng phía sau. Sử dụng `pad=False` hoặc VAD-gate.
- **Thông số mel sai.** Sử dụng mels của librosa thay vì của Whisper tạo ra đầu ra gần như ngẫu nhiên. Sử dụng `whisper.audio.log_mel_spectrogram`.

## Triển khai

Lưu dưới dạng `outputs/skill-whisper-tuner.md`. Thiết kế một pipeline fine-tune hoặc suy luận Whisper cho một miền dữ liệu cụ thể.

## Bài tập

1. **Dễ.** Chạy `code/main.py`. Nó thực hiện token hóa một prompt kiểu Whisper, tính toán ngân sách hình dạng giải mã và in lịch trình chunk cho một clip 10 phút.
2. **Trung bình.** Cài đặt `faster-whisper`, phiên âm một podcast 10 phút, so sánh WER với bản ghi chép của con người. Thử `language="auto"` so với `language="en"` cưỡng bức.
3. **Khó.** Sử dụng HF `datasets`, chọn một ngôn ngữ mà Whisper gặp khó khăn (ví dụ: tiếng Urdu), fine-tune Medium với LoRA trong 2 epoch trên 2 giờ dữ liệu và báo cáo mức chênh lệch WER.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| 30-sec window | Giới hạn của Whisper | Giới hạn đầu vào cứng; chia nhỏ âm thanh dài hơn. |
| SOT | Start-of-transcript | `<\|startoftranscript\|>` khởi động prompt của decoder. |
| Timestamps token | Căn chỉnh thời gian | Mỗi độ lệch 0.02 giây là một token đặc biệt trong từ vựng 51k. |
| Turbo | Biến thể nhanh | 4 lớp decoder, nhanh hơn 8 lần, giảm WER <1%. |
| WhisperX | Wrapper cho long-form | VAD + Whisper + căn chỉnh wav2vec + diarization. |
| LoRA fine-tune | Tinh chỉnh hiệu quả | Thêm các adapter hạng thấp vào attention; huấn luyện ~0.3% tham số. |
| Hallucination | Lỗi im lặng | Whisper tạo ra tiếng Anh trôi chảy từ nhiễu/khoảng lặng. |

## Đọc thêm

- [Radford et al. (2022). Whisper paper](https://arxiv.org/abs/2212.04356) — kiến trúc gốc và công thức huấn luyện.
- [OpenAI (2024). Whisper Large-v3-turbo release](https://github.com/openai/whisper/discussions/2363) — decoder 4 lớp, tăng tốc 8 lần.
- [Bain et al. (2023). WhisperX](https://arxiv.org/abs/2303.00747) — long-form, căn chỉnh từ, diarized.
- [Systran — faster-whisper repo](https://github.com/SYSTRAN/faster-whisper) — hỗ trợ CTranslate2, nhanh hơn 4 lần.
- [HuggingFace — Whisper fine-tune tutorial](https://huggingface.co/blog/fine-tune-whisper) — hướng dẫn LoRA / full-FT chuẩn.