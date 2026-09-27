# Nhận dạng giọng nói (ASR) — CTC, RNN-T, Attention

> Nhận dạng giọng nói là quá trình phân loại âm thanh tại mỗi bước thời gian (timestep), được kết nối bởi một mô hình trình tự hiểu được ngôn ngữ và khoảng lặng. CTC, RNN-T và Attention là ba phương pháp để thực hiện việc này. Hãy chọn một và hiểu lý do tại sao.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms & Mel), Phase 5 · 08 (CNNs & RNNs for Text), Phase 5 · 10 (Attention)
**Time:** ~45 phút

## Vấn đề

Bạn có một đoạn clip 10 giây với tần số 16 kHz. Bạn muốn nhận được một chuỗi văn bản: "turn on the kitchen lights". Thách thức nằm ở cấu trúc: các khung âm thanh (audio frames) không khớp một-một với các ký tự. Từ "okay" có thể mất 200 ms hoặc 1200 ms. Khoảng lặng ngắt quãng giữa các câu nói. Một số âm vị dài hơn những âm vị khác. Số lượng token đầu ra không được biết trước.

Ba công thức giải quyết vấn đề này:

1. **CTC (Connectionist Temporal Classification).** Phát ra xác suất token trên mỗi khung hình bao gồm một token *blank* đặc biệt. Hợp nhất các ký tự lặp lại và các khoảng trống (blank) tại thời điểm giải mã. Không tự hồi quy (non-autoregressive), tốc độ nhanh. Được sử dụng bởi wav2vec 2.0, MMS.
2. **RNN-T (Recurrent Neural Network Transducer).** Mạng kết hợp (joint network) dự đoán token tiếp theo dựa trên khung hình từ encoder và các token trước đó. Có khả năng stream. Được sử dụng bởi ASR trên thiết bị của Google, NVIDIA Parakeet.
3. **Attention encoder-decoder.** Encoder nén âm thanh thành các trạng thái ẩn (hidden states), decoder sử dụng cross-attention để tạo token một cách tự hồi quy (autoregressive). Được sử dụng bởi Whisper, SeamlessM4T.

Vào năm 2026, SOTA WER trên LibriSpeech test-clean là 1.4% (Parakeet-TDT-1.1B, NVIDIA) và 1.58% (Whisper-Large-v3-turbo). Sự khác biệt là rất nhỏ; nhưng sự khác biệt trong triển khai thực tế lại rất lớn.

## Khái niệm

![Three ASR formulations: CTC, RNN-T, attention-encoder-decoder](../assets/asr-formulations.svg)

**Trực giác về CTC.** Giả sử encoder xuất ra `T` phân phối cấp khung hình trên `V+1` token (V ký tự + blank). Đối với một chuỗi mục tiêu `y` có độ dài `U < T`, bất kỳ sự căn chỉnh khung hình nào hợp nhất thành `y` đều được tính. Hàm mất mát CTC tính tổng trên tất cả các căn chỉnh như vậy. Suy luận: argmax trên mỗi khung hình, hợp nhất các ký tự lặp lại, loại bỏ các blank.

Ưu điểm: không tự hồi quy, có thể stream, không cần nhìn trước (zero lookahead). Nhược điểm: *giả định độc lập có điều kiện* — mỗi dự đoán khung hình độc lập với các khung hình khác, do đó không có mô hình ngôn ngữ nội tại. Khắc phục bằng LM bên ngoài thông qua beam search hoặc shallow fusion.

**Trực giác về RNN-T.** Thêm một mạng *predictor* nhúng lịch sử token và một mạng *joiner* kết hợp trạng thái predictor với khung hình encoder thành một phân phối chung trên `V+1` (trong đó `+1` là null / không phát ra gì). Mô hình hóa rõ ràng sự phụ thuộc có điều kiện mà CTC đã bỏ qua. Có khả năng stream vì mỗi bước chỉ phụ thuộc vào các khung hình quá khứ và các token quá khứ.

Ưu điểm: có thể stream + LM nội tại. Nhược điểm: huấn luyện phức tạp hơn và tốn bộ nhớ (lưới mất mát 3D); các kernel mất mát RNN-T là một danh mục thư viện riêng biệt.

**Attention encoder-decoder.** Encoder (6-32 lớp transformer) trên các khung log-mel. Decoder (6-32 lớp transformer) sử dụng cross-attention vào đầu ra của encoder để tạo token một cách tự hồi quy. Không có ràng buộc căn chỉnh — attention có thể nhìn vào bất kỳ đâu trong âm thanh. Không thể stream trừ khi bạn giới hạn attention (chunked Whisper-Streaming, 2024).

Ưu điểm: chất lượng cao nhất trên ASR offline, dễ huấn luyện với các công cụ seq2seq tiêu chuẩn. Nhược điểm: độ trễ tự hồi quy tỷ lệ thuận với độ dài đầu ra; không thể stream nếu không có kỹ thuật bổ sung.

### WER: con số duy nhất

**Word Error Rate** = `(S + D + I) / N`, trong đó S=thay thế (substitutions), D=xóa (deletions), I=chèn (insertions), N=số lượng từ tham chiếu. Khớp với khoảng cách chỉnh sửa Levenshtein ở cấp độ từ. Chỉ số càng thấp càng tốt. WER trên 20% thường không thể sử dụng; dưới 5% là ngang bằng với con người đối với giọng đọc. Các con số năm 2026 trên các benchmark tiêu chuẩn:

| Mô hình | LibriSpeech test-clean | LibriSpeech test-other | Kích thước |
|-------|------------------------|------------------------|------|
| Parakeet-TDT-1.1B | 1.40% | 2.78% | 1.1B tham số |
| Whisper-Large-v3-turbo | 1.58% | 3.03% | 809M |
| Canary-1B Flash | 1.48% | 2.87% | 1B |
| Seamless M4T v2 | 1.7% | 3.5% | 2.3B |

Tất cả các mô hình này đều dựa trên encoder-decoder hoặc RNN-T. Các hệ thống CTC thuần túy (wav2vec 2.0) nằm ở khoảng 1.8–2.1% trên test-clean.

```figure
ctc-collapse
```

## Xây dựng

### Bước 1: giải mã CTC tham lam (greedy)

```python
def ctc_greedy(frame_logits, blank=0, vocab=None):
    # frame_logits: list of per-frame probability vectors
    preds = [max(range(len(p)), key=lambda i: p[i]) for p in frame_logits]
    out = []
    prev = -1
    for p in preds:
        if p != prev and p != blank:
            out.append(p)
        prev = p
    return "".join(vocab[i] for i in out) if vocab else out
```

Hai quy tắc: hợp nhất các ký tự lặp lại liên tiếp, loại bỏ các blank. Ví dụ: `a a _ _ a b b _ c` → `a a b c`.

### Bước 2: beam-search CTC

```python
def ctc_beam(frame_logits, beam=8, blank=0):
    import math
    beams = [([], 0.0)]  # (tokens, log_prob)
    for p in frame_logits:
        log_p = [math.log(max(pi, 1e-10)) for pi in p]
        candidates = []
        for seq, lp in beams:
            for t, lpt in enumerate(log_p):
                new = seq[:] if t == blank else (seq + [t] if not seq or seq[-1] != t else seq)
                candidates.append((new, lp + lpt))
        candidates.sort(key=lambda x: -x[1])
        beams = candidates[:beam]
    return beams[0][0]
```

Trong sản xuất, người ta sử dụng tìm kiếm beam trên cây tiền tố (prefix tree) với LM fusion; đây là khung khái niệm.

### Bước 3: WER

```python
def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        dp[i][0] = i
    for j in range(len(h) + 1):
        dp[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[len(r)][len(h)] / max(1, len(r))
```

### Bước 4: suy luận với Whisper

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("clip.wav")
print(result["text"])
```

Một dòng lệnh cho ASR tổng quát mạnh mẽ nhất vào năm 2026. Chạy trên GPU 24 GB với tốc độ ~20× thời gian thực.

### Bước 5: streaming với Parakeet hoặc wav2vec 2.0

```python
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="nvidia/parakeet-tdt-1.1b")
for chunk in streaming_audio():
    print(asr(chunk, return_timestamps=True))
```

ASR streaming cần attention encoder theo từng đoạn (chunked) và trạng thái lưu trữ (carryover state); hãy sử dụng thư viện hỗ trợ (NeMo cho Parakeet, `transformers` pipeline với `chunk_length_s`).

## Sử dụng

Stack công nghệ năm 2026:

| Tình huống | Lựa chọn |
|-----------|------|
| Tiếng Anh, offline, chất lượng tối đa | Whisper-large-v3-turbo |
| Đa ngôn ngữ, mạnh mẽ | SeamlessM4T v2 |
| Streaming, độ trễ thấp | Parakeet-TDT-1.1B hoặc Riva |
| Edge, di động, độ trễ <500 ms | Whisper-Tiny đã lượng tử hóa hoặc Moonshine (2024) |
| Dạng dài (Long-form) | Whisper với chunking dựa trên VAD (WhisperX) |
| Chuyên ngành (y tế, pháp lý) | Fine-tune wav2vec 2.0 + LM fusion chuyên ngành |

## Những cạm bẫy vẫn tồn tại vào năm 2026

- **Không có VAD.** Chạy Whisper trên khoảng lặng tạo ra các ảo giác ("Thanks for watching!"). Luôn chặn bằng VAD.
- **WER ký tự vs từ vs subword.** Báo cáo WER cấp độ từ *sau khi* chuẩn hóa (chuyển thường, loại bỏ dấu câu).
- **Lệch ID ngôn ngữ.** Tính năng tự động nhận diện ngôn ngữ (auto LID) của Whisper có thể định tuyến sai các clip nhiễu sang tiếng Nhật hoặc tiếng Wales; hãy ép buộc `language="en"` khi bạn đã biết ngôn ngữ.
- **Clip dài không chia đoạn.** Whisper có cửa sổ 30 giây. Sử dụng `chunk_length_s=30, stride=5` cho bất kỳ nội dung nào dài hơn.

## Triển khai

Lưu dưới dạng `outputs/skill-asr-picker.md`. Chọn mô hình, chiến lược giải mã, chia đoạn và LM fusion cho mục tiêu triển khai cụ thể.

## Bài tập

1. **Dễ.** Chạy `code/main.py`. Nó giải mã tham lam một đầu ra CTC thủ công và tính toán WER so với tham chiếu.
2. **Trung bình.** Triển khai tìm kiếm beam trên cây tiền tố trong Bước 2 một cách chính xác (tính đến quy tắc hợp nhất blank). So sánh với giải mã tham lam trên tập dữ liệu tổng hợp gồm 10 ví dụ.
3. **Khó.** Sử dụng `whisper-large-v3-turbo` trên [LibriSpeech test-clean](https://www.openslr.org/12). Tính WER trên 100 câu đầu tiên. So sánh với các con số đã công bố.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| CTC | Hàm mất mát token blank | Biên trên tất cả các căn chỉnh khung-đến-token; không tự hồi quy. |
| RNN-T | Hàm mất mát streaming | CTC + dự đoán token tiếp theo; xử lý thứ tự từ. |
| Attention enc-dec | Kiểu Whisper | Encoder + decoder cross-attention; chất lượng offline tốt nhất. |
| WER | Con số bạn báo cáo | `(S+D+I)/N` ở cấp độ từ. |
| Blank | Sự trống rỗng | Token đặc biệt trong CTC báo hiệu "không phát ra gì trong khung này". |
| LM fusion | Mô hình ngôn ngữ bên ngoài | Thêm log-probs LM có trọng số trong quá trình beam search. |
| VAD | Cổng khoảng lặng | Bộ phát hiện hoạt động giọng nói; cắt bỏ phần không phải giọng nói. |

## Đọc thêm

- [Graves et al. (2006). Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf) — bài báo gốc về CTC.
- [Graves (2012). Sequence Transduction with RNNs](https://arxiv.org/abs/1211.3711) — bài báo gốc về RNN-T.
- [Radford et al. / OpenAI (2022). Whisper: Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — bài báo chuẩn năm 2022; mở rộng v3-turbo năm 2024.
- [NVIDIA NeMo — Parakeet-TDT card](https://huggingface.co/nvidia/parakeet-tdt-1.1b) — dẫn đầu bảng xếp hạng Open ASR 2026.
- [Hugging Face — Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) — bảng xếp hạng trực tiếp trên hơn 25 mô hình.