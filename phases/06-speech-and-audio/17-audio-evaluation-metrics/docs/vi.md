# Đánh giá âm thanh — WER, MOS, UTMOS, MMAU, FAD và các bảng xếp hạng mở

> Bạn không thể triển khai những gì bạn không thể đo lường. Bài học này liệt kê các chỉ số năm 2026 cho mọi tác vụ âm thanh: ASR (WER, CER, RTFx), TTS (MOS, UTMOS, SECS, WER-on-ASR-round-trip), audio-language (MMAU, LongAudioBench), âm nhạc (FAD, CLAP) và người nói (EER). Cùng với đó là các bảng xếp hạng nơi bạn có thể so sánh.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 6 · 04, 06, 07, 09, 10; Phase 2 · 09 (Model Evaluation)
**Time:** ~60 phút

## Vấn đề

Mỗi tác vụ âm thanh đều có nhiều chỉ số, mỗi chỉ số đo lường một khía cạnh khác nhau. Sử dụng sai chỉ số là cách nhanh nhất để triển khai một mô hình trông có vẻ tuyệt vời trên bảng điều khiển nhưng lại hoạt động tệ hại trong thực tế. Danh sách chuẩn năm 2026:

| Tác vụ | Chính | Phụ |
|------|---------|-----------|
| ASR | WER | CER · RTFx · độ trễ token đầu tiên |
| TTS | MOS / UTMOS | SECS · WER-on-ASR-round-trip · CER · TTFA |
| Nhân bản giọng nói | SECS (ECAPA cosine) | MOS · CER |
| Xác thực người nói | EER | minDCF · FAR / FRR tại điểm vận hành |
| Phân đoạn hội thoại | DER | JER · nhầm lẫn người nói |
| Phân loại âm thanh | top-1 · mAP | macro F1 · recall theo lớp |
| Tạo nhạc | FAD | CLAP · MOS từ hội đồng nghe |
| Mô hình ngôn ngữ âm thanh | MMAU-Pro | LongAudioBench · AudioCaps FENSE |
| Streaming S2S | độ trễ P50/P95 | WER · MOS |

## Khái niệm

![Audio evaluation matrix — metrics vs tasks vs 2026 leaderboards](../assets/eval-landscape.svg)

### Các chỉ số ASR

**WER (Word Error Rate).** `(S + D + I) / N`. Chuyển về chữ thường, loại bỏ dấu câu, chuẩn hóa số trước khi tính điểm. Sử dụng `jiwer` hoặc `whisper_normalizer` của OpenAI. &lt; 5% = tương đương con người đối với giọng đọc.

**CER (Character Error Rate).** Công thức tương tự, ở cấp độ ký tự. Được sử dụng cho các ngôn ngữ có thanh điệu (tiếng Quan Thoại, tiếng Quảng Đông) nơi việc phân tách từ ngữ không rõ ràng.

**RTFx (hệ số thời gian thực nghịch đảo).** Số giây âm thanh được xử lý trên mỗi giây đồng hồ thực tế. Chỉ số càng cao càng tốt. Parakeet-TDT đạt 3380×. Whisper-large-v3 đạt ~30×.

**Độ trễ token đầu tiên.** Thời gian từ khi nhập âm thanh đến khi có token văn bản đầu tiên. Rất quan trọng cho streaming. Deepgram Nova-3: ~150 ms.

### Các chỉ số TTS

**MOS (Mean Opinion Score).** Đánh giá của con người từ 1-5. Tiêu chuẩn vàng nhưng chậm. Thu thập 20+ người nghe mỗi mẫu, 100+ mẫu mỗi mô hình.

**UTMOS (2022-2026).** Bộ dự đoán MOS đã qua huấn luyện. Tương quan ~0.9 với MOS của con người trên các benchmark tiêu chuẩn. F5-TTS: UTMOS 3.95; ground truth: 4.08.

**SECS (Speaker Encoder Cosine Similarity).** Dành cho nhân bản giọng nói. Cosine của embedding ECAPA giữa mẫu tham chiếu và đầu ra nhân bản. &gt; 0.75 = bản sao có thể nhận diện được.

**WER-on-ASR-round-trip.** Chạy Whisper trên đầu ra TTS, tính WER so với văn bản đầu vào. Giúp phát hiện sự suy giảm về độ rõ nét. 2026 SOTA: &lt; 2% CER.

**TTFA (time-to-first-audio).** Độ trễ thời gian thực. Kokoro-82M: ~100 ms; F5-TTS: ~1 s.

### Đặc thù cho nhân bản giọng nói

**SECS + MOS + CER** như một bộ ba. Nhân bản đạt SECS cao nhưng MOS thấp nghĩa là đúng âm sắc nhưng không tự nhiên; ngược lại nghĩa là giọng tự nhiên nhưng sai người nói.

### Xác thực người nói

**EER (Equal Error Rate).** Ngưỡng nơi tỷ lệ chấp nhận sai (False Accept Rate) bằng tỷ lệ từ chối sai (False Reject Rate). ECAPA trên VoxCeleb1-O: 0.87%.

**minDCF (min Detection Cost).** Chi phí có trọng số tại một điểm vận hành đã chọn (thường là FAR=0.01). Liên quan đến thực tế sản xuất hơn EER.

### Phân đoạn hội thoại (Diarization)

**DER (Diarization Error Rate).** `(FA + Miss + Confusion) / total_speaker_time`. Bao gồm: bỏ sót giọng nói + báo động giả giọng nói + nhầm lẫn người nói, mỗi loại là một phần nhỏ. Các cuộc họp AMI: DER ~10-20% là thực tế. pyannote 3.1 + Precision-2 thương mại: &lt;10% DER trên âm thanh ghi âm tốt.

**JER (Jaccard Error Rate).** Thay thế cho DER, mạnh mẽ hơn với các phân đoạn ngắn.

### Phân loại âm thanh

Đa nhãn: **mAP (mean Average Precision)** trên tất cả các lớp. AudioSet: 0.548 mAP cho BEATs-iter3.

Đa lớp độc quyền: **top-1, top-5 accuracy**. Speech Commands v2: 99.0% top-1 (Audio-MAE).

Mất cân bằng: **macro F1** + **recall theo lớp**. Báo cáo theo từng lớp — độ chính xác tổng thể che giấu các lớp bị lỗi.

### Tạo nhạc

**FAD (Fréchet Audio Distance).** Khoảng cách giữa các phân phối embedding VGGish của âm thanh thực so với âm thanh được tạo ra. MusicGen-small trên MusicCaps: 4.5. MusicLM: 4.0. Chỉ số thấp hơn là tốt hơn.

**CLAP Score.** Điểm căn chỉnh văn bản-âm thanh sử dụng embedding CLAP. &gt; 0.3 = căn chỉnh hợp lý.

**Listening panel MOS.** Vẫn là tiếng nói cuối cùng cho âm nhạc thương mại. Suno v5 ELO 1293 trên TTS Arena (từ các ưu tiên của con người theo cặp).

### Benchmark ngôn ngữ âm thanh

**MMAU (Massive Multi-Audio Understanding).** 10k cặp audio-QA.

**MMAU-Pro.** 1800 mục khó, bốn danh mục: lời nói / âm thanh / âm nhạc / đa âm thanh. Xác suất ngẫu nhiên 25% cho 4 lựa chọn. Gemini 2.5 Pro tổng thể ~60%; đa âm thanh ~22% trên tất cả các mô hình.

**LongAudioBench.** Các đoạn clip dài nhiều phút với các truy vấn ngữ nghĩa. Audio Flamingo Next vượt qua Gemini 2.5 Pro.

**AudioCaps / Clotho.** Các benchmark chú thích. Các chỉ số SPICE, CIDEr, FENSE.

### Streaming speech-to-speech

**Độ trễ P50 / P95 / P99.** Thời gian thực từ khi người dùng kết thúc nói đến phản hồi âm thanh đầu tiên. Moshi: 200 ms; GPT-4o Realtime: 300 ms.

**WER / MOS** trên đầu ra.

**Khả năng ngắt lời (Barge-in).** Thời gian từ khi người dùng ngắt lời đến khi trợ lý im lặng. Mục tiêu &lt; 150 ms.

### Các bảng xếp hạng năm 2026

| Bảng xếp hạng | Tracks | URL |
|------------|--------|-----|
| Open ASR Leaderboard (HF) | Tiếng Anh + đa ngôn ngữ + dạng dài | `huggingface.co/spaces/hf-audio/open_asr_leaderboard` |
| TTS Arena (HF) | TTS tiếng Anh | `huggingface.co/spaces/TTS-AGI/TTS-Arena` |
| Artificial Analysis Speech | TTS + STT, ELO từ các phiếu bầu theo cặp | `artificialanalysis.ai/speech` |
| MMAU-Pro | LALM reasoning | `mmaubenchmark.github.io` |
| SpeakerBench / VoxSRC | Nhận diện người nói | `voxsrc.github.io` |
| MMAU music subset | Music LALM | (trong MMAU) |
| HEAR benchmark | Âm thanh tự giám sát | `hearbenchmark.com` |

```figure
sp-wer-align
```

## Xây dựng

### Bước 1: WER với chuẩn hóa

```python
from jiwer import wer, Compose, ToLowerCase, RemovePunctuation, Strip

transform = Compose([ToLowerCase(), RemovePunctuation(), Strip()])
score = wer(
    truth="Please turn on the lights.",
    hypothesis="please turn on the light",
    truth_transform=transform,
    hypothesis_transform=transform,
)
# ~0.17
```

### Bước 2: TTS round-trip WER

```python
def ttr_wer(tts_model, asr_model, texts):
    errors = []
    for txt in texts:
        audio = tts_model.synthesize(txt)
        recog = asr_model.transcribe(audio)
        errors.append(wer(truth=txt, hypothesis=recog))
    return sum(errors) / len(errors)
```

### Bước 3: SECS cho nhân bản giọng nói

```python
from speechbrain.inference.speaker import EncoderClassifier
sv = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb")

emb_ref = sv.encode_batch(load_wav("reference.wav"))
emb_clone = sv.encode_batch(load_wav("cloned.wav"))
secs = torch.nn.functional.cosine_similarity(emb_ref, emb_clone, dim=-1).item()
```

### Bước 4: FAD cho tạo nhạc

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()
score = fad.get_fad_score("generated_folder/", "reference_folder/")
```

### Bước 5: EER cho xác thực người nói (mã tương tự Bài 6)

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        frr = sum(1 for s in same_scores if s < t) / len(same_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

## Sử dụng

Kết hợp mỗi lần triển khai với một bộ đánh giá cố định chạy trên mỗi bản cập nhật mô hình. Ba quy tắc cốt lõi:

1. **Chuẩn hóa trước khi tính điểm.** Chuyển chữ thường, loại bỏ dấu câu, mở rộng số. Báo cáo quy tắc chuẩn hóa.
2. **Báo cáo phân phối, không phải trung bình.** P50/P95/P99 cho độ trễ. Recall theo lớp cho phân loại. Theo danh mục cho MMAU.
3. **Chạy một benchmark công khai chuẩn.** Ngay cả khi dữ liệu sản xuất của bạn khác biệt, việc báo cáo trên Open ASR / TTS Arena / MMAU cho phép người đánh giá so sánh một cách công bằng.

## Cạm bẫy

- **Ngoại suy UTMOS.** Được huấn luyện trên giọng nói sạch kiểu VCTK; chấm điểm kém cho âm thanh ồn / nhân bản / cảm xúc.
- **Thiên kiến hội đồng MOS.** 20 người làm trên Amazon Mechanical Turk ≠ 20 người dùng mục tiêu. Hãy trả phí cho một hội đồng chuyên môn nếu rủi ro cao.
- **FAD phụ thuộc vào tập tham chiếu.** So sánh với cùng một phân phối tham chiếu trên các mô hình.
- **WER tổng hợp.** WER 5% tổng thể có thể che giấu WER 30% trên giọng nói có ngữ điệu. Hãy báo cáo theo phân khúc nhân khẩu học.
- **Bão hòa benchmark công khai.** Hầu hết các mô hình tiên phong đều gần đạt trần trên các benchmark tiêu chuẩn. Hãy xây dựng một tập dữ liệu giữ lại (held-out set) nội bộ phản ánh lưu lượng truy cập của bạn.

## Triển khai

Lưu dưới dạng `outputs/skill-audio-evaluator.md`. Chọn các chỉ số, benchmark và định dạng báo cáo cho bất kỳ bản phát hành mô hình âm thanh nào.

## Bài tập

1. **Dễ.** Chạy `code/main.py`. Tính WER / CER / EER / SECS / FAD-ish / MMAU-ish trên các đầu vào mẫu.
2. **Trung bình.** Xây dựng bộ đánh giá TTS round-trip WER. Chạy đầu ra Kokoro hoặc F5-TTS của bạn qua Whisper. Tính WER trên 50 câu lệnh. Đánh dấu các câu lệnh có WER &gt; 10%.
3. **Khó.** Chấm điểm mô hình LALM bạn đã chọn ở Bài 10 trên các tập con speech + multi-audio của MMAU-Pro (mỗi tập 50 mục). Báo cáo độ chính xác theo danh mục và so sánh với con số đã công bố.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| WER | Điểm ASR | `(S+D+I)/N` ở cấp độ từ sau khi chuẩn hóa. |
| CER | WER ký tự | Cho ngôn ngữ có thanh điệu hoặc hệ thống cấp ký tự. |
| MOS | Ý kiến con người | Đánh giá 1-5; 20+ người nghe × 100 mẫu. |
| UTMOS | Bộ dự đoán MOS ML | Mô hình đã học; tương quan ~0.9 với MOS con người. |
| SECS | Độ tương đồng nhân bản | Cosine ECAPA giữa tham chiếu và bản sao. |
| EER | Điểm xác thực người nói | Ngưỡng nơi FAR = FRR. |
| DER | Điểm phân đoạn | (FA + Miss + Confusion) / tổng. |
| FAD | Chất lượng tạo nhạc | Khoảng cách Fréchet trên embedding VGGish. |
| RTFx | Thông lượng | Số giây âm thanh trên mỗi giây đồng hồ thực tế. |

## Đọc thêm

- [jiwer](https://github.com/jitsi/jiwer) — Thư viện WER/CER với các tiện ích chuẩn hóa.
- [UTMOS (Saeki et al. 2022)](https://arxiv.org/abs/2204.02152) — Bộ dự đoán MOS đã học.
- [Fréchet Audio Distance (Kilgour et al. 2019)](https://arxiv.org/abs/1812.08466) — Tiêu chuẩn tạo nhạc.
- [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) — Bảng xếp hạng trực tiếp năm 2026.
- [TTS Arena](https://huggingface.co/spaces/TTS-AGI/TTS-Arena) — Bảng xếp hạng TTS dựa trên bình chọn của con người.
- [MMAU-Pro benchmark](https://mmaubenchmark.github.io/) — Bảng xếp hạng suy luận LALM.
- [HEAR benchmark](https://hearbenchmark.com/) — Các benchmark âm thanh SSL.