# Voice Cloning & Voice Conversion

> Voice cloning đọc văn bản của bạn bằng giọng của người khác. Voice conversion viết lại giọng nói của bạn thành giọng của người khác trong khi vẫn giữ nguyên nội dung bạn đã nói. Cả hai đều dựa trên cùng một sự phân tách: tách biệt danh tính người nói khỏi nội dung.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 06 (Speaker Recognition), Phase 6 · 07 (TTS)
**Time:** ~75 phút

## Vấn đề

Vào năm 2026, một đoạn âm thanh dài 5 giây là đủ để tạo ra bản sao giọng nói chất lượng cao của bất kỳ ai với GPU tiêu dùng. ElevenLabs, F5-TTS, OpenVoice v2, VoiceBox đều hỗ trợ cloning zero-shot hoặc few-shot. Công nghệ này vừa là một lợi ích (TTS hỗ trợ tiếp cận, lồng tiếng, giọng nói hỗ trợ) vừa là một vũ khí (cuộc gọi lừa đảo, deepfake chính trị, đánh cắp sở hữu trí tuệ).

Hai tác vụ liên quan chặt chẽ:

- **Voice cloning (phía TTS):** văn bản + 5 giây giọng nói tham chiếu → âm thanh bằng giọng đó.
- **Voice conversion (phía speech):** âm thanh nguồn (người A nói X) + giọng nói tham chiếu của người B → âm thanh của B nói X.

Cả hai đều phân tích dạng sóng thành (nội dung, người nói, ngữ điệu) và kết hợp lại nội dung từ nguồn này với người nói từ nguồn khác.

Ràng buộc chính mà bạn phải tuân thủ vào năm 2026: **watermarking (đóng dấu bản quyền) và cổng kiểm soát sự đồng ý (consent gates) là yêu cầu bắt buộc về mặt pháp lý tại EU (AI Act, có hiệu lực từ tháng 8 năm 2026) và tại California (AB 2905, có hiệu lực từ năm 2025)**. Pipeline của bạn phải phát ra một watermark không thể nghe thấy và từ chối các bản clone không có sự đồng ý.

## Khái niệm

![Voice cloning vs conversion: factorize, swap speaker, recombine](../assets/voice-cloning.svg)

**Zero-shot cloning.** Truyền một đoạn clip 5 giây vào một mô hình đã được huấn luyện trên hàng ngàn người nói. Bộ mã hóa người nói (speaker encoder) ánh xạ clip đó thành một speaker embedding; bộ giải mã TTS sẽ điều kiện hóa dựa trên embedding đó cộng với văn bản.

Được sử dụng bởi: F5-TTS (2024), YourTTS (2022), XTTS v2 (2024), OpenVoice v2 (2024).

**Few-shot fine-tuning.** Ghi âm 5-30 phút giọng nói mục tiêu. LoRA-fine-tune một mô hình cơ sở trong một giờ. Chất lượng nhảy vọt từ "tạm ổn" sang "không thể phân biệt được". Coqui và ElevenLabs đều hỗ trợ mô hình này; cộng đồng sử dụng nó với F5-TTS.

**Voice conversion (VC).** Hai nhóm chính:

- **Recognition-synthesis.** Chạy mô hình kiểu ASR để trích xuất biểu diễn nội dung (ví dụ: soft phoneme posteriors, PPGs), sau đó tổng hợp lại với speaker embedding mục tiêu. Mạnh mẽ với ngôn ngữ và giọng địa phương. Được sử dụng bởi KNN-VC (2023), Diff-HierVC (2023).
- **Disentanglement.** Huấn luyện một autoencoder tách biệt nội dung, người nói và ngữ điệu trong không gian tiềm ẩn (latent space) tại bottleneck. Hoán đổi speaker embedding khi inference. Chất lượng thấp hơn nhưng nhanh hơn. Được sử dụng bởi AutoVC (2019), các biến thể VITS-VC.

**Neural codec-based cloning (2024+).** VALL-E, VALL-E 2, NaturalSpeech 3, VoiceBox — coi âm thanh như các token rời rạc từ SoundStream / EnCodec, huấn luyện một mô hình tự hồi quy (autoregressive) hoặc flow-matching lớn trên các token codec. Chất lượng tương đương ElevenLabs trên các prompt ngắn.

### Vấn đề đạo đức, không phải là tính năng phụ

**Watermarking.** PerTh (Perth) và SilentCipher (2024) nhúng một ID khoảng 16-32 bit một cách không thể nhận thấy vào âm thanh. Có khả năng chống chịu việc re-encoding, streaming và các chỉnh sửa thông thường. Mã nguồn mở sẵn sàng cho sản xuất.

**Consent gates.** Phải ghép mọi đầu ra clone với một bản ghi đồng ý có thể xác minh. "Tôi, Rohit, vào ngày 22-04-2026, cho phép sử dụng giọng nói này cho mục đích X." Lưu trữ trong một nhật ký chống giả mạo.

**Detection.** AASIST, RawNet2 và Wav2Vec2-AASIST được sử dụng làm bộ phát hiện. Thử thách ASVspoof 2025 đã công bố EER từ 0.8–2.3% cho các bộ phát hiện hiện đại nhất chống lại đầu ra của ElevenLabs, VALL-E 2 và Bark.

### Các con số (2026)

| Mô hình | Zero-shot? | SECS (độ tương đồng) | WER (độ thông minh) | Tham số |
|-------|-----------|--------------------|--------------|--------|
| F5-TTS | Có | 0.72 | 2.1% | 335M |
| XTTS v2 | Có | 0.65 | 3.5% | 470M |
| OpenVoice v2 | Có | 0.70 | 2.8% | 220M |
| VALL-E 2 | Có | 0.77 | 2.4% | 370M |
| VoiceBox | Có | 0.78 | 2.1% | 330M |

SECS > 0.70 thường là không thể phân biệt được với mục tiêu đối với hầu hết người nghe.

```figure
sp-voice-factorize
```

## Xây dựng

### Bước 1: phân tách với recognition-synthesis (demo code trong main.py)

```python
def clone_pipeline(ref_audio, text, target_embedder, tts_model):
    speaker_emb = target_embedder.encode(ref_audio)
    mel = tts_model(text, speaker=speaker_emb)
    return vocoder(mel)
```

Về mặt khái niệm thì đơn giản; khối lượng triển khai nằm ở `tts_model` và bộ mã hóa người nói.

### Bước 2: zero-shot clone với F5-TTS

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="rohit_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please add milk and bread to my list.",
)
```

Bản ghi tham chiếu phải khớp chính xác với âm thanh; sự sai lệch sẽ làm hỏng quá trình căn chỉnh (alignment).

### Bước 3: voice conversion với KNN-VC

```python
import torch
from knnvc import KNNVC  # 2023 model, https://github.com/bshall/knn-vc
vc = KNNVC.load("wavlm-base-plus")
out_wav = vc.convert(source="my_voice.wav", target_pool=["alice_1.wav", "alice_2.wav"])
```

KNN-VC chạy WavLM để trích xuất các embedding theo khung (per-frame) cho nguồn và tập hợp mục tiêu, sau đó thay thế mỗi khung nguồn bằng khung gần nhất trong tập hợp. Không tham số (non-parametric), hoạt động với một phút giọng nói mục tiêu.

### Bước 4: nhúng watermark

```python
from silentcipher import SilentCipher
sc = SilentCipher(model="2024-06-01")
payload = b"consent_id:abc123;ts:1745353200"
watermarked = sc.embed(wav, sr=24000, message=payload)
detected = sc.detect(watermarked, sr=24000)   # returns payload bytes
```

~32 bit payload, có thể phát hiện sau khi re-encode MP3 và nhiễu nhẹ.

### Bước 5: cổng kiểm soát sự đồng ý

```python
def cloned_inference(text, ref_audio, consent_record):
    assert verify_signature(consent_record), "Signed consent required"
    assert consent_record["speaker_id"] == hash_speaker(ref_audio)
    wav = tts.infer(ref_file=ref_audio, gen_text=text)
    wav = watermark(wav, payload=consent_record["id"])
    return wav
```

## Sử dụng

Stack năm 2026:

| Tình huống | Lựa chọn |
|-----------|------|
| 5-giây zero-shot clone, mã nguồn mở | F5-TTS hoặc OpenVoice v2 |
| Cloning thương mại | ElevenLabs Instant Voice Clone v2.5 |
| Voice conversion (viết lại) | KNN-VC hoặc Diff-HierVC |
| Fine-tune nhiều người nói | StyleTTS 2 + speaker adapter |
| Cloning đa ngôn ngữ | XTTS v2 hoặc VALL-E X |
| Phát hiện deepfake | Wav2Vec2-AASIST |

## Những cạm bẫy

- **Bản ghi tham chiếu bị lệch.** F5-TTS và các mô hình tương tự yêu cầu văn bản tham chiếu phải khớp chính xác với âm thanh tham chiếu, bao gồm cả dấu câu.
- **Tham chiếu có tiếng vang.** Tiếng vang làm hỏng bản clone. Hãy ghi âm trong môi trường khô, mic gần.
- **Sai lệch cảm xúc.** Tham chiếu huấn luyện "vui vẻ" sẽ tạo ra các bản clone vui vẻ cho mọi thứ. Hãy khớp cảm xúc tham chiếu với mục đích sử dụng.
- **Rò rỉ ngôn ngữ.** Clone một người nói tiếng Anh rồi yêu cầu mô hình nói tiếng Pháp thường vẫn mang theo giọng Anh; hãy sử dụng các mô hình đa ngôn ngữ (XTTS, VALL-E X).
- **Không có watermark.** Không thể phát hành hợp pháp tại EU từ tháng 8 năm 2026.

## Xuất xưởng

Lưu dưới dạng `outputs/skill-voice-cloner.md`. Thiết kế một pipeline cloning hoặc conversion với cổng đồng ý + watermark + mục tiêu chất lượng.

## Bài tập

1. **Dễ.** Chạy `code/main.py`. Minh họa việc hoán đổi speaker-embedding bằng cách tính cosine giữa hai "người nói" trước và sau khi hoán đổi.
2. **Trung bình.** Sử dụng OpenVoice v2 để clone giọng nói của chính bạn. Đo SECS giữa tham chiếu và bản clone. Đo CER thông qua Whisper.
3. **Khó.** Áp dụng watermark SilentCipher cho 20 bản clone, chạy chúng qua mã hóa/giải mã MP3 128 kbps, phát hiện payload. Báo cáo độ chính xác bit.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| Zero-shot clone | 5 giây là đủ | Mô hình đã huấn luyện + speaker embedding; không cần train thêm. |
| PPG | Phonetic posteriorgram | Các posterior ASR theo khung được dùng làm biểu diễn nội dung độc lập ngôn ngữ. |
| KNN-VC | Chuyển đổi láng giềng gần nhất | Thay thế mỗi khung nguồn bằng khung gần nhất trong tập hợp mục tiêu. |
| Neural codec TTS | Kiểu VALL-E | Mô hình AR trên các token EnCodec/SoundStream. |
| Watermark | Chữ ký không thể nghe thấy | Các bit nhúng trong âm thanh, tồn tại sau khi re-encode. |
| SECS | Độ trung thực của clone | Cosine giữa speaker embedding của mục tiêu và bản clone. |
| AASIST | Bộ phát hiện deepfake | Mô hình chống giả mạo; phát hiện giọng nói tổng hợp. |

## Đọc thêm

- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885) — SOTA zero-shot cloning mã nguồn mở.
- [Baevski et al. / Microsoft (2023). VALL-E](https://arxiv.org/abs/2301.02111) và [VALL-E 2 (2024)](https://arxiv.org/abs/2406.05370) — neural-codec TTS.
- [Qian et al. (2019). AutoVC](https://arxiv.org/abs/1905.05879) — voice conversion dựa trên disentanglement.
- [Baas, Waubert de Puiseau, Kamper (2023). KNN-VC](https://arxiv.org/abs/2305.18975) — VC dựa trên truy xuất.
- [SilentCipher (2024) — Audio Watermarking](https://github.com/sony/silentcipher) — watermark âm thanh 32-bit sẵn sàng cho sản xuất.
- [Kết quả ASVspoof 2025](https://www.asvspoof.org/) — cuộc đua giữa bộ phát hiện và bộ tổng hợp, cập nhật 2026.