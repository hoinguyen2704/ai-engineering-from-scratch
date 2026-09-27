# Audio-Language Models: từ Whisper đến Audio Flamingo 3 Arc

> Whisper (Radford và cộng sự, tháng 12 năm 2022) đã giải quyết bài toán nhận dạng giọng nói — 680 nghìn giờ dữ liệu giọng nói đa ngôn ngữ được giám sát yếu, một kiến trúc encoder-decoder transformer đơn giản, một chuẩn mực mà mọi bản phát hành ASR sau đó đều phải trích dẫn. Nhưng nhận dạng không phải là suy luận. Việc hỏi "những nhạc cụ nào có trong bản ghi âm này", "người nói đang thể hiện cảm xúc gì" hoặc "điều gì đã xảy ra ở phút thứ 3" đòi hỏi khả năng hiểu âm thanh, chứ không chỉ là chuyển đổi thành văn bản. Qwen-Audio, SALMONN, LTU và Audio Flamingo 3 (AF3, tháng 7 năm 2025) của NVIDIA đã dần xây dựng nên hệ thống đó: giữ lại các encoder đẳng cấp Whisper, gắn thêm Q-former, huấn luyện trên dữ liệu hướng dẫn âm thanh-văn bản, và bổ sung suy luận chuỗi tư duy (chain-of-thought). Bài học này sẽ đi qua chặng đường đó.

**Type:** Build
**Languages:** Python (stdlib, log-Mel spectrogram + audio Q-former skeleton)
**Prerequisites:** Phase 6 (Speech and Audio), Phase 12 · 03 (Q-Former)
**Time:** ~180 phút

## Mục tiêu học tập

- Tính toán log-Mel spectrogram từ một dạng sóng (waveform): windowing, FFT, filter banks, log transform.
- So sánh các tùy chọn encoder: Whisper encoder, BEATs, AF-Whisper hybrid. Khi nào mỗi loại chiếm ưu thế.
- Xây dựng một audio Q-former: N truy vấn (queries) có thể học được thực hiện cross-attention với các bản vá (patches) của spectrogram.
- Giải thích sự khác biệt giữa huấn luyện theo tầng (Whisper-rồi-đến-LLM) và huấn luyện end-to-end audio-LLM: tại sao end-to-end lại mở rộng tốt hơn cho suy luận.

## Vấn đề

Nhận dạng giọng nói đã được Whisper giải quyết. OCR-của-âm thanh đã trở thành một loại hàng hóa phổ thông. Nhưng "hàng hóa" chỉ dừng lại ở việc chuyển đổi thành văn bản. Nếu mô hình không thể suy luận về những gì nó nghe được — thời điểm, người nói, cảm xúc, cấu trúc âm nhạc, âm thanh môi trường — thì việc chuyển đổi văn bản đơn thuần không thể thúc đẩy các tính năng sản phẩm.

Ba hướng đi rõ ràng:

1. Cascaded (Theo tầng): Whisper chuyển đổi âm thanh thành văn bản, LLM suy luận dựa trên văn bản đó. Hoạt động tốt cho các kịch bản chỉ có giọng nói. Thất bại với âm nhạc, âm thanh môi trường, chồng lấn nhiều người nói, cảm xúc.

2. End-to-end audio-LLM: một audio encoder đưa các token âm thanh trực tiếp vào LLM, bỏ qua bước chuyển đổi văn bản. Bảo toàn thông tin âm thanh (cảm xúc, người nói, môi trường). Cần dữ liệu huấn luyện mới.

3. Hybrid (Lai): audio encoder + text decoder có khả năng vừa chuyển đổi vừa suy luận. Qwen-Audio và Audio Flamingo chọn hướng đi này.

## Khái niệm

### Log-Mel spectrogram: đặc trưng đầu vào

Mọi audio encoder đều bắt đầu với cùng một đặc trưng: log-Mel spectrogram.

1. Resample về 16 kHz.
2. Short-time Fourier transform với cửa sổ 25ms, bước nhảy (hop) 10ms.
3. Lấy độ lớn (magnitude) của kết quả FFT.
4. Áp dụng Mel filter banks (thường là 80 bộ lọc cách đều theo thang log từ 0-8000 Hz) để chuyển đổi sang tần số cảm nhận.
5. Nén log (log(1 + x)) để có dải động (dynamic range).

Kết quả: một mảng 2D có hình dạng (T, 80) trong đó T là số khung thời gian. Đối với clip 30 giây ở tốc độ khung hình 100 Hz: (3000, 80).

### Encoder của Whisper

Encoder của Whisper là một transformer kiểu ViT 12 lớp xử lý log-Mel spectrogram như một chuỗi các khung thời gian. Đầu ra: một vector trạng thái ẩn cho mỗi khung thời gian.

Đối với ASR, decoder của Whisper là một transformer cross-attention tạo ra các token văn bản dựa trên đầu ra của encoder. Đây là kiến trúc encoder-decoder tiêu chuẩn.

Đối với ALMs (audio-LLMs), bạn muốn đầu ra của encoder làm đầu vào cho một LLM khác. Mô hình: Whisper encoder đóng băng, Q-former có thể huấn luyện, LLM đóng băng hoặc tinh chỉnh.

### BEATs và các encoder chuyên biệt cho âm thanh

Whisper được huấn luyện trên dữ liệu chủ yếu là giọng nói. Nó yếu hơn đối với âm nhạc và âm thanh môi trường.

BEATs (Chen và cộng sự, 2022) là một transformer tự giám sát được huấn luyện trên AudioSet. Nó nắm bắt âm nhạc và âm thanh môi trường tốt hơn Whisper với cùng số lượng tham số.

AF-Whisper (Hybrid của Audio Flamingo 3): nối các đặc trưng của Whisper + BEATs làm đầu vào âm thanh. Whisper mang tín hiệu ngôn ngữ, BEATs mang tín hiệu âm thanh.

### Audio Q-former

Cùng mô hình với visual Q-former của BLIP-2. Một số lượng cố định các truy vấn có thể học được (thường là 32 hoặc 64) thực hiện cross-attention trên các khung đầu ra của audio encoder. Các truy vấn này trở thành các token âm thanh được LLM tiêu thụ.

Giai đoạn căn chỉnh huấn luyện: chỉ Q-former, sử dụng hàm mất mát contrastive + captioning trên các cặp âm thanh-văn bản (AudioCaps, Clotho). Giai đoạn hướng dẫn: end-to-end, mở băng LLM, huấn luyện trên dữ liệu hướng dẫn.

### Chặng đường — SALMONN, Qwen-Audio, AF3

SALMONN (Tang và cộng sự, 2023): Whisper + BEATs + Q-former + LLaMA. Audio-LLM mã nguồn mở đầu tiên có khả năng suy luận nghiêm túc. Các benchmark trên MMAU cho thấy điểm tổng hợp ~0.55.

Qwen-Audio (Chu và cộng sự, 2023): kiến trúc tương tự, được huấn luyện trên tập dữ liệu phong phú hơn, tinh chỉnh cho đối thoại đa vòng. MMAU ~0.60.

LTU — Listen, Think, Understand (Gong và cộng sự, 2023): dữ liệu suy luận rõ ràng, tập trung vào chuỗi tư duy trên các clip âm thanh. Nhỏ hơn nhưng tập trung hơn.

Audio Flamingo 3 (Goel và cộng sự, tháng 7 năm 2025): SOTA mã nguồn mở hiện tại. Backbone LLM 8B (Qwen2 7B), encoder Whisper-large nối BEATs, Q-former 64 truy vấn, huấn luyện trên hơn 1 triệu cặp hướng dẫn âm thanh-văn bản. MMAU 0.72, ngang bằng với các mô hình thương mại tiên phong trong một số tác vụ phụ.

AF3 cũng giới thiệu khả năng suy luận chuỗi tư duy theo yêu cầu cho âm thanh: mô hình có thể tùy chọn phát ra các token suy nghĩ ("để tôi xác định các nhạc cụ trước: ...") trước khi đưa ra câu trả lời cuối cùng. Độ chính xác trên các tác vụ suy luận phức tạp tăng 3-5 điểm khi bật tính năng suy nghĩ.

### Cascaded vs end-to-end

Pipeline Cascaded:

1. Whisper chuyển đổi âm thanh → văn bản.
2. LLM suy luận dựa trên văn bản.

Hoạt động hoàn hảo cho "tóm tắt podcast này". Thất bại với:
- "Tâm trạng của bài hát này là gì?" — tâm trạng nằm trong âm thanh, không phải từ ngữ.
- "Ai đang nói, Alice hay Bob?" — đòi hỏi nhận dạng người nói.
- "Vụ nổ xảy ra ở giây thứ mấy?" — thông tin thời gian bị mất trong văn bản.
- "Đây là âm thanh thật hay do AI tạo ra?" — phát hiện deepfake cần các đặc trưng âm thanh.

End-to-end bảo toàn tín hiệu âm thanh. Qwen-Audio và AF3 xử lý âm nhạc, môi trường và cảm xúc một cách tự nhiên.

### Công thức sản xuất năm 2026

Cho một sản phẩm hiểu âm thanh mới:

- Cascaded nếu: mục tiêu là chuyển đổi văn bản, không có âm nhạc, không suy luận cảm xúc.
- AF3 / Qwen-Audio-family nếu: có âm nhạc, cảm xúc, nhiều người nói, hoặc suy luận âm thanh phức tạp.

Cascaded rẻ hơn và đơn giản hơn. End-to-end có khả năng mạnh mẽ hơn.

### MMAU — benchmark suy luận âm thanh

MMAU (Massive Multimodal Audio Understanding) là benchmark suy luận âm thanh giai đoạn 2024-2025:

- 10.000 cặp QA âm thanh-văn bản bao gồm giọng nói, âm nhạc, âm thanh môi trường.
- Bao gồm phân loại, suy luận thời gian, suy luận nhân quả, QA mở.
- Kiểm tra những gì các pipeline cascaded thường xuyên bỏ lỡ.

SOTA mở (AF3) ở mức 0.72; các mô hình thương mại tiên phong ~0.78 (Gemini 2.5 Pro, Claude Opus 4.7). Khoảng cách nhỏ hơn so với chênh lệch mở-đóng của VideoMME, cho thấy các audio-LLM đang dần trưởng thành.

```figure
audio-text-ctc
```

## Sử dụng

`code/main.py`:

- Triển khai tính toán log-Mel spectrogram trong stdlib: windowing, DFT cơ bản, Mel filter-bank.
- Khung audio Q-former: với các khung đầu ra của encoder, tính toán Q, K, V, attention và phát ra N token.
- So sánh cascaded-vs-end-to-end trên một tác vụ thử nghiệm.

## Triển khai

Bài học này tạo ra `outputs/skill-audio-llm-pipeline-picker.md`. Với một tác vụ âm thanh (chuyển đổi văn bản, gắn thẻ âm nhạc, suy luận cảm xúc, phân đoạn người nói, phân loại môi trường), nó sẽ chọn cascaded, end-to-end AF3, hoặc một phương pháp lai.

## Bài tập

1. Tính toán kích thước log-Mel spectrogram cho một clip 30 giây ở 16kHz, cửa sổ 25ms, bước nhảy 10ms, 80 Mel bins. Điều này thay đổi như thế nào ở 48kHz?

2. Tại sao Whisper hoạt động kém trên âm nhạc? BEATs nắm bắt được những đặc trưng âm thanh nào mà Whisper không có?

3. Audio Q-former với 64 truy vấn so với 32: ở độ phức tạp tác vụ nào thì 64 mang lại hiệu quả? 32 tiết kiệm tính toán cho việc gì?

4. Đọc AF3 Phần 4 về suy nghĩ theo yêu cầu. Đề xuất ba tác vụ âm thanh mà chuỗi tư duy (chain-of-thought) giúp ích nhiều nhất.

5. Triển khai một pipeline phân đoạn người nói (diarization) tối giản sử dụng đầu ra của AF3. Bạn báo hiệu sự thay đổi người nói như thế nào?

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|------------------------|
| Log-Mel spectrogram | "Đặc trưng Mel" | Mảng 2D (thời gian, tần số) của các giá trị log-magnitude sau khi qua Mel filter banks |
| Audio Q-former | "Audio Perceiver" | Nút thắt cross-attention từ đầu ra audio encoder đến các truy vấn độ dài cố định cung cấp cho LLM |
| Cascaded | "ASR-rồi-đến-LLM" | Pipeline nơi Whisper chuyển đổi văn bản và LLM văn bản suy luận; mất thông tin âm thanh |
| End-to-end | "Audio-LLM" | Các đặc trưng âm thanh đi trực tiếp vào LLM qua Q-former; bảo toàn tín hiệu âm thanh |
| BEATs | "AudioSet encoder" | SSL transformer được huấn luyện trên AudioSet; mạnh về âm nhạc + âm thanh môi trường |
| MMAU | "Benchmark suy luận âm thanh" | 10k cặp QA về giọng nói, âm nhạc, môi trường; chuẩn đánh giá năm 2024 |
| On-demand thinking | "Audio CoT" | Mô hình có thể tùy chọn phát ra các token suy luận trước câu trả lời cuối cùng, tăng độ chính xác 3-5 điểm |

## Đọc thêm

- [Radford và cộng sự — Whisper (arXiv:2212.04356)](https://arxiv.org/abs/2212.04356)
- [Chu và cộng sự — Qwen-Audio (arXiv:2311.07919)](https://arxiv.org/abs/2311.07919)
- [Goel và cộng sự — Audio Flamingo 3 (arXiv:2507.08128)](https://arxiv.org/abs/2507.08128)
- [Tang và cộng sự — SALMONN (arXiv:2310.13289)](https://arxiv.org/abs/2310.13289)
- [Gong và cộng sự — LTU (arXiv:2305.10790)](https://arxiv.org/abs/2305.10790)