# MIO và các mô hình đa phương thức streaming Any-to-Any

> GPT-4o mang đến một sản phẩm mà hầu hết các mô hình mã nguồn mở chưa thể sao chép: một tác nhân (agent) có khả năng nghe giọng nói, nhìn video và phản hồi bằng giọng nói trong thời gian thực. Câu trả lời từ hệ sinh thái mã nguồn mở vào cuối năm 2024 là MIO (Wang và cộng sự, tháng 9 năm 2024). MIO thực hiện token hóa văn bản, hình ảnh, giọng nói và âm nhạc, huấn luyện một causal transformer trên các chuỗi xen kẽ, và tạo ra bất kỳ phương thức nào từ bất kỳ phương thức nào. AnyGPT (Zhan và cộng sự, tháng 2 năm 2024) là bằng chứng khái niệm; MIO là phiên bản mở rộng; Unified-IO 2 (Allen AI, tháng 12 năm 2023) là "người anh em" với khả năng vision + action grounding. Bài học này tìm hiểu về mô hình any-to-any — bốn bộ token hóa, một transformer, và giải mã hỗ trợ streaming.

**Type:** Learn
**Languages:** Python (stdlib, four-modality token allocator + streaming decode loop)
**Prerequisites:** Phase 12 · 11 (Chameleon), Phase 6 (Speech and Audio)
**Time:** ~120 phút

## Mục tiêu học tập

- Thiết kế một từ vựng chung (shared vocabulary) chứa các token văn bản, hình ảnh, giọng nói và âm nhạc mà không bị xung đột.
- So sánh SEED-Tokenizer (hình ảnh) và SpeechTokenizer residual-VQ (giọng nói) về sự đánh đổi giữa nén và tái tạo.
- Giải thích chương trình huấn luyện bốn giai đoạn để xây dựng khả năng tạo any-to-any.
- Liệt kê ba công thức any-to-any mã nguồn mở và những sự đánh đổi chính của chúng: MIO, AnyGPT, Unified-IO 2.

## Vấn đề

Một mô hình đa phương thức thống nhất rất dễ để tuyên bố nhưng lại khó để xây dựng ở quy mô lớn. Hầu hết các hệ thống "any-to-any" cho đến năm 2024 đều hoạt động theo đường ống (pipelined): mô hình thị giác → biểu diễn văn bản → mô hình giọng nói → âm thanh. Mỗi bước chuyển đổi đều làm mất thông tin, tăng độ trễ và làm phức tạp quá trình huấn luyện. Video demo của GPT-4o cho thấy một giải pháp thay thế bằng mô hình đơn lẻ với phản hồi dưới một giây; các hệ thống mã nguồn mở đã chậm chân hơn nhiều tháng.

Các thách thức kỹ thuật:

- Các bộ token hóa phải tồn tại cho mọi phương thức, nén đủ tốt để tái tạo mà không mất dữ liệu, và tạo ra các token với tốc độ mà transformer có thể xử lý.
- Một từ vựng duy nhất phải phân bổ không gian cho văn bản (32k+), hình ảnh (16k+), giọng nói (4k+), âm nhạc (8k+). Tối thiểu hơn 40.000 mục.
- Dữ liệu huấn luyện phải bao phủ mọi cặp đầu vào-đầu ra (văn bản→hình ảnh, hình ảnh→giọng nói, giọng nói→hình ảnh, v.v.) hoặc mô hình phải có khả năng tự kết hợp.
- Quá trình suy luận (inference) phải stream các token đầu ra đủ nhanh để đạt độ trễ hội thoại (<500ms time-to-first-audio-byte).

## Khái niệm

### Bốn bộ token hóa cho bốn phương thức

Ngăn xếp bộ token hóa của MIO:

- Văn bản: BPE tiêu chuẩn, từ vựng ~32.000.
- Hình ảnh: SEED-Tokenizer (2023) — VAE lượng tử hóa với codebook rời rạc, 4.096 mục, 32x32 token mỗi ảnh.
- Giọng nói: SpeechTokenizer residual-VQ (2023) — mã hóa dạng sóng 16kHz thành 8 codebook phân cấp; cấp độ đầu tiên là nội dung thô, các cấp độ sau thêm vào ngữ điệu và đặc điểm người nói.
- Âm nhạc: residual-VQ tương tự (họ MusicGen / Encodec của Meta), 4-8 codebook.

Mỗi phương thức tạo ra các token số nguyên. Các token này nhận các dải ID riêng biệt trong từ vựng chung:

```
text:   0..31999
image:  32000..36095  (4096 image tokens)
speech: 36096..40191  (4096 speech base tokens, plus residual layers)
music:  40192..48383  (8192 music tokens)
sep:    48384..48390  (<image>, <speech>, <music>, </...>, etc.)
```

Tổng cộng: ~48k từ vựng. Embedding đầu vào và projection đầu ra bao phủ toàn bộ dải này.

### Giải mã Streaming

Tạo giọng nói sử dụng residual-VQ. Transformer dự đoán các token giọng nói cơ sở (lớp 0); một bộ lượng tử hóa dư (residual quantizer) được giải mã song song sẽ dự đoán các lớp tiếp theo. Mỗi token lớp 0 tương đương khoảng 50ms âm thanh ở 16kHz.

Mô hình streaming:

1. Người dùng nói vào micro; bộ token hóa âm thanh thời gian thực phát ra các token giọng nói mỗi 50ms.
2. MIO tiêu thụ các token khi chúng đến (prompt prefill + incremental forward).
3. Các token đầu ra được stream ra khi được tạo; một bộ giải mã giọng nói song song chuyển đổi chúng thành các mẫu âm thanh với độ trễ ~50-150ms.
4. Time-to-first-audio-byte: ~300-500ms trong bài báo MIO, tiệm cận mức ~250ms của GPT-4o.

Mini-Omni (arXiv:2408.16725), GLM-4-Voice (arXiv:2412.02612), và Moshi (arXiv:2410.00037) là các thiết kế speech-LLM streaming bổ trợ. Đặc biệt, Moshi đạt được độ trễ khứ hồi 160ms trên một GPU đơn lẻ.

### Chương trình huấn luyện bốn giai đoạn

Chương trình huấn luyện của MIO:

1. Giai đoạn 1 — căn chỉnh (alignment). Các tập dữ liệu cặp phương thức quy mô lớn: văn bản-hình ảnh, văn bản-giọng nói, văn bản-âm nhạc. Mỗi cặp sử dụng phân đoạn từ vựng riêng. Huấn luyện từ vựng chung.
2. Giai đoạn 2 — xen kẽ (interleaved). Các tài liệu đa phương thức xen kẽ (blog với hình ảnh + video, podcast với bản ghi chép, v.v.). Huấn luyện ngữ cảnh liên phương thức.
3. Giai đoạn 3 — tăng cường giọng nói (speech-enhanced). Dữ liệu âm thanh bổ sung để nâng cao chất lượng giọng nói mà không làm mất khả năng văn bản.
4. Giai đoạn 4 — SFT. Tinh chỉnh hướng dẫn (instruction tuning) trên các phương thức: VQA, chú thích, tường thuật, đối thoại giọng nói-giọng nói.

Thiếu một giai đoạn sẽ làm giảm các khả năng cụ thể: bỏ qua giai đoạn 2, mô hình mất ngữ cảnh liên phương thức; bỏ qua giai đoạn 3, chất lượng giọng nói sẽ kém.

### Chain-of-visual-thought

MIO giới thiệu chain-of-visual-thought: mô hình phát ra các token hình ảnh trung gian như một bước suy luận. Đối với câu hỏi "con mèo có đang leo cây không?", mô hình sẽ:

1. Phát ra các token `<image>` để dựng lại cảnh (từ hình ảnh đầu vào hoặc bản phác thảo).
2. Phát ra văn bản phân tích bản phác thảo đó.
3. Phát ra câu trả lời cuối cùng.

Hình ảnh trung gian được dựng lại đóng vai trò như một nháp vẽ. Các điểm chuẩn (benchmark) cải thiện trên các tác vụ suy luận không gian. Ý tưởng này phản chiếu chain-of-thought trong suy luận văn bản.

### Các đối thủ trong lĩnh vực any-to-any

- AnyGPT (arXiv:2402.12226): 4 phương thức (văn bản, hình ảnh, giọng nói, âm nhạc), thiết kế tương tự.
- Unified-IO 2 (arXiv:2312.17172): thêm đầu ra hành động thị giác, độ sâu, normals. Đa dạng tác vụ hơn, quy mô nhỏ hơn.
- NExT-GPT (arXiv:2309.05519): LLM + bộ giải mã khuếch tán (diffusion) chuyên biệt cho phương thức. Không phải cách tiếp cận mô hình đơn lẻ.
- CoDi (arXiv:2305.11846): khuếch tán có thể kết hợp; any-to-any thông qua latent chung.

MIO là mô hình gần nhất với khái niệm any-to-any thuần token. AnyGPT là tiền thân về mặt khái niệm của nó.

### Ngân sách độ trễ

Đối với một sản phẩm hội thoại, độ trễ của mọi thành phần đều quan trọng:

- Micro đến token âm thanh: ~50ms.
- Prefill (token âm thanh + lịch sử): ~100ms trên mô hình 8B.
- Token đầu ra đầu tiên: ~50ms.
- Residual-VQ song song + bộ giải mã giọng nói: ~100-150ms.

Tổng thời gian đến byte âm thanh đầu tiên: tối thiểu ~300ms. GPT-4o tuyên bố ~250ms. Moshi tuyên bố 160ms. MIO/AnyGPT nằm trong khoảng 400-600ms theo các điểm chuẩn công khai.

### Tại sao any-to-any vẫn khó

Ngay cả vào năm 2026, các mô hình any-to-any mã nguồn mở vẫn đi sau các mô hình đóng trên hai trục:

- Chất lượng giọng nói. Bộ token hóa residual-VQ bị mất dữ liệu; giọng nói hội thoại nghe có vẻ máy móc so với các giọng nói đẳng cấp ElevenLabs.
- Suy luận liên phương thức. Việc yêu cầu mô hình "hát về những gì bạn thấy" vẫn thất bại thường xuyên hơn so với các tác vụ thị giác thuần túy.

Đây là những vấn đề nghiên cứu mở. Qwen3-Omni (Bài học 12.20) là nỗ lực mã nguồn mở tiên tiến nhất vào năm 2025.

```figure
any-to-any-stream
```

## Sử dụng

`code/main.py`:

- Định nghĩa và in ra sự phân bổ từ vựng bốn phương thức.
- Định tuyến danh sách các đầu vào đa phương thức (văn bản, hình ảnh, đoạn âm thanh, âm nhạc) thông qua bộ định tuyến token.
- Mô phỏng giải mã streaming cho phản hồi văn bản-thành-giọng nói với việc đếm độ trễ.
- Tính toán thời gian dự kiến đến byte âm thanh đầu tiên dựa trên độ trễ của bộ mã hóa, prefill và bộ giải mã.

## Triển khai

Bài học này tạo ra `outputs/skill-any-to-any-pipeline-auditor.md`. Với một đặc tả sản phẩm hội thoại (đầu vào/đầu ra phương thức, mục tiêu độ trễ), nó kiểm tra các lựa chọn thiết kế của dòng MIO và tính toán ngân sách độ trễ.

## Bài tập

1. Sản phẩm của bạn chấp nhận đầu vào giọng nói và trả về đầu ra giọng nói. Mục tiêu ngân sách độ trễ end-to-end là bao nhiêu? Liệt kê các thành phần tiêu tốn thời gian.

2. SpeechTokenizer residual-VQ sử dụng 8 codebook. Hãy đề xuất lý do tại sao việc giải mã song song các cấp độ dư (residual levels) là cần thiết (so với tuần tự) và nó mang lại sự tiết kiệm độ trễ như thế nào.

3. Từ vựng của bạn có 32k văn bản + 4k hình ảnh + 4k giọng nói. Thêm 8k âm nhạc và ~10 ký tự phân cách. Chi phí tham số ma trận embedding ở hidden dim 4096 là bao nhiêu?

4. Chain-of-visual-thought phát ra một hình ảnh trung gian. Những loại câu hỏi nào được hưởng lợi? Những loại nào bị ảnh hưởng bởi các token bổ sung?

5. Đọc Moshi (arXiv:2410.00037). Mô tả kỹ thuật "độc thoại nội tâm" (inner monologue) của nó và so sánh với chain-of-visual-thought của MIO.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|------------------------|
| Any-to-any | "Đa phương thức vào/ra" | Một mô hình duy nhất chấp nhận và phát ra văn bản, hình ảnh, giọng nói và âm nhạc theo bất kỳ hướng nào |
| Residual-VQ | "Ngăn xếp token giọng nói" | Token hóa đa codebook trong đó mỗi lớp thêm thông tin; lớp cơ sở là nội dung, các lớp sau là ngữ điệu |
| SEED-Tokenizer | "Mã hình ảnh" | Bộ token hóa hình ảnh rời rạc với codebook 4096 mục được MIO sử dụng |
| Chain-of-visual-thought | "Nháp vẽ thị giác" | Mô hình tạo ra một hình ảnh trung gian như một bước suy luận trước câu trả lời cuối cùng |
| Time-to-first-audio-byte | "TTFAB" | Độ trễ từ giọng nói người dùng đến đầu ra âm thanh đầu tiên; <500ms để tạo cảm giác hội thoại |
| Four-stage curriculum | "Công thức huấn luyện" | Căn chỉnh -> xen kẽ -> tăng cường giọng nói -> SFT, theo thứ tự đó |

## Đọc thêm

- [Wang và cộng sự — MIO (arXiv:2409.17692)](https://arxiv.org/abs/2409.17692)
- [Zhan và cộng sự — AnyGPT (arXiv:2402.12226)](https://arxiv.org/abs/2402.12226)
- [Lu và cộng sự — Unified-IO 2 (arXiv:2312.17172)](https://arxiv.org/abs/2312.17172)
- [Wu và cộng sự — NExT-GPT (arXiv:2309.05519)](https://arxiv.org/abs/2309.05519)
- [Tang và cộng sự — CoDi (arXiv:2305.11846)](https://arxiv.org/abs/2305.11846)