# Capstone 12 — Pipeline Hiểu Video (Scene, QA, Search)

> Twelve Labs đã thương mại hóa Marengo + Pegasus. VideoDB đã ra mắt API CRUD-for-video. Molmo 2 của AI2 đã công bố các checkpoint VLM mã nguồn mở. Gemini với ngữ cảnh dài (long-context) có thể xử lý hàng giờ video một cách tự nhiên. TimeLens-100K đã định nghĩa việc xác định thời gian (temporal grounding) ở quy mô lớn. Pipeline năm 2026 đã được chốt: phân đoạn cảnh (scene segmentation), chú thích + embedding cho mỗi cảnh, căn chỉnh transcript, chỉ mục đa vector (multi-vector index), và truy vấn trả về kết quả kèm dấu thời gian (start, end) cùng ảnh xem trước. Capstone này yêu cầu xử lý 100 giờ video, đạt các benchmark công khai, và đo lường hiện tượng ảo giác (hallucination) đối với các câu hỏi về đếm số lượng và hành động.

**Type:** Capstone
**Languages:** Python (pipeline), TypeScript (UI)
**Prerequisites:** Phase 4 (CV), Phase 6 (speech), Phase 7 (transformers), Phase 11 (LLM engineering), Phase 12 (multimodal), Phase 17 (infrastructure)
**Phases exercised:** P4 · P6 · P7 · P11 · P12 · P17
**Time:** 30 giờ

## Vấn đề

QA cho video dài là bài toán đa phương thức tiêu tốn băng thông nhất ở quy mô năm 2026. Gemini 2.5 Pro có thể đọc video 2 giờ một cách tự nhiên, nhưng việc nạp 100 giờ video vào một kho dữ liệu có thể truy vấn vẫn đòi hỏi một chỉ mục ở cấp độ cảnh (scene-level index). Cấu trúc sản phẩm kết hợp phân đoạn cảnh (TransNetV2 hoặc PySceneDetect), chú thích mỗi cảnh bằng VLM (Gemini 2.5, Qwen3-VL-Max, hoặc Molmo 2), căn chỉnh transcript (Whisper-v3-turbo với dấu thời gian từng từ), và chỉ mục đa vector lưu trữ chú thích, embedding khung hình và transcript song song. Pipeline truy vấn trả về kết quả kèm dấu thời gian (start, end) và ảnh xem trước.

Các benchmark công khai (ActivityNet-QA, NeXT-GQA) cộng với bộ 100 câu hỏi tùy chỉnh của bạn. Ảo giác đối với các câu hỏi về đếm số lượng và loại hành động là lớp lỗi khó đã biết; capstone này sẽ đo lường cụ thể hiện tượng đó.

## Khái niệm

Ba pipeline chạy song song khi nạp dữ liệu. **Phân đoạn cảnh** cắt video thành các cảnh. **Chú thích VLM** tạo chú thích cho mỗi cảnh và embedding khung hình từ một khung hình chính (keyframe). **Căn chỉnh ASR** tạo ra dấu thời gian ở cấp độ từ. Ba luồng được kết hợp bởi (scene_id, khoảng thời gian). Mỗi cảnh nhận ba loại vector trong chỉ mục đa vector (Qdrant): embedding chú thích, embedding khung hình chính, embedding transcript.

Tại thời điểm truy vấn, câu hỏi ngôn ngữ tự nhiên được thực hiện trên cả ba vector; kết quả được hợp nhất bằng RRF; một bộ điều hợp xác định thời gian (kiểu TimeLens) tinh chỉnh cửa sổ (start, end) trong cảnh hàng đầu. Bộ tổng hợp VLM (Gemini 2.5 Pro hoặc Qwen3-VL-Max) nhận câu hỏi + các cảnh hàng đầu + khung hình đã cắt và trả lời kèm trích dẫn dấu thời gian và ảnh xem trước.

Việc đo lường ảo giác rất quan trọng. Các câu hỏi về đếm ("có bao nhiêu người vào phòng?") và loại hành động ("đầu bếp có đổ trước khi khuấy không?") nổi tiếng là không đáng tin cậy. Hãy báo cáo độ chính xác riêng biệt so với các câu hỏi mô tả.

## Kiến trúc

```
video file / URL
      |
      v
PySceneDetect / TransNetV2  (scene segmentation)
      |
      +--- per-scene keyframe --- VLM caption + frame embedding
      |                            (Gemini 2.5 Pro / Qwen3-VL-Max / Molmo 2)
      |
      +--- audio channel --- Whisper-v3-turbo ASR + word timestamps
      |
      v
multi-vector Qdrant: {caption_emb, keyframe_emb, transcript_emb}
      |
query:
  dense queries against all three -> RRF merge -> top-k scenes
      |
      v
TimeLens / VideoITG temporal grounding (refine start/end within scene)
      |
      v
VLM synth: query + top scenes + frame previews
      |
      v
answer + (start, end) timestamps + frame thumbs + citations
```

## Stack

- Phân đoạn cảnh: TransNetV2 (state-of-the-art 2024-26) hoặc PySceneDetect
- ASR: Whisper-v3-turbo qua faster-whisper với dấu thời gian từng từ
- VLM captioner + answerer: Gemini 2.5 Pro hoặc Qwen3-VL-Max hoặc Molmo 2
- Temporal grounding: Bộ điều hợp được huấn luyện kiểu TimeLens-100K hoặc VideoITG
- Index: Qdrant với hỗ trợ đa vector (caption / frame / transcript)
- UI: Next.js 15 với trình phát video HTML5 và ảnh thu nhỏ của cảnh
- Eval: ActivityNet-QA, NeXT-GQA, bộ 100 câu hỏi tùy chỉnh được dán nhãn thủ công
- Benchmark ảo giác: các tập con về đếm và loại hành động với nhãn thủ công

```figure
cf-scene-index
```

## Xây dựng

1. **Ingest walker.** Chấp nhận URL YouTube hoặc MP4 cục bộ. Giảm độ phân giải xuống 720p nếu cần. Lưu trữ `{video_id, file_path}`.

2. **Phân đoạn cảnh.** Chạy TransNetV2 hoặc PySceneDetect để tạo `[{scene_id, start_ms, end_ms, keyframe_path}]`. Mục tiêu 100 giờ: ~6k-8k cảnh.

3. **ASR pass.** Chạy Whisper-v3-turbo trên âm thanh; xuất dấu thời gian từng từ; chia thành các đoạn transcript theo cảnh.

4. **Chú thích VLM.** Với mỗi cảnh, gọi Gemini 2.5 Pro (hoặc Qwen3-VL-Max) với khung hình chính và mẫu chú thích ngắn. Tạo chú thích + embedding khung hình.

5. **Chỉ mục đa vector.** Bộ sưu tập Qdrant với ba vector được đặt tên. Payload: `{video_id, scene_id, start_ms, end_ms, keyframe_url}`.

6. **Truy vấn.** Câu hỏi ngôn ngữ tự nhiên kích hoạt ba truy vấn dày đặc (dense queries); hợp nhất với RRF; top-k=5 cảnh.

7. **Xác định thời gian (Temporal grounding).** Chạy bộ điều hợp kiểu TimeLens trên cảnh hàng đầu để tinh chỉnh cửa sổ (start, end) trong cảnh đó.

8. **Tổng hợp VLM.** Gọi Gemini 2.5 Pro với câu hỏi + 3 clip cảnh hàng đầu (dưới dạng hình ảnh hoặc clip ngắn) + transcript. Yêu cầu trích dẫn `(video_id, start_ms, end_ms)`.

9. **Đánh giá.** Chạy ActivityNet-QA và NeXT-GQA. Xây dựng bộ 100 câu hỏi tùy chỉnh. Báo cáo độ chính xác tổng thể + phân tích theo lớp (đếm, hành động, mô tả).

## Sử dụng

```
$ video-qa ask --url=https://youtube.com/watch?v=X "how many cars pass the intersection in the first minute?"
[scene]    23 scenes detected
[asr]      transcript complete, 4m12s
[index]    69 vectors written (23 scenes x 3)
[query]    top scene: scene 3 [01:32-01:54], confidence 0.84
[ground]   refined window: [00:12-00:58]
[synth]    gemini 2.5 pro, 1.4s
answer:    5 cars pass the intersection between 00:12 and 00:58.
citations: [scene 3: 00:12-00:58]
          [frame preview at 00:14, 00:27, 00:44, 00:51, 00:57]
```

## Triển khai

`outputs/skill-video-qa.md` là sản phẩm bàn giao. Với một URL YouTube hoặc video đã tải lên, pipeline sẽ lập chỉ mục các cảnh và trả lời câu hỏi kèm trích dẫn có dấu thời gian.

| Trọng số | Tiêu chí | Cách đo lường |
|:-:|---|---|
| 25 | Temporal grounding IoU | Intersection-over-union trên tập grounding giữ lại |
| 20 | Độ chính xác QA | NeXT-GQA và bộ 100 câu hỏi tùy chỉnh |
| 20 | Thông lượng nạp dữ liệu | Số giờ video trên mỗi đô la chi phí |
| 20 | UI và UX trích dẫn | Liên kết dấu thời gian, dải ảnh thu nhỏ, nhảy đến khung hình |
| 15 | Tỷ lệ ảo giác | Độ chính xác của câu hỏi đếm và loại hành động riêng biệt |
| **100** | | |

## Bài tập

1. Thay thế Gemini 2.5 Pro bằng Qwen3-VL-Max trong bước chú thích. Báo cáo sự khác biệt về chất lượng chú thích trên mẫu 50 cảnh được con người đánh giá.

2. Giảm embedding khung hình mỗi cảnh xuống một vector gộp thay vì đa vector. Đo lường sự suy giảm hiệu suất truy xuất.

3. Xây dựng chế độ "đếm nghiêm ngặt": bộ tổng hợp trích xuất từng trường hợp được đếm kèm dấu thời gian và người dùng nhấp để xác minh. Đo lường xem việc xác minh của người dùng có làm giảm ảo giác hay không.

4. Benchmark chi phí nạp dữ liệu: số giờ video trên mỗi đô la trên ba lựa chọn VLM. Chọn điểm tối ưu.

5. Thêm transcript phân đoạn người nói (speaker-diarized): chạy pyannote speaker diarization trên âm thanh và nhúng transcript theo từng người nói. Thực hiện truy vấn "Alice đã nói gì về X?".

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|------------------------|
| Phân đoạn cảnh | "Shot detection" | Cắt video thành các cảnh tại ranh giới shot |
| Chỉ mục đa vector | "Caption + frame + transcript" | Bộ sưu tập Qdrant với các vector được đặt tên cho mỗi biểu diễn |
| Xác định thời gian | "When exactly did it happen" | Tinh chỉnh cửa sổ (start, end) cho câu trả lời truy vấn |
| Embedding khung hình | "Visual representation" | Vector embedding của một khung hình chính; dùng cho độ tương đồng hình ảnh cảnh |
| RRF fusion | "Reciprocal rank fusion" | Chiến lược hợp nhất trên nhiều danh sách xếp hạng; thủ thuật truy xuất lai cổ điển |
| Ảo giác đếm | "Miscount" | Chế độ lỗi đã biết của VLM đối với các câu hỏi "có bao nhiêu X" |
| ActivityNet-QA | "Video-QA benchmark" | Benchmark độ chính xác QA cho video dài |

## Đọc thêm

- [AI2 Molmo 2](https://allenai.org/blog/molmo2) — các checkpoint VLM mã nguồn mở
- [TimeLens (CVPR 2026)](https://github.com/TencentARC/TimeLens) — xác định thời gian ở quy mô lớn
- [Gemini Video long-context](https://deepmind.google/technologies/gemini) — tài liệu tham khảo được lưu trữ
- [VideoDB](https://videodb.io) — tài liệu tham khảo API CRUD-for-video
- [Twelve Labs Marengo + Pegasus](https://www.twelvelabs.io) — tài liệu tham khảo thương mại
- [TransNetV2](https://github.com/soCzech/TransNetV2) — mô hình phân đoạn cảnh
- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) — giải pháp thay thế mã nguồn mở cổ điển
- [ActivityNet-QA](https://arxiv.org/abs/1906.02467) — benchmark đánh giá tham chiếu