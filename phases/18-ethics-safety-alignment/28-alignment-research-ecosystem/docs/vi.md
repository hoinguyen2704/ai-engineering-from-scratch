# Hệ sinh thái Nghiên cứu Alignment — MATS, Redwood, Apollo, METR

> Năm tổ chức định hình lớp nghiên cứu alignment phi phòng thí nghiệm (non-lab) năm 2026. MATS (ML Alignment & Theory Scholars): hơn 527 nhà nghiên cứu từ cuối năm 2021, hơn 180 bài báo, hơn 10K trích dẫn, h-index 47; khóa hè 2024 được hợp nhất thành tổ chức 501(c)(3) với khoảng 90 học giả và 40 cố vấn; 80% cựu học viên trước năm 2025 làm việc trong lĩnh vực an toàn/bảo mật với hơn 200 người tại Anthropic, DeepMind, OpenAI, UK AISI, RAND, Redwood, METR, Apollo. Redwood Research: phòng thí nghiệm alignment ứng dụng do Buck Shlegeris sáng lập; giới thiệu AI Control (Bài 10); hợp tác với UK AISI về các trường hợp an toàn kiểm soát (control safety cases). Apollo Research: đánh giá âm mưu (scheming) trước khi triển khai cho các phòng thí nghiệm tiên phong; tác giả của In-Context Scheming (Bài 8) và Towards Safety Cases for AI Scheming. METR (Model Evaluation and Threat Research): đánh giá năng lực dựa trên tác vụ, nghiên cứu khung thời gian hoàn thành tác vụ tự hành; "Common Elements of Frontier AI Safety Policies" so sánh các khung chính sách của các phòng thí nghiệm. Eleos AI Research: đánh giá phúc lợi mô hình trước khi triển khai (Bài 19); thực hiện đánh giá phúc lợi cho Claude Opus 4.

**Type:** Learn
**Languages:** none
**Prerequisites:** Phase 18 · 01-27 (các bài học trước của Phase 18)
**Time:** ~45 phút

## Mục tiêu học tập

- Xác định năm tổ chức trong hệ sinh thái nghiên cứu alignment phi phòng thí nghiệm và kết quả cốt lõi của họ.
- Mô tả quy mô của MATS (số lượng học giả, bài báo, h-index) và vai trò của nó như một đường ống nhân tài.
- Mô tả chương trình nghị sự AI Control của Redwood và quan hệ đối tác của họ với UK AISI.
- Mô tả phương pháp đánh giá dựa trên tác vụ của METR.

## Vấn đề

Các phòng thí nghiệm tiên phong (Bài 18) tự thực hiện các đánh giá an toàn nội bộ và công bố các kết quả chọn lọc. Hệ sinh thái bên ngoài các phòng thí nghiệm là nơi các đánh giá được xác thực, nơi các dạng lỗi mới được phát hiện lần đầu và nơi nhân tài được đào tạo. Hiểu về hệ sinh thái này giúp giải mã những kết quả nghiên cứu nào được tin tưởng bởi ai.

## Khái niệm

### MATS (ML Alignment & Theory Scholars)

Bắt đầu từ cuối năm 2021. Chương trình cố vấn nghiên cứu; các học giả dành 10-12 tuần làm việc cùng một nhà nghiên cứu cấp cao về một vấn đề alignment cụ thể.

Quy mô (2026):
- Hơn 527 nhà nghiên cứu kể từ khi thành lập.
- Hơn 180 bài báo được xuất bản.
- Hơn 10K trích dẫn.
- h-index 47.
- Hè 2024: 90 học giả + 40 cố vấn; hợp nhất thành tổ chức 501(c)(3).

Kết quả sự nghiệp: ~80% cựu học viên trước năm 2025 đang làm việc trong lĩnh vực an toàn/bảo mật. Hơn 200 người tại Anthropic, DeepMind, OpenAI, UK AISI, RAND, Redwood, METR, Apollo.

### Redwood Research

Phòng thí nghiệm alignment ứng dụng. Do Buck Shlegeris sáng lập. Giới thiệu chương trình nghị sự AI Control (Bài 10). Hợp tác với UK AISI về các trường hợp an toàn kiểm soát. Cố vấn cho DeepMind và Anthropic về thiết kế đánh giá.

Các bài báo tiêu biểu: Greenblatt, Shlegeris và cộng sự, "AI Control" (arXiv:2312.06942, ICML 2024); Alignment Faking (Greenblatt, Denison, Wright và cộng sự, arXiv:2412.14093, hợp tác với Anthropic).

Phong cách: các mô hình đe dọa cụ thể, đối thủ trong trường hợp xấu nhất, các giao thức cụ thể có thể được kiểm tra áp lực.

### Apollo Research

Đánh giá âm mưu trước khi triển khai cho các phòng thí nghiệm tiên phong. Tác giả của In-Context Scheming (Bài 8, arXiv:2412.04984). Đối tác trong chương trình hợp tác đào tạo chống âm mưu của OpenAI năm 2025. Sản xuất tài liệu Towards Safety Cases for AI Scheming (2024).

Phong cách: đánh giá trong môi trường tác nhân (agentic-setting) nơi sự lừa dối có thể xuất hiện; phân tách ba trụ cột (lệch lạc mục tiêu, tính định hướng mục tiêu, nhận thức tình huống).

### METR (Model Evaluation and Threat Research)

Đánh giá năng lực dựa trên tác vụ. Nghiên cứu khung thời gian hoàn thành tác vụ tự hành. "Common Elements of Frontier AI Safety Policies" (metr.org/common-elements, 2025) so sánh các khung chính sách của các phòng thí nghiệm.

Đồng tác giả bản phác thảo trường hợp an toàn về AI Scheming cùng với Apollo.

Phong cách: đánh giá tác vụ dài hạn, đo lường năng lực thực nghiệm, tổng hợp khung chính sách.

### Eleos AI Research

Đánh giá phúc lợi mô hình trước khi triển khai. Thực hiện đánh giá phúc lợi cho Claude Opus 4 được ghi lại trong phần 5.3 của system card. Cung cấp kiểm tra phương pháp luận bên ngoài cho các tuyên bố liên quan đến phúc lợi trong Bài 19.

### Luồng vận hành

MATS đào tạo các nhà nghiên cứu. Tốt nghiệp, họ chuyển đến Anthropic, DeepMind, OpenAI (các nhóm an toàn phòng thí nghiệm) hoặc đến Redwood, Apollo, METR, Eleos (đánh giá bên ngoài). Các đơn vị đánh giá bên ngoài hợp tác với các phòng thí nghiệm và với UK AISI / CAISI. Các ấn phẩm phản hồi lại hệ sinh thái để chuẩn bị cho khóa MATS tiếp theo.

### Tại sao lớp này quan trọng

Các đánh giá từ một nguồn duy nhất không đáng tin cậy: các phòng thí nghiệm tự đánh giá mô hình của chính mình có xung đột lợi ích về mặt cấu trúc. Các đơn vị đánh giá bên ngoài có thể nêu ra và xác thực các dạng lỗi mà phòng thí nghiệm có thể báo cáo thiếu. Bài báo Sleeper Agents năm 2024 (Bài 7) là sự hợp tác giữa Anthropic + Redwood; Alignment Faking là Anthropic + Redwood; In-Context Scheming là Apollo; Anti-Scheming là Apollo + OpenAI. Cấu trúc đa tổ chức chính là sự kiểm soát chất lượng.

### Vị trí trong Phase 18

Các bài 7-11 tham chiếu công trình của Redwood và Apollo; Bài 18 tham chiếu so sánh khung chính sách của METR; Bài 19 tham chiếu Eleos. Bài 28 là bản đồ tổ chức rõ ràng cho hệ sinh thái mà phần còn lại của Phase này dựa vào.

```figure
sae-features
```

## Sử dụng

Không có mã nguồn. Hãy đọc "Common Elements of Frontier AI Safety Policies" của METR như một ví dụ về cách việc tổng hợp từ bên ngoài tạo thêm giá trị cho công tác chính sách nội bộ của phòng thí nghiệm.

## Triển khai

Bài học này tạo ra `outputs/skill-ecosystem-map.md`. Với một tuyên bố hoặc đánh giá về alignment, nó xác định tổ chức, địa điểm công bố, phong cách phương pháp luận và đối chiếu chéo với các tổ chức đối tác đã biết.

## Bài tập

1. Chọn một bài báo từ các Bài 7-15 và xác định các tổ chức liên quan. Đối chiếu chéo các tác giả với cựu học viên MATS và các đơn vị liên kết trong hệ sinh thái hiện tại.

2. Đọc "Common Elements of Frontier AI Safety Policies" của METR. Xác định ba điểm hội tụ giữa các phòng thí nghiệm mà họ nhấn mạnh và hai điểm khác biệt lớn nhất.

3. Kết quả sự nghiệp của MATS là ~80% an toàn/bảo mật. Hãy lập luận xem áp lực lựa chọn này là thích nghi (đào tạo cho lĩnh vực) hay thiên kiến (loại bỏ các quan điểm không chính thống).

4. Redwood và Apollo đều thực hiện công việc về kiểm soát/âm mưu nhưng với phong cách khác nhau. Hãy chọn một dạng lỗi và mô tả cách mỗi bên sẽ điều tra nó.

5. Eleos AI là tổ chức duy nhất thuần túy về phúc lợi mô hình. Hãy thiết kế một tổ chức thứ hai giả định tập trung vào một câu hỏi khác liên quan đến phúc lợi (tự do nhận thức, hiện thân robot, v.v.) và trình bày phương pháp luận của nó.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|------------------------|
| MATS | "chương trình cố vấn" | ML Alignment & Theory Scholars; hơn 527 nhà nghiên cứu từ 2021 |
| Redwood Research | "phòng lab kiểm soát" | Alignment ứng dụng; tác giả AI Control; đối tác UK AISI |
| Apollo Research | "đánh giá âm mưu" | Đánh giá âm mưu trước khi triển khai cho các phòng lab tiên phong |
| METR | "đánh giá tác vụ dài hạn" | Đánh giá năng lực dựa trên tác vụ; tổng hợp khung chính sách |
| Eleos AI | "phòng lab phúc lợi" | Đánh giá phúc lợi mô hình trước khi triển khai |
| Đường ống nhân tài | "MATS -> labs" | Học viên MATS chuyển đến Anthropic, DM, OpenAI, Redwood, Apollo, METR |
| Đánh giá bên ngoài | "kiểm tra phi phòng lab" | Đánh giá không thực hiện bởi nhà sản xuất mô hình; tăng độ tin cậy |

## Đọc thêm

- [MATS (ML Alignment & Theory Scholars)](https://www.matsprogram.org/) — chương trình cố vấn
- [Redwood Research](https://www.redwoodresearch.org/) — các bài báo về AI Control
- [Apollo Research](https://www.apolloresearch.ai/) — các đánh giá về âm mưu
- [METR — Common Elements of Frontier AI Safety Policies](https://metr.org/blog/2025-03-26-common-elements-of-frontier-ai-safety-policies/) — so sánh khung chính sách
- [Eleos AI Research](https://www.eleosai.org/research) — phương pháp luận về phúc lợi mô hình