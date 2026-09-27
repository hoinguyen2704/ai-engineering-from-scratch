# Capstone 17 — Personal AI Tutor (Thích ứng, Đa phương thức, có Bộ nhớ)

> Khanmigo (Khan Academy), Duolingo Max, Google LearnLM / Gemini for Education, Quizlet Q-Chat và Synthesis Tutor đều đã triển khai thành công các hệ thống gia sư đa phương thức thích ứng ở quy mô lớn vào năm 2026. Mô hình chung bao gồm: chính sách Socratic (không bao giờ đưa ra đáp án trực tiếp), mô hình người học cập nhật sau mỗi tương tác (theo phong cách Bayesian knowledge tracing), đầu vào đa phương thức (giọng nói + văn bản + ảnh chụp toán học), truy xuất đồ thị chương trình học, lập lịch lặp lại ngắt quãng (spaced-repetition) và các bộ lọc an toàn nghiêm ngặt cho nội dung phù hợp với lứa tuổi. Capstone này yêu cầu bạn xây dựng một gia sư chuyên biệt cho một môn học (đại số K-12 hoặc Python nhập môn), thực hiện nghiên cứu hiệu quả trong hai tuần với 10 người học và vượt qua bài kiểm tra đánh giá an toàn nội dung.

**Type:** Capstone
**Languages:** Python (backend, mô hình người học), TypeScript (web app), SQL (đồ thị chương trình học qua Postgres + Neo4j)
**Prerequisites:** Phase 5 (NLP), Phase 6 (speech), Phase 11 (LLM engineering), Phase 12 (multimodal), Phase 14 (agents), Phase 17 (infrastructure), Phase 18 (safety)
**Phases exercised:** P5 · P6 · P11 · P12 · P14 · P17 · P18
**Time:** 30 giờ

## Vấn đề

Gia sư thích ứng từng là một lĩnh vực nghiên cứu ngách trong ed-tech. Đến năm 2026, nó đã trở thành một sản phẩm tiêu dùng phổ biến. Khanmigo được triển khai tại hầu hết các khu học chánh ở Mỹ. Duolingo Max đạt hàng chục triệu MAU. Google LearnLM / Gemini for Education hỗ trợ việc dạy kèm trong Google Classroom. Quizlet Q-Chat tích hợp cùng với flashcards. Synthesis Tutor trở nên viral với mô hình gia sư cho trẻ em hiếu kỳ. Các yếu tố chung bao gồm: đầu vào đa phương thức (gõ, nói, chụp ảnh phương trình), phương pháp sư phạm Socratic (hỏi trước, giải thích sau), mô hình người học cập nhật sau mỗi tương tác và các quy tắc an toàn nghiêm ngặt phù hợp với lứa tuổi.

Bạn sẽ xây dựng một hệ thống như vậy cho một nhóm đối tượng cụ thể. Thước đo thành công là một nghiên cứu hiệu quả thực tế: điểm số trước và sau kiểm tra trong hai tuần với 10 người học. Vòng lặp giọng nói phải tạo cảm giác tự nhiên (sub-stack của capstone 03). Bộ nhớ phải tôn trọng quyền riêng tư. Bộ lọc an toàn phải vượt qua bài kiểm tra red-team tuân thủ COPPA cho cấp K-12.

## Khái niệm

Bốn thành phần chính. **Chính sách gia sư (Tutor policy)** là một vòng lặp Socratic: khi người học hỏi đáp án, chính sách sẽ đưa ra câu hỏi gợi mở; khi họ trả lời đúng, hệ thống chuyển sang khái niệm tiếp theo; khi họ gặp khó khăn, hệ thống cung cấp gợi ý từng bước. **Mô hình người học (Learner model)** là Bayesian knowledge tracing (hoặc một biến thể đơn giản) cập nhật xác suất nắm vững kiến thức trên mỗi nút của chương trình học sau mỗi tương tác. **Đồ thị chương trình học (Curriculum graph)** là một Neo4j chứa các khái niệm với các cạnh điều kiện tiên quyết; chính sách sẽ duyệt đồ thị để chọn khái niệm tiếp theo. **Bộ nhớ (Memory)** là kho lưu trữ theo tình tiết + ngữ nghĩa (phong cách agentmemory) chứa các tương tác, lỗi sai và sở thích trong quá khứ.

UX là đa phương thức. Đầu vào văn bản cho các câu trả lời gõ phím. Đầu vào giọng nói qua LiveKit + Whisper (tái sử dụng capstone 03). Đầu vào ảnh cho các bài toán qua dots.ocr hoặc PaliGemma 2. Đầu ra giọng nói qua Cartesia Sonic-2. An toàn sử dụng Llama Guard 4 cộng với bộ lọc phù hợp lứa tuổi (chặn nội dung người lớn, bạo lực, tự hại) và chính sách lưu trữ bộ nhớ tuân thủ COPPA.

Nghiên cứu hiệu quả là sản phẩm bàn giao. 10 người học, kiểm tra trước và sau, trong hai tuần. Báo cáo mức độ tăng trưởng học tập và khoảng tin cậy. So sánh với một nhóm đối chứng không thích ứng (cùng nội dung được truyền tải tuyến tính mà không có chính sách gia sư).

## Kiến trúc

```
learner device
  |
  +-- text         -> web app
  +-- voice        -> LiveKit Agents (ASR + TTS)
  +-- photo math   -> dots.ocr / PaliGemma 2
       |
       v
  tutor policy (LangGraph)
       - Socratic decision head
       - next-concept chooser (curriculum graph walk)
       - hint scaffolder
       - mastery update
       |
       v
  learner model (BKT / item-response theory)
       - per-concept mastery probability
       - spaced-repetition scheduler (SM-2 or FSRS)
       |
       v
  memory (agentmemory-style)
       - episodic: every interaction
       - semantic: learned mistakes, preferences
       - retention policy: COPPA / GDPR aware
       |
       v
  curriculum graph (Neo4j)
       - prerequisite edges
       - OER content attached
       |
       v
  safety:
    Llama Guard 4 + age-appropriate filter
    memory access guarded by learner ID scope
```

## Stack

- Lựa chọn môn học: Đại số K-12 hoặc Python nhập môn (chọn một để đi sâu)
- Chính sách gia sư: LangGraph trên Claude Sonnet 4.7 (với prompt caching)
- Mô hình người học: Bayesian knowledge tracing (cổ điển) hoặc FSRS để lập lịch lặp lại
- Đồ thị chương trình học: Neo4j các khái niệm + cạnh điều kiện tiên quyết + nội dung OER
- Bộ nhớ: Vector lưu trữ bền vững phong cách agentmemory + lưu trữ tình tiết + ngữ nghĩa
- Giọng nói: LiveKit Agents 1.0 + Cartesia Sonic-2 (tái sử dụng sub-stack capstone 03)
- Toán học qua ảnh: dots.ocr hoặc PaliGemma 2 để nhận diện phương trình
- An toàn: Llama Guard 4 + bộ lọc tùy chỉnh phù hợp lứa tuổi
- Đánh giá: Tạo câu hỏi cấp độ Bloom, bộ công cụ kiểm tra trước/sau, công cụ nghiên cứu hiệu quả

```figure
cf-tutor-loop
```

## Xây dựng

1. **Đồ thị chương trình học.** Xây dựng Neo4j gồm 50-150 nút khái niệm (ví dụ: đại số K-12 từ "trục số" đến "công thức nghiệm bậc hai") với các cạnh điều kiện tiên quyết. Gắn nội dung OER cho mỗi nút (Open Textbook, OpenStax).

2. **Mô hình người học.** Khởi tạo Bayesian knowledge tracing với các tham số tiên nghiệm: đoán (guess), trượt (slip), tốc độ học (learn-rate). Cập nhật mức độ nắm vững theo khái niệm sau mỗi tương tác. Lưu trữ theo từng người học.

3. **Chính sách gia sư.** LangGraph với các nút: `read_signal` (câu trả lời của người học đúng / một phần / bế tắc?), `select_concept` (duyệt đồ thị chương trình học để chọn khái niệm ưu tiên cao nhất), `scaffold` (câu hỏi Socratic), `update_mastery`.

4. **Bộ nhớ.** Mỗi tương tác được ghi vào kho lưu trữ tình tiết. Các lỗi sai và sở thích được nâng cấp lên bộ nhớ ngữ nghĩa. Chính sách lưu trữ tuân thủ COPPA: tự động xóa sau 1 năm, phụ huynh có quyền truy cập.

5. **Luồng giọng nói.** LiveKit Agents worker gắn với chính sách gia sư. ASR qua Whisper-v3-turbo. TTS qua Cartesia Sonic-2. Hỗ trợ ngắt lời (barge-in) (tái sử dụng cơ chế capstone 03).

6. **Luồng toán học qua ảnh.** Tải lên hoặc chụp ảnh; chạy dots.ocr hoặc PaliGemma 2 để nhận diện phương trình; đưa vào gia sư dưới dạng đầu vào có cấu trúc.

7. **An toàn.** Mọi đầu ra của mô hình đều đi qua Llama Guard 4 + bộ lọc phù hợp lứa tuổi (chặn tự hại, nội dung người lớn, bạo lực). Truy cập bộ nhớ được giới hạn theo ID người học; giao diện truy cập cho phụ huynh để xóa dữ liệu.

8. **Nghiên cứu hiệu quả.** 10 người học, kiểm tra trước (bài kiểm tra chuẩn hóa 30 câu hỏi), hai tuần tương tác với gia sư (3 phiên/tuần), kiểm tra sau. So sánh với nhóm đối chứng 10 người học không thích ứng với cùng nội dung.

9. **Báo cáo tiến độ hàng tuần.** Đối với mỗi người học, tự động tạo tóm tắt PDF về các chủ đề đã học, quỹ đạo nắm vững kiến thức và các bước tiếp theo được đề xuất.

## Sử dụng

```
learner: "I don't understand why 3x + 6 = 12 means x = 2"
[signal]   stuck
[concept]  'isolating variables' (prerequisite: addition-subtraction-equality)
[scaffold] "what number would you subtract from both sides to start?"
learner: "6"
[signal]   correct
[mastery]  addition-subtraction-equality: 0.62 -> 0.77
[concept]  continue 'isolating variables'
[scaffold] "great. now what is 3x / 3 equal to?"
```

## Triển khai

`outputs/skill-ai-tutor.md` là sản phẩm bàn giao. Một gia sư thích ứng chuyên biệt cho môn học với đầu vào đa phương thức, mô hình người học, bộ nhớ, tính an toàn và hiệu quả đã được đo lường.

| Trọng số | Tiêu chí | Cách đo lường |
|:-:|---|---|
| 25 | Mức độ tăng trưởng học tập | Delta điểm kiểm tra trước/sau trong nghiên cứu 10 người học/2 tuần |
| 20 | Độ trung thực Socratic | Điểm rubric trên các mẫu bản ghi hội thoại |
| 20 | UX đa phương thức | Sự mạch lạc của giọng nói + ảnh + văn bản từ đầu đến cuối |
| 20 | Tư thế an toàn + quyền riêng tư | Tỷ lệ vượt qua Llama Guard 4 + lưu trữ tuân thủ COPPA |
| 15 | Độ rộng chương trình và chất lượng đồ thị | Độ bao phủ khái niệm + tính nhất quán của đồ thị điều kiện tiên quyết |
| **100** | | |

## Bài tập

1. Chạy nghiên cứu hiệu quả với và không có mô hình người học thích ứng (thứ tự khái niệm ngẫu nhiên). Báo cáo delta. Dự kiến mô hình thích ứng sẽ thắng, nhưng con số chênh lệch mới là điều thú vị.

2. Thêm một thử nghiệm đa phương thức: cùng một câu hỏi khái niệm được truyền tải dưới dạng văn bản, giọng nói và ảnh. Đo lường xem người học có tiếp thu nhanh hơn với phương thức họ ưa thích hay không.

3. Xây dựng bảng điều khiển cho phụ huynh: các chủ đề đã luyện tập, quỹ đạo nắm vững kiến thức, các khái niệm sắp tới, các sự kiện an toàn (bất kỳ lần kích hoạt bộ lọc nào). Tuân thủ COPPA.

4. Thêm chế độ chuyển đổi ngôn ngữ: gia sư chấp nhận đầu vào tiếng Tây Ban Nha và giảng dạy bằng tiếng Tây Ban Nha. Đo lường độ bao phủ của X-Guard.

5. Kiểm tra quyền riêng tư của bộ nhớ: xác minh rằng người học A không thể xem dữ liệu của người học B ngay cả thông qua tấn công tái nhập clip giọng nói. Ghi nhật ký nỗ lực truy cập và gửi cảnh báo.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|------------------------|
| Socratic policy | "Hỏi, đừng đưa đáp án" | Gia sư đặt câu hỏi gợi mở thay vì đưa ra câu trả lời |
| Bayesian knowledge tracing | "BKT" | Các phương trình mô hình người học cổ điển cho xác suất nắm vững kiến thức |
| FSRS | "Free Spaced Repetition Scheduler" | Bộ lập lịch lặp lại ngắt quãng năm 2024, tốt hơn SM-2 |
| Curriculum graph | "Concept DAG" | Neo4j các khái niệm với các cạnh điều kiện tiên quyết |
| Episodic memory | "Nhật ký tương tác" | Mọi tương tác được lưu trữ để truy xuất sau này |
| Semantic memory | "Kho lưu trữ mẫu đã học" | Các lỗi sai và sở thích được nén và nâng cấp từ bộ nhớ tình tiết |
| COPPA | "Luật quyền riêng tư trẻ em" | Luật Mỹ hạn chế thu thập dữ liệu từ trẻ em dưới 13 tuổi |

## Đọc thêm

- [Khanmigo (Khan Academy)](https://www.khanmigo.ai) — gia sư K-12 tiêu dùng tham khảo
- [Duolingo Max](https://blog.duolingo.com/duolingo-max/) — gia sư học ngôn ngữ tham khảo
- [Google LearnLM / Gemini for Education](https://blog.google/technology/google-deepmind/learnlm) — mô hình tham khảo được lưu trữ
- [Quizlet Q-Chat](https://quizlet.com) — tham khảo thay thế
- [Synthesis Tutor](https://www.synthesis.com) — startup tham khảo
- [FSRS algorithm](https://github.com/open-spaced-repetition/fsrs4anki) — bộ lập lịch lặp lại ngắt quãng
- [Bayesian Knowledge Tracing](https://en.wikipedia.org/wiki/Bayesian_knowledge_tracing) — mô hình người học cổ điển
- [LiveKit Agents](https://github.com/livekit/agents) — stack giọng nói