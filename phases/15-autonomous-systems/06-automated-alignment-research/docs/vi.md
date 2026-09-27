# Automated Alignment Research (Anthropic AAR)

> Anthropic đã vận hành các nhóm Claude Opus 4.6 Autonomous Alignment Researchers (AAR) song song trong các sandbox độc lập, phối hợp thông qua một diễn đàn chung có nhật ký lưu trữ nằm ngoài bất kỳ sandbox nào (để các tác nhân không thể xóa hồ sơ của chính mình). Đối với bài toán weak-to-strong training, các AAR đã vượt qua các nhà nghiên cứu con người. Bản tóm tắt của Anthropic chỉ ra rằng các quy trình làm việc được quy định sẵn thường hạn chế sự linh hoạt của AAR và làm giảm hiệu suất. Tự động hóa nghiên cứu căn chỉnh (alignment research) là bước nén giúp rút ngắn thời gian dẫn đến các rủi ro sai lệch (misalignment) mà RSP dự định phát hiện.

**Type:** Learn
**Languages:** Python (stdlib, parallel-research-forum simulator)
**Prerequisites:** Phase 15 · 05 (AI Scientist v2), Phase 15 · 04 (DGM)
**Time:** ~60 minutes

## Vấn đề

Nghiên cứu căn chỉnh (alignment research) rất tốn kém về thời gian của các nhà nghiên cứu con người. Các vấn đề như scalable oversight, reward specification, hoặc weak-to-strong training đòi hỏi các thí nghiệm kéo dài hàng tuần cho mỗi lần lặp lại. Khi các năng lực tiên phong (frontier capabilities) tiến bộ, khối lượng công việc căn chỉnh tăng nhanh hơn nguồn cung các nhà nghiên cứu có trình độ.

Automated Alignment Research (AAR) đặt câu hỏi liệu chính các mô hình tiên phong đang có năng lực vượt xa khả năng căn chỉnh có thể góp phần thu hẹp khoảng cách này hay không. Báo cáo năm 2026 của Anthropic về một nghiên cứu weak-to-strong-training do AAR thực hiện (alignment.anthropic.com/2026/automated-w2s-researcher/) là một trong những kết quả công khai đầu tiên từ một hệ thống được triển khai thuộc loại này.

Kết quả thực sự tích cực: Các AAR đã giải quyết một vấn đề nghiên cứu tốt hơn so với kết quả cơ sở từ con người. Kết quả này cũng tập trung vào một hệ thống duy nhất, mọi mối quan ngại mà giai đoạn này đã phát triển. Nếu nghiên cứu căn chỉnh có thể được tự động hóa, thì các phần làm tổn hại đến các biện pháp bảo vệ cũng có thể bị tự động hóa. Các ngưỡng của RSP đối với năng lực R&D AI được viết với vòng lặp này trong tâm trí.

## Khái niệm

### Kiến trúc (như được mô tả công khai)

- **Các tác nhân song song.** Nhiều instance Claude Opus 4.6 chạy dưới dạng AAR, mỗi instance trong một sandbox độc lập. Các sandbox giới hạn hệ thống tệp, mạng và bề mặt tiến trình.
- **Diễn đàn chung.** Các phát hiện được công bố lên một diễn đàn chung có backend lưu trữ nằm ngoài mọi sandbox. Các tác nhân có thể đọc diễn đàn; các tác nhân không thể xóa hoặc chỉnh sửa hồ sơ diễn đàn từ bên trong sandbox của chính chúng.
- **Phân bổ nhiệm vụ.** Một hàng đợi nhiệm vụ do con người thiết kế sẽ gán các vấn đề nghiên cứu cho các AAR. Các AAR đề xuất phân rã, chạy thí nghiệm và viết báo cáo kết quả.
- **Trọng tâm weak-to-strong training.** Nghiên cứu được công bố tập trung vào weak-to-strong training — sử dụng một mô hình giám sát yếu hơn để huấn luyện một mô hình học viên mạnh hơn. Các AAR đã vượt qua các nhà nghiên cứu con người trong nhiệm vụ này.

### Tại sao nhật ký ngoài sandbox lại quan trọng

Tính toàn vẹn của nhật ký là sự khác biệt giữa một thí nghiệm được kiểm soát và một thí nghiệm có khả năng bị xâm phạm. Nếu các AAR có thể chỉnh sửa nhật ký của chính mình, chúng có thể che đậy các thí nghiệm thất bại hoặc đáng ngờ. Anthropic chỉ ra điều này trực tiếp: thuộc tính này là điều kiện cần để tin tưởng vào kết quả nghiên cứu, và việc xác minh nó không hề đơn giản.

Tương tự về mặt cơ sở hạ tầng là các nhật ký chỉ cho phép ghi thêm (append-only) với cơ chế ghi trực tiếp vào lưu trữ bên ngoài (ví dụ: S3 object-lock, nhật ký Datadog bất biến). Tác nhân không có thông tin xác thực để sửa đổi các hồ sơ cũ. Đây là mô hình tương tự mà các đội ngũ bảo mật doanh nghiệp năm 2026 sử dụng để tạo ra các dấu vết kiểm toán chống ransomware.

### Sự đánh đổi của quy trình làm việc được quy định sẵn

Báo cáo của Anthropic lưu ý rằng các quy trình làm việc được quy định sẵn thường hạn chế sự linh hoạt của AAR và làm giảm hiệu suất. Nói cách khác: nếu nhà thiết kế con người viết kế hoạch nghiên cứu từng bước một, các AAR sẽ làm kém hơn so với khi được giao các mục tiêu cấp cao và quyền tự do phân rã. Điều này nhất quán với các kết quả trong nghiên cứu tác nhân (LangGraph, Microsoft's Agent Framework) — các khung công tác được chỉ định quá mức sẽ gây hại cho các tác nhân có năng lực.

Hệ quả về an toàn rất tinh vi. Sự linh hoạt giúp cải thiện hiệu suất AAR cũng chính là sự linh hoạt cho phép phân rã không được phê duyệt, sử dụng công cụ mới lạ và trôi dạt mục tiêu (goal-drift). Sự đánh đổi này là một quyết định về không gian thiết kế: bao nhiêu sự linh hoạt là đủ để đánh bại con người mà không tạo điều kiện cho tác nhân khái quát hóa các mục tiêu ngoài phân phối (off-distribution)?

### Rủi ro nén thời gian

RSP v3.0 (Bài 19) giới thiệu một ngưỡng năng lực R&D AI: khả năng tự động hóa hoàn toàn quy trình R&D AI với chi phí cạnh tranh so với con người + công cụ AI. FSF v3 của DeepMind bao gồm một cấp độ tự chủ R&D ML tương tự. Cả hai khung công tác đều coi ngưỡng này là yếu tố kích hoạt các biện pháp kiểm soát nâng cao.

AAR chỉ còn một bước nữa là đạt đến ngưỡng này: nó tự động hóa một phần của quy trình (nghiên cứu căn chỉnh trên các nhiệm vụ cụ thể, được xác định rõ ràng) nhưng không phải là vòng lặp phát triển năng lực end-to-end. Câu hỏi về dòng thời gian là khoảng cách này sẽ được thu hẹp nhanh như thế nào.

Dòng thời gian bị nén là mối lo ngại về sự thất bại kép. Nếu nghiên cứu căn chỉnh và nghiên cứu năng lực cùng phát triển với tốc độ tương tự, bề mặt rủi ro sai lệch sẽ tăng nhanh ít nhất bằng tốc độ tăng năng lực. Nếu năng lực tăng nhanh hơn (xu hướng lịch sử), khoảng cách sẽ ngày càng rộng. Đây là lập luận cho việc AAR là một lợi ích có điều kiện: mỗi kết quả căn chỉnh bổ sung sẽ thu hẹp khoảng cách nếu và chỉ nếu quy trình nghiên cứu là đáng tin cậy.

### Những gì AAR không thay thế

Các nhà nghiên cứu con người thiết lập hàng đợi nhiệm vụ, xem xét kết quả và nắm giữ thẩm quyền hiến định. Các AAR tăng tốc phần giữa của quy trình, không phải hai đầu. Các kết quả đầu ra được công bố của Anthropic bao gồm cả đóng góp của AAR và sự đánh giá của nhà nghiên cứu con người về việc nên công bố cái gì, thu hồi cái gì và tinh chỉnh cái gì.

Điều này khớp với mô hình propose-then-commit từ Bài 15 áp dụng cho chính nghiên cứu: AAR đề xuất; con người cam kết.

```figure
aar-forum
```

## Sử dụng

`code/main.py` mô phỏng một diễn đàn nghiên cứu đa AAR. Ba "tác nhân nghiên cứu" độc lập chạy song song, đăng các phát hiện lên một diễn đàn chỉ cho phép ghi thêm với các nỗ lực giả mạo tùy chọn.

Quan sát:

- Cách ghi nhật ký ngoài sandbox phát hiện các nỗ lực giả mạo nhật ký.
- Cách phân bổ nhiệm vụ với sự phân rã tự do vượt trội hơn các quy trình làm việc cố định trên các vấn đề khó hơn.
- Cách một AAR bị xâm phạm có thể làm ô nhiễm tín hiệu của diễn đàn nếu nhật ký không có bằng chứng giả mạo.

## Triển khai

`outputs/skill-aar-deployment-review.md` là một đánh giá trước khi triển khai của một quy trình nghiên cứu căn chỉnh tự động.

## Bài tập

1. Chạy `code/main.py`. So sánh cài đặt "quy trình làm việc cố định" (fixed-workflow) và "phân rã tự do" (free-decomposition). Liệu phân rã tự do có luôn thắng, hay có những lớp vấn đề mà quy trình làm việc cố định có tính cạnh tranh?

2. Sửa đổi trình mô phỏng để một tác nhân cố gắng giả mạo nhật ký. Xác nhận nhật ký chỉ cho phép ghi thêm phát hiện ra điều đó. Viết một đoạn văn mô tả chính xác chữ ký phát hiện trông như thế nào trong nhật ký.

3. Đọc báo cáo AAR weak-to-strong của Anthropic. Xác định nhiệm vụ phụ cụ thể mà các AAR đã đánh bại các nhà nghiên cứu con người. Điều gì làm cho nó phù hợp với tự động hóa?

4. Thiết kế một chính sách phân bổ hàng đợi nhiệm vụ cân bằng giữa sự linh hoạt của AAR (kết quả tốt hơn) và các ràng buộc quy trình làm việc được quy định sẵn (kiểm toán dễ dàng hơn). Mô tả cách bạn sẽ thực hiện A/B test cho hai chính sách này.

5. Đọc ngưỡng AI R&D-4 của RSP v3.0. Trong một đoạn văn, hãy mô tả những gì bạn nghĩ sẽ vượt qua ngưỡng đó mà AAR hiện tại chưa làm được.

## Thuật ngữ chính

| Thuật ngữ | Mọi người nói | Ý nghĩa thực tế |
|---|---|---|
| AAR | "Automated Alignment Researcher" | Instance Claude Opus 4.6 vận hành tự động trên các vấn đề căn chỉnh |
| Weak-to-strong training | "Huấn luyện mô hình mạnh hơn với giám sát yếu hơn" | Benchmark scalable-oversight kinh điển mà AAR đã vượt qua con người |
| Shared forum | "Nơi các tác nhân công bố phát hiện" | Lưu trữ chỉ cho phép ghi thêm, nằm ngoài sandbox |
| Out-of-sandbox log | "Tác nhân không thể chỉnh sửa hồ sơ của chính mình" | Ghi trực tiếp vào lưu trữ bên ngoài có bằng chứng giả mạo |
| Prescribed workflow | "Kế hoạch từng bước từ nhà thiết kế con người" | Hạn chế AAR; thường làm giảm hiệu suất so với phân rã tự do |
| Free decomposition | "Tác nhân quyết định cách chia nhỏ nhiệm vụ" | Có năng lực hơn, khó kiểm toán hơn |
| AI R&D threshold | "Cấp độ năng lực RSP/FSF" | Tự động hóa hoàn toàn quy trình R&D với chi phí cạnh tranh |
| Compressed timeline | "Cuộc đua căn chỉnh vs năng lực" | Nếu năng lực tăng nhanh hơn căn chỉnh, rủi ro sai lệch sẽ tăng |

## Đọc thêm

- [Anthropic — Automated Weak-to-Strong Researcher](https://alignment.anthropic.com/2026/automated-w2s-researcher/) — nguồn chính.
- [Anthropic Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — khung ngưỡng R&D AI.
- [Anthropic — Measuring AI agent autonomy](https://www.anthropic.com/research/measuring-agent-autonomy) — khung tự chủ tác nhân rộng hơn.
- [DeepMind Frontier Safety Framework v3](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — các cấp độ tự chủ R&D ML song song với RSP.
- [Burns et al. (2023). Weak-to-Strong Generalization (OpenAI)](https://openai.com/index/weak-to-strong-generalization/) — vấn đề cơ bản mà các AAR đã tấn công.