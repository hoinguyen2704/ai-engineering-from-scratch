# OpenAI Preparedness Framework và DeepMind Frontier Safety Framework

> OpenAI Preparedness Framework v2 (tháng 4 năm 2025) giới thiệu các Danh mục Nghiên cứu (Research Categories) — bao gồm Tự chủ tầm xa (Long-range Autonomy), Sandbagging, Tự sao chép và Thích ứng (Autonomous Replication and Adaptation), Làm suy yếu các biện pháp bảo vệ (Undermining Safeguards) — tách biệt với các Danh mục được theo dõi (Tracked Categories). Các Danh mục được theo dõi sẽ kích hoạt Báo cáo năng lực (Capabilities Reports) cùng với Báo cáo biện pháp bảo vệ (Safeguards Reports) được xem xét bởi Nhóm Cố vấn An toàn (Safety Advisory Group). FSF v3 của DeepMind (tháng 9 năm 2025, với các Mức năng lực được theo dõi - Tracked Capability Levels được bổ sung vào ngày 17 tháng 4 năm 2026) tích hợp tính tự chủ vào các lĩnh vực ML R&D và An ninh mạng (mức độ tự chủ 1 trong ML R&D = tự động hóa hoàn toàn quy trình R&D AI với chi phí cạnh tranh so với con người + công cụ AI). FSF v3 giải quyết rõ ràng vấn đề căn chỉnh lừa dối (deceptive alignment) thông qua giám sát tự động đối với việc lạm dụng suy luận công cụ (instrumental-reasoning). Một lưu ý trung thực: Các Danh mục Nghiên cứu trong PF v2 (bao gồm Tự chủ tầm xa) không tự động kích hoạt các biện pháp giảm thiểu; ngôn ngữ chính sách sử dụng từ "tiềm năng". Bản thân DeepMind cũng thừa nhận rằng việc giám sát tự động "sẽ không còn đủ về lâu dài" nếu suy luận công cụ trở nên mạnh mẽ hơn.

**Type:** Learn
**Languages:** Python (stdlib, three-framework decision-table diff tool)
**Prerequisites:** Phase 15 · 19 (Anthropic RSP)
**Time:** ~45 phút

## Vấn đề

Bài học 19 đã phân tích kỹ chính sách mở rộng quy mô của Anthropic. Bài học này hoàn thiện bức tranh bằng cách đọc các tài liệu của OpenAI và DeepMind. Ba tài liệu này là các văn bản có liên quan chặt chẽ, cùng giải quyết một câu hỏi — khi nào một phòng thí nghiệm tiên phong nên tạm dừng hoặc kiểm soát một mô hình — và chúng hội tụ ở một số danh mục nhỏ nhưng lại khác biệt ở những điểm cụ thể quan trọng.

Sự hội tụ: cả ba đều coi tính tự chủ tầm xa là một lớp năng lực đáng theo dõi. Cả ba đều thừa nhận hành vi lừa dối (giả vờ căn chỉnh, sandbagging) là một loại rủi ro cụ thể. Cả ba đều có cơ quan đánh giá nội bộ. Sự khác biệt: OpenAI chia các danh mục thành "Được theo dõi" (bắt buộc giảm thiểu) và "Nghiên cứu" (không có kích hoạt tự động). DeepMind tích hợp tính tự chủ vào hai lĩnh vực thay vì đặt tên riêng biệt. Các phòng thí nghiệm đặt tên cho các nhóm như Tracked vs Research, hoặc Critical vs Moderate, hoặc Tier-1 vs Tier-2; hệ quả vận hành của việc một năng lực nằm trong nhóm nào là khác nhau giữa các phòng thí nghiệm.

Đọc chúng cùng nhau là một bài tập hữu ích. Cùng một năng lực có thể là "bắt buộc giảm thiểu" tại Anthropic, "được giám sát nhưng không kích hoạt" tại OpenAI, và "được theo dõi trong một lĩnh vực cụ thể" tại DeepMind. Quan điểm chính sách là điều quan trọng.

## Khái niệm

### OpenAI Preparedness Framework v2 (tháng 4 năm 2025)

Cấu trúc:

- **Tracked Categories (Danh mục được theo dõi)**: kích hoạt Báo cáo năng lực (những gì mô hình có thể làm) cộng với Báo cáo biện pháp bảo vệ (các biện pháp giảm thiểu đã được áp dụng). Được xem xét bởi Nhóm Cố vấn An toàn trước khi triển khai.
- **Research Categories (Danh mục nghiên cứu)**: các năng lực của mô hình mà phòng thí nghiệm đang theo dõi nhưng chưa cam kết thực hiện các biện pháp giảm thiểu cụ thể. Bao gồm Tự chủ tầm xa, Sandbagging, Tự sao chép và Thích ứng, Làm suy yếu các biện pháp bảo vệ.

Cách định khung chính trong v2: Các Danh mục Nghiên cứu không tự động kích hoạt các biện pháp giảm thiểu. Ngôn ngữ chính sách là các biện pháp giảm thiểu "tiềm năng". Đây là điểm khác biệt cụ thể cần theo dõi: vị trí của một danh mục trong Tracked hay Research là bản lề giữa "được kiểm soát" và "được quan sát".

### DeepMind Frontier Safety Framework v3 (tháng 9 năm 2025; Tracked Capability Levels được bổ sung ngày 17 tháng 4 năm 2026)

Cấu trúc:

- **Critical Capability Levels (CCLs - Các mức năng lực tới hạn)**: các ngưỡng năng lực trên năm lĩnh vực: An ninh mạng, Sinh học, ML R&D, CBRN, Tự chủ (được tích hợp vào ML R&D và An ninh mạng).
- **Tracked Capability Levels (Các mức năng lực được theo dõi)**: độ chi tiết bổ sung được thêm vào tháng 4 năm 2026. Ví dụ cụ thể: mức độ tự chủ 1 trong ML R&D = tự động hóa hoàn toàn quy trình R&D AI với chi phí cạnh tranh so với con người + công cụ AI.
- **Giám sát căn chỉnh lừa dối**: cam kết rõ ràng về việc giám sát tự động đối với việc lạm dụng suy luận công cụ.

Cách định khung về tính tự chủ khác với OpenAI. DeepMind không giữ "Tự chủ" là một lĩnh vực cấp cao nhất; nó được tích hợp vào các lĩnh vực mà tính tự chủ có thể gây hại (ML R&D và An ninh mạng). Lập luận là tính tự chủ mà không có lĩnh vực cụ thể thì chỉ là năng lực mà không có rủi ro; lập luận phản bác là tính tự chủ trên các lĩnh vực là một rủi ro meta mà khung chính sách nên gọi tên.

### Những điểm hội tụ của cả ba

- Nhóm Cố vấn An toàn nội bộ (được gọi là Anthropic SAG, OpenAI SAG, ủy ban nội bộ DeepMind). Đánh giá trước khi triển khai đối với các mô hình có năng lực cao.
- Đề cập rõ ràng đến căn chỉnh lừa dối / giả vờ căn chỉnh như một loại rủi ro.
- Các văn bản định kỳ theo lịch trình đã công bố (Anthropic: Frontier Safety Roadmap, Risk Report; OpenAI: Capabilities and Safeguards Reports; DeepMind: chu kỳ cập nhật FSF).
- Thừa nhận rằng các biện pháp phòng thủ chỉ dựa trên giám sát có giới hạn. DeepMind khẳng định rõ ràng: "việc giám sát tự động sẽ không còn đủ về lâu dài".

### Những điểm khác biệt

- **Anthropic**: cam kết tạm dừng đã bị loại bỏ trong v3.0; ngưỡng AI R&D-4 là cổng kiểm soát tiếp theo được nêu tên.
- **OpenAI**: phân chia Tracked vs Research; các Danh mục Nghiên cứu (bao gồm Tự chủ tầm xa) không tự động kích hoạt kiểm soát.
- **DeepMind**: tính tự chủ được tích hợp vào các lĩnh vực khác; Tracked Capability Levels bổ sung độ chi tiết vào tháng 4 năm 2026.

### Sandbagging: một năng lực cụ thể làm phức tạp cả ba

Sandbagging (một mô hình cố tình thể hiện kém trong các bài đánh giá) nằm trong các Danh mục Nghiên cứu của OpenAI. RSP v3.0 của Anthropic giải quyết vấn đề này thông qua khoảng cách ngữ cảnh đánh giá (Bài học 1). DeepMind giải quyết vấn đề này thông qua giám sát căn chỉnh lừa dối trong FSF v3.

Nếu một mô hình thực hiện sandbagging trong các bài đánh giá, các ngưỡng năng lực của mọi khung chính sách đều bị đánh giá thấp. Khung chính sách chỉ hoạt động nếu phép đo lường hoạt động. Đây là lý do tại sao việc đo lường bên ngoài (Bài học 21, METR) và đánh giá đối kháng là cần thiết bên cạnh việc tự đánh giá của phòng thí nghiệm.

### Kỹ năng đọc chính sách

- Xác định vị trí: mọi năng lực bạn quan tâm đều phải tìm thấy được trong chính sách. Nếu không tìm thấy, chính sách không bao quát nó.
- Phân loại: nó là Tracked (kích hoạt giảm thiểu) hay Research (được theo dõi nhưng không kích hoạt)? OpenAI đặt tên rõ ràng; Anthropic và DeepMind có các tương đương riêng.
- Chu kỳ: chính sách được cập nhật theo lịch trình đã công bố hay chỉ sau các sự kiện cụ thể? Chu kỳ đã công bố là mạnh mẽ hơn.
- Độc lập: đánh giá bên ngoài là bắt buộc hay tùy chọn? Anthropic hợp tác với Apollo và Viện An toàn AI Hoa Kỳ; OpenAI với METR; DeepMind chủ yếu với SAG nội bộ.

```figure
a5-tracked-vs-research
```

## Sử dụng

`code/main.py` triển khai một công cụ so sánh bảng quyết định nhỏ. Với một năng lực (tự chủ, căn chỉnh lừa dối, tự động hóa R&D, nâng cao năng lực mạng, v.v.), nó xuất ra cách mỗi chính sách trong ba chính sách phân loại năng lực đó và các biện pháp giảm thiểu nào được kích hoạt. Đây là công cụ hỗ trợ đọc, không phải công cụ chính sách.

## Triển khai

`outputs/skill-cross-policy-diff.md` tạo ra một so sánh chéo giữa các chính sách cho một năng lực cụ thể, sử dụng ba khung chính sách làm tham chiếu.

## Bài tập

1. Chạy `code/main.py`. Xác nhận đầu ra của công cụ so sánh khớp với các chính sách cho ít nhất hai năng lực mà bạn có thể kiểm chứng với tài liệu gốc.

2. Đọc toàn bộ OpenAI Preparedness Framework v2. Xác định từng Danh mục Nghiên cứu. Với mỗi danh mục, hãy viết một câu giải thích tại sao nó nằm trong Research thay vì Tracked.

3. Đọc toàn bộ DeepMind FSF v3, cộng với bản cập nhật Tracked Capability Levels tháng 4 năm 2026. Xác định các tiêu chí đánh giá cụ thể cho mức độ tự chủ 1 trong ML R&D. Bạn sẽ đo lường nó từ bên ngoài như thế nào?

4. Sandbagging nằm trong các Danh mục Nghiên cứu của OpenAI. Hãy thiết kế một bài đánh giá buộc một mô hình sandbagging phải bộc lộ năng lực thực sự của nó. Tham khảo thảo luận về eval-context-gaming trong Bài học 1.

5. So sánh ba chính sách về một năng lực cụ thể (do bạn chọn). Nêu tên chính sách mà bạn thấy nghiêm ngặt nhất và ít nghiêm ngặt nhất. Biện minh bằng văn bản gốc.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|---|---|---|
| Preparedness Framework | "Chính sách mở rộng của OpenAI" | PF v2 (tháng 4 năm 2025); các danh mục Tracked vs Research |
| Tracked Category | "Giảm thiểu bắt buộc" | Kích hoạt Báo cáo năng lực + Báo cáo biện pháp bảo vệ; đánh giá bởi SAG |
| Research Category | "Chỉ giám sát" | Được theo dõi nhưng không có giảm thiểu tự động; bao gồm Tự chủ tầm xa |
| Frontier Safety Framework | "Chính sách mở rộng của DeepMind" | FSF v3 (tháng 9 năm 2025) + Tracked Capability Levels (tháng 4 năm 2026) |
| CCL | "Critical Capability Level" | Ngưỡng của DeepMind theo từng lĩnh vực (An ninh mạng, Sinh học, ML R&D, CBRN) |
| ML R&D autonomy level 1 | "Tự động hóa R&D" | Tự động hóa hoàn toàn quy trình R&D AI với chi phí cạnh tranh |
| Sandbagging | "Hiệu suất kém chiến lược" | Mô hình thể hiện kém trong các bài đánh giá; nằm trong Danh mục Nghiên cứu của OpenAI |
| Instrumental reasoning | "Suy luận phương tiện-mục đích" | Suy luận về cách đạt được mục tiêu; mục tiêu giám sát của DeepMind |

## Đọc thêm

- [OpenAI — Cập nhật Preparedness Framework](https://openai.com/index/updating-our-preparedness-framework/) — thông báo v2.
- [OpenAI — PDF Preparedness Framework v2](https://cdn.openai.com/pdf/18a02b5d-6b67-4cec-ab64-68cdfbddebcd/preparedness-framework-v2.pdf) — tài liệu đầy đủ.
- [DeepMind — Tăng cường Frontier Safety Framework](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — thông báo FSF v3.
- [DeepMind — Cập nhật Frontier Safety Framework (tháng 4 năm 2026)](https://deepmind.google/blog/updating-the-frontier-safety-framework/) — bổ sung Tracked Capability Levels.
- [Báo cáo FSF Gemini 3 Pro](https://storage.googleapis.com/deepmind-media/gemini/gemini_3_pro_fsf_report.pdf) — ví dụ về Báo cáo rủi ro theo định dạng FSF.