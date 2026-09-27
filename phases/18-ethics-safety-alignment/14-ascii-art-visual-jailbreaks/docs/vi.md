# ASCII Art và Visual Jailbreaks

> Jiang, Xu, Niu, Xiang, Ramasubramanian, Li, Poovendran, "ArtPrompt: ASCII Art-based Jailbreak Attacks against Aligned LLMs" (ACL 2024, arXiv:2402.11753). Che giấu các token liên quan đến an toàn trong một yêu cầu độc hại, thay thế chúng bằng các hình vẽ ASCII của cùng các chữ cái đó, và gửi prompt đã được ngụy trang. GPT-3.5, GPT-4, Gemini, Claude, Llama-2 đều thất bại trong việc nhận diện một cách mạnh mẽ các token ASCII-art. Cuộc tấn công này vượt qua được PPL (bộ lọc perplexity), các biện pháp phòng thủ Paraphrase và Retokenization. Liên quan: benchmark ViTC đo lường khả năng nhận diện các visual prompt phi ngữ nghĩa; StructuralSleight khái quát hóa thành các Uncommon Text-Encoded Structures (cây, đồ thị, JSON lồng nhau) như một họ các cuộc tấn công mã hóa.

**Type:** Build
**Languages:** Python (stdlib, ArtPrompt token-masking harness)
**Prerequisites:** Phase 18 · 12 (PAIR), Phase 18 · 13 (MSJ)
**Time:** ~60 phút

## Mục tiêu học tập

- Mô tả cuộc tấn công ArtPrompt: bước nhận diện từ, thay thế bằng ASCII-art, và prompt ngụy trang cuối cùng.
- Giải thích lý do tại sao các biện pháp phòng thủ tiêu chuẩn (PPL, Paraphrase, Retokenization) thất bại trước ArtPrompt.
- Định nghĩa ViTC và mô tả những gì nó đo lường.
- Mô tả StructuralSleight như một sự khái quát hóa đối với các Uncommon Text-Encoded Structures tùy ý.

## Vấn đề

Các cuộc tấn công thông qua paraphrase và nhập vai (Bài 12) và thông qua ngữ cảnh dài (Bài 13) hoạt động ở cấp độ mẫu văn bản. ArtPrompt hoạt động ở cấp độ nhận diện: mô hình không phân tích cú pháp token bị cấm. Nó phân tích một hình ảnh được hiển thị bằng các ký tự. Bộ lọc an toàn chỉ thấy các dấu câu vô hại. Mô hình thấy một từ.

## Khái niệm

### ArtPrompt, hai bước

Bước 1. Nhận diện từ. Với một yêu cầu độc hại, kẻ tấn công sử dụng một LLM để xác định các từ liên quan đến an toàn (ví dụ: "bomb" trong "how to make a bomb").

Bước 2. Tạo Prompt ngụy trang. Thay thế mỗi từ đã xác định bằng hình vẽ ASCII của nó (một khối ký tự 7x5 hoặc 7x7 tạo thành hình dạng chữ cái). Mô hình nhận được một lưới các dấu câu và khoảng trắng mà một mô hình đủ khả năng có thể nhận diện thành từ; bộ lọc an toàn chỉ thấy lưới đó.

Kết quả: GPT-4, Gemini, Claude, Llama-2, GPT-3.5 đều thất bại. Tỷ lệ tấn công thành công trên tập benchmark của họ là trên 75%.

### Tại sao các biện pháp phòng thủ tiêu chuẩn thất bại

- **PPL (bộ lọc perplexity).** ASCII art có perplexity cao — nhưng tất cả các đầu vào mới lạ cũng vậy. Việc chọn ngưỡng chặn ArtPrompt cũng sẽ chặn cả các đầu vào có cấu trúc hợp lệ.
- **Paraphrase.** Việc diễn giải lại prompt sẽ phá hủy ASCII art. Trên thực tế, các LLM dùng để paraphrase thường bảo toàn hoặc tái tạo lại hình ảnh đó.
- **Retokenization.** Việc chia tách token theo cách khác không làm thay đổi thực tế là thị giác của mô hình đang nhận diện hình dạng chữ cái.

Vấn đề cốt lõi là các bộ lọc an toàn hoạt động ở cấp độ token hoặc ngữ nghĩa; ArtPrompt hoạt động ở cấp độ nhận diện thị giác.

### Benchmark ViTC

Nhận diện các visual prompt phi ngữ nghĩa. Đo lường khả năng đọc ASCII-art, wingdings và các nội dung thị giác phi ngữ nghĩa văn bản khác của mô hình. Hiệu quả của ArtPrompt tương quan với độ chính xác của ViTC: mô hình đọc văn bản thị giác càng tốt, ArtPrompt càng hoạt động hiệu quả trên mô hình đó. Đây là sự đánh đổi giữa khả năng và an toàn.

### StructuralSleight

Khái quát hóa ArtPrompt: Uncommon Text-Encoded Structures (UTES). Cây, đồ thị, JSON lồng nhau, CSV-in-JSON, các khối mã kiểu diff. Nếu một cấu trúc hiếm gặp trong dữ liệu an toàn huấn luyện nhưng mô hình có thể phân tích cú pháp, nó có thể che giấu nội dung độc hại.

Hệ quả đối với phòng thủ: an toàn phải khái quát hóa trên các biểu diễn có cấu trúc mà mô hình có thể phân tích. Tập hợp này rất lớn và đang ngày càng mở rộng.

### Tương tự với phương thức hình ảnh

Các Visual LLM (GPT-5.2, Gemini 3 Pro, Claude Opus 4.5, Grok 4.1) mở rộng bề mặt tấn công. Các cuộc tấn công kiểu ArtPrompt với hình ảnh thực tế mạnh hơn các bản tương tự ASCII-art vì các bộ mã hóa hình ảnh tạo ra tín hiệu phong phú hơn.

### Vị trí trong Phase 18

Các bài 12-14 mô tả ba vectơ tấn công trực giao: tinh chỉnh lặp lại (PAIR), độ dài ngữ cảnh (MSJ), và mã hóa (ArtPrompt/StructuralSleight). Bài 15 chuyển từ các cuộc tấn công tập trung vào mô hình sang các cuộc tấn công vào ranh giới hệ thống (indirect prompt injection). Bài 16 mô tả phản ứng của công cụ phòng thủ.

```figure
al-ascii-cloak
```

## Sử dụng

`code/main.py` xây dựng một ArtPrompt đơn giản. Bạn có thể ngụy trang các từ cụ thể trong một truy vấn độc hại bằng các ký tự ASCII-art, xác minh chuỗi đã ngụy trang vượt qua bộ lọc từ khóa, và (tùy chọn) giải mã chuỗi đã ngụy trang bằng một bộ nhận diện đơn giản.

## Triển khai

Bài học này tạo ra `outputs/skill-encoding-audit.md`. Với một báo cáo phòng thủ jailbreak, nó liệt kê các họ tấn công mã hóa được đề cập (ASCII art, base64, leet-speak, UTF-8 homoglyph, UTES) và lớp phòng thủ bắt được từng loại.

## Bài tập

1. Chạy `code/main.py`. Xác minh chuỗi đã ngụy trang vượt qua bộ lọc từ khóa đơn giản. Báo cáo sự thay đổi ở cấp độ ký tự cần thiết.

2. Triển khai một kiểu mã hóa thứ hai: base64 cho cùng một từ mục tiêu. So sánh tỷ lệ vượt qua bộ lọc với ArtPrompt và độ khó khôi phục.

3. Đọc Jiang et al. 2024 Mục 4.3 (kết quả của năm mô hình). Đề xuất lý do tại sao khả năng kháng ArtPrompt của Claude cao hơn Gemini trên cùng một benchmark.

4. Thiết kế một biện pháp phòng thủ trước khi tạo (pre-generation) để phát hiện các vùng có hình dạng ASCII-art trong prompt. Đo lường tỷ lệ dương tính giả trên mã nguồn, bảng và ký hiệu toán học hợp lệ.

5. StructuralSleight liệt kê 10 cấu trúc mã hóa. Phác thảo một biện pháp phòng thủ khái quát xử lý cả 10 loại và ước tính chi phí tính toán cho mỗi prompt được bảo vệ.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|------------------------|
| ArtPrompt | "tấn công ASCII-art" | Jailbreak hai bước che giấu các từ an toàn bằng hình vẽ ASCII |
| Cloaking | "ẩn từ" | Thay thế một token bị cấm bằng biểu diễn thị giác mà mô hình đọc được nhưng bộ lọc thì không |
| UTES | "cấu trúc không phổ biến" | Uncommon Text-Encoded Structure — cây, đồ thị, JSON lồng nhau, v.v. được dùng để tuồn nội dung |
| ViTC | "khả năng văn bản thị giác" | Benchmark đo khả năng đọc mã hóa thị giác phi ngữ nghĩa của mô hình |
| Perplexity filter | "phòng thủ PPL" | Từ chối các prompt có perplexity cao; thất bại vì đầu vào có cấu trúc hợp lệ cũng có điểm cao |
| Retokenization | "phòng thủ chuyển đổi tokenizer" | Tiền xử lý prompt bằng một tokenizer khác; thất bại vì nhận diện là dựa trên thị giác |
| Homoglyph | "ký tự trông giống nhau" | Các ký tự Unicode trông giống hệt chữ cái Latin; vượt qua kiểm tra chuỗi con |

## Đọc thêm

- [Jiang et al. — ArtPrompt (ACL 2024, arXiv:2402.11753)](https://arxiv.org/abs/2402.11753) — bài báo về jailbreak ASCII-art
- [Li et al. — StructuralSleight (arXiv:2406.08754)](https://arxiv.org/abs/2406.08754) — khái quát hóa UTES
- [Chao et al. — PAIR (Bài 12, arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — tấn công lặp lại bổ trợ
- [Anil et al. — Many-shot Jailbreaking (Bài 13)](https://www.anthropic.com/research/many-shot-jailbreaking) — tấn công độ dài bổ trợ