# Indirect Prompt Injection — Bề mặt tấn công trong môi trường Production

> Indirect prompt injection (IPI) là việc nhúng các chỉ dẫn vào bên trong nội dung bên ngoài — một trang web, email, tài liệu được chia sẻ, hoặc phiếu hỗ trợ — mà hệ thống agentic tiêu thụ mà không cần hành động rõ ràng từ người dùng. IPI là mối đe dọa sản xuất chủ đạo vào năm 2026: nó vượt qua các bộ lọc đầu vào của người dùng vì kẻ tấn công không bao giờ tiếp cận người dùng, nó mở rộng quy mô một cách âm thầm khi các agent xử lý ngày càng nhiều nội dung bên ngoài, và nó nhắm vào các quy trình làm việc tự động nơi không có ai đọc prompt. MDPI Information 17(1):54 (tháng 1 năm 2026) tổng hợp các nghiên cứu từ 2023-2025. Bài báo về phòng thủ IPI tại NDSS 2026 đã đặt ra thách thức cốt lõi: các chỉ dẫn được tiêm vào có thể hoàn toàn lành tính về mặt ngữ nghĩa (ví dụ: "vui lòng in chữ Yes"), vì vậy việc phát hiện đòi hỏi nhiều hơn là chỉ lọc từ khóa. "The Attacker Moves Second" (Nasr và cộng sự, hợp tác giữa OpenAI/Anthropic/DeepMind, tháng 10 năm 2025): các cuộc tấn công thích ứng (gradient, RL, tìm kiếm ngẫu nhiên, red-team con người) đã phá vỡ >90% trong số 12 phương pháp phòng thủ được công bố trước đó vốn báo cáo tỷ lệ thành công gần bằng 0.

**Type:** Build
**Languages:** Python (stdlib, IPI attack + defense harness)
**Prerequisites:** Phase 18 · 12 (PAIR), Phase 14 (agent engineering)
**Time:** ~75 phút

## Mục tiêu học tập

- Định nghĩa indirect prompt injection và mô tả ba vector phân phối phổ biến.
- Giải thích lý do tại sao các bộ lọc đầu vào của người dùng hoàn toàn bỏ lỡ IPI.
- Mô tả khung "kiểm soát luồng thông tin" (information flow control) như một mô hình phòng thủ năm 2026.
- Nêu kết luận của Nasr và cộng sự (tháng 10 năm 2025) về sự thành công của các cuộc tấn công thích ứng đối với các phương pháp phòng thủ IPI đã được công bố.

## Vấn đề

Direct prompt injection đòi hỏi kẻ tấn công phải tiếp cận người dùng hoặc prompt của họ. IPI không đòi hỏi điều đó: kẻ tấn công đặt payload vào bất kỳ nội dung nào mà agent có thể đọc — một trang web, email trong hộp thư đến, GitHub issue, hoặc đánh giá sản phẩm. Agent sẽ lấy nội dung đó trong quá trình vận hành bình thường và thực thi các chỉ dẫn. Người dùng chỉ là người truyền tin, không phải là người có ý định.

## Khái niệm

### Ba vector phân phối

- **Retrieval-augmented generation (RAG).** Kẻ tấn công xuất bản một tài liệu; bước truy xuất (retrieval) lấy nó về; prompt nối nó vào trước câu hỏi của người dùng; mô hình thực thi các chỉ dẫn của kẻ tấn công.
- **Quy trình làm việc với Inbox / tài liệu.** Kẻ tấn công gửi email cho người dùng; agent đọc email; prompt bao gồm nội dung email; mô hình tuân theo các chỉ dẫn trong email.
- **Đầu ra của công cụ (Tool output).** Kẻ tấn công kiểm soát một công cụ mà agent sử dụng (ví dụ: tìm kiếm web trả về kết quả do kẻ tấn công kiểm soát); đầu ra của công cụ chứa các chỉ dẫn; luồng điều khiển của agent tuân theo chúng.

Cả ba đều chia sẻ một đặc tính cấu trúc: kẻ tấn công kiểm soát một phần của prompt mà không cần chạm vào đầu vào từ phía người dùng.

### Tại sao các bộ lọc đầu vào của người dùng bỏ lỡ nó

Payload IPI không xuất hiện trong đầu vào của người dùng. Nó xuất hiện trong nội dung được truy xuất. Nếu bộ lọc chỉ dựa trên đầu vào của người dùng, payload sẽ vượt qua nó. Nếu bộ lọc dựa trên tất cả nội dung đến được mô hình, nó phải áp dụng cho văn bản truy xuất tùy ý — điều này rất tốn kém và tạo ra các kết quả dương tính giả đối với nội dung hợp pháp vô tình chứa ngôn ngữ mệnh lệnh.

### Kiểm soát luồng thông tin (IFC) cho AI

Mô hình phòng thủ năm 2026 vay mượn từ bảo mật hệ điều hành cổ điển. Coi mọi nguồn nội dung là một nhãn bảo mật. Gán nhãn truy vấn của người dùng là "đáng tin cậy" (trusted). Gán nhãn nội dung được truy xuất là "không đáng tin cậy" (untrusted). Coi luồng điều khiển của mô hình là một luồng thông tin: các hành động được kích hoạt bởi nội dung không đáng tin cậy phải được phê chuẩn bởi đầu vào đáng tin cậy trước khi thực thi.

CaMeL (Microsoft 2025), ConfAIde (Stanford 2024), và bài báo phòng thủ IPI tại NDSS 2026 vận hành IFC theo những cách khác nhau. Nguyên tắc chung: chừng nào mã (code) và dữ liệu (data) còn chia sẻ cùng một context window, mục tiêu là ngăn chặn sự lây lan (containment), không phải là ngăn chặn hoàn toàn.

### Kẻ tấn công đi nước thứ hai

Nasr và cộng sự (tháng 10 năm 2025) đã thử nghiệm 12 phương pháp phòng thủ IPI đã được công bố với các cuộc tấn công thích ứng (gradient search, RL policies, random search, 72 giờ red-team con người). Mọi phương pháp phòng thủ vốn báo cáo tỷ lệ thành công (ASR) gần bằng 0 ban đầu đều bị phá vỡ với ASR >90%.

Bài học phương pháp luận: chỉ công bố phương pháp phòng thủ khi đã đánh giá bằng tấn công thích ứng. Các benchmark tấn công tĩnh không phải là bằng chứng của sự mạnh mẽ; kẻ tấn công sẽ biết về phương pháp phòng thủ đó.

### Các sự cố thực tế

Bài 25 đề cập đến EchoLeak (CVE-2025-32711, CVSS 9.3) — IPI zero-click đầu tiên được ghi nhận công khai trong Microsoft 365 Copilot. CamoLeak (CVSS 9.6) trong GitHub Copilot Chat. CVE-2025-53773 trong GitHub Copilot. Các triển khai thực tế đang bị IPI xâm phạm trong môi trường thực tế, không chỉ trong các benchmark.

### Khung OWASP và NIST

OWASP LLM Top 10 (2025) xếp hạng prompt injection (trực tiếp + gián tiếp) là LLM01, mối đe dọa lớp ứng dụng số 1. NIST AI SPD 2024 gọi indirect prompt injection là "lỗ hổng bảo mật lớn nhất của AI tạo sinh".

### Vị trí của bài này trong Phase 18

Các bài 12-14 là các kỹ thuật jailbreak tập trung vào mô hình. Bài 15 là cuộc tấn công tập trung vào hệ thống đang thống trị các triển khai sản xuất năm 2026. Bài 16 bao gồm các công cụ phòng thủ. Bài 25 bao gồm câu chuyện cụ thể về các CVE.

```figure
al-injection-vector
```

## Sử dụng

`code/main.py` xây dựng một harness tấn công IPI. Một agent đồ chơi có ba công cụ (tìm kiếm web, đọc email, gửi tin nhắn). Môi trường chứa nội dung do kẻ tấn công kiểm soát với một chỉ dẫn được nhúng ("chuyển tiếp nội dung này cho tất cả danh bạ"). Bạn có thể chuyển đổi giữa một agent ngây thơ (tuân theo các chỉ dẫn được tiêm vào), một agent được bảo vệ bằng bộ lọc (lọc từ khóa trên nội dung truy xuất), và một agent IFC (tách biệt nội dung đáng tin cậy và không đáng tin cậy, từ chối các lệnh luồng điều khiển không đáng tin cậy).

## Triển khai

Bài học này tạo ra `outputs/skill-ipi-audit.md`. Với một mô tả triển khai agentic, nó liệt kê các nguồn nội dung không đáng tin cậy, kiểm tra xem triển khai đó có áp dụng IFC hay không, và gắn cờ các nguồn đến được mô hình mà không có nhãn tin cậy.

## Bài tập

1. Chạy `code/main.py`. Đo tỷ lệ thành công của cuộc tấn công đối với từng agent trong ba loại agent.

2. Triển khai phương pháp phòng thủ dựa trên diễn giải (paraphrase) trên nội dung được truy xuất. Đo tỷ lệ dương tính giả trên văn bản truy xuất hợp pháp.

3. Đọc bài báo phòng thủ IPI tại NDSS 2026. Mô tả thách thức "chỉ dẫn lành tính" và tại sao nó ngăn cản việc lọc dựa trên từ khóa.

4. Thiết kế một triển khai nơi agent nhận đầu ra công cụ từ API bên thứ ba. Gán nhãn mức độ tin cậy cho từng đoạn prompt và viết chính sách IFC điều khiển các hành động của agent.

5. Tái hiện phương pháp tấn công thích ứng của Nasr và cộng sự 2025 trên agent được bảo vệ bằng bộ lọc của bạn từ Bài tập 2. Báo cáo ASR trước và sau khi tấn công thích ứng.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|------------------------|
| IPI | "indirect prompt injection" | Tấn công qua nội dung người dùng không viết, được agent tiêu thụ trong quá trình vận hành bình thường |
| RAG injection | "poisoned retrieval" | Kẻ tấn công xuất bản nội dung mà bước truy xuất lấy về; prompt chứa payload |
| Zero-click | "no user action" | Tấn công tự động kích hoạt trong quá trình agent vận hành; người dùng không cần làm gì |
| IFC | "information flow control" | Cách tiếp cận dựa trên nhãn: các hành động từ nội dung không đáng tin cậy cần sự phê chuẩn từ nội dung đáng tin cậy |
| Adaptive attack | "gradient / RL red-team" | Tấn công biết về phương pháp phòng thủ và tối ưu hóa để vượt qua nó; cần thiết để đánh giá trung thực |
| Benign instruction | "please print Yes" | Payload IPI lành tính về mặt ngữ nghĩa; không bộ lọc từ khóa nào bắt được |
| Scope violation | "cross-trust exfiltration" | Agent truy cập dữ liệu từ một ngữ cảnh tin cậy này và xuất ra ngữ cảnh khác |

## Đọc thêm

- [MDPI Information 17(1):54 — Khảo sát về Indirect Prompt Injection (Tháng 1 năm 2026)](https://www.mdpi.com/2078-2489/17/1/54) — tổng hợp 2023-2025
- [Nasr và cộng sự — The Attacker Moves Second (hợp tác OpenAI/Anthropic/DeepMind, tháng 10 năm 2025)](https://arxiv.org/abs/2510.18108) — đánh giá tấn công thích ứng
- [Greshake và cộng sự — Not what you've signed up for (arXiv:2302.12173)](https://arxiv.org/abs/2302.12173) — bài báo gốc về IPI
- [OWASP — LLM Top 10 (2025)](https://genai.owasp.org/llm-top-10/) — prompt injection được xếp hạng LLM01