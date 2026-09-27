# Kinh tế học về Nền tảng Inference — Fireworks, Together, Baseten, Modal, Replicate, Anyscale

> Thị trường inference năm 2026 không còn đơn thuần là cho thuê thời gian GPU. Nó đã phân hóa thành silicon tùy chỉnh (Groq, Cerebras, SambaNova), các nền tảng GPU (Baseten, Together, Fireworks, Modal) và các marketplace ưu tiên API (Replicate, DeepInfra). Việc Fireworks tăng giá với định giá $1/hr per GPU on May 1, 2026, and $4 tỷ USD trên lưu lượng 10 nghìn tỷ token/ngày cho thấy mô hình dựa trên khối lượng (volume-driven) thực sự hiệu quả. Baseten đã chốt $300M Series E at $5 tỷ USD vào tháng 1 năm 2026. Quy tắc định vị cạnh tranh rất đơn giản: Fireworks tối ưu hóa độ trễ (latency), Together tối ưu hóa độ rộng danh mục, Baseten tối ưu hóa sự chỉn chu cho doanh nghiệp, Modal tối ưu hóa trải nghiệm lập trình viên (DX) thuần Python, Replicate tối ưu hóa phạm vi đa phương thức (multimodal), và Anyscale tối ưu hóa Python phân tán. Bài học này cung cấp cho bạn một ma trận mà bạn có thể đưa cho bất kỳ nhà sáng lập nào.

**Type:** Học tập
**Languages:** Python (thư viện chuẩn, trình so sánh kinh tế học mô phỏng)
**Prerequisites:** Phase 17 · 01 (Các nền tảng LLM được quản lý), Phase 17 · 04 (Nội bộ Serving Engine)
**Time:** ~60 phút

## Mục tiêu học tập

- Nêu tên ba phân khúc thị trường (silicon tùy chỉnh, nền tảng GPU, ưu tiên API) và ánh xạ từng nhà cung cấp vào phân khúc tương ứng.
- Giải thích lý do tại sao mô hình định giá API "theo token" lại nén về đường cong chi phí của serving engine thay vì chi phí phần cứng.
- Tính toán chi phí hiệu dụng trên mỗi yêu cầu (request) của ít nhất ba nhà cung cấp và giải thích khi nào định giá theo phút (Baseten, Modal) sẽ ưu việt hơn theo token.
- Xác định nền tảng mặc định phù hợp cho một workload cụ thể (serverless bùng nổ, thông lượng cao ổn định, các biến thể fine-tuned, đa phương thức).

## Vấn đề

Bạn đã đánh giá các nền tảng hyperscaler được quản lý. Bạn quyết định rằng mình cần một nhà cung cấp hẹp hơn, nhanh hơn — Fireworks cho độ trễ, Together cho độ rộng, Baseten cho mô hình tùy chỉnh đã fine-tune. Bây giờ bạn có sáu lựa chọn thực tế và các trang bảng giá không khớp nhau. Fireworks hiển thị $/M tokens; Baseten shows $/phút; Modal hiển thị $/second; Replicate shows $/lần dự đoán (prediction). Bạn không thể so sánh trực tiếp nếu không mô hình hóa workload.

Tệ hơn nữa, mô hình kinh doanh đằng sau mỗi trang bảng giá đều khác nhau. Fireworks chạy engine tùy chỉnh của riêng họ (FireAttention) trên các GPU chia sẻ; mức giá theo token phản ánh đường cong sử dụng của họ. Baseten cung cấp cho bạn Truss + GPU chuyên dụng; giá theo phút phản ánh tính độc quyền. Modal là serverless Python thực thụ — thanh toán theo giây với cold start dưới một giây. Cùng một đầu ra (phản hồi LLM), ba hàm chi phí khác nhau.

Bài học này mô hình hóa sáu nền tảng trên và cho bạn biết khi nào mỗi nền tảng sẽ thắng thế.

## Khái niệm

### Ba phân khúc

**Silicon tùy chỉnh** — Groq (LPU), Cerebras (WSE), SambaNova (RDU). Thường nhanh hơn 5-10 lần trong quá trình giải mã (decode) so với cụm GPU trên cùng một mô hình. Giá mỗi token cao hơn (Groq khoảng ~$0,99/triệu token trên Llama-70B cuối năm 2025) nhưng không thể đánh bại về độ trễ. Groq là lựa chọn sản xuất cho các voice agent và dịch thuật thời gian thực.

**Nền tảng GPU** — Baseten, Together, Fireworks, Modal, Anyscale. Chạy trên NVIDIA (H100, H200, B200 vào năm 2026) hoặc đôi khi là AMD. Lớp kinh tế nằm giữa "thuê GPU thô" (RunPod, Lambda) và "dịch vụ quản lý hyperscaler" (Bedrock).

**Marketplace ưu tiên API** — Replicate, DeepInfra, OpenRouter, Fal. Danh mục rộng, thanh toán theo lần dự đoán hoặc theo giây, nhấn mạnh vào thời gian phản hồi cuộc gọi đầu tiên (time-to-first-call).

### Fireworks — nền tảng GPU tối ưu hóa độ trễ

- Engine FireAttention (tùy chỉnh); được quảng bá là có độ trễ thấp hơn 4 lần so với vLLM trên các cấu hình tương đương.
- Tầng Batch ở mức ~50% giá serverless cho các workload không tương tác.
- Mô hình fine-tuned được phục vụ với cùng mức giá như mô hình cơ sở — một điểm khác biệt thực sự so với các nhà cung cấp tính phí cao cho LoRA của bạn.
- Giữa năm 2026: tăng giá thuê GPU on-demand thêm 1 USD/giờ kể từ ngày 1 tháng 5 năm 2026. Giá theo khối lượng có thể thương lượng ở quy mô lớn.
- Tín hiệu tài chính: định giá 4 tỷ USD, xử lý hơn 10 nghìn tỷ token/ngày.

### Together — tối ưu hóa độ rộng

- Hơn 200 mô hình bao gồm các bản phát hành mã nguồn mở chỉ trong vài ngày sau khi công bố.
- Rẻ hơn 50-70% so với Replicate trên các mô hình LLM tương đương — định vị "AI Native Cloud" tập trung vào khối lượng và danh mục.
- Inference + fine-tuning + training trong một API duy nhất.

### Baseten — tối ưu hóa sự chỉn chu cho doanh nghiệp

- Framework Truss: đóng gói mô hình với các phụ thuộc, bí mật, cấu hình phục vụ trong một manifest duy nhất.
- Dải GPU từ T4 đến B200. Thanh toán theo phút với khả năng giảm thiểu cold-start hợp lý.
- Đạt chuẩn SOC 2 Type II, sẵn sàng cho HIPAA. Lựa chọn phổ biến cho fintech và y tế.
- $5B valuation, January 2026 Series E ($300 triệu USD từ CapitalG, IVP, NVIDIA).

### Modal — tối ưu hóa thuần Python

- Infrastructure-as-code bằng Python thuần túy. Trang trí một hàm với `@modal.function(gpu="A100")` và triển khai bằng một lệnh duy nhất.
- Thanh toán theo giây. Cold start từ 2-4 giây với tính năng làm nóng trước (pre-warming); <1 giây cho các mô hình nhỏ.
- $87M Series B at $1,1 tỷ USD định giá (2025). Điểm trải nghiệm lập trình viên cao nhất trong các khảo sát độc lập.

### Replicate — độ rộng đa phương thức

- Thanh toán theo lần dự đoán. Nền tảng mặc định cho các mô hình hình ảnh, video và âm thanh.
- Hệ sinh thái tích hợp (Zapier, Vercel, các plugin CMS).
- Ít cạnh tranh hơn về giá mỗi token LLM nhưng thắng thế về sự đa dạng đa phương thức.

### Anyscale — Ray-native

- Được xây dựng trên Ray; RayTurbo là engine inference độc quyền của Anyscale (cạnh tranh với vLLM).
- Tốt nhất cho các workload Python phân tán nơi bước inference là một nút trong một đồ thị lớn hơn.
- Các cụm Ray được quản lý; tích hợp chặt chẽ với Ray AIR và Ray Serve.

### Theo token so với theo phút — khi nào mỗi loại thắng thế

Theo token hợp lý khi workload không nhạy cảm với độ trễ và có tính bùng nổ — bạn chỉ trả tiền cho những gì bạn sử dụng. Theo phút hợp lý khi mức độ sử dụng cao và có thể dự đoán được — bạn sẽ có lợi hơn so với theo token khi đã bão hòa GPU.

Quy tắc thô: đối với các workload sử dụng duy trì trên ~30% công suất GPU chuyên dụng, định giá theo phút (Baseten, Modal) bắt đầu thắng thế so với theo token (Fireworks, Together). Dưới mức đó, theo token thắng vì bạn tránh được việc trả tiền cho thời gian nhàn rỗi.

### Engine tùy chỉnh là hào kỹ thuật thực sự

Mọi nền tảng phía trên vLLM và SGLang đều tuyên bố có engine tùy chỉnh. FireAttention, RayTurbo, stack inference của Baseten. Các tuyên bố về engine tùy chỉnh thường mang tính tiếp thị — cách nhìn nhận trung thực là vLLM + SGLang chiếm khoảng 80% inference mã nguồn mở trong sản xuất, và các yếu tố khác biệt ở lớp nền tảng là DX, phân bổ chi phí và SLA.

### Các con số bạn cần nhớ

- Thuê GPU Fireworks: tăng 1 USD/giờ kể từ ngày 1 tháng 5 năm 2026.
- Tuyên bố của Fireworks: độ trễ thấp hơn 4 lần so với vLLM trên các cấu hình tương đương.
- Together: rẻ hơn 50-70% so với Replicate trên các LLM.
- Định giá Baseten: $5B (Series E, Jan 2026, $300 triệu USD).
- Định giá Modal: 1,1 tỷ USD (Series B, 2025).
- Theo phút thắng theo token trên ~30% mức sử dụng duy trì.

```figure
cost-per-token
```

## Sử dụng

`code/main.py` so sánh sáu nhà cung cấp trên một workload tổng hợp qua các mô hình định giá. Báo cáo $/day and effective $/triệu token. Hãy chạy nó để tìm điểm hòa vốn giữa theo token và theo phút.

## Triển khai

Bài học này tạo ra `outputs/skill-inference-platform-picker.md`. Dựa trên hồ sơ workload, SLA và ngân sách, chọn nền tảng inference chính và nêu tên nền tảng dự phòng.

## Bài tập

1. Chạy `code/main.py`. Tại mức sử dụng duy trì nào thì Baseten (theo phút) thắng Fireworks (theo token) cho mô hình 70B trên một H100? Tự rút ra điểm giao cắt và so sánh với quy tắc ngón tay cái.
2. Sản phẩm của bạn phục vụ tạo hình ảnh cộng với chat và chuyển đổi giọng nói thành văn bản. Chọn nền tảng cho từng phương thức và nêu tên mẫu gateway hợp nhất chúng.
3. Fireworks tăng giá 1 USD/giờ trên mô hình chính của bạn. Mô hình hóa tác động chi phí hỗn hợp nếu 40% lưu lượng truy cập của bạn chuyển sang tầng batch (giảm 50%).
4. Một khách hàng được quản lý yêu cầu SOC 2 Type II + HIPAA + GPU chuyên dụng. Ba nền tảng nào khả thi và nền tảng nào thắng về FinOps?
5. So sánh chi phí cho 1.000 lần dự đoán Llama 3.1 70B trên Fireworks serverless, Together on-demand, Baseten dedicated và Replicate API. Cái nào rẻ nhất ở mức 10 lần dự đoán/ngày? Ở mức 10.000 lần?

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|------------------------|
| Custom silicon | "chip không phải GPU" | Groq LPU, Cerebras WSE, SambaNova RDU — tối ưu cho giải mã |
| FireAttention | "engine Fireworks" | Kernel attention tùy chỉnh; quảng bá độ trễ thấp hơn 4 lần vLLM |
| Truss | "định dạng Baseten" | Manifest đóng gói mô hình; phụ thuộc + bí mật + cấu hình phục vụ |
| Per-token | "định giá API" | Tính phí theo token tiêu thụ; không trả tiền khi nhàn rỗi |
| Per-minute | "định giá chuyên dụng" | Tính phí theo thời gian GPU thực tế; thắng ở mức sử dụng cao |
| Per-prediction | "định giá Replicate" | Tính phí mỗi lần gọi mô hình; phổ biến cho ảnh/video |
| RayTurbo | "engine Anyscale" | Inference độc quyền trên Ray; cạnh tranh với vLLM trên cụm Ray |
| Batch tier | "giảm 50%" | Hàng đợi không tương tác với mức giá giảm; phổ biến trên Fireworks, OpenAI |
| Fine-tuned at base rate | "Fireworks LoRA" | Tính phí yêu cầu LoRA theo giá mô hình cơ sở (điểm khác biệt) |

## Đọc thêm

- [Bảng giá Fireworks](https://fireworks.ai/pricing) — giá theo token, tầng batch, thuê GPU.
- [Bảng giá Baseten](https://www.baseten.co/pricing/) — giá theo phút, công suất cam kết, các tầng doanh nghiệp.
- [Bảng giá Modal](https://modal.com/pricing) — giá GPU theo giây và tầng miễn phí.
- [Bảng giá Together AI](https://www.together.ai/pricing) — danh mục mô hình và giá theo token.
- [Bảng giá Anyscale](https://www.anyscale.com/pricing) — giá RayTurbo và Ray được quản lý.
- [Northflank — Các lựa chọn thay thế Fireworks AI](https://northflank.com/blog/7-best-fireworks-ai-alternatives-for-inference) — đánh giá so sánh.
- [Infrabase — Các nhà cung cấp API AI Inference 2026](https://infrabase.ai/blog/ai-inference-api-providers-compared) — bối cảnh nhà cung cấp.