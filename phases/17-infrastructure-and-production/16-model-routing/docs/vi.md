# Model Routing như một Primitive giúp giảm chi phí

> Một broker động sẽ đánh giá mọi yêu cầu (loại tác vụ, độ dài token, độ tương đồng embedding, độ tin cậy) và gửi các truy vấn đơn giản đến một model giá rẻ, sau đó leo thang các truy vấn phức tạp lên model frontier. Kỹ thuật này còn được gọi là model cascading. Các nghiên cứu thực tế trong sản xuất cho thấy mức giảm chi phí từ 20-60% trong khi vẫn giữ nguyên chất lượng trên các triển khai tại Mỹ/Anh/EU; cải thiện 30% hiệu suất định tuyến trên các SaaS có lưu lượng lớn có thể chuyển đổi thành khoản tiết kiệm hàng trăm nghìn đô la mỗi năm. Bối cảnh năm 2026 là giá inference của LLM đã giảm ~10 lần mỗi năm — một token thuộc phân khúc GPT-4 đã giảm từ $20/M to ~$0.40/M vào cuối năm 2022 xuống còn 2026. Phần lớn sự sụt giảm này đến từ các serving stack tốt hơn (Phase 17 · 04-09), chứ không phải từ phần cứng. Routing là cách bạn chuyển đổi mức giảm giá đó thành lợi nhuận mà không làm giảm chất lượng sản phẩm. Chế độ lỗi (failure mode) thường gặp là cheap-model drift: lộ trình đẩy 40% yêu cầu sang một model yếu hơn, chất lượng giảm 3-5% đối với các tác vụ suy luận, và không ai nhận ra điều đó trong suốt một quý. Hãy kiểm soát các lộ trình bằng các chỉ số chất lượng trực tuyến (online quality metrics), thay vì chỉ dựa vào các bộ eval ngoại tuyến.

**Type:** Learn
**Languages:** Python (stdlib, toy cascading router simulator)
**Prerequisites:** Phase 17 · 01 (Managed LLM Platforms), Phase 17 · 19 (AI Gateways)
**Time:** ~60 phút

## Mục tiêu học tập

- Giải thích model cascading: ưu tiên model giá rẻ với kiểm tra độ tin cậy, leo thang khi độ tin cậy thấp.
- Liệt kê bốn tín hiệu định tuyến (phân loại tác vụ, độ dài prompt, độ tương đồng embedding với tập dữ liệu khó đã biết, độ tự tin từ lần chạy đầu tiên).
- Tính toán chi phí hỗn hợp dự kiến tại mức phân bổ định tuyến mục tiêu và mức chấp nhận giảm chất lượng.
- Gọi tên chỉ số giám sát drift (online quality gate) giúp phát hiện tình trạng cheap-model creep.

## Vấn đề

Dịch vụ của bạn tốn 80.000 USD/tháng trên GPT-5. Phân tích cho thấy 70% truy vấn là đơn giản: "mấy giờ rồi ở Paris?", "diễn đạt lại câu này". Một model thuộc phân khúc Haiku xử lý những yêu cầu đó hoàn hảo với chi phí chỉ bằng 3%. 30% còn lại cần khả năng suy luận của GPT-5 — lập trình, toán học, lập kế hoạch nhiều bước.

Nếu bạn định tuyến 70% sang model giá rẻ và 30% sang model đắt tiền, hóa đơn của bạn sẽ giảm ~65% với cùng chất lượng sản phẩm. Đây chính là routing. Bí quyết nằm ở việc xây dựng broker mà không làm giảm chất lượng.

## Khái niệm

### Bốn tín hiệu định tuyến

1. **Phân loại tác vụ**: đơn giản/phức tạp/codegen/toán học/chat. Có thể là bộ phân loại dựa trên quy tắc, một LLM nhỏ (phân khúc Haiku với giá $0.25/M), hoặc độ tương đồng embedding với các bucket đã được gán nhãn. Đầu ra: route = giá rẻ / cân bằng / frontier.

2. **Độ dài prompt**: các prompt >4K token thường cần model frontier để đảm bảo tính mạch lạc. Các prompt <500 token thường không cần.

3. **Độ tương đồng embedding với tập dữ liệu khó đã biết**: nếu truy vấn gần giống (cosine > 0.88) với một bucket khó đã biết, hãy leo thang trực tiếp lên frontier.

4. **Độ tự tin từ lần chạy đầu tiên**: gửi đến model giá rẻ; nếu log-probs của model cho thấy độ tin cậy thấp HOẶC nó từ chối HOẶC đưa ra ngôn ngữ mơ hồ, hãy thử lại trên frontier. Điều này làm tăng độ trễ P95 trên ~10% lưu lượng nhưng tiết kiệm 50%+ trên 90% còn lại.

### Ba mô hình

**Pre-route** (phân loại ngay từ đầu): thêm ~5-10ms độ trễ; nhanh nhất về tổng thể.

**Cascade** (ưu tiên giá rẻ, leo thang khi độ tin cậy thấp): ~1.2x độ trễ trung bình (chạy model giá rẻ cộng với xác minh), ~2x khi leo thang. Đảm bảo chất lượng tốt nhất.

**Ensemble route** (chạy song song giá rẻ và frontier cho một mẫu, reward-model chọn kết quả): chất lượng cao nhất, chi phí cao nhất; chỉ sử dụng cho các A/B test quan trọng.

### Triển khai

Các AI gateway (Phase 17 · 19) cung cấp khả năng định tuyến. LiteLLM có `router` config với fallback và cost-routing. Portkey có guards + routing. Kong AI Gateway có định tuyến dựa trên plugin. Model marketplace của OpenRouter cung cấp API gợi ý.

Mã nguồn mở: RouteLLM (LMSYS), Not Diamond (thương mại), Prompt Mule.

### Đường cong giá năm 2026

| Phân khúc model | Cuối 2022 | 2026 | Thay đổi |
|-------------|-----------|------|--------|
| Chất lượng ngang GPT-4 | ~$20/M | ~$0.40/M | Rẻ hơn 50 lần |
| Frontier (GPT-5, Claude 4) | — | ~$3-10/M | phân khúc mới |

Phần lớn sự cải thiện đến từ hiệu suất phục vụ — các bài học cốt lõi trong Phase 17 · 04-09 đã chuyển thành mức giảm chi phí từ phía nhà cung cấp. Routing cho phép bạn nắm bắt những lợi ích đó ở tầng ứng dụng thay vì phải chờ đợi tất cả người dùng của bạn chuyển sang phân khúc giá rẻ.

### Drift là rủi ro thực sự

Lộ trình của bạn gửi 40% yêu cầu đến model giá rẻ. Sau sáu tháng, phân phối tác vụ thay đổi (người dùng trở nên tinh vi hơn, đặt câu hỏi dài hơn). Router không nhận ra điều đó vì bộ phân loại của nó được huấn luyện trên dữ liệu Q1. Chất lượng giảm âm thầm. Không ai phàn nàn đủ lớn. Bạn chỉ phát hiện ra khi thua trong một bài benchmark của đối thủ cạnh tranh.

Hãy kiểm soát các lộ trình bằng các chỉ số chất lượng trực tuyến:

- Số lượt thumbs-up / thumbs-down của người dùng trên mỗi route.
- LLM-judge tự động trên một mẫu dữ liệu giữ lại (5%) trên mỗi route.
- Tỷ lệ leo thang (escalation rate): nếu cascade đang đẩy >30% lên route cao hơn, model giá rẻ đang bị quá tải.
- Tỷ lệ từ chối trên mỗi route.

### Các con số bạn nên nhớ

- Tiết kiệm từ routing năm 2026 ở mức chất lượng tương đương: 20-60% theo các nghiên cứu thực tế.
- Mức giảm giá LLM 2022-2026: ~10 lần mỗi năm tổng cộng.
- Chất lượng ngang GPT-4 2022 so với 2026: ~$20/M → ~$0.40/M.
- Tác động độ trễ của cascade: ~1.2x trung bình, ~2x khi leo thang (~10% lưu lượng).

```figure
model-cascade-router
```

## Sử dụng

`code/main.py` mô phỏng pre-route, cascade và ensemble trên một khối lượng công việc hỗn hợp. Báo cáo chi phí hỗn hợp, mức giảm chất lượng và tỷ lệ leo thang.

## Triển khai

Bài học này tạo ra `outputs/skill-router-plan.md`. Dựa trên khối lượng công việc và ngân sách chất lượng, chọn một mô hình định tuyến và các tín hiệu.

## Bài tập

1. Chạy `code/main.py`. Tại mức độ chính xác nào thì cascade vượt qua pre-route?
2. Cơ sở người dùng của bạn là 30% doanh nghiệp (truy vấn phức tạp), 70% miễn phí (đơn giản). Hãy thiết kế phân bổ định tuyến. Chỉ số trực tuyến nào sẽ kiểm soát nó?
3. Một route làm giảm 2% chất lượng nhưng tiết kiệm 40% chi phí. Đó có phải là một quyết định nên triển khai không? Tùy thuộc vào sản phẩm — hãy lập luận cho cả hai trường hợp.
4. Triển khai kiểm tra độ tin cậy bằng cách sử dụng logprobs từ OpenAI / Anthropic API. Ngưỡng bạn bắt đầu là bao nhiêu?
5. Sau sáu tháng, tỷ lệ leo thang tăng từ 8% lên 22%. Hãy chẩn đoán ba nguyên nhân và cách khắc phục cho từng nguyên nhân.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|------------------------|
| Model routing | "cost broker" | Lựa chọn model động cho mỗi yêu cầu |
| Model cascade | "cheap-first escalate" | Chạy giá rẻ, leo thang lên frontier khi độ tin cậy thấp |
| Pre-route | "classify first" | Phân loại trước; không chạy lại |
| Ensemble route | "parallel pick" | Chạy nhiều model, reward-model chọn kết quả tốt nhất |
| Escalation rate | "uprouted %" | Tỷ lệ phần trăm các yêu cầu cascade đã leo thang |
| RouteLLM | "LMSYS router" | Thư viện router mã nguồn mở |
| Not Diamond | "commercial router" | Sản phẩm model-routing thương mại (SaaS) |
| Drift | "cheap creep" | Sự thay đổi phân phối mà router không nhận ra |
| Online quality gate | "live check" | Lấy mẫu LLM-judge tự động trên lưu lượng thực tế |

## Đọc thêm

- [AbhyashSuchi — Model Routing LLM 2026 Best Practices](https://abhyashsuchi.in/model-routing-llm-2026-best-practices/)
- [Lukas Brunner — Rise of Inference Optimization 2026](https://dev.to/lukas_brunner/the-rise-of-inference-optimization-the-real-llm-infra-trend-shaping-2026-4e4o)
- [RouteLLM paper / code](https://github.com/lm-sys/RouteLLM)
- [Not Diamond — model routing](https://www.notdiamond.ai/)
- [OpenRouter](https://openrouter.ai/) — gateway đa model với các primitive định tuyến.