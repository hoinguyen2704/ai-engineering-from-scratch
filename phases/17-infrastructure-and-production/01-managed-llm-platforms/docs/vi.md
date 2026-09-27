# Managed LLM Platforms — Bedrock, Vertex AI, Azure OpenAI

> Ba nhà cung cấp hyperscaler, ba chiến lược khác biệt. AWS Bedrock là một marketplace mô hình — Claude, Llama, Titan, Stability, Cohere cùng nằm sau một API. Azure OpenAI là mối quan hệ đối tác độc quyền với OpenAI cộng với Provisioned Throughput Units (PTUs) cho năng lực tính toán chuyên dụng. Vertex AI ưu tiên Gemini với lợi thế về ngữ cảnh dài (long-context) và khả năng đa phương thức (multimodal). Năm 2026, Artificial Analysis đo lường Azure OpenAI ở mức trung vị ~50 ms và Bedrock ở mức ~75 ms trên các mô hình tương đương Llama 3.1 405B — PTUs giải thích cho khoảng cách này vì năng lực chuyên dụng luôn vượt trội hơn so với on-demand dùng chung. Quy tắc ra quyết định không phải là "cái nào nhanh nhất" mà là "danh mục mô hình và bề mặt FinOps nào phù hợp với sản phẩm của tôi". Bài học này dạy bạn cách lựa chọn dựa trên các đánh đổi thực tế, không phải dựa trên cảm tính.

**Type:** Learn
**Languages:** Python (stdlib, toy cost-and-latency comparator)
**Prerequisites:** Phase 11 (LLM Engineering), Phase 13 (Tools & Protocols)
**Time:** ~60 minutes

## Mục tiêu học tập

- Nêu tên ba chiến lược nền tảng (marketplace vs độc quyền vs ưu tiên Gemini) và khớp mỗi chiến lược với một trường hợp sử dụng sản phẩm.
- Giải thích Provisioned Throughput Units (PTUs) mang lại lợi ích gì trong Azure OpenAI và tại sao Bedrock on-demand thường có độ trễ cao hơn khoảng 25 ms ở quy mô 405B.
- Vẽ sơ đồ bề mặt phân bổ FinOps cho từng nền tảng (Bedrock Application Inference Profiles vs Vertex project-per-team vs Azure scopes + PTU reservations).
- Viết ra chính sách "tối thiểu hai nhà cung cấp" và giải thích tại sao việc phụ thuộc vào một nhà cung cấp duy nhất (vendor lock-in) là sai lầm đắt giá vào năm 2026.

## Vấn đề

Bạn đã chọn Claude 3.7 Sonnet cho sản phẩm của mình. Bây giờ bạn cần triển khai nó. Bạn có thể gọi trực tiếp Anthropic API, hoặc gọi thông qua AWS Bedrock, hoặc thông qua một gateway. API trực tiếp là đơn giản nhất; Bedrock bổ sung thêm BAA, VPC endpoints, IAM và phân bổ CloudWatch. Gateway bổ sung khả năng dự phòng (failover), thanh toán hợp nhất và giới hạn tốc độ (rate limits) trên các nhà cung cấp.

Câu hỏi sâu sắc hơn nằm ở danh mục mô hình. Nếu bạn cần Claude, Llama và Gemini trong cùng một sản phẩm, bạn không thể mua tất cả từ một nơi trừ khi nơi đó là Bedrock cộng với Vertex cộng với Azure OpenAI cùng lúc. Các hyperscaler không thể thay thế cho nhau — mỗi bên đặt cược khác nhau vào việc ai sẽ sở hữu lớp mô hình.

Bài học này vạch ra ba chiến lược đặt cược, khoảng cách về độ trễ, khoảng cách về FinOps và rủi ro bị khóa chặt (lock-in).

## Khái niệm

### Ba chiến lược

**AWS Bedrock** — marketplace. Claude (Anthropic), Llama (Meta), Titan (AWS first-party), Stability (hình ảnh), Cohere (embeddings), Mistral, cùng với các danh mục con về hình ảnh và embedding. Một API, một bề mặt IAM, một xuất dữ liệu CloudWatch. Bedrock đặt cược rằng khách hàng muốn sự tùy chọn hơn là chỉ một mô hình duy nhất.

**Azure OpenAI** — đối tác độc quyền. Bạn nhận được GPT-4 / 4o / 5 / o-series, DALL·E, Whisper và khả năng tinh chỉnh (fine-tuning) các mô hình OpenAI trong các trung tâm dữ liệu của Azure. Không có mô hình nào không phải của OpenAI trong danh mục "Azure OpenAI Service" — những mô hình đó thuộc về Azure AI Foundry (sản phẩm riêng biệt). Azure đặt cược rằng OpenAI vẫn là tiên phong và khách hàng muốn các kiểm soát doanh nghiệp trên mối quan hệ cụ thể đó.

**Vertex AI** — ưu tiên Gemini, mọi thứ khác là thứ yếu. Gemini 1.5 / 2.0 / 2.5 Flash và Pro, cộng với Model Garden (bên thứ ba). Vertex đặt cược vào khả năng đa phương thức với ngữ cảnh dài — ngữ cảnh 1M-token của Gemini là điểm khác biệt.

### Khoảng cách độ trễ ở quy mô lớn

Artificial Analysis thực hiện các bài kiểm tra hiệu năng liên tục. Trên các triển khai Llama 3.1 405B tương đương (shared on-demand), độ trễ token đầu tiên (TTFT) trung vị của Azure OpenAI là khoảng 50 ms; Bedrock là khoảng 75 ms. Khoảng cách này không phải là lỗi của AWS — đó là sự khác biệt về mô hình năng lực. Azure bán PTUs (Provisioned Throughput Units), giúp dành riêng năng lực GPU cho tenant của bạn. Tương đương của Bedrock (Provisioned Throughput) cũng tồn tại nhưng bắt đầu từ khoảng $21/giờ mỗi đơn vị, và hầu hết khách hàng vẫn sử dụng shared on-demand.

Năng lực dùng chung on-demand cạnh tranh với lưu lượng truy cập của mọi khách hàng khác. Năng lực chuyên dụng thì không. Nếu SLA sản phẩm của bạn là TTFT < 100 ms ở P99, bạn phải mua PTUs trên Azure, mua Provisioned Throughput trên Bedrock, hoặc chấp nhận sự biến thiên mặc định.

### Kinh tế học của Provisioned Throughput

Azure PTUs: một khối tính toán suy luận (inference) được đặt trước. Tiết kiệm tới ~70% so với on-demand cho các khối lượng công việc có thể dự đoán được. Chi phí cố định mỗi giờ bất kể lưu lượng — bạn trả tiền cho việc đặt trước ngay cả khi nhàn rỗi. Điểm hòa vốn thường ở mức khoảng 40-60% mức sử dụng bền vững.

Bedrock Provisioned Throughput: $21-$50 mỗi giờ tùy thuộc vào mô hình và khu vực. Phép tính tương tự — điểm hòa vốn ở mức khoảng một nửa mức sử dụng đỉnh. Yêu cầu cam kết hàng tháng.

Năng lực dự phòng của Vertex được bán theo SKU Gemini; giá cả thay đổi theo mô hình và khu vực và ít được công khai hơn.

### Bề mặt FinOps — điểm khác biệt thực sự

**Bedrock Application Inference Profiles** là cách phân bổ sạch nhất trong marketplace. Gắn thẻ một profile với `team`, `product`, `feature`; định tuyến tất cả các lệnh gọi mô hình qua đó; CloudWatch tách chi phí theo từng profile mà không cần xử lý hậu kỳ. Được thêm vào năm 2025, đây vẫn là tính năng gốc của hyperscaler có độ chi tiết cao nhất.

Phân bổ của **Vertex** là mỗi dự án cho mỗi nhóm cộng với nhãn (labels) ở khắp mọi nơi. Bạn mô hình hóa mỗi nhóm như một dự án GCP, đặt nhãn trên mọi tài nguyên và sử dụng BigQuery Billing Export + DataStudio để tổng hợp. Tốn công hơn, nhưng BigQuery cho phép bạn chạy SQL tùy ý trên dữ liệu chi phí.

**Azure** dựa trên phạm vi subscription/resource-group cộng với thẻ (tags), với các đặt trước PTU là đối tượng chi phí hạng nhất. Thẻ được kế thừa từ các nhóm tài nguyên, không phải từ các yêu cầu, vì vậy việc phân bổ theo yêu cầu đòi hỏi các chỉ số tùy chỉnh của Application Insights hoặc một gateway đóng dấu tiêu đề.

Mô hình: Bedrock là sạch nhất, Vertex linh hoạt nhất thông qua BigQuery, Azure mờ nhạt nhất trừ khi bạn thực hiện đo lường (instrumentation).

### Lock-in là rủi ro của năm 2026

Cam kết với một hyperscaler duy nhất là ổn khi một mô hình thống trị. Vào năm 2026, công nghệ tiên phong thay đổi hàng tháng — Claude 3.7 quý này, Gemini 2.5 quý sau, GPT-5 quý tiếp theo. Khóa mình vào một nền tảng sẽ khiến bạn mất cơ hội tiếp cận hai phần ba công nghệ tiên phong.

Mô hình mà các đội ngũ hiệu quả áp dụng: tối thiểu hai nhà cung cấp cho bất kỳ lệnh gọi LLM quan trọng nào của sản phẩm. Bedrock cộng với Azure OpenAI là cặp đôi phổ biến — Claude từ bên này, GPT từ bên kia, dự phòng giữa chúng, cùng một gateway. Chi phí tăng thêm không đáng kể vì gateway định tuyến tối ưu; khả năng sẵn sàng trong các sự cố (như sự cố Azure OpenAI tháng 1 năm 2025, sự cố AWS us-east-1) là yếu tố quyết định.

### Lưu trú dữ liệu, BAA và các ngành được quản lý

Bedrock: BAA ở hầu hết các khu vực; VPC endpoints; guardrails. Mặc định phổ biến cho fintech.
Azure OpenAI: HIPAA, SOC 2, ISO 27001; lưu trú dữ liệu EU; mặc định cho doanh nghiệp được quản lý.
Vertex: HIPAA, GDPR, lưu trú dữ liệu theo khu vực; hệ sinh thái tuân thủ của Google Cloud.

Cả ba đều đáp ứng các kiểm tra cơ bản. Sự khác biệt nằm ở chính sách lưu giữ dữ liệu, cách xử lý nhật ký và liệu việc giám sát lạm dụng có đọc lưu lượng truy cập của bạn hay không (mặc định là opt-in trên hầu hết; có sẵn opt-out cho doanh nghiệp).

### Các con số bạn nên nhớ

- TTFT trung vị của Azure OpenAI trên các mô hình tương đương Llama 3.1 405B: ~50 ms (với PTUs).
- TTFT trung vị của Bedrock on-demand: ~75 ms.
- Bedrock Provisioned Throughput: $21-$50/giờ mỗi đơn vị.
- Điểm hòa vốn PTU của Azure: ~40-60% mức sử dụng bền vững.
- Tiết kiệm PTU so với on-demand ở mức sử dụng cao: lên tới 70%.

```figure
i4-platform-lanes
```

## Sử dụng

`code/main.py` so sánh ba nền tảng trên một khối lượng công việc tổng hợp — nó mô hình hóa kinh tế học on-demand vs PTU, độ biến thiên TTFT và độ trung thực của phân bổ chi phí. Hãy chạy nó để xem PTUs mang lại lợi ích ở đâu và nơi nào sự đa dạng mô hình của marketplace vượt trội hơn khoảng cách TTFT.

## Triển khai

Bài học này tạo ra `outputs/skill-managed-platform-picker.md`. Với một hồ sơ khối lượng công việc (các mô hình cần thiết, SLA TTFT, khối lượng hàng ngày, yêu cầu tuân thủ), nó đề xuất một nền tảng chính, một phương án dự phòng và kế hoạch đo lường FinOps.

## Bài tập

1. Chạy `code/main.py`. Ở mức sử dụng bền vững nào thì Azure PTU vượt trội hơn on-demand cho mô hình lớp 70B? Tính điểm hòa vốn và so sánh với dải 40-60% được quảng cáo.
2. Sản phẩm của bạn cần Claude 3.7 Sonnet và GPT-4o. Thiết kế một triển khai hai nhà cung cấp — cái nào đi với hyperscaler nào, gateway nào đứng trước, chính sách dự phòng là gì?
3. Một khách hàng chăm sóc sức khỏe được quản lý yêu cầu BAA, lưu trú dữ liệu US-East và P99 TTFT dưới 100ms. Chọn một nền tảng và biện minh bằng ba tính năng cụ thể.
4. Bạn phát hiện hóa đơn Bedrock của mình tăng gấp 4 lần trong tháng này mà không có thay đổi về lưu lượng. Nếu không có Application Inference Profiles, làm thế nào bạn tìm ra thủ phạm? Với profiles, mất bao lâu?
5. Đọc các trang giá của Azure OpenAI và Bedrock. Đối với khối lượng công việc 100M-token/tháng của Claude, cái nào rẻ hơn — Anthropic API trực tiếp, Bedrock on-demand, hay Bedrock Provisioned Throughput?

## Thuật ngữ chính

| Thuật ngữ | Mọi người nói | Ý nghĩa thực tế |
|------|----------------|------------------------|
| Bedrock | "Dịch vụ LLM của AWS" | Marketplace mô hình bao gồm Claude, Llama, Titan, Mistral, Cohere |
| Azure OpenAI | "ChatGPT của Azure" | Các mô hình OpenAI độc quyền trong trung tâm dữ liệu Azure với kiểm soát doanh nghiệp |
| Vertex AI | "LLM của Google" | Nền tảng ưu tiên Gemini với Model Garden cho các mô hình bên thứ ba |
| PTU | "năng lực chuyên dụng" | Provisioned Throughput Unit — GPU suy luận được đặt trước, tính phí theo giờ |
| Application Inference Profile | "Gắn thẻ Bedrock" | Hồ sơ chi phí/sử dụng theo từng sản phẩm với thẻ, gốc CloudWatch |
| Model Garden | "Danh mục Vertex" | Phần mô hình bên thứ ba của Vertex AI, tách biệt với Gemini |
| Two-provider minimum | "Dự phòng LLM" | Chính sách chạy mọi đường dẫn LLM quan trọng trên ≥2 hyperscalers |
| BAA | "Giấy tờ HIPAA" | Business Associate Agreement; bắt buộc cho PHI; được cung cấp bởi cả ba |
| Abuse monitoring | "Trình theo dõi nhật ký" | Quét an toàn phía nhà cung cấp trên các prompt/output; opt-out trong doanh nghiệp |

## Đọc thêm

- [AWS Bedrock Pricing](https://aws.amazon.com/bedrock/pricing/) — bảng giá chính thức và giá Provisioned Throughput.
- [Azure OpenAI Service Pricing](https://azure.microsoft.com/en-us/pricing/details/azure-openai/) — kinh tế học PTU và bảng giá.
- [Vertex AI Generative AI Pricing](https://cloud.google.com/vertex-ai/generative-ai/pricing) — các tầng Gemini và phụ phí Model Garden.
- [Artificial Analysis LLM Leaderboard](https://artificialanalysis.ai/) — các bài kiểm tra độ trễ và thông lượng liên tục trên các nhà cung cấp.
- [The AI Journal — AWS Bedrock vs Azure OpenAI CTO Guide 2026](https://theaijournal.co/2026/03/aws-bedrock-vs-azure-openai/) — khung quyết định cho doanh nghiệp.
- [Finout — Bedrock vs Vertex vs Azure FinOps](https://www.finout.io/blog/bedrock-vs.-vertex-vs.-azure-cognitive-a-finops-comparison-for-ai-spend) — cơ chế phân bổ so sánh song song.