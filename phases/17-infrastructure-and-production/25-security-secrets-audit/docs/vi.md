# Bảo mật — Secrets, Xoay vòng API Key, Nhật ký kiểm toán (Audit Logs), Guardrails

> Loại bỏ tình trạng rò rỉ secret thông qua các vault tập trung (HashiCorp Vault, AWS Secrets Manager, Azure Key Vault). Tuyệt đối không lưu trữ thông tin xác thực trong các tệp cấu hình, tệp env trong VCS, hoặc bảng tính. Sử dụng IAM role thay vì các khóa tĩnh; sử dụng OIDC cho CI/CD. Mô hình AI-gateway là giải pháp tiêu chuẩn năm 2026: ứng dụng → gateway → nhà cung cấp mô hình, với gateway tự động lấy thông tin xác thực từ vault tại thời điểm runtime. Thực hiện xoay vòng trong vault và tất cả ứng dụng sẽ cập nhật trong vài phút — không cần redeploy, không còn những tin nhắn Slack kiểu "ai đang giữ key mới vậy". Chính sách xoay vòng ≤ 90 ngày; quét bằng TruffleHog / GitGuardian / Gitleaks trên mỗi commit. Zero-trust: MFA, SSO, RBAC/ABAC, token tồn tại trong thời gian ngắn, kiểm tra trạng thái thiết bị. Việc làm sạch PII sử dụng nhận dạng thực thể để che giấu PHI/PII trước khi chuyển tiếp; token hóa nhất quán (cách tiếp cận Mesh) ánh xạ các giá trị nhạy cảm sang các placeholder ổn định để LLM vẫn giữ được ngữ nghĩa của mã/mối quan hệ. Egress mạng: các dịch vụ LLM nằm trong subnet VPC/VNet chuyên dụng chỉ cho phép whitelist `api.openai.com`, `api.anthropic.com`, v.v.; chặn tất cả các kết nối outbound khác. Sự cố bảo mật tiêu biểu năm 2026: tấn công chuỗi cung ứng Vercel thông qua thông tin xác thực CI/CD bị xâm nhập, dẫn đến rò rỉ các biến môi trường trên hàng ngàn triển khai của khách hàng.

**Type:** Học tập
**Languages:** Python (thư viện chuẩn, toy PII-scrubber + trình ghi audit-log)
**Prerequisites:** Phase 17 · 19 (AI Gateways), Phase 17 · 13 (Khả năng quan sát - Observability)
**Time:** ~60 phút

## Mục tiêu học tập

- Liệt kê bốn anti-pattern trong quản lý secret (tệp cấu hình trong VCS, env hardcode, bảng tính, khóa tĩnh) và nêu tên các giải pháp thay thế.
- Giải thích mô hình AI-gateway lấy secret từ vault như là tiêu chuẩn sản xuất năm 2026.
- Triển khai trình làm sạch PII với token hóa nhất quán (cùng giá trị → cùng placeholder) để bảo toàn ngữ nghĩa.
- Nêu tên sự cố chuỗi cung ứng Vercel năm 2026 và bài học về vệ sinh thông tin xác thực CI/CD.

## Vấn đề

Một thực tập sinh commit `.env` chứa các API key. Họ xóa nó ngay sau đó. Các khóa đã nằm trong lịch sử git — bản quét GitGuardian phát hiện ra, và quy trình xoay vòng của bạn là "nhắn tin Slack cho cả nhóm, cập nhật 40 tệp cấu hình, redeploy tất cả dịch vụ." 8 giờ sau, một nửa dịch vụ của bạn đã chạy, nửa còn lại vẫn đang chờ cửa sổ triển khai.

Ngoài ra, các prompt của người dùng bao gồm "SSN của tôi là 123-45-6789." Prompt được gửi đến OpenAI. Bạn có BAA nhưng chính sách nội bộ yêu cầu phải che giấu PII trước khi chuyển tiếp. Bạn đã không làm vậy.

Ngoài ra, pod LLM trong cụm EKS của bạn có thể truy cập bất kỳ host nào trên internet. Ai đó đã exfil dữ liệu thông qua tra cứu DNS đến một tên miền do kẻ tấn công kiểm soát. Không có gì chặn nó cả.

Bảo mật cho các dịch vụ LLM phải giải quyết cả ba vectơ này: thông tin xác thực được hỗ trợ bởi Vault, làm sạch PII, lọc egress mạng và nhật ký kiểm toán.

## Khái niệm

### Vault tập trung + IAM-role

**Vault**: HashiCorp Vault, AWS Secrets Manager, Azure Key Vault, GCP Secret Manager. Một nguồn sự thật duy nhất.

**IAM role**: ứng dụng/gateway xác thực thông qua danh tính IAM của nó, không phải khóa tĩnh. Vault trả về secret trong thời hạn của token.

**Mô hình AI-gateway**: gateway lấy `OPENAI_API_KEY` từ vault tại thời điểm yêu cầu. Xoay vòng trong vault; yêu cầu tiếp theo sẽ nhận được khóa mới. Không cần redeploy.

### Chính sách xoay vòng ≤ 90 ngày

Tất cả API key, vault root token, thông tin xác thực CI/CD. Tự động hóa xoay vòng bất cứ khi nào có thể. Xoay vòng thủ công phải được ghi nhật ký và theo dõi.

### Quét secret

- **TruffleHog** — regex + entropy trên các commit.
- **GitGuardian** — thương mại, độ chính xác cao.
- **Gitleaks** — mã nguồn mở, chạy trong CI.

Chạy trên mỗi commit. Chặn PR nếu phát hiện secret mới.

### Tư duy Zero-trust

- Yêu cầu MFA trên tất cả tài khoản.
- SSO thông qua SAML/OIDC.
- RBAC (dựa trên vai trò) hoặc ABAC (dựa trên thuộc tính) để kiểm soát truy cập chi tiết.
- Token tồn tại trong thời gian ngắn (tính bằng giờ, không phải ngày).
- Trạng thái thiết bị — chỉ các thiết bị công ty có mã hóa ổ đĩa.

### Làm sạch PII / PHI

Trước khi prompt rời khỏi hạ tầng của bạn:

1. Nhận dạng thực thể (spaCy NER, Presidio, thương mại).
2. Che giấu các thực thể khớp: `"My SSN is 123-45-6789"` → `"My SSN is [SSN_TOKEN_A3F]"`.
3. Token hóa nhất quán (cách tiếp cận Mesh): cùng giá trị ánh xạ tới cùng một placeholder để LLM bảo toàn các mối quan hệ.
4. Ánh xạ ngược tùy chọn cho phản hồi của LLM.

Các bộ lọc regex tĩnh bắt các mẫu cơ bản; NER bắt được nhiều hơn. Hãy sử dụng cả hai.

### Input + output guardrails

Input: chặn các jailbreak đã biết, các chủ đề bị cấm; giới hạn tốc độ (rate-limit) theo người dùng.

Output: quét regex để tìm secret bị rò rỉ (các mẫu API key, mẫu email trong ngữ cảnh từ chối), bộ phân loại cho các vi phạm chính sách.

### Whitelist egress mạng

Các dịch vụ LLM trong một subnet chuyên dụng:
- Whitelist: `api.openai.com`, `api.anthropic.com`, các endpoint vector DB, các endpoint vault.
- Mọi thứ khác: chặn.
- DNS thông qua trình phân giải chỉ cho phép whitelist (tránh exfil qua DNS-tunneling).

### Nhật ký kiểm toán (Audit log)

Nhật ký bất biến của mọi cuộc gọi LLM với:
- Dấu thời gian.
- Người dùng / khách hàng.
- Hash của prompt (không phải prompt gốc để đảm bảo quyền riêng tư).
- Mô hình + phiên bản.
- Số lượng token.
- Chi phí.
- Hash của phản hồi.
- Bất kỳ lần kích hoạt guardrail nào.

Lưu giữ theo yêu cầu quy định (SOC 2 là 1 năm, HIPAA là 6 năm).

### Sự cố Vercel năm 2026

Tấn công chuỗi cung ứng: thông tin xác thực CI/CD bị xâm nhập đã làm rò rỉ các biến môi trường trên hàng ngàn triển khai của khách hàng. Bài học: thông tin xác thực CI/CD tương đương với môi trường production. Lưu trữ trong vault. Giới hạn phạm vi chặt chẽ. Xoay vòng quyết liệt.

### Các con số cần nhớ

- Chính sách xoay vòng: ≤ 90 ngày.
- Quét trên mỗi commit: TruffleHog / GitGuardian / Gitleaks.
- Vercel 2026: Creds CI/CD bị xâm nhập → hàng ngàn biến môi trường của khách hàng bị rò rỉ.
- Lưu giữ nhật ký kiểm toán: SOC 2 = 1 năm, HIPAA = 6 năm.

```figure
i4-vault-rotation
```

## Sử dụng

`code/main.py` triển khai một trình làm sạch PII đơn giản với token hóa nhất quán và nhật ký kiểm toán chỉ ghi thêm (append-only).

## Triển khai

Bài học này tạo ra `outputs/skill-llm-security-plan.md`. Dựa trên phạm vi quy định và trạng thái hiện tại, lập kế hoạch di chuyển vault, trình làm sạch, egress, nhật ký kiểm toán.

## Bài tập

1. Chạy `code/main.py`. Gửi hai prompt tham chiếu đến cùng một SSN. Xác nhận cả hai đều nhận được cùng một placeholder.
2. Thiết kế chính sách egress mạng cho một triển khai vLLM-on-EKS gọi OpenAI + Anthropic + Weaviate.
3. Bạn phát hiện một khóa trong lịch sử git (đã 2 năm). Phản ứng đúng là gì — xoay vòng khóa, làm sạch lịch sử, hay cả hai? Giải thích.
4. Nhật ký kiểm toán của bạn tăng 10 GB/ngày. Thiết kế các tầng lưu giữ (nóng 30 ngày, ấm 12 tháng, lạnh 6 năm).
5. Tranh luận xem liệu token hóa ngược (thay thế các giá trị thực trở lại phản hồi của LLM) có đáng giá với sự phức tạp của nó so với việc giữ nguyên các placeholder hay không.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|------------------------|
| Vault | "kho chứa secret" | Dịch vụ quản lý thông tin xác thực tập trung |
| IAM role | "xác thực dựa trên danh tính" | Vai trò được ứng dụng đảm nhận; trả về creds ngắn hạn |
| OIDC cho CI/CD | "token do cloud cấp" | Không có khóa tĩnh trong CI — danh tính thông qua OIDC |
| TruffleHog / GitGuardian / Gitleaks | "trình quét secret" | Phát hiện secret tại thời điểm commit |
| RBAC / ABAC | "kiểm soát truy cập" | Dựa trên vai trò so với dựa trên thuộc tính |
| PII scrubbing | "che giấu dữ liệu" | Loại bỏ hoặc token hóa các thực thể nhạy cảm |
| Token hóa nhất quán | "placeholder ổn định" | Cùng giá trị → cùng token mỗi lần |
| Cách tiếp cận Mesh | "Mesh tokenization" | Mô hình token hóa bảo toàn ngữ nghĩa |
| Egress whitelist | "danh sách cho phép outbound" | Chỉ các tên miền được phép mới có thể truy cập |
| Audit log | "lịch sử bất biến" | Bản ghi chỉ ghi thêm để tuân thủ |

## Đọc thêm

- [Doppler — Bảo mật LLM nâng cao](https://www.doppler.com/blog/advanced-llm-security)
- [Portkey — Quản lý API key LLM với tham chiếu secret](https://portkey.ai/blog/secret-references-ai-api-key-management/)
- [Datadog — Các phương pháp hay nhất về LLM Guardrails](https://www.datadoghq.com/blog/llm-guardrails-best-practices/)
- [JumpServer — Các phương pháp hay nhất về quản lý Secret 2026](https://www.jumpserver.com/blog/secret-management-best-practices-2026)
- [Microsoft Presidio](https://github.com/microsoft/presidio) — Phát hiện và ẩn danh PII.
- [Tài liệu HashiCorp Vault](https://developer.hashicorp.com/vault/docs)