# Kinh tế học Agent, Token Incentives, Reputation

> Các autonomous agent có tầm nhìn dài hạn (đường cong công việc từ 1 đến 8 giờ theo METR) cần có năng lực kinh tế. **Stack 5 lớp** đang nổi lên bao gồm: **DePIN** (tính toán vật lý) → **Identity** (W3C DIDs + vốn danh tiếng) → **Cognition** (RAG + MCP) → **Settlement** (trừu tượng hóa tài khoản) → **Governance** (Agentic DAOs). Các mạng lưới khuyến khích agent trong sản xuất bao gồm **Bittensor** (các subnet TAO thưởng cho các model chuyên biệt), **Fetch.ai / ASI Alliance** (ASI-1 Mini LLM + token FET), và **Gonka** (PoW dựa trên transformer giúp tái phân bổ tài nguyên tính toán cho các tác vụ AI hiệu quả). Các nghiên cứu học thuật: LaMAS phi tập trung tại AAMAS 2025 sử dụng **Shapley-value credit attribution** để thưởng công bằng cho các agent đóng góp; nghiên cứu của Google "Mechanism design for large language models" đề xuất **đấu giá token** với cơ chế thanh toán giá thứ hai (second-price) dưới sự tổng hợp đơn điệu (monotone aggregation). Bài học này xây dựng một marketplace agent tối giản, áp dụng Shapley-value credit attribution vào một pipeline đa agent, và chạy một phiên đấu giá token giá thứ hai để các cơ chế lý thuyết trò chơi được hiện thực hóa cụ thể.

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 16 (Negotiation and Bargaining), Phase 16 · 09 (Parallel Swarm Networks)
**Time:** ~75 phút

## Vấn đề

Các hệ thống đa agent trở nên phức tạp khi các agent tạo ra giá trị chung nhưng cần được khen thưởng riêng lẻ. Các cơ chế cổ điển — chia đều, người đóng góp cuối cùng nhận tất cả — thường không công bằng hoặc dễ bị thao túng. Việc khen thưởng dựa trên liên minh thông qua Shapley values là công bằng theo cấu trúc nhưng tốn kém để tính toán. Các tài liệu giai đoạn 2025-2026 thúc đẩy các phương pháp xấp xỉ hữu ích: Shapley sampling, đấu giá tổng hợp đơn điệu, và danh tiếng trên chuỗi (on-chain reputation) tích lũy từ các đóng góp đã được xác nhận.

Ngoài việc phân bổ tín dụng, lĩnh vực này đã chuyển sang các tác nhân kinh tế thực tế: Bittensor TAO thưởng cho việc khai thác tài nguyên tính toán để tinh chỉnh các model chuyên biệt cho subnet, Fetch.ai/ASI thưởng cho việc sử dụng ASI-1 Mini LLM bằng token FET, Gonka tái phân bổ proof-of-work của transformer vào các tác vụ AI hiệu quả. Các agent giao dịch tự chủ đã tồn tại ngày nay; câu hỏi đặt ra là làm thế nào để căn chỉnh các động lực (incentives).

Bài học này coi nền kinh tế agent là một nhóm vấn đề cụ thể — phân bổ tín dụng, thiết kế cơ chế, và danh tiếng — và xây dựng từng phần với toán học tối giản để các ý tưởng dễ nắm bắt.

## Khái niệm

### Stack kinh tế agent 5 lớp

1. **DePIN (tính toán vật lý).** Hạ tầng phi tập trung cho thuê GPU, lưu trữ, băng thông. Bittensor subnets, Render Network, Akash. Không dành riêng cho agent; các agent sử dụng nó.
2. **Identity.** W3C Decentralized Identifiers (DIDs) cung cấp cho mỗi agent một ID bền vững độc lập với bất kỳ nền tảng nào. Danh tiếng tích lũy vào DID. Agent Network Protocol (ANP) sử dụng DID làm lớp khám phá.
3. **Cognition.** Vòng lặp suy luận của agent: LLM + RAG + MCP. Đây là những gì các giai đoạn khác xây dựng.
4. **Settlement.** Trừu tượng hóa tài khoản (ERC-4337) cho phép các agent thanh toán phí gas từ số dư của chính chúng mà không cần giữ ETH. Các agent có thể trả tiền cho dịch vụ, cho nhau, hoặc cho tài nguyên tính toán.
5. **Governance.** Agentic DAOs: các cấu trúc quản trị nơi con người *và* agent bỏ phiếu cho các thay đổi giao thức, với quyền biểu quyết gắn liền với danh tiếng.

Không phải mọi hệ thống sản xuất đều sử dụng cả năm lớp. Bittensor sử dụng 1, 2, một phần 3, một phần 4, không có 5. Các agent của OpenAI không sử dụng gì ngoài 3. Stack này là bản đồ tham chiếu, không phải yêu cầu bắt buộc.

### Bittensor, Fetch.ai, Gonka — những gì đang vận hành

**Bittensor (TAO).** Các subnet là các tác vụ chuyên biệt (mô hình hóa ngôn ngữ, tạo ảnh, dự báo). Các thợ đào (miners) gửi kết quả đầu ra của model. Các validator xếp hạng chúng; điểm số dựa trên stake sẽ phân phối phần thưởng TAO. Mỗi subnet có đánh giá riêng. Bài học kinh tế: trả tiền cho chất lượng đầu ra chuyên biệt cho tác vụ, không phải cho tài nguyên tính toán đã sử dụng.

**Fetch.ai / ASI Alliance.** ASI-1 Mini LLM chạy trên mạng lưới của Fetch.ai; người dùng trả token FET cho việc suy luận (inference). Câu chuyện về các agent như những thực thể ngang hàng (peers) mạnh mẽ hơn ở đây: một agent trên Fetch có thể gọi một agent khác cho một tác vụ và trả bằng FET.

**Gonka.** Transformer proof-of-work: "công việc" là các lượt forward pass của một transformer. Các thợ đào kiếm tiền bằng cách chạy các tác vụ suy luận có kết quả đúng đã biết (từ dữ liệu huấn luyện). PoW hiệu quả về tài nguyên thay vì PoW dựa trên băm (hash).

Cả ba đều ở cấp độ sản xuất tính đến tháng 4 năm 2026. Việc phân phối phần thưởng khác nhau. Bittensor thưởng cho chất lượng so với các validator của subnet; Fetch thưởng cho tiện ích được đo lường bởi người dùng trả phí; Gonka thưởng cho công việc suy luận có thể xác minh.

### Phân bổ tín dụng Shapley-value

Ba agent cộng tác trong một tác vụ. Kết quả đạt 0.8 điểm. Ai đã đóng góp những gì?

Shapley value: phân bổ tín dụng duy nhất thỏa mãn bốn tiên đề (hiệu quả, đối xứng, tuyến tính, rỗng). Đối với agent `i`:

```
shapley(i) = (1/N!) * sum over all orderings O of (v(S_i_O ∪ {i}) - v(S_i_O))
```

trong đó `S_i_O` là tập hợp các agent trước `i` trong thứ tự `O`. Trong thực tế: liệt kê tất cả các hoán vị, ghi lại đóng góp biên của mỗi agent trong mỗi hoán vị, sau đó lấy trung bình.

Với N=3 agent, có 6 hoán vị. Với N=10, có 3.6 triệu — vì vậy trong thực tế, bạn lấy mẫu các thứ tự thay vì liệt kê toàn bộ.

### Đấu giá giá thứ hai cho tổng hợp

Nghiên cứu của Google ("Mechanism design for large language models") đề xuất đấu giá token giá thứ hai để tổng hợp các kết quả đầu ra của LLM. Thiết lập: N agent, mỗi agent đề xuất một kết quả; mỗi agent có một giá trị riêng tư cho việc được chọn. Người đấu giá chọn đề xuất có giá trị cao nhất và trả số tiền bằng giá trị *cao thứ hai*. Dưới sự tổng hợp đơn điệu (giá trị phụ thuộc vào đề xuất nào được chọn, không phải bao nhiêu đề xuất được đặt giá), đây là cơ chế trung thực — các agent đặt giá đúng với giá trị thực của chúng.

Tại sao điều này quan trọng đối với các hệ thống LLM: bạn có thể thuê ngoài các tác vụ hoàn thành cho nhiều agent với mức giá khác nhau; phiên đấu giá chọn kết quả tốt nhất + trả tiền công bằng, và các agent không có động lực để báo cáo sai.

### Vốn danh tiếng (Reputation capital)

Điểm danh tiếng gắn liền với DID tích lũy từ các đóng góp đã được xác nhận. Một quy tắc cập nhật đơn giản:

```
rep(i, t+1) = alpha * rep(i, t) + (1 - alpha) * contribution_quality(i, t)
```

Với hệ số suy giảm `alpha` gần bằng 1. Danh tiếng:

- Rẻ để đọc cho các quyết định định tuyến ("gửi các tác vụ khó cho các agent có danh tiếng cao").
- Đắt để giả mạo (tích lũy theo thời gian, gắn liền với DID).
- Có thể bị cắt giảm (slashed): các đóng góp không vượt qua xác minh sẽ bị trừ điểm.

### LaMAS phi tập trung AAMAS 2025

Đề xuất LaMAS (AAMAS 2025) kết hợp: định danh DID, phân bổ tín dụng Shapley-value, và một cơ chế đấu giá đơn giản. Tuyên bố chính: phi tập trung hóa bước phân bổ tín dụng làm cho hệ thống có thể kiểm toán và miễn nhiễm với sự thao túng tại một điểm duy nhất.

### Khi nào nền kinh tế agent sụp đổ

- **Thao túng oracle giá.** Nếu hàm tín dụng có thể bị thao túng, các agent sẽ thao túng nó. Mỗi cơ chế cần một bài kiểm tra đối kháng.
- **Tấn công Sybil.** Một người vận hành tạo ra N agent giả để thổi phồng đóng góp của chính mình. DIDs làm chậm nhưng không ngăn chặn được điều này; chi phí để giả mạo danh tiếng là biện pháp giảm thiểu.
- **Chi phí xác minh.** Phân bổ tín dụng chỉ công bằng khi người xác minh công bằng. Nếu xác minh rẻ (LLM nhỏ), nó có thể bị thao túng; nếu đắt (hội đồng con người), hệ thống không thể mở rộng.
- **Rào cản pháp lý.** Nền kinh tế agent giao thoa với quy định tài chính. Bittensor, Fetch, và Gonka đều hoạt động trong các vùng xám pháp lý ở một số khu vực pháp lý tính đến năm 2026.

### Khi nào nền kinh tế agent trở nên hợp lý

- **Mạng lưới mở với các nhà vận hành không đồng nhất.** Không có đội ngũ duy nhất nào kiểm soát tất cả các agent.
- **Kết quả đầu ra có thể xác minh.** Nếu không có xác minh, phân bổ tín dụng chỉ là phỏng đoán.
- **Quy trình làm việc dài hạn.** Các tác vụ một lần không được hưởng lợi từ việc tích lũy danh tiếng.
- **Thanh toán bằng token khả thi về mặt pháp lý** trong khu vực pháp lý của bạn.

Trong các hệ thống doanh nghiệp đóng, kinh tế học nhường chỗ cho việc phân bổ đơn giản hơn (quản lý giao việc, các chỉ số là nội bộ). Văn học kinh tế học áp dụng chủ yếu cho các mạng lưới mở.

```figure
swarm-auction
```

## Xây dựng

`code/main.py` triển khai:

- `shapley(value_fn, agents)` — tính toán Shapley chính xác bằng cách liệt kê cho N nhỏ.
- `second_price_auction(bids)` — cơ chế trung thực; người thắng trả giá cao thứ hai.
- `Reputation` — danh tiếng gắn liền với DID với suy giảm lũy thừa và cắt giảm.
- Demo 1: ba agent cộng tác, Shapley chính xác phân bổ tín dụng.
- Demo 2: năm agent đấu giá cho một vị trí tác vụ; đấu giá giá thứ hai chọn người thắng + thanh toán.
- Demo 3: 100 vòng phân bổ tác vụ cho các agent có danh tiếng không đồng nhất; định tuyến theo trọng số danh tiếng tốt hơn ngẫu nhiên.

Chạy:

```
python3 code/main.py
```

Kết quả mong đợi: Các giá trị Shapley cho mỗi agent; kết quả đấu giá cho thấy trạng thái cân bằng đặt giá trung thực; định tuyến theo trọng số danh tiếng cho thấy mức tăng chất lượng 10-20% so với ngẫu nhiên sau giai đoạn khởi động.

## Sử dụng

`outputs/skill-economy-designer.md` thiết kế một nền kinh tế agent tối giản: lựa chọn lớp định danh, cơ chế phân bổ tín dụng, cơ chế thanh toán, quy tắc danh tiếng.

## Triển khai

Vận hành một nền kinh tế agent vào năm 2026:

- **Bắt đầu với danh tiếng, không phải token.** Danh tiếng rẻ để triển khai và có giá trị riêng; token thêm vào sự phức tạp về pháp lý và kinh tế.
- **Xác minh trước khi thưởng.** Không bao giờ phân phối tín dụng mà không có bước xác minh độc lập. Chất lượng tự báo cáo sẽ dẫn đến các trò chơi Sybil.
- **Shapley-sample, không phải Shapley-exact.** Lấy mẫu 100-1000 thứ tự; liệt kê chính xác không thể mở rộng.
- **Giới hạn hệ số suy giảm và đặt sàn danh tiếng.** Suy giảm không giới hạn sẽ xóa sạch những người đóng góp hợp pháp; suy giảm quá chậm sẽ thưởng cho các agent có danh tiếng cao nhưng đã cũ.
- **Kiểm toán cơ chế theo hướng đối kháng.** Chạy các kịch bản red-team trước khi mở mạng lưới. Mọi cơ chế đều có lý thuyết trò chơi; bạn muốn tìm ra các lỗ hổng, không phải những kẻ tấn công.

## Bài tập

1. Chạy `code/main.py`. Xác nhận các giá trị Shapley cộng lại bằng tổng giá trị (tiên đề hiệu quả). Thay đổi hàm giá trị; liệu các phân bổ Shapley có thay đổi theo hướng mong đợi không?
2. Triển khai Shapley *sampling* (Monte Carlo qua K thứ tự). K ảnh hưởng như thế nào đến độ chính xác xấp xỉ? So sánh với tính toán chính xác cho N=4.
3. Triển khai bước hình thành liên minh trước khi đấu giá: các agent có thể hợp nhất thành các đội và đặt giá như một đơn vị. Những liên minh nào hình thành? Kết quả có tốt hơn theo Pareto so với việc đặt giá cá nhân không?
4. Đọc bài viết về thiết kế cơ chế của Google Research. Xác định một giả định mà nếu bị vi phạm, sẽ phá vỡ tính trung thực. Chế độ thất bại đó trông như thế nào trong môi trường LLM?
5. Đọc bài báo LaMAS phi tập trung AAMAS 2025. Triển khai bước Shapley của họ trên 10 agent cho một tác vụ tổng hợp. Tính toán chính xác mất bao lâu? Lấy mẫu đạt được kết quả gần như thế nào với 100 lần rút mẫu?

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|------------------------|
| DePIN | "Hạ tầng vật lý phi tập trung" | Tính toán/lưu trữ/băng thông được khuyến khích bằng token. Bittensor, Akash, Render. |
| DID | "Định danh phi tập trung" | Đặc tả W3C cho ID di động. Danh tiếng agent gắn với DID, không phải nền tảng. |
| ERC-4337 | "Trừu tượng hóa tài khoản" | Tài khoản hợp đồng có thể tài trợ phí gas, cho phép thanh toán agent. |
| Shapley value | "Phân bổ tín dụng công bằng" | Phân bổ duy nhất thỏa mãn hiệu quả, đối xứng, tuyến tính, rỗng. |
| Second-price auction | "Đấu giá Vickrey" | Cơ chế trung thực: người thắng trả giá cao thứ hai. Tương thích với tổng hợp đơn điệu. |
| Reputation capital | "Điểm chất lượng tích lũy" | Điểm gắn với DID từ các đóng góp đã xác nhận; suy giảm theo thời gian. |
| Agentic DAO | "Agent + con người quản trị" | DAO với các agent bỏ phiếu như thành viên hạng nhất, quyền biểu quyết gắn với danh tiếng. |
| TAO / FET / GPU credits | "Đơn vị token" | Bittensor TAO, Fetch.ai FET, các token DePIN khác nhau. |

## Đọc thêm

- [The Agent Economy](https://arxiv.org/abs/2602.14219) — Khảo sát năm 2026 về stack kinh tế agent 5 lớp
- [Google Research — Mechanism design for large language models](https://research.google/blog/mechanism-design-for-large-language-models/) — đấu giá token với tổng hợp đơn điệu
- [AAMAS 2025 — decentralized LaMAS](https://www.ifaamas.org/Proceedings/aamas2025/pdfs/p2896.pdf) — phân bổ tín dụng Shapley-value
- [Bittensor TAO documentation](https://docs.bittensor.com/) — cấu trúc subnet và phân phối phần thưởng
- [Fetch.ai / ASI Alliance](https://fetch.ai/) — ASI-1 Mini LLM và token FET
- [W3C Decentralized Identifiers (DIDs) spec](https://www.w3.org/TR/did-core/) — nền tảng định danh