# Failure Modes — MAST, Groupthink, Monoculture, Cascading Errors

> Phân loại tham chiếu cho năm 2026 là **MAST** (Cemri và cộng sự, NeurIPS 2025, arXiv:2503.13657), được rút ra từ 1642 dấu vết thực thi (execution traces) trên 7 hệ thống MAS mã nguồn mở hiện đại nhất, cho thấy **tỷ lệ thất bại từ 41–86,7%**. Ba danh mục gốc bao gồm: **Specification Problems** (41,77%) — mơ hồ về vai trò, định nghĩa nhiệm vụ không rõ ràng; **Coordination Failures** (36,94%) — gián đoạn giao tiếp, mất đồng bộ trạng thái; **Verification Gaps** (21,30%) — thiếu xác thực, không có kiểm tra chất lượng. Nhóm **Groupthink** (arXiv:2508.05687) bổ sung thêm: sụp đổ do độc canh (cùng một base model → lỗi tương quan), thiên kiến tuân thủ (các agent củng cố lỗi của nhau), thiếu hụt theory of mind, động lực hỗn hợp, và các lỗi tin cậy dây chuyền (cascading reliability failures). Ví dụ về lỗi dây chuyền: các đợt retry storm (bão thử lại) xảy ra khi một lỗi thanh toán kích hoạt việc thử lại đơn hàng, từ đó kích hoạt thử lại kho hàng, làm quá tải dịch vụ kho (tải gấp 10 lần trong vài giây — cần có circuit breakers). Memory poisoning: ảo giác của một agent đi vào bộ nhớ chia sẻ, các agent hạ nguồn coi đó là sự thật; độ chính xác suy giảm dần, khiến việc chẩn đoán nguyên nhân gốc rễ trở nên khó khăn. **STRATUS** (NeurIPS 2025) báo cáo mức cải thiện thành công trong giảm thiểu lỗi lên tới 1,5 lần thông qua các agent chuyên biệt về phát hiện / chẩn đoán / xác thực. Bài học này coi các failure mode là các mục tiêu kỹ thuật hạng nhất.

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 13 (Shared Memory), Phase 16 · 14 (Consensus and BFT), Phase 16 · 15 (Voting and Debate Topology)
**Time:** ~75 phút

## Problem

Các hệ thống đa tác tử (multi-agent systems) thất bại từ 41-86,7% thời gian trên các tác vụ thực tế (Cemri và cộng sự 2025 đã đo lường điều này trên 7 MAS mã nguồn mở). Điều này không thể debug chỉ bằng cách "thêm nhiều agent hơn". Các thất bại này có nguyên nhân cấu trúc. Phân loại MAST cung cấp cho bạn các danh mục. Bài học này ánh xạ từng danh mục tới một mô hình phát hiện, chẩn đoán và giảm thiểu cụ thể để các con số không còn vẻ ngẫu nhiên.

Thực tiễn sản xuất năm 2026 là coi các failure mode như là đầu vào thiết kế. Kiến trúc của bạn chưa "đủ tốt" cho đến khi bạn có thể chỉ ra từng danh mục MAST và nêu tên biện pháp giảm thiểu mà bạn đã triển khai.

## Concept

### Các danh mục MAST

**Specification Problems (41,77% các thất bại).** Nhiệm vụ của agent không được định nghĩa đủ chặt chẽ. Ví dụ:

- Mơ hồ về vai trò: hai agent cùng nghĩ rằng mình là người đánh giá (reviewer).
- Nhiệm vụ không được chỉ định rõ: "tóm tắt cái này" khi người dùng muốn một góc nhìn cụ thể.
- Tiêu chí thành công ngầm định: agent không thể biết liệu nó đã thành công hay chưa.

Các biện pháp giảm thiểu:
- Viết hợp đồng vai trò rõ ràng. Prompt của mỗi agent nêu rõ những gì nó làm *và những gì nó không làm*.
- Kiểm thử chấp nhận (acceptance tests) cho mỗi tác vụ. Trước khi agent bắt đầu, hãy định nghĩa "hoàn thành trông như thế nào".
- Kiểm tra đặc tả trước khi chạy (pre-flight spec check): một agent riêng biệt xem xét định nghĩa tác vụ trước khi phân phối.

**Coordination Failures (36,94%).** Sự cố về giao tiếp hoặc trạng thái.

Ví dụ:
- Hai agent cập nhật trạng thái chia sẻ mà không đồng bộ hóa.
- Tin nhắn bị mất giữa các agent (lỗi hàng đợi, timeout).
- Trôi trạng thái (state drift): agent A nghĩ tác vụ đã xong; agent B vẫn đang thực thi.

Các biện pháp giảm thiểu:
- Trạng thái chia sẻ có phiên bản với optimistic concurrency.
- Xác nhận rõ ràng (explicit acknowledgment) cho các tin nhắn quan trọng (thử lại cho đến khi nhận được xác nhận).
- Các điểm kiểm tra đồng bộ trạng thái định kỳ; phát hiện trôi trạng thái sớm.

**Verification Gaps (21,30%).** Không có kiểm tra độc lập đối với các đầu ra.

Ví dụ:
- Một agent tuyên bố thành công; không ai xác minh.
- Chuỗi các agent tin tưởng vào đầu ra của agent trước đó.
- Thiếu độ bao phủ kiểm thử (test coverage) trên hành vi tổng hợp mới nổi.

Các biện pháp giảm thiểu:
- Agent xác minh độc lập (Bài học 13). Chỉ đọc, truy cập nguồn độc lập.
- Hợp đồng bàn giao rõ ràng: "Đầu ra của A phải vượt qua bộ kiểm tra C trước khi B bắt đầu."
- Ghi nhật ký kết quả để phân tích hậu kiểm.

### Nhóm Groupthink (arXiv:2508.05687)

Năm thất bại liên quan khi các agent trở nên đồng nhất hoặc bắt chước lẫn nhau:

**Sụp đổ do độc canh (Monoculture collapse).** Cùng base model hoặc dữ liệu huấn luyện → lỗi tương quan. Khi ba agent chia sẻ một LLM, chúng chia sẻ cả ảo giác của nó.

**Thiên kiến tuân thủ (Conformity bias).** Các agent điều chỉnh theo hướng của đồng nghiệp ồn ào hoặc tự tin nhất, ngay cả khi sai.

**Thiếu hụt ToM (Deficient ToM).** Các agent không mô hình hóa được niềm tin của nhau; sự phối hợp bị phá vỡ (Bài học 18).

**Động lực hỗn hợp (Mixed-motive dynamics).** Các agent với các ưu đãi được căn chỉnh một phần sẽ trôi về hướng thỏa hiệp ở giữa, điều không làm hài lòng ai cả.

**Lỗi tin cậy dây chuyền (Cascading reliability failures).** Mô hình lỗi của một thành phần kích hoạt các mô hình lỗi trong các thành phần phụ thuộc.

### Ví dụ về lỗi dây chuyền — retry storm

Một mô hình sự cố kinh điển năm 2026:

```
payment service fails 10% of requests
   ↓
order agent retries payment (exponential backoff but naive)
   ↓
each retry is a new order-inventory check
   ↓
inventory service sees 2x normal load
   ↓
inventory service starts timing out
   ↓
every order retries inventory check
   ↓
inventory service sees 10x normal load
   ↓
cluster goes down
```

Giải pháp là kinh điển: **circuit breakers** (cầu dao). Khi tỷ lệ lỗi hạ nguồn vượt quá ngưỡng, hãy ngắt mạch với các kết quả được lưu trong bộ nhớ đệm hoặc kết quả mặc định. Cộng với ngân sách thử lại (retry budget) giới hạn cho mỗi yêu cầu.

Circuit breakers là một trong số ít các biện pháp giảm thiểu lỗi đa tác tử mà bạn mượn trực tiếp từ các hệ thống phân tán mà không cần sửa đổi.

### Memory poisoning (xem lại)

Từ Bài học 13: ảo giác của một agent trở thành sự thật trong bộ nhớ chia sẻ; các agent hạ nguồn suy luận dựa trên sự thật bị nhiễm độc đó. Theo thuật ngữ MAST, đây là một lỗ hổng xác minh ở lớp bộ nhớ chia sẻ.

Sự suy giảm độ chính xác dần dần là triệu chứng. Bạn không gặp sự cố sập hệ thống; bạn gặp sự trôi dạt chậm chạp khó tìm ra nguyên nhân gốc rễ.

Biện pháp giảm thiểu: nhật ký chỉ ghi (append-only log), truy xuất nguồn gốc, bộ xác minh không thể ghi. Đã được đề cập trong Bài học 13.

### STRATUS — các agent chuyên biệt để phát hiện lỗi

STRATUS (NeurIPS 2025) báo cáo mức cải thiện thành công trong giảm thiểu lỗi lên tới 1,5 lần khi bạn triển khai:

- **Detection agent.** Theo dõi các mô hình triệu chứng (bất đồng cao, đột biến thử lại, trôi độ chính xác).
- **Diagnosis agent.** Dựa trên các triệu chứng, suy luận nguyên nhân gốc rễ có khả năng nhất từ phân loại MAST.
- **Validation agent.** Sau khi biện pháp giảm thiểu được áp dụng, kiểm tra xem các triệu chứng đã biến mất chưa.

Đây là phản ứng sự cố theo phong cách SRE, áp dụng cho các hệ thống agent. Cả ba vai trò đều có thể là các LLM agent với các prompt chuyên biệt.

### Kiểm toán failure-mode

Một thực tiễn tốt nhất năm 2026 là kiểm toán failure-mode hàng năm (hoặc mỗi lần phát hành lớn):

1. **Trace sample.** Thu thập khoảng 1000 dấu vết thực thi thực tế.
2. **Categorize.** Đối với các lỗi của mỗi dấu vết, ánh xạ vào các danh mục MAST + Groupthink.
3. **Compute failure-by-category rate.** Danh mục nào chiếm ưu thế trong hệ thống của bạn?
4. **Rank mitigations.** Giải pháp nào sẽ loại bỏ được nhiều lỗi nhất?
5. **Pick 2-3 mitigations.** Triển khai; kiểm toán lại vào quý sau.

Kỷ luật quan trọng hơn các lựa chọn cụ thể. Nếu không có kiểm toán, các lỗi sẽ hòa lẫn vào nhiễu và không bao giờ được giải quyết một cách hệ thống.

### Khi hệ thống thất bại trong im lặng

Danh mục thất bại nguy hiểm nhất là thất bại đúng đắn trong im lặng (silent correctness failure). Một hệ thống thất bại ồn ào (crash, ngoại lệ, cảnh báo) có thể được giám sát. Một hệ thống tạo ra các đầu ra có vẻ hợp lý nhưng sai thì không thể bị phát hiện bởi nhật ký ngoại lệ. Đây là lý do tại sao các lỗ hổng xác minh là danh mục đắt đỏ nhất trên mỗi lỗi mặc dù chúng chỉ chiếm 21,30% theo số lượng.

Hãy đầu tư vào:
- Đánh giá của con người dựa trên mẫu.
- Kiểm thử hồi quy (regression tests) với tập dữ liệu vàng (golden-dataset).
- Kiểm tra chéo giữa các agent đối với các đầu ra quan trọng.

### Thất bại vs thất bại chậm

Một số thất bại là tức thời; một số là chậm. Các thất bại tức thời (timeout, không khớp schema, lỗi xác thực) rất rẻ để phát hiện. Các thất bại chậm (memory poisoning, trôi dạt độc canh, mơ hồ vai trò) rất đắt đỏ để phát hiện và ngăn chặn.

Động thái kỹ thuật năm 2026: đo lường các proxy thất bại chậm để bạn có thể bắt được sự trôi dạt trước khi nó trở thành một lỗi hiển thị. Tỷ lệ đồng thuận, tỷ lệ thử lại, phân phối độ dài đầu ra và khoảng cách chỉnh sửa (edit-distance) giữa các phiên bản agent liên tiếp đều là các proxy hữu ích.

```figure
a5-retry-cascade
```

## Build It

`code/main.py` triển khai:

- `FailureTaxonomy` — phân loại các sự cố mô phỏng vào các danh mục MAST + Groupthink.
- `CircuitBreaker` — mô hình kinh điển; mở khi tỷ lệ lỗi vượt quá ngưỡng.
- `RetryStormSimulator` — hiển thị lỗi dây chuyền; bật / tắt circuit breaker.
- `DetectionAgent` — trình khớp triệu chứng theo phong cách STRATUS được viết kịch bản.

Chạy:

```
python3 code/main.py
```

Đầu ra mong đợi:
- retry storm không có circuit breaker: lỗi kho hàng bùng nổ (mô phỏng).
- với circuit breaker: giới hạn ở ngưỡng; các phản hồi chế độ suy giảm được phục vụ.
- detection agent gắn cờ mô hình và đặt tên danh mục MAST.

## Use It

`outputs/skill-mast-auditor.md` chạy kiểm toán failure-mode theo phong cách MAST trên một hệ thống đa tác tử. Dấu vết → phân loại → xếp hạng giảm thiểu.

## Ship It

Kỷ luật failure-mode trong sản xuất:

- **Kiểm toán MAST mỗi quý.** Không phải hàng năm. Các danh mục thay đổi khi hệ thống của bạn phát triển.
- **Circuit breakers ở khắp mọi nơi.** Mỗi cuộc gọi đi đến bất kỳ dịch vụ phụ thuộc nào. Ngưỡng mở mặc định ở tỷ lệ lỗi 5-10%.
- **Golden datasets.** Nhỏ, chất lượng cao, được kiểm toán thủ công. Kiểm thử hồi quy chống lại chúng hàng tuần.
- **Bộ ba STRATUS.** Các agent Phát hiện + Chẩn đoán + Xác thực giám sát sản xuất. Bắt đầu chỉ với agent phát hiện; thêm chẩn đoán khi các triệu chứng trở nên nhiễu.
- **Ngân sách thất bại (Failure budget).** SLO rõ ràng cho tỷ lệ lỗi theo danh mục. Vượt quá ngân sách sẽ kích hoạt cuộc trò chuyện dừng vận chuyển.

## Exercises

1. Chạy `code/main.py`. Xác nhận circuit breaker giới hạn retry storm. Thay đổi ngưỡng lỗi và quan sát sự đánh đổi.
2. Triển khai một **proxy thất bại chậm**: tỷ lệ đồng thuận trên 3 agent song song. Khi nó giảm mạnh, hãy kích hoạt cảnh báo. Mô phỏng sự trôi dạt độc canh bằng cách dần dần tương quan các đầu ra của agent.
3. Đọc Cemri và cộng sự (arXiv:2503.13657). Chọn một trong 7 hệ thống MAS của họ và ánh xạ 3 danh mục thất bại hàng đầu của nó. Chúng so sánh thế nào với những gì MAST dự đoán?
4. Đọc bài báo về Groupthink (arXiv:2508.05687). Xác định mô hình nào trong năm mô hình là khó phát hiện nhất trong sản xuất. Đề xuất một chỉ số proxy.
5. Thiết kế bộ ba phát hiện-chẩn đoán-xác thực theo phong cách STRATUS cho một hệ thống đa tác tử cụ thể mà bạn biết. Detection theo dõi những triệu chứng nào? Diagnosis đề xuất những biện pháp giảm thiểu nào? Validation xác nhận chúng hoạt động như thế nào?

## Key Terms

| Thuật ngữ | Mọi người nói | Ý nghĩa thực sự |
|------|----------------|------------------------|
| MAST | "Phân loại năm 2026" | Cemri 2025; 3 danh mục gốc + 14 loại phụ của thất bại. |
| Specification Problem | "Mơ hồ vai trò" | Nhiệm vụ hoặc vai trò chưa được định nghĩa; agent không biết phải làm gì. |
| Coordination Failure | "Trôi trạng thái" | Sự cố giao tiếp hoặc đồng bộ giữa các agent. |
| Verification Gap | "Không ai kiểm tra" | Đầu ra được chấp nhận mà không có xác thực độc lập. |
| Groupthink family | "Thất bại đồng nhất" | Độc canh, tuân thủ, thiếu hụt ToM, động lực hỗn hợp, dây chuyền. |
| Monoculture collapse | "Cùng model, cùng ảo giác" | Lỗi tương quan từ base model hoặc dữ liệu huấn luyện chia sẻ. |
| Retry storm | "Khuếch đại lỗi dây chuyền" | Một thất bại kích hoạt các lần thử lại làm khuếch đại tải hạ nguồn. |
| Circuit breaker | "Thất bại nhanh khi có lỗi" | Mở khi tỷ lệ lỗi vượt ngưỡng; ngắt mạch với kết quả mặc định. |
| STRATUS | "Bộ ba phản ứng sự cố" | Các agent Phát hiện + Chẩn đoán + Xác thực. Thành công giảm thiểu 1,5 lần. |
| Memory poisoning | "Ảo giác lan truyền" | Sự thật trong bộ nhớ chia sẻ bị nhiễm độc; agent hạ nguồn suy luận trên chất độc. |

## Further Reading

- [Cemri và cộng sự — Tại sao các hệ thống LLM đa tác tử thất bại?](https://arxiv.org/abs/2503.13657) — Phân loại MAST, NeurIPS 2025
- [Thất bại Groupthink trong LLM đa tác tử](https://arxiv.org/abs/2508.05687) — độc canh, tuân thủ và phân loại năm gia đình
- [STRATUS — các agent chuyên biệt cho phản ứng sự cố MAS](https://neurips.cc/) — mục nhập kỷ yếu NeurIPS 2025 (phát hiện + chẩn đoán + xác thực)
- [Release It! — các mô hình ổn định (Nygard)](https://pragprog.com/titles/mnee2/release-it-second-edition/) — tài liệu tham khảo kinh điển về circuit-breaker
- [Anthropic — Hệ thống nghiên cứu đa tác tử](https://www.anthropic.com/engineering/multi-agent-research-system) — ghi chú về failure-mode trong sản xuất