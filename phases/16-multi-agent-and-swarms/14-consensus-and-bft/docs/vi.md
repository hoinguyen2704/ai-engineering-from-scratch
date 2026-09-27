# Consensus và Byzantine Fault Tolerance cho các Agent

> Sự kết hợp giữa BFT trong hệ thống phân tán cổ điển và các LLM mang tính ngẫu nhiên. Trong giai đoạn 2025-2026, ba hướng nghiên cứu đã xuất hiện: **CP-WBFT** (arXiv:2511.10400) trọng số hóa mỗi phiếu bầu bằng một confidence probe; **DecentLLMs** (arXiv:2507.14928) loại bỏ vai trò leader với các đề xuất worker song song và tổng hợp bằng geometric-median; **WBFT** (arXiv:2505.05103) kết hợp bỏ phiếu có trọng số với Hierarchical Structure Clustering để phân tách các node Core và Edge. Kết quả thực nghiệm trung thực từ "Can AI Agents Agree?" (arXiv:2603.01213) cho thấy ngay cả sự đồng thuận về giá trị vô hướng (scalar) hiện nay cũng rất mong manh — một agent lừa đảo duy nhất có thể làm tổn hại đến một Mixture-of-Agents. BFT là cần thiết nhưng chưa đủ. Bài học này xây dựng một giao thức BFT tối giản, đưa vào ba kiểu tấn công đặc thù của agent (byzantine lie, sycophantic conformity, correlated-error monoculture), và đo lường cách mỗi biến thể đồng thuận đối phó với chúng.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 07 (Society of Mind and Debate), Phase 16 · 13 (Shared Memory)
**Time:** ~75 phút

## Vấn đề

Bạn có N LLM agent, mỗi agent đưa ra một câu trả lời. Chúng không đồng ý với nhau. Bỏ phiếu đa số chọn sai vì hai agent có sự tương quan (cùng base model, cùng dữ liệu huấn luyện, cùng kiểu lỗi). Một agent thứ ba lại sai theo một cách mới lạ — vì vậy đa số trở thành một đa số giả.

Bây giờ hãy thêm một agent lừa đảo: nó cố tình nói dối. Hoặc một agent nịnh hót (sycophantic): nó đồng ý với bất kỳ ai lên tiếng cuối cùng. Trong BFT cổ điển, giả định là các node Byzantine chiếm một phần `f < n/3` và hành xử tùy ý. Thực tế năm 2026 là các node LLM mang tính ngẫu nhiên ngay cả khi trung thực, có sự tương quan giữa các model và bị ảnh hưởng bởi đầu ra của nhau. Bạn không thể coi chúng là các cử tri Bernoulli độc lập.

BFT cổ điển (PBFT, 1999) không sai — nó chỉ chưa đầy đủ. Nó xử lý việc lật bit tùy ý. Nó không xử lý việc "ba agent trung thực chia sẻ một ảo giác vì chúng chia sẻ dữ liệu huấn luyện". Bài học này xây dựng dựa trên nền tảng của PBFT và bổ sung ba phương pháp thích ứng của giai đoạn 2025-2026.

## Khái niệm

### BFT cổ điển mang lại cho bạn điều gì

Practical Byzantine Fault Tolerance (Castro & Liskov, OSDI 1999) chịu lỗi được `f < n/3` node Byzantine. Giao thức có ba giai đoạn (pre-prepare, prepare, commit) và hai nguyên hàm (signed messages, quorum certificates). Sự đồng thuận về một giá trị duy nhất giữa `n >= 3f + 1` node trung thực hoặc độc hại.

Các đảm bảo rất mạnh mẽ nhưng giả định rằng:

1. **Các lỗi độc lập.** Các node Byzantine không phối hợp với nhau.
2. **Các node trung thực là thực sự trung thực.** Tính đúng đắn của đầu ra từ node trung thực không phải là vấn đề; giao thức chỉ điều chỉnh sự bất đồng.
3. **Câu hỏi có câu trả lời đúng (ground-truth).** Sự đồng thuận về một sự thật sai vẫn là sự đồng thuận.

Các LLM agent vi phạm cả ba điều trên. Hai agent chạy cùng một base model sẽ chia sẻ lỗi. Một LLM "trung thực" vẫn có thể ảo giác. Và đối với các câu hỏi mơ hồ, "sự thật" là những gì các agent quyết định — không có oracle bên ngoài nào cả.

### Ba kiểu tấn công đặc thù của LLM

**Byzantine lie.** Một agent đưa ra câu trả lời sai một cách cố ý. BFT cổ điển xử lý được điều này nếu `f < n/3`.

**Sycophantic conformity.** Một agent đọc câu trả lời của các agent khác trước khi bỏ phiếu và đồng ý với bất kỳ ai lên tiếng cuối cùng. Không độc hại, nhưng tạo ra sự tương quan với tiếng nói lớn nhất. BFT cổ điển không ngăn chặn được điều này vì agent vượt qua mọi kiểm tra chữ ký.

**Correlated-error monoculture.** Ba agent chia sẻ cùng một base model. Chúng ảo giác ra cùng một câu trả lời sai. Đa số là sai. BFT cổ điển không giúp ích gì vì cả ba đều "trung thực" đồng ý với nhau.

### Các phản hồi giai đoạn 2025-2026

**CP-WBFT** (arXiv:2511.10400) — Confidence-Probed Weighted BFT. Mỗi cử tri đính kèm một confidence probe vào câu trả lời của mình (một xác suất tự báo cáo, hoặc dự đoán từ một model hiệu chuẩn riêng). Trọng số phiếu bầu tỉ lệ thuận với độ tin cậy. Cải thiện 85.71% BFT trên các đồ thị đầy đủ. Giảm thiểu cho: sycophantic conformity (các agent nịnh hót thường có độ tin cậy thấp đối với vị trí mà chúng tự nguyện đưa ra).

**DecentLLMs** (arXiv:2507.14928) — Không leader. Các agent worker đề xuất song song, các agent đánh giá chấm điểm các đề xuất, câu trả lời cuối cùng là geometric median của các vị trí đã được chấm điểm. Mạnh mẽ khi `f < n/2`. Giảm thiểu cho: Byzantine lie và các lỗi tương quan (geometric median mạnh mẽ trước các giá trị ngoại lai và kéo về phía cụm dày đặc, thay vì trung bình bị lệch theo model).

**WBFT** (arXiv:2505.05103) — Weighted BFT với Hierarchical Structure Clustering. Trọng số phiếu bầu được gán bởi chất lượng phản hồi cộng với điểm tin cậy được học từ lịch sử. Phân cụm các agent thành Core và Edge; các agent Core phải đạt được sự đồng thuận trước, các agent Edge theo sau. Giảm thiểu cho: khả năng mở rộng (sự đồng thuận của Core nhỏ và nhanh) và một phần cho monoculture (Core có thể được chọn để đảm bảo tính đa dạng).

### Thực nghiệm: "Can AI Agents Agree?" (arXiv:2603.01213)

Bài báo đo lường sự đồng thuận vô hướng (các LLM agent đồng ý về một giá trị số duy nhất) trên nhiều model tiên phong. Kết quả khá đáng lo ngại:

- Ngay cả khi không có đối thủ, các LLM agent vẫn bất đồng về các câu hỏi vô hướng với tỉ lệ trên 30% ở nhiều benchmark.
- Một agent duy nhất áp dụng persona lừa đảo có thể kéo sự đồng thuận của Mixture-of-Agents lệch khỏi đường cơ sở trung thực hơn 40 điểm phần trăm.
- Tỉ lệ bất đồng tương quan với sự đa dạng của model — các ensemble không đồng nhất bất đồng nhiều hơn các ensemble đồng nhất (tốt: các lỗi không tương quan) nhưng cũng trôi (drift) chậm hơn (xấu: thời gian đạt đồng thuận lâu hơn).

Kết luận: BFT cung cấp cho bạn cơ chế để căn chỉnh đầu ra, nhưng nó không cho bạn biết liệu đầu ra đã căn chỉnh đó có đúng hay không. Hãy kết hợp với xác minh (Phase 16 · 08 role specialization), tính đa dạng (Phase 16 · 15 debate variants), và các agent đánh giá (Phase 16 · 24 benchmarks).

### Giao thức cốt lõi, được tinh giản

Một vòng BFT tối giản cho các LLM agent:

```
1. task arrives; each agent i produces answer a_i
2. each agent attaches confidence probe c_i in [0, 1]
3. aggregator collects (a_i, c_i) from all n agents
4. aggregator groups by semantic cluster (equivalent answers)
5. aggregator computes weight for each cluster C:
     w(C) = sum_{i in C} c_i
6. winner = cluster with max weight, if max > threshold * sum(c_i)
   else: retry or escalate
7. minority clusters logged with provenance for post-hoc audit
```

Bước phân cụm ngữ nghĩa là điểm nhấn đặc thù của LLM. Hai câu trả lời "nghiên cứu báo cáo 4.2%" và "cải thiện 4.2%" là cùng một cụm. Một kiểm tra bằng chuỗi ký tự đơn thuần sẽ bỏ lỡ điều này. Trong thực tế, hãy sử dụng một embedding model giá rẻ hoặc chuẩn hóa tường minh.

### Điều chỉnh ngưỡng (Threshold tuning)

Tham số `threshold` quyết định khi nào chấp nhận và khi nào thử lại. Quá thấp: bạn chấp nhận các đa số yếu. Quá cao: bạn không bao giờ chấp nhận bất cứ điều gì. Phạm vi thực nghiệm: 0.5-0.67 cho `n=5-7` agent, cao hơn cho `n` nhỏ hơn. Dưới một ngưỡng, hãy leo thang lên con người hoặc một ensemble agent khác.

### Nơi sự đồng thuận không giúp ích

- **Câu hỏi mơ hồ.** Nếu câu hỏi không có sự thật khách quan, sự đồng thuận chỉ là một ý kiến. Hãy gọi nó là như vậy.
- **Câu hỏi phức hợp.** "Viết code và giải thích nó" — hai câu trả lời. Hãy bỏ phiếu cho từng phần một cách độc lập.
- **Đối kháng nhiều vòng.** Nếu các agent có thể quan sát các vòng trước và bắt chước (Du 2023 debate), chúng bắt đầu đồng ý với nhau bất kể sự thật. Hãy giới hạn số vòng (thường là 2-3).

```figure
swarm-consensus-wave
```

## Xây dựng

`code/main.py` triển khai:

- `AgentVoter` — một chính sách kịch bản với (câu trả lời, độ tin cậy).
- `MajorityVote` — đa số cổ điển (plurality).
- `CPWBFT` — bỏ phiếu có trọng số theo độ tin cậy với phân cụm ngữ nghĩa.
- `DecentLLMs` — tổng hợp geometric-median trên các đề xuất đã được chấm điểm.
- `Scenario` — chạy mỗi bộ tổng hợp dưới ba kiểu tấn công.

Các kiểu tấn công được triển khai:

1. `byzantine`: một agent nói dối với độ tin cậy cao.
2. `sycophancy`: một agent sao chép câu trả lời đầu tiên nó thấy, với độ tin cậy tương ứng.
3. `monoculture`: ba agent chia sẻ một câu trả lời sai (lỗi tương quan) với độ tin cậy trung bình.

Chạy:

```
python3 code/main.py
```

Kết quả mong đợi: một bảng (tấn công, bộ tổng hợp) -> câu trả lời cuối cùng, với câu trả lời đúng được làm nổi bật. Plurality thất bại trong trường hợp monoculture. Trọng số độ tin cậy của CPWBFT giảm thiểu sự nịnh hót. Geometric-median của DecentLLMs kéo về phía cụm trung thực khi monoculture chiếm ít hơn một nửa dân số.

## Sử dụng

`outputs/skill-consensus-designer.md` thiết kế một giao thức đồng thuận cho một ensemble đa agent: phương pháp phân cụm, trọng số, ngưỡng, và chính sách leo thang cho các vòng dưới ngưỡng.

## Triển khai

Trước khi triển khai bất kỳ cơ chế đồng thuận nào:

- **Kiểm thử tấn công với ít nhất ba kiểu** ở trên. Giao thức của bạn nên thất bại một cách có thể dự đoán được, không phải âm thầm.
- **Ghi lại mọi cụm thiểu số** cùng với nguồn gốc. Các cụm thiểu số là hệ thống cảnh báo sớm cho các lỗi tương quan.
- **Thực thi các vòng giới hạn.** Không có chuyện "tiếp tục tranh luận cho đến khi đồng ý" — điều đó khuyến khích sự nịnh hót.
- **Tách biệt sự đồng thuận khỏi tính đúng đắn.** Đầu ra đồng thuận đi đến một bộ xác minh; bộ xác minh độc lập với ensemble.
- **Giám sát tỉ lệ đồng thuận.** Sự gia tăng đột ngột có nghĩa là thiên kiến tuân thủ; sự sụt giảm đột ngột có nghĩa là model bị trôi (drift).

## Bài tập

1. Chạy `code/main.py`. Xác nhận rằng plurality thất bại trong cuộc tấn công monoculture nhưng CPWBFT giảm thiểu một phần khi độ tin cậy của monoculture dưới 0.7.
2. Thêm kiểu tấn công thứ tư: **im lặng bỏ phiếu** — một agent từ chối trả lời ("Tôi không biết"). Mỗi bộ tổng hợp nên xử lý việc bỏ phiếu trắng như thế nào? Hãy triển khai lựa chọn của bạn.
3. Thay đổi phân cụm ngữ nghĩa từ chuẩn hóa chuỗi sang độ tương đồng embedding (sử dụng bất kỳ embedding model mã nguồn mở nào). Điều gì xảy ra với cuộc tấn công nịnh hót?
4. Đọc CP-WBFT (arXiv:2511.10400). Triển khai bước hiệu chuẩn confidence-probe (một model hiệu chuẩn riêng kiểm tra độ tin cậy tự báo cáo của mỗi agent). Đo lường mức tăng độ chính xác trong kịch bản monoculture.
5. Đọc "Can AI Agents Agree?" (arXiv:2603.01213). Tái tạo một thí nghiệm đồng thuận vô hướng đơn giản: ba agent, một câu hỏi vô hướng, prompt persona lừa đảo. CPWBFT hay DecentLLMs có bắt được nó không?

## Thuật ngữ chính

| Thuật ngữ | Mọi người nói | Ý nghĩa thực sự |
|------|----------------|------------------------|
| BFT | "Byzantine fault tolerance" | Giao thức Castro-Liskov 1999 cho sự đồng thuận với `f < n/3` lỗi tùy ý. |
| Byzantine | "Bất kỳ hành vi xấu nào" | Một node có thể nói dối, bỏ tin nhắn, lỗi im lặng — bất cứ điều gì ngoại trừ crash an toàn. |
| Confidence probe | "Bạn chắc chắn đến mức nào?" | Xác suất tự báo cáo hoặc dự đoán bởi bộ hiệu chuẩn được đính kèm vào phiếu bầu. |
| Semantic clustering | "Cùng câu trả lời, khác từ ngữ" | Nhóm các câu trả lời tương đương trước khi đếm phiếu. |
| Geometric median | "Trung tâm mạnh mẽ" | Điểm tối thiểu hóa tổng khoảng cách đến các điểm mẫu. Mạnh mẽ trước các giá trị ngoại lai, không giống như trung bình. |
| Monoculture | "Cùng model, cùng lỗi" | Các lỗi tương quan khi các agent chia sẻ dữ liệu huấn luyện hoặc base model. |
| Sycophantic conformity | "Đồng ý với tiếng nói lớn" | Phiếu bầu của agent bị thiên lệch về phía bất kỳ ai lên tiếng đầu tiên/lớn nhất. |
| Core/Edge | "Hierarchical BFT" | Phân tách WBFT: đồng thuận Core nhỏ trước, các node Edge theo sau. Giới hạn độ trễ. |

## Đọc thêm

- [Castro & Liskov — Practical Byzantine Fault Tolerance (OSDI 1999)](https://pmg.csail.mit.edu/papers/osdi99.pdf) — nền tảng
- [CP-WBFT — Confidence-Probe Weighted BFT](https://arxiv.org/abs/2511.10400) — trọng số phiếu bầu theo độ tin cậy
- [DecentLLMs — leaderless multi-agent consensus](https://arxiv.org/abs/2507.14928) — tổng hợp geometric-median
- [WBFT — Weighted BFT with Hierarchical Structure Clustering](https://arxiv.org/abs/2505.05103) — phân tách Core/Edge cho độ trễ giới hạn
- [Can AI Agents Agree?](https://arxiv.org/abs/2603.01213) — sự mong manh của đồng thuận vô hướng và tấn công deceptive-persona