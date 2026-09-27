# Voting, Self-Consistency, và Topology của Debate

> Cách tổng hợp rẻ nhất: lấy mẫu N tác nhân độc lập, sau đó bỏ phiếu đa số (majority-vote). Phương pháp self-consistency của Wang và cộng sự (2022) đã thực hiện điều này với một model được lấy mẫu N lần. Multi-agent mở rộng phương pháp này bằng các tác nhân **không đồng nhất (heterogeneous)** để thoát khỏi sự đơn điệu (monoculture) — các model khác nhau, prompt khác nhau, temperature khác nhau, ngữ cảnh khác nhau. Ngoài việc bỏ phiếu đa số, topology của debate đóng vai trò quan trọng: MultiAgentBench (arXiv:2503.01935, ACL 2025) đã đánh giá các cấu trúc điều phối star / chain / tree / graph và nhận thấy **graph là tốt nhất cho nghiên cứu**, với "thuế điều phối" (coordination tax) xuất hiện sau khoảng 4 tác nhân. AgentVerse (ICLR 2024) ghi lại hai mô hình hành vi nổi bật — hành vi tình nguyện (volunteer) và hành vi tuân thủ (conformity) — trong đó sự tuân thủ vừa là một tính năng (để đạt được đồng thuận) vừa là một rủi ro (tư duy nhóm, Bài 24). Bài học này vạch ra không gian topology, xây dựng từng biến thể và đo lường thuế điều phối.

**Type:** Học + Xây dựng
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 07 (Society of Mind and Debate), Phase 16 · 14 (Consensus and BFT)
**Time:** ~75 phút

## Vấn đề

Debate có thể cải thiện độ chính xác (Du và cộng sự, arXiv:2305.14325). Nó cũng có thể làm giảm độ chính xác đó. Việc debate có giúp ích hay không phụ thuộc vào bốn lựa chọn cấu trúc:

1. Ai nói chuyện với ai (topology).
2. Bao nhiêu vòng (Du 2023: cả số vòng và số tác nhân đều quan trọng một cách độc lập).
3. Các tác nhân có không đồng nhất hay không (các base model khác nhau giúp phá vỡ sự đơn điệu).
4. Có sự hiện diện của tiếng nói phản biện hay không (steel-manning so với straw-manning).

Các nhóm thường áp dụng kiểu "chạy 5 tác nhân rồi bỏ phiếu" vào một tác vụ thường cho kết quả kém hơn so với một tác nhân đơn lẻ. Những thất bại này không ngẫu nhiên. Chúng tuân theo topology và tính không đồng nhất. Bài học này là bản đồ topology.

## Khái niệm

### Self-consistency, baseline của single-model

Wang và cộng sự (2022) ("Self-Consistency Improves Chain of Thought Reasoning") đã lấy mẫu cùng một model N lần ở temperature > 0 và bỏ phiếu đa số cho các câu trả lời theo chuỗi suy luận. Kết quả trên GSM8K: đạt mức tăng đáng kể với N=40 mẫu so với giải mã greedy đơn lẻ. Self-consistency là tiền thân của bỏ phiếu multi-agent.

Hạn chế: self-consistency sử dụng một base model duy nhất. Các lỗi bị tương quan theo cấu trúc. Nếu model có thiên kiến hệ thống, tất cả N mẫu đều chia sẻ thiên kiến đó.

### Bỏ phiếu multi-agent, mở rộng không đồng nhất

Thay thế N mẫu bằng N tác nhân *khác nhau*. Các base model khác nhau (Claude, GPT, Llama), prompt khác nhau, quyền truy cập công cụ khác nhau. Lợi ích: các lỗi không tương quan. Chi phí: các tác nhân khác nhau có chi phí khác nhau; việc điều phối chúng làm tăng chi phí vận hành.

Tên gọi chuẩn năm 2026 cho debate không đồng nhất là **A-HMAD** — Adversarial Heterogeneous Multi-Agent Debate. Dù chưa được áp dụng phổ biến, các bài báo sử dụng thuật ngữ này cho việc "các model khác nhau tranh luận, giúp giảm các lỗi tương quan do sự sụp đổ của monoculture."

### Bốn loại topology

```
star                chain               tree                graph

    ┌─A─┐           A─B─C─D         ┌──A──┐              A───B
    │   │                           │     │              │ × │
    B   C                           B     C              D───C
    │   │                          / \   / \
    D   E                         D   E F   G           (fully connected)
```

Star: một trung tâm (hub), tất cả các tác nhân khác chỉ nói chuyện với trung tâm. Tương đương với supervisor-worker mà không có kênh phản hồi ngược.
Chain: tuyến tính, mỗi tác nhân thấy đầu ra của tác nhân trước đó. Giống như pipeline.
Tree: phân cấp, được sử dụng bởi các hệ thống tác nhân phân cấp (Bài 06).
Graph: bất kỳ-đến-bất kỳ. Bao gồm clique kết nối đầy đủ và các DAG tùy ý.

### Thuế điều phối (MultiAgentBench)

MultiAgentBench (MARBLE, ACL 2025, arXiv:2503.01935) đã benchmark các cấu trúc star, chain, tree, graph trên một bộ tác vụ bao gồm nghiên cứu, lập trình và lập kế hoạch. Các kết quả đo lường chính:

- **Graph** topology thắng trong các tác vụ nghiên cứu. Thông tin luân chuyển bất kỳ-đến-bất kỳ; các tác nhân có thể phản biện lẫn nhau.
- **Star** thắng trong các tác vụ thực tế cần câu trả lời nhanh. Hub lọc và hợp nhất thông tin.
- **Chain** thắng trong các pipeline từng bước (tinh chỉnh theo giai đoạn).
- **Thuế điều phối** xuất hiện sau khoảng 4 tác nhân trong topology graph. Thời gian thực (wall-clock) và chi phí token tăng nhanh hơn chất lượng.

Giới hạn 4 tác nhân là thực nghiệm, không phải cơ bản. Nó phản ánh dung lượng ngữ cảnh của LLM năm 2026: ngữ cảnh của mỗi tác nhân bị lấp đầy bởi đầu ra của các đồng nghiệp, và giá trị biên của việc thêm tác nhân thứ N+1 giảm xuống khi mọi người đều có thể thấy mọi người.

### Chiến lược Multi-Agent Debate ("Chúng ta có nên thực hiện MAD?")

arXiv:2311.17371 là khảo sát năm 2023 về các chiến lược MAD. Phát hiện chính được những người khác lặp lại: các biến thể MAD *tương tự về cấu trúc* với self-consistency (lấy mẫu độc lập + tổng hợp) thường hoạt động kém hơn self-consistency khi sử dụng cùng ngân sách. MAD giúp ích nhiều nhất khi các tác nhân thực sự không đồng nhất và cuộc tranh luận có cấu trúc phản biện (một tác nhân tranh luận ngược lại).

### Các mô hình hành vi nổi bật của AgentVerse

AgentVerse (ICLR 2024, https://proceedings.iclr.cc/paper_files/paper/2024/file/578e65cdee35d00c708d4c64bce32971-Paper-Conference.pdf) ghi lại hai hành vi nảy sinh từ debate multi-agent ngay cả khi không được thiết kế rõ ràng:

- **Volunteer (Tình nguyện).** Một tác nhân đề nghị giúp đỡ ("Tôi có thể thực hiện bước tiếp theo") mà không cần nhắc nhở. Hữu ích: nó phân bổ công việc cho tác nhân có năng lực nhất cho một tác vụ con.
- **Conformity (Tuân thủ).** Một tác nhân điều chỉnh lập trường của mình để khớp với người phản biện, ngay cả khi người phản biện sai. Đây là phiên bản debate của sự nịnh hót (sycophancy, Bài 14).

Sự tuân thủ là lý do tại sao việc debate cho đến khi đạt được thỏa thuận thường thưởng cho những kẻ bắt nạt. Các vòng debate giới hạn với một giám khảo riêng biệt sẽ giảm thiểu điều này.

### Tính không đồng nhất: nút vặn thực sự thay đổi độ chính xác

Một mô hình năm 2024-2026 trong tài liệu thực tế: thay thế một trong N tác nhân của bạn bằng một base model khác mang lại mức tăng độ chính xác lớn hơn so với việc tăng N thêm 1. Trực giác ở đây là sự đơn điệu — mỗi nguồn lỗi độc lập mới có giá trị hơn một mẫu tương quan bổ sung.

Trong giới hạn, tính không đồng nhất đánh bại số lượng. Ba model khác nhau đánh bại năm bản sao của một model trong hầu hết các tác vụ có ground truth rõ ràng.

### Phương pháp Jury (Bồi thẩm đoàn)

Khung Sibyl (được trích dẫn trong tài liệu Minsky-LLM) chính thức hóa một "bồi thẩm đoàn" — một tập hợp nhỏ các tác nhân chuyên biệt tinh chỉnh câu trả lời bằng cách bỏ phiếu ở mỗi giai đoạn. Không giống như bỏ phiếu đa số đơn thuần, bồi thẩm đoàn có các vai trò: một tác nhân thẩm vấn chéo, một tác nhân cung cấp ngữ cảnh, một tác nhân chấm điểm tính hợp lý. Phương pháp bồi thẩm đoàn là điểm trung gian giữa bỏ phiếu đơn thuần (rẻ, dễ bị đơn điệu) và MAD đầy đủ (đắt, dễ bị tuân thủ).

### Khi nào bỏ phiếu kèm debate chiếm ưu thế

- Câu hỏi có ground truth (sự kiện, toán học, hành vi code). Sự hội tụ bỏ phiếu là có ý nghĩa.
- Các tác nhân có thể truy cập các nguồn hoặc công cụ khác nhau (tính không đồng nhất khả dụng).
- Các vòng debate bị giới hạn (thường là 2-3) và có một giám khảo hoặc người xác minh riêng biệt.
- Ngân sách cho phép 3-5 tác nhân. Vượt quá 5-7 trong topology graph, thuế điều phối sẽ chiếm ưu thế.

### Khi nào bỏ phiếu kèm debate gây hại

- Câu hỏi mang tính ý kiến. Các tác nhân hội tụ về câu trả lời trông có vẻ tự tin nhất, không phải đúng nhất.
- Tất cả các tác nhân chia sẻ cùng một base model. Sự đơn điệu làm cho sự đồng thuận trở nên vô nghĩa.
- Các vòng debate không giới hạn. Sự tuân thủ luôn thắng.
- Tác vụ đơn giản. Một tác nhân đơn lẻ với self-consistency ở N=5 rẻ hơn và chính xác tương đương.

```figure
sw-debate-topology
```

## Xây dựng

`code/main.py` triển khai:

- `run_star(agents, hub, question)` — hub thăm dò ý kiến từng worker, tổng hợp.
- `run_chain(agents, question)` — tinh chỉnh tuần tự.
- `run_tree(root, children, question)` — phân cấp với tổng hợp độ sâu 2.
- `run_graph(agents, question, rounds)` — tranh luận tất cả-với-tất cả, các vòng giới hạn.
- Một nút vặn không đồng nhất được lập trình: mỗi tác nhân có một `error_bias` chỉ ra sai số hệ thống của nó.
- Một bộ đo lường chạy từng topology ở N=3, 5, 7 và báo cáo (độ chính xác, tổng số token, thời gian mô phỏng).

Chạy:

```
python3 code/main.py
```

Kết quả mong đợi: một bảng topology × N → (độ chính xác, token, độ trễ). Graph thắng ở N=3-5 trên các tác vụ kiểu nghiên cứu; star thắng trên các tác vụ thực tế nhanh; graph ở N=7 cho thấy thuế điều phối (độ trễ tăng nhanh hơn độ chính xác).

## Sử dụng

`outputs/skill-topology-picker.md` là một kỹ năng đọc mô tả tác vụ và đề xuất topology (star / chain / tree / graph), số lượng N (số tác nhân), hồ sơ không đồng nhất (các base model cần sử dụng) và giới hạn vòng debate.

## Triển khai

Đối với bất kỳ ensemble nào:

- Bắt đầu với **self-consistency ở N=5** sử dụng một base model mạnh. Đó là baseline rẻ tiền.
- Nâng cấp lên **bỏ phiếu không đồng nhất ở N=3** nếu độ chính xác quan trọng. Đo lường delta.
- Chỉ nâng cấp lên **debate topology** nếu tác vụ có cấu trúc (nghiên cứu, đa bước) và các vòng giới hạn là khả thi.
- Luôn ghi lại cụm thiểu số. Khi một nhóm thiểu số liên tục đúng, bạn có một tín hiệu về sự đa dạng.
- Benchmark thời gian thực và token cùng với độ chính xác. "Độ chính xác tốt hơn với chi phí gấp 10 lần" là một quyết định kinh doanh.

## Bài tập

1. Chạy `code/main.py`. Vẽ đường cong thuế điều phối cho topology graph: độ chính xác vs N, token vs N. Tại N nào đường cong bắt đầu uốn cong?
2. Triển khai A-HMAD: ba tác nhân với các thiên kiến khác nhau một cách cố ý. Baseline với cùng thiên kiến so sánh thế nào với A-HMAD trong cuộc tấn công vào sự đơn điệu từ Bài 14?
3. Thêm vai trò "giám khảo" vào topology graph, vai trò này không bỏ phiếu mà chỉ chấm điểm sự đồng thuận cuối cùng. Điều này có thay đổi hành vi tuân thủ nảy sinh không?
4. Đọc bài báo AgentVerse (ICLR 2024). Xác định hành vi nảy sinh nào mà triển khai của bạn thể hiện mạnh mẽ nhất. Bạn có thể gợi ra hành vi ngược lại bằng cách thay đổi prompt không?
5. Đọc MultiAgentBench (arXiv:2503.01935) Phần 4 (thí nghiệm topology). Tái tạo kết quả "graph-thắng-nghiên cứu" trên một tác vụ từ bài báo bằng bộ đo lường của bạn.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|------------------------|
| Self-consistency | "Lấy mẫu N lần, bỏ phiếu" | Wang 2022. Một model, N mẫu temperature>0, bỏ phiếu đa số trên các chuỗi suy luận. |
| Heterogeneity | "Các model khác nhau" | Ensemble của các base model hoặc họ prompt khác nhau. Phá vỡ sự đơn điệu. |
| MAD | "Multi-agent debate" | Thuật ngữ chung cho các tác nhân trao đổi phản biện qua các vòng. Xem Du 2023. |
| A-HMAD | "Adversarial Heterogeneous MAD" | Biến thể MAD nhấn mạnh các model khác nhau + cấu trúc phản biện. |
| Topology | "Ai nói chuyện với ai" | Star, chain, tree, graph. Xác định luồng thông tin. |
| Coordination tax | "Lợi nhuận giảm dần" | Trên ~4 tác nhân trong graph, chi phí tăng nhanh hơn chất lượng. |
| Volunteer behavior | "Giúp đỡ không cần nhắc" | Mô hình nảy sinh của AgentVerse: tác nhân đề nghị thực hiện một bước. |
| Conformity behavior | "Đồng thuận dưới áp lực" | Mô hình nảy sinh của AgentVerse: tác nhân điều chỉnh theo người phản biện. |
| Jury | "Hội đồng chuyên biệt nhỏ" | Ensemble kiểu Sibyl với các vai trò (thẩm vấn, ngữ cảnh, chấm điểm). |

## Đọc thêm

- [Wang và cộng sự — Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/abs/2203.11171) — baseline single-model
- [Du và cộng sự — Improving Factuality and Reasoning via Multiagent Debate](https://arxiv.org/abs/2305.14325) — cả tác nhân VÀ vòng debate đều quan trọng độc lập
- [MultiAgentBench / MARBLE](https://arxiv.org/abs/2503.01935) — benchmark topology cho thấy graph tốt nhất cho nghiên cứu, chain cho pipeline
- [Should we be going MAD?](https://arxiv.org/abs/2311.17371) — khảo sát chiến lược MAD; thấy rằng MAD thường thua self-consistency ở cùng ngân sách
- [AgentVerse (ICLR 2024)](https://proceedings.iclr.cc/paper_files/paper/2024/file/578e65cdee35d00c708d4c64bce32971-Paper-Conference.pdf) — các mô hình hành vi nảy sinh volunteer và conformity
- [MARBLE repo](https://github.com/ulab-uiuc/MARBLE) — triển khai benchmark tham chiếu