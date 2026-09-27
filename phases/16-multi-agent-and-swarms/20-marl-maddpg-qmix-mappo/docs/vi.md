# MARL — MADDPG, QMIX, MAPPO

> Di sản học tăng cường (reinforcement learning) trong phối hợp đa tác nhân, vốn vẫn là nền tảng cho các hệ thống LLM-agent vào năm 2026. **MADDPG** (Lowe và cộng sự, NeurIPS 2017, arXiv:1706.02275) đã giới thiệu mô hình Huấn luyện tập trung, Thực thi phi tập trung (Centralized Training, Decentralized Execution - CTDE): mỗi critic quan sát trạng thái và hành động của tất cả các tác nhân trong quá trình huấn luyện; tại thời điểm kiểm thử, chỉ các actor cục bộ được vận hành. Phương pháp này hiệu quả cho các môi trường hợp tác, cạnh tranh và hỗn hợp. **QMIX** (Rashid và cộng sự, ICML 2018, arXiv:1803.11485) là phương pháp phân rã giá trị với mạng trộn đơn điệu (monotonic mixing network); các Q-value của từng tác nhân được kết hợp thành Q-value chung để `argmax` phân phối một cách sạch sẽ — chiếm ưu thế trong StarCraft Multi-Agent Challenge (SMAC). **MAPPO** (Yu và cộng sự, NeurIPS 2022, arXiv:2103.01955) là PPO với hàm giá trị tập trung; "hiệu quả đến bất ngờ" trên các môi trường particle-world, SMAC, Google Research Football, Hanabi với mức tinh chỉnh tối thiểu. Đây là những nền tảng để huấn luyện các chính sách cho các đội tác nhân cần hành động phi tập trung. MAPPO là **baseline cooperative-MARL mặc định năm 2026**. Bài học này xây dựng từng phương pháp từ một grid-world nhỏ và giúp bạn ghi nhớ ba ý tưởng này vào "bộ nhớ cơ bắp" trước khi chạm vào việc huấn luyện LLM-agent.

**Type:** Learn
**Languages:** Python (stdlib, các triển khai nhỏ không dùng NumPy)
**Prerequisites:** Phase 09 (Reinforcement Learning), Phase 16 · 09 (Parallel Swarm Networks)
**Time:** ~90 phút

## Vấn đề

Các hệ thống LLM-agent ngày càng huấn luyện nhiều chính sách cho việc phối hợp giữa các tác nhân: khi nào nên trì hoãn, khi nào nên hành động, nên gọi đồng nghiệp nào. Tài liệu hướng dẫn cách huấn luyện các chính sách như vậy chính là Multi-Agent Reinforcement Learning (MARL), vốn đã tồn tại trước làn sóng LLM và có một tập hợp nhỏ các thuật toán chiếm ưu thế.

Việc đọc các bài báo MARL mà không có vốn từ vựng về mô hình là một cực hình. Huấn luyện tập trung với thực thi phi tập trung (CTDE), phân rã giá trị và critic tập trung không phải là những từ ngữ thời thượng — chúng là câu trả lời cụ thể cho các vấn đề cụ thể:

- Independent RL (mỗi tác nhân tự học) là không dừng (non-stationary) từ góc nhìn của mỗi tác nhân. Điều này không tốt.
- Centralized RL (một tác nhân điều khiển tất cả) không thể mở rộng và vi phạm các ràng buộc thực thi.
- CTDE đạt được lợi ích của cả hai: huấn luyện với thông tin toàn cục, triển khai với các chính sách cục bộ.

## Khái niệm

### Ba môi trường mà các bài báo thường sử dụng

- **Particle World (multi-agent particle env).** Vật lý 2D đơn giản với các tác vụ hợp tác/cạnh tranh. Đây là môi trường thử nghiệm gốc của MADDPG.
- **StarCraft Multi-Agent Challenge (SMAC).** Quản lý vi mô hợp tác, quan sát một phần. Môi trường thử nghiệm của QMIX. Hành động rời rạc, trạng thái liên tục.
- **Google Research Football, Hanabi, MPE.** Các baseline của MAPPO.

Các môi trường khác nhau có các loại hành động/quan sát khác nhau. Các thuật toán sẽ được lựa chọn tương ứng.

### MADDPG (2017) — mô hình CTDE

Mỗi tác nhân `i` có một actor `mu_i(o_i)` ánh xạ quan sát của chính nó thành hành động. Mỗi tác nhân cũng có một critic `Q_i(x, a_1, ..., a_n)` quan sát tất cả các quan sát và tất cả các hành động trong quá trình huấn luyện. Actor được cập nhật bằng policy gradient dựa trên đánh giá của critic.

```
actor update:    grad_theta_i J = E[grad_theta mu_i(o_i) * grad_a_i Q_i(x, a_1..n) at a_i=mu_i(o_i)]
critic update:   TD on Q_i(x, a_1..n) given next-state joint estimate
```

Tại sao lại là CTDE: tại thời điểm huấn luyện, chúng ta biết hành động của mọi người; chúng ta sử dụng điều đó để giảm phương sai trong mỗi critic. Tại thời điểm triển khai, mỗi tác nhân chỉ nhìn thấy `o_i` và gọi `mu_i(o_i)`.

Chế độ thất bại: các critic phát triển theo số lượng N tác nhân (đầu vào bao gồm tất cả các hành động). Không thể mở rộng quá ~10 tác nhân nếu không có các phép xấp xỉ.

### QMIX (2018) — phân rã giá trị

Chỉ dành cho hợp tác. Phần thưởng toàn cục là tổng của một hàm đơn điệu của các Q-value của từng tác nhân:

```
Q_tot(tau, a) = f(Q_1(tau_1, a_1), ..., Q_n(tau_n, a_n)),   df/dQ_i >= 0
```

Tính đơn điệu đảm bảo `argmax_a Q_tot` có thể được tính toán bởi mỗi tác nhân khi chọn `argmax_{a_i} Q_i` một cách độc lập. Đó chính xác là **đặc tính thực thi phi tập trung** mà bạn cần. Tại thời điểm huấn luyện, một mạng trộn (mixing network) tạo ra `Q_tot` từ các Q-value của từng tác nhân.

Tại sao QMIX thắng trên SMAC: quản lý vi mô StarCraft hợp tác có các tác nhân đồng nhất, quan sát cục bộ, phần thưởng toàn cục — hoàn toàn phù hợp cho phân rã giá trị.

Chế độ thất bại: ràng buộc đơn điệu khá hạn chế; một số tác vụ có cấu trúc phần thưởng không thể phân rã đơn điệu (ví dụ: một tác nhân hy sinh vì cả đội). Các phần mở rộng (QTRAN, QPLEX) giúp nới lỏng điều này.

### MAPPO (2022) — mặc định bị bỏ quên

Multi-Agent PPO: PPO với hàm giá trị tập trung. Mỗi tác nhân có chính sách riêng; tất cả các tác nhân chia sẻ (hoặc có riêng) các hàm giá trị nhìn thấy trạng thái đầy đủ. Yu và cộng sự 2022 đã benchmark MAPPO so với MADDPG, QMIX và các phần mở rộng của chúng trên năm bộ benchmark và nhận thấy:

- MAPPO ngang bằng hoặc vượt qua các phương pháp MARL off-policy trên particle-world, SMAC, Google Research Football, Hanabi, MPE.
- Yêu cầu tinh chỉnh siêu tham số tối thiểu.
- Huấn luyện ổn định; có thể tái lập trên các seed khác nhau.

Cộng đồng đã đánh giá thấp on-policy MARL cho đến bài báo này. Vào năm 2026, MAPPO là baseline mặc định cho cooperative-MARL; bất kỳ phương pháp mới nào cũng phải vượt qua nó.

### Tại sao kỹ sư LLM-agent nên quan tâm

Ba ứng dụng trực tiếp:

1. **Huấn luyện bộ định tuyến (Router training).** Một meta-agent chọn sub-agent nào xử lý tác vụ. Đây là bài toán MARL với N sub-agent phi tập trung và một bộ định tuyến tập trung. MAPPO rất phù hợp.
2. **Sự xuất hiện vai trò (Role emergence).** Trong các mô phỏng generative-agent, việc huấn luyện các tác nhân đảm nhận các vai trò bổ sung theo thời gian là một bài toán MARL ẩn. Phân rã giá trị kiểu QMIX buộc sự bổ sung bằng cấu trúc.
3. **Sử dụng công cụ đa tác nhân.** Khi các tác nhân chia sẻ công cụ và cạnh tranh ngân sách, việc huấn luyện chúng thông qua CTDE tạo ra các chính sách cục bộ có thể triển khai, tôn trọng các ràng buộc tài nguyên.

Lưu ý thực tế: vào năm 2026, hầu hết các hệ thống LLM-agent sản xuất đều prompt các chính sách thay vì huấn luyện chúng. MARL xuất hiện khi bạn có (a) nhiều dữ liệu tương tác, (b) tín hiệu phần thưởng rõ ràng, và (c) sẵn sàng đầu tư vào cơ sở hạ tầng huấn luyện.

### CTDE như một mẫu thiết kế ngoài RL

Ngay cả khi không huấn luyện, CTDE là một mẫu kiến trúc hữu ích:

- Trong quá trình *thiết kế*, giả định sự hiển thị đầy đủ của cả đội.
- Tại *thời điểm chạy*, thực thi thực thi phi tập trung: mỗi tác nhân chỉ nhìn thấy `o_i`.

Mẫu này buộc bạn phải giữ trạng thái của từng tác nhân một cách rõ ràng và suy nghĩ về khả năng quan sát một phần ngay từ đầu. Nhiều hệ thống đa tác nhân sản xuất giả định ngầm về trạng thái chia sẻ ở khắp mọi nơi — kỷ luật CTDE ngăn chặn điều đó.

### Vấn đề không dừng (non-stationarity)

Khi nhiều tác nhân học cùng lúc, môi trường của mỗi tác nhân (bao gồm cả chính sách của những người khác) là không dừng. Các bằng chứng RL đơn tác nhân cổ điển bị phá vỡ. Tất cả các thuật toán MARL trong bài học này đều giải quyết vấn đề này:

- MADDPG: critic toàn cục nhìn thấy tất cả các hành động, vì vậy ước tính giá trị của nó là dừng.
- QMIX: phân rã giá trị chuyển việc học sang không gian Q chung nơi tính tối ưu được xác định rõ ràng.
- MAPPO: hàm giá trị tập trung làm giảm phương sai từ những thay đổi chính sách của người khác.

Trong các hệ thống LLM-agent, tính không dừng biểu hiện như "tác nhân của tôi hoạt động tháng trước, bây giờ tác nhân khác ở thượng nguồn thay đổi, tác nhân của tôi lại hoạt động sai". Huấn luyện MARL với CTDE là cách sửa chữa có nguyên tắc; các bản sửa lỗi ở cấp độ prompt nhanh hơn nhưng kém bền vững hơn.

### Những gì bài học này KHÔNG bao gồm

Huấn luyện các mạng thực tế là chủ đề của Phase 09. Bài học này xây dựng các phiên bản chính sách kịch bản (scripted-policy) minh họa các mẫu CTDE, phân rã giá trị và giá trị tập trung mà không cần cập nhật gradient. Mục tiêu là nội hóa các mẫu trước khi bạn chọn một thư viện MARL đầy đủ (PyMARL, MARLlib, RLlib multi-agent).

```figure
sw-ctde
```

## Xây dựng

`code/main.py` triển khai ba minh họa mẫu, tất cả trên một grid-world hợp tác 2 tác nhân nhỏ:

- Môi trường: 2 tác nhân trên lưới 4x4, một viên phần thưởng. Phần thưởng = 1 nếu bất kỳ tác nhân nào chạm vào viên phần thưởng; tác vụ kết thúc.
- `IndependentAgents` — mỗi tác nhân coi những người khác là môi trường. Baseline.
- `MADDPGStyle` — critic tập trung tính toán giá trị chung; các chính sách actor cập nhật từ đó. Cải thiện chính sách kịch bản.
- `QMIXStyle` — phân rã giá trị với bộ trộn đơn điệu.
- `MAPPOStyle` — hàm giá trị tập trung; các chính sách cập nhật dựa trên baseline chia sẻ.

Cả bốn đều chạy cùng các tập (episodes) và báo cáo số bước trung bình để đạt mục tiêu. Các biến thể CTDE hội tụ về các đường đi ngắn hơn so với baseline độc lập.

Chạy:

```
python3 code/main.py
```

Kết quả mong đợi: các tác nhân độc lập mất trung bình ~6 bước; các biến thể CTDE hội tụ về ~3.5 bước (tối ưu cho lưới 4x4 là 3). Sự khác biệt về mẫu xuất hiện ngay cả với các chính sách kịch bản.

## Sử dụng

`outputs/skill-marl-picker.md` là một kỹ năng chọn thuật toán MARL cho một tác vụ đa tác nhân nhất định: hợp tác vs cạnh tranh, đồng nhất vs không đồng nhất, loại không gian hành động, quy mô, tín hiệu phần thưởng.

## Triển khai

MARL trong sản xuất rất hiếm. Khi bạn sử dụng nó:

- **Bắt đầu với MAPPO.** Bài báo năm 2022 đã thiết lập đây là baseline; việc tái lập nó trước tiên giúp tiết kiệm hàng tuần chạy theo các phương pháp hào nhoáng hơn.
- **Ghi lại luồng quan sát và hành động của mọi tác nhân.** Gỡ lỗi MARL mà không có dấu vết của từng tác nhân là vô vọng.
- **Tách biệt mã huấn luyện khỏi mã thực thi.** CTDE là một kỷ luật; hãy để đường dẫn thực thi chỉ thực sự nhìn thấy `o_i`.
- **Cảnh báo về định hình phần thưởng (reward shaping).** MARL cực kỳ nhạy cảm với thiết kế phần thưởng. Một lỗi phối hợp trong việc định hình và các tác nhân sẽ học cách khai thác nó. Chạy các bài kiểm tra đối nghịch.
- **Đối với LLM-agent**, hãy cân nhắc các chính sách cấp prompt trước. Chỉ đầu tư vào huấn luyện MARL khi dữ liệu tương tác + tín hiệu phần thưởng + cơ sở hạ tầng đều có sẵn.

## Bài tập

1. Chạy `code/main.py`. Đo khoảng cách số bước đến mục tiêu giữa các tác nhân độc lập và tác nhân kiểu MAPPO. Khoảng cách đó tăng hay giảm trên lưới 6x6?
2. Triển khai một biến thể cạnh tranh: hai tác nhân, một viên phần thưởng, chỉ người đầu tiên chạm vào mới nhận được phần thưởng. Mẫu nào xử lý cạnh tranh một cách sạch sẽ? MADDPG về mặt lịch sử.
3. Đọc MADDPG (arXiv:1706.02275) Mục 3. Triển khai quy tắc cập nhật critic chính xác bằng mã giả theo cách của riêng bạn.
4. Đọc MAPPO (arXiv:2103.01955). Tại sao các tác giả lập luận rằng giá trị tập trung + PPO đánh bại MARL off-policy trên các benchmark của họ? Liệt kê ba tuyên bố mạnh mẽ nhất.
5. Áp dụng CTDE như một mẫu thiết kế cho một hệ thống LLM-agent giả định (ví dụ: tác nhân nghiên cứu + tác nhân tóm tắt + tác nhân lập trình). Thông tin chung nào có sẵn tại thời điểm thiết kế mà không có sẵn tại thời điểm chạy?

## Thuật ngữ chính

| Thuật ngữ | Mọi người nói | Ý nghĩa thực sự |
|------|----------------|------------------------|
| MARL | "Multi-Agent RL" | Học tăng cường cho các hệ thống đa tác nhân. |
| CTDE | "Centralized Training, Decentralized Execution" | Huấn luyện với thông tin toàn cục; triển khai với chính sách cục bộ. |
| MADDPG | "Multi-Agent DDPG" | CTDE với critic từng tác nhân nhìn thấy tất cả quan sát + hành động. |
| QMIX | "Value decomposition" | Trộn đơn điệu các Q-value của từng tác nhân. Hợp tác. |
| MAPPO | "Multi-Agent PPO" | PPO với hàm giá trị tập trung. Baseline mặc định năm 2026. |
| Value decomposition | "Sum of individual Qs" | Q chung được biểu diễn dưới dạng hàm đơn điệu của Q từng tác nhân. |
| Non-stationarity | "Moving targets" | Môi trường của mỗi tác nhân thay đổi khi những người khác học. Vấn đề cốt lõi của MARL. |
| On-policy / off-policy | "Learn from current / replay" | PPO là on-policy (MAPPO); DDPG và Q-learning là off-policy. |
| SMAC | "StarCraft Multi-Agent Challenge" | Benchmark quản lý vi mô hợp tác; sân nhà của QMIX. |

## Đọc thêm

- [Lowe và cộng sự — Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275) — MADDPG; NeurIPS 2017
- [Rashid và cộng sự — QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485) — QMIX; ICML 2018
- [Yu và cộng sự — The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://arxiv.org/abs/2103.01955) — MAPPO; NeurIPS 2022
- [Bài đăng blog BAIR về MAPPO](https://bair.berkeley.edu/blog/2021/07/14/mappo/) — cách diễn đạt dễ hiểu về kết quả của MAPPO
- [Kho lưu trữ SMAC](https://github.com/oxwhirl/smac) — StarCraft Multi-Agent Challenge