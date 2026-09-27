# Sim-to-Real Transfer

> Một chính sách (policy) được huấn luyện trong môi trường mô phỏng nhưng thất bại trên phần cứng thực tế là một chính sách đã "học vẹt" môi trường mô phỏng đó. Domain randomization, domain adaptation và system identification là ba công cụ giúp các bộ điều khiển đã học vượt qua khoảng cách thực tế (reality gap).

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 9 · 08 (PPO), Phase 2 · 10 (Bias/Variance)
**Time:** ~45 phút

## Vấn đề

Huấn luyện một robot thực tế rất chậm, nguy hiểm và tốn kém. Một robot hai chân cần hàng triệu tập huấn luyện để học cách đi; một robot thực tế chỉ cần ngã một lần là có thể hỏng phần cứng. Mô phỏng (simulation) cung cấp cho bạn khả năng reset không giới hạn, tính tái lập xác định, môi trường song song và không gây hư hại vật lý.

Nhưng các trình mô phỏng thường không chính xác. Các ổ bi có ma sát lớn hơn so với các mô hình MuJoCo. Camera có độ méo ống kính mà trình mô phỏng không bao gồm. Động cơ có độ trễ, độ rơ và sự bão hòa mà 99% các mô hình mô phỏng bỏ qua. Gió, bụi và ánh sáng thay đổi phá hoại một chính sách được huấn luyện trong môi trường kết xuất (rendering) vô trùng. **Reality gap** — sự khác biệt hệ thống giữa phân phối trong mô phỏng và phân phối thực tế — là vấn đề trung tâm của RL được triển khai cho robot.

Bạn cần một chính sách *có khả năng chống chịu với sự dịch chuyển phân phối từ sim sang real*. Ba phương pháp lịch sử: ngẫu nhiên hóa trình mô phỏng (domain randomization), thích nghi chính sách với một ít dữ liệu thực tế (domain adaptation / fine-tuning), hoặc xác định các tham số của hệ thống thực và khớp chúng (system identification). Vào năm 2026, công thức chủ đạo kết hợp cả ba với mô phỏng song song quy mô lớn (Isaac Sim, Isaac Lab, Mujoco MJX trên GPU).

## Khái niệm

![Three sim-to-real regimes: domain randomization, adaptation, system identification](../assets/sim-to-real.svg)

**Domain Randomization (DR).** Tobin và cộng sự 2017, Peng và cộng sự 2018. Trong quá trình huấn luyện, hãy ngẫu nhiên hóa mọi tham số mô phỏng có thể khác biệt trên robot thực: khối lượng, hệ số ma sát, độ lợi PD của động cơ, nhiễu cảm biến, vị trí camera, ánh sáng, kết cấu, mô hình tiếp xúc. Chính sách học một phân phối có điều kiện về "môi trường mô phỏng nào đang diễn ra hôm nay" và tổng quát hóa trên toàn bộ phạm vi đó. Nếu robot thực nằm trong phạm vi huấn luyện, chính sách sẽ hoạt động.

- **Ưu điểm:** không cần dữ liệu thực. Một công thức, nhiều robot.
- **Nhược điểm:** huấn luyện quá mức ngẫu nhiên tạo ra một chính sách "vạn năng" nhưng quá thận trọng. Quá nhiều nhiễu ≈ quá nhiều regularization.

**System Identification (SI).** Khớp các tham số của trình mô phỏng với dữ liệu thế giới thực trước khi huấn luyện. Nếu bạn có thể đo ma sát khớp cánh tay trên robot thực, hãy đưa nó vào mô phỏng. Sau đó huấn luyện một chính sách kỳ vọng các giá trị đó. Cần quyền truy cập vào hệ thống thực nhưng giảm trực tiếp khoảng cách thực tế.

- **Ưu điểm:** mục tiêu huấn luyện chính xác, ít nhiễu.
- **Nhược điểm:** sai số mô hình còn lại không hiển thị với chính sách; các hiệu ứng nhỏ chưa được xác định (ví dụ: vùng chết của động cơ) vẫn làm hỏng việc triển khai.

**Domain Adaptation.** Huấn luyện trong mô phỏng, tinh chỉnh (fine-tune) với một lượng nhỏ dữ liệu thực. Hai biến thể:

- **Real2Sim2Real:** học một trình mô phỏng phần dư (residual simulator) `f(s, a, z) - f_sim(s, a)` sử dụng các lượt chạy thực tế, huấn luyện trong trình mô phỏng đã được hiệu chỉnh. Thu hẹp khoảng cách mà không cần nhiều dữ liệu thực.
- **Observation adaptation:** huấn luyện một chính sách ánh xạ quan sát thực → quan sát giống mô phỏng thông qua một bộ trích xuất đặc trưng đã học (ví dụ: GAN pixel-to-pixel). Bộ điều khiển vẫn nằm trong mô phỏng.

**Privileged learning / teacher-student.** Miki và cộng sự 2022 (robot bốn chân ANYmal). Huấn luyện một *giáo viên* (teacher) trong mô phỏng có quyền truy cập vào thông tin đặc quyền (ma sát thực tế, độ cao địa hình, độ trôi IMU). Chưng cất (distill) một *học viên* (student) chỉ nhìn thấy các quan sát từ cảm biến thực. Học viên học cách suy luận các đặc trưng đặc quyền từ lịch sử, có khả năng chống chịu tốt với các tham số vật lý.

**Massively parallel simulation.** 2024–2026. Isaac Lab, Mujoco MJX, Brax đều chạy hàng ngàn robot song song trên một GPU duy nhất. PPO với 4.096 robot hình người song song thu thập hàng năm kinh nghiệm trong vài giờ. "Reality gap" thu hẹp khi phân phối huấn luyện mở rộng; DR trở nên gần như miễn phí khi mỗi môi trường trong số 4.096 môi trường đó có các tham số ngẫu nhiên khác nhau.

**Công thức thế giới thực năm 2026 (ví dụ đi bộ của robot bốn chân):**

1. Mô phỏng song song quy mô lớn với trọng lực, ma sát, độ lợi động cơ, tải trọng được ngẫu nhiên hóa theo miền (domain-randomized).
2. Chính sách giáo viên được huấn luyện với thông tin đặc quyền (bản đồ địa hình, vận tốc cơ thể thực tế).
3. Chính sách học viên được chưng cất từ giáo viên chỉ sử dụng cảm biến nội tại (bộ mã hóa khớp chân).
4. Tùy chọn thích nghi quan sát thông qua autoencoder trên IMU thực.
5. Triển khai. Zero-shot trên 10+ môi trường. Nếu thất bại, thực hiện tinh chỉnh thế giới thực trong vài phút với PPO có ràng buộc an toàn.

```figure
f3-reality-gap
```

## Xây dựng

Mã nguồn của bài học này là một minh chứng nhỏ về domain randomization trên GridWorld với các chuyển đổi *nhiễu*. Chúng ta huấn luyện một chính sách trải nghiệm các xác suất trượt ngẫu nhiên trong "mô phỏng" và đánh giá trên "thực tế" với mức độ trượt mà nó chưa từng thấy trong quá trình huấn luyện. Hình dạng này ánh xạ trực tiếp đến việc chuyển đổi từ MuJoCo sang phần cứng.

### Bước 1: mô phỏng có tham số

```python
def step(state, action, slip):
    if rng.random() < slip:
        action = random_perpendicular(action)
    ...
```

`slip` là một tham số mà trình mô phỏng phơi bày. Trong robot thực tế, nó có thể là ma sát, khối lượng, độ lợi động cơ — bất cứ thứ gì thay đổi giữa mô phỏng và thực tế.

### Bước 2: huấn luyện với DR

Khi bắt đầu mỗi tập, lấy mẫu `slip ~ Uniform[0.0, 0.4]`. Huấn luyện PPO / Q-learning / bất cứ thứ gì. Thực hiện điều này cho nhiều tập.

### Bước 3: đánh giá zero-shot trên các mức trượt "thực"

Đánh giá trên `slip ∈ {0.0, 0.1, 0.2, 0.3, 0.5, 0.7}`. Bốn mức đầu tiên nằm trong phạm vi hỗ trợ huấn luyện; `0.5` và `0.7` nằm ngoài. Một chính sách được huấn luyện bằng DR sẽ duy trì hiệu suất gần tối ưu bên trong phạm vi hỗ trợ và suy giảm một cách ổn định bên ngoài. Một chính sách được huấn luyện với độ trượt cố định sẽ rất giòn (brittle) bên ngoài phạm vi trượt huấn luyện của nó.

### Bước 4: so sánh với huấn luyện hẹp

Huấn luyện chính sách thứ hai chỉ với `slip = 0.0`. Đánh giá trên cùng một dải `slip`. Bạn sẽ thấy sự sụt giảm thảm hại ngay khi độ trượt thực tế > 0.

## Các cạm bẫy

- **Quá nhiều ngẫu nhiên hóa.** Huấn luyện trên `slip ∈ [0, 0.9]` và chính sách của bạn trở nên quá sợ rủi ro đến mức không bao giờ thử con đường tối ưu. Hãy khớp với phân phối *kỳ vọng* của thế giới thực, không phải "bất cứ điều gì cũng có thể xảy ra".
- **Quá ít ngẫu nhiên hóa.** Huấn luyện trên một lát cắt mỏng và chính sách không thể tổng quát hóa chút nào. Sử dụng chương trình giảng dạy thích nghi (Automatic Domain Randomization) để mở rộng phân phối khi chính sách cải thiện.
- **Không gian tham số bị xác định sai.** Ngẫu nhiên hóa sai thứ (ví dụ: màu sắc camera khi khoảng cách thực tế là độ trễ động cơ) và DR không giúp ích gì. Hãy lập hồ sơ robot thực trước.
- **Rò rỉ thông tin đặc quyền.** Một giáo viên sử dụng trạng thái toàn cục cho các hành động, thay vì chỉ quan sát, có thể tạo ra một học viên không thể theo kịp. Đảm bảo chính sách của giáo viên có thể thực hiện được bởi học viên dựa trên lịch sử quan sát.
- **Thất bại khi chuyển đổi sim-to-sim.** Nếu chính sách của bạn không đủ mạnh với một biến thể mô phỏng khó hơn, nó cũng sẽ không đủ mạnh với thế giới thực. Luôn kiểm tra trên một biến thể mô phỏng dự phòng trước khi triển khai.
- **Không có lớp bảo vệ thế giới thực.** Một chính sách hoạt động trong mô phỏng và "hoạt động trong thực tế" mà không có lớp bảo vệ cấp thấp vẫn có thể làm hỏng phần cứng. Hãy thêm giới hạn tốc độ, giới hạn mô-men xoắn, giới hạn khớp trong một bộ điều khiển không học (non-learned controller).

## Sử dụng

Stack sim-to-real năm 2026:

| Lĩnh vực | Stack |
|--------|-------|
| Di chuyển bằng chân (ANYmal, Spot, humanoid) | Isaac Lab + DR + giáo viên/học viên đặc quyền |
| Thao tác (bàn tay khéo léo, gắp và đặt) | Isaac Lab + DR + DR-GAN cho thị giác |
| Xe tự lái | CARLA / NVIDIA DRIVE Sim + DR + tinh chỉnh thực tế |
| Đua drone | RotorS / Flightmare + DR + thích nghi trực tuyến |
| Thao tác bằng ngón tay/trong lòng bàn tay | OpenAI Dactyl (DR ở quy mô chưa từng có) |
| Cánh tay công nghiệp | MuJoCo-Warp + SI + tinh chỉnh thực tế nhỏ |

Đối với điều khiển ở mọi quy mô, quy trình làm việc là nhất quán: khớp mô phỏng tốt nhất có thể, ngẫu nhiên hóa những gì bạn không thể khớp, huấn luyện các chính sách khổng lồ, chưng cất, triển khai với lớp bảo vệ an toàn.

## Ship It

Lưu dưới dạng `outputs/skill-sim2real-planner.md`:

```markdown
---
name: sim2real-planner
description: Plan a sim-to-real transfer pipeline for a given robot + task, covering DR, SI, and safety.
version: 1.0.0
phase: 9
lesson: 11
tags: [rl, sim2real, robotics, domain-randomization]
---

Given a robot platform, a task, and access to real hardware time, output:

1. Reality gap inventory. Suspected sources ranked by expected impact (contact, sensing, actuation delay, vision).
2. DR parameters. Exact list, ranges, distribution. Justify each range against real measurements.
3. SI steps. Which parameters to measure; measurement method.
4. Teacher/student split. What privileged info the teacher uses; what obs the student uses.
5. Safety envelope. Low-level limits, emergency stops, backup controller.

Refuse to deploy without (a) a zero-shot sim-variant test, (b) a safety shield, (c) a rollback plan. Flag any DR range wider than 3× measured real variability as likely over-randomized.
```

## Bài tập

1. **Dễ.** Huấn luyện một tác nhân Q-learning trên GridWorld trượt cố định (slip=0.0). Đánh giá trên slip ∈ {0.0, 0.1, 0.3, 0.5}. Vẽ biểu đồ lợi nhuận theo độ trượt.
2. **Trung bình.** Huấn luyện một tác nhân Q-learning DR lấy mẫu `slip ~ Uniform[0, 0.3]`. Đánh giá trên cùng dải đó. DR mang lại bao nhiêu lợi ích tại slip=0.5 (ngoài phân phối)?
3. **Khó.** Triển khai một chương trình giảng dạy: bắt đầu với slip=0.0, mở rộng phạm vi DR mỗi khi chính sách đạt 90% tối ưu. Đo tổng số bước môi trường để đạt slip=0.3 zero-shot so với baseline DR cố định.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| Reality gap | "Sự khác biệt sim-to-real" | Dịch chuyển phân phối giữa vật lý/cảm biến huấn luyện và triển khai. |
| Domain randomization (DR) | "Huấn luyện qua các sim ngẫu nhiên" | Ngẫu nhiên hóa các tham số mô phỏng trong quá trình huấn luyện để chính sách tổng quát hóa. |
| System identification (SI) | "Đo thực tế và khớp mô phỏng" | Ước tính các tham số vật lý thực; thiết lập mô phỏng để khớp. |
| Domain adaptation | "Tinh chỉnh trên dữ liệu thực" | Tinh chỉnh thế giới thực nhỏ sau khi huấn luyện mô phỏng; có thể thích nghi quan sát hoặc động lực học. |
| Privileged info | "Sự thật cơ bản cho giáo viên" | Thông tin chỉ mô phỏng mới có; học viên phải suy luận từ lịch sử quan sát. |
| Teacher/student | "Chưng cất đặc quyền -> quan sát được" | Giáo viên được huấn luyện với các lối tắt; học viên học cách bắt chước mà không cần chúng. |
| ADR | "Automatic Domain Randomization" | Chương trình giảng dạy mở rộng phạm vi DR khi chính sách cải thiện. |
| Real2Sim | "Thu hẹp khoảng cách bằng dữ liệu thực" | Học một phần dư để làm cho mô phỏng bắt chước các lượt chạy thực tế. |

## Đọc thêm

- [Tobin và cộng sự (2017). Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907) — bài báo DR gốc (thị giác cho robot).
- [Peng và cộng sự (2018). Sim-to-Real Transfer of Robotic Control with Dynamics Randomization](https://arxiv.org/abs/1710.06537) — DR cho động lực học, di chuyển robot bốn chân.
- [OpenAI và cộng sự (2019). Solving Rubik's Cube with a Robot Hand](https://arxiv.org/abs/1910.07113) — Dactyl, ADR ở quy mô lớn.
- [Miki và cộng sự (2022). Learning robust perceptive locomotion for quadrupedal robots in the wild](https://www.science.org/doi/10.1126/scirobotics.abk2822) — giáo viên-học viên cho ANYmal.
- [Makoviychuk và cộng sự (2021). Isaac Gym: High Performance GPU Based Physics Simulation for Robot Learning](https://arxiv.org/abs/2108.10470) — mô phỏng song song quy mô lớn thúc đẩy triển khai 2025–2026.
- [Akkaya và cộng sự (2019). Automatic Domain Randomization](https://arxiv.org/abs/1910.07113) — phương pháp chương trình giảng dạy ADR.
- [Sutton & Barto (2018). Ch. 8 — Planning and Learning with Tabular Methods](http://incompleteideas.net/book/RLbook2020.pdf) — khung Dyna (sử dụng mô hình để lập kế hoạch + lượt chạy) làm nền tảng cho các pipeline sim-to-real hiện đại.
- [Zhao, Queralta & Westerlund (2020). Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics: a Survey](https://arxiv.org/abs/2009.13303) — phân loại các phương pháp sim-to-real với kết quả benchmark.