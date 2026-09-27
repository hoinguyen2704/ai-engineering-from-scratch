# Tối ưu hóa bầy đàn cho LLM (PSO, ACO)

> Tối ưu hóa lấy cảm hứng từ sinh học đang quay trở lại với LLM. **LMPSO** (arXiv:2504.09247) sử dụng PSO, trong đó vận tốc của mỗi hạt là một prompt và LLM tạo ra ứng viên tiếp theo; phương pháp này hoạt động tốt trên các đầu ra có cấu trúc (biểu thức toán học, chương trình). **Model Swarms** (arXiv:2410.11163) coi mỗi chuyên gia LLM là một hạt PSO trên đa tạp trọng số mô hình và báo cáo **mức tăng trung bình 13,3%** so với 12 mô hình cơ sở trên 9 tập dữ liệu chỉ với 200 thực thể. **SwarmPrompt** (ICAART 2025) kết hợp PSO + Grey Wolf để tối ưu hóa prompt. **AMRO-S** (arXiv:2603.12933) lấy cảm hứng từ ACO với các chuyên gia pheromone cho việc định tuyến LLM đa tác nhân — **tăng tốc 4,7 lần**, bằng chứng định tuyến có thể giải thích được, cập nhật bất đồng bộ có kiểm soát chất lượng giúp tách biệt suy luận khỏi học tập. Bài học này triển khai PSO trên không gian tham số prompt và ACO trên định tuyến tác nhân, đo lường lý do tại sao các thuật toán cổ điển này phù hợp với kỷ nguyên LLM và khi nào thì không.

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 09 (Parallel Swarm Networks), Phase 16 · 14 (Consensus and BFT)
**Time:** ~75 phút

## Vấn đề

Bạn có một prompt đạt điểm 62% trong bài đánh giá tác vụ của mình. Bạn muốn cải thiện nó. Cách tiếp cận ngây thơ là tinh chỉnh thủ công không dựa trên gradient, điều này rất khó mở rộng. Học tăng cường (Reinforcement Learning) cần tín hiệu phần thưởng và đủ số lần thử nghiệm để huấn luyện. Backprop qua các prompt thực sự không khả thi — prompt là một chuỗi rời rạc, không phải là tham số có thể vi phân.

Tối ưu hóa lấy cảm hứng từ sinh học cổ điển — PSO cho không gian tìm kiếm liên tục, ACO cho lựa chọn đường đi — được thiết kế chính xác cho chế độ này: không cần gradient, dựa trên quần thể, chi phí thấp cho mỗi lần đánh giá. Kết hợp chúng với LLM cho bước tìm kiếm không cần gradient, bạn sẽ có một bộ tối ưu hóa thực tế đến bất ngờ.

Các mô hình tương tự áp dụng cho việc *định tuyến* tác nhân trong các hệ thống đa tác nhân. Một vệt pheromone kiểu ACO ghi lại tác nhân nào hoạt động tốt nhất trên loại tác vụ nào, cho phép bộ định tuyến khai thác vệt đó và làm suy giảm pheromone để các tuyến đường có thể được khám phá lại.

## Khái niệm

### Ôn tập về PSO (Kennedy & Eberhart 1995)

Particle Swarm Optimization (Tối ưu hóa bầy đàn hạt): quần thể các hạt trong không gian tìm kiếm liên tục. Mỗi hạt có vị trí `x_i` và vận tốc `v_i`. Mỗi vòng lặp:

```
v_i <- w * v_i + c1 * r1 * (p_best_i - x_i) + c2 * r2 * (g_best - x_i)
x_i <- x_i + v_i
evaluate fitness(x_i)
update p_best_i if improved
update g_best if global best
```

Trong đó `p_best` là vị trí tốt nhất của chính hạt đó, `g_best` là vị trí tốt nhất của cả bầy, `w, c1, c2` là các trọng số quán tính + nhận thức + xã hội, `r1, r2` là các yếu tố ngẫu nhiên.

### PSO trên đầu ra LLM — LMPSO

arXiv:2504.09247 điều chỉnh PSO cho các đầu ra có cấu trúc do LLM tạo ra (biểu thức toán học, chương trình). Mỗi hạt là một đầu ra ứng viên. Vận tốc là một *prompt* mô tả cách sửa đổi đầu ra hiện tại hướng tới vị trí tốt nhất của cá nhân/toàn cục. LLM tạo ra đầu ra mới từ prompt vận tốc. "Quán tính" của vận tốc là một prompt như "thực hiện các thay đổi nhỏ dần".

Cách này hoạt động tốt khi:
- Đầu ra có cấu trúc (có thể phân tích cú pháp, có thể đánh giá).
- Độ thích nghi (fitness) là tự động (chạy thử nghiệm, đánh giá số học).
- Quần thể nhỏ (~10-30 hạt) để tổng số lần gọi LLM vẫn nằm trong tầm kiểm soát.

Nó không hoạt động tốt khi độ thích nghi cần sự đánh giá của con người — chi phí mỗi vòng lặp trở nên quá đắt đỏ.

### Model Swarms

arXiv:2410.11163 đưa PSO ra khỏi lớp đầu ra và đi vào lớp *mô hình*. Mỗi "hạt" là một LLM chuyên gia (các tham số). Bầy đàn di chuyển các tham số về phía tốt nhất tập thể thông qua cập nhật không cần gradient. Báo cáo: mức tăng trung bình 13,3% so với 12 mô hình cơ sở trên 9 tập dữ liệu, với chỉ 200 thực thể mỗi vòng lặp.

Điểm mấu chốt là các mô hình chuyên gia LLM đã ở gần nhau trong một đa tạp tham số chia sẻ (trọng số adapter, LoRA deltas). PSO trên không gian con chiều thấp này rẻ và hiệu quả.

### Ôn tập về ACO (Dorigo 1992)

Ant Colony Optimization (Tối ưu hóa đàn kiến): kiến đi qua một đồ thị; mỗi đường đi có một vệt pheromone. Xác suất di chuyển của kiến được trọng số bởi cường độ pheromone. Những con kiến hoàn thành tác vụ sẽ gửi pheromone tỷ lệ thuận với chất lượng giải pháp. Pheromone suy giảm theo thời gian.

### AMRO-S — ACO cho định tuyến tác nhân

arXiv:2603.12933 sử dụng ACO cho định tuyến đa tác nhân. Mỗi loại tác vụ là một "đích đến"; mỗi tác nhân là một tuyến đường khả thi. Pheromone củng cố các tuyến đường tạo ra đầu ra tốt. Các đóng góp chính:

- **Bằng chứng định tuyến có thể giải thích được.** Cường độ pheromone là một tín hiệu mà con người có thể đọc được.
- **Cập nhật bất đồng bộ có kiểm soát chất lượng.** Pheromone chỉ cập nhật sau khi vượt qua các kiểm tra chất lượng, tách biệt suy luận khỏi học tập.
- **Tăng tốc 4,7 lần** trên benchmark định tuyến đa tác nhân.

Cổng kiểm soát chất lượng rất quan trọng: nếu không có nó, các tác nhân nhanh-nhưng-sai sẽ tích lũy pheromone và hệ thống sẽ bị khóa vào các tuyến đường xấu.

### Khi nào nên sử dụng PSO / ACO cho LLM

**Sử dụng PSO khi:**
- Không gian tìm kiếm liên tục hoặc ánh xạ tới các tham số liên tục (nhúng prompt, trọng số LoRA, tham số tạo số).
- Độ thích nghi rẻ và tự động.
- Quần thể có thể nhỏ (10-30).

**Sử dụng ACO khi:**
- Bạn có vấn đề về định tuyến hoặc lựa chọn đường đi.
- Các quyết định củng cố theo thời gian (các loại tác vụ tương tự quay trở lại).
- Bạn cần bằng chứng có thể giải thích được cho các quyết định định tuyến.

**Không sử dụng khi:**
- Độ thích nghi cần sự đánh giá của con người (quá đắt đỏ mỗi vòng lặp).
- Không gian tìm kiếm rời rạc và tổ hợp theo cách mà PSO không bao phủ được (hãy sử dụng thuật toán di truyền thay thế).
- Các quyết định thời gian thực cần độ trễ nghiêm ngặt (PSO/ACO hội tụ chậm so với các heuristic một lần).

### Tại sao lấy cảm hứng từ sinh học vẫn thắng thế

Các phương pháp dựa trên gradient cần các tín hiệu có thể vi phân. Đầu ra của LLM và các quyết định định tuyến không thể vi phân một cách đơn giản. Các phương pháp giả gradient (bộ định tuyến học tăng cường, bộ tinh chỉnh prompt kiểu DPO) hoạt động nhưng cần huấn luyện đắt đỏ.

PSO và ACO chỉ cần một hàm *đánh giá*. Nếu bạn có thể chấm điểm một đầu ra ứng viên hoặc một quyết định định tuyến, bạn có thể tối ưu hóa trên không gian đó. Điều đó làm cho rào cản áp dụng thấp hơn nhiều.

### Giới hạn thực tế

- **Ngân sách quần thể.** N hạt × T vòng lặp × chi phí mỗi lần đánh giá. Đối với đánh giá LLM ở mức ~$0.02 / call, a 20-particle PSO running 50 iterations costs ~$20. Hãy lập kế hoạch phù hợp.
- **Khám phá vs Khai thác.** Tỷ lệ suy giảm pheromone và quán tính PSO đánh đổi lẫn nhau; suy giảm quá nhanh → quên giải pháp; quá chậm → mắc kẹt ở các tối ưu cục bộ sớm.
- **Trôi dạt thảm khốc (Catastrophic drift).** Cả hai thuật toán đều có thể hội tụ và sau đó phân kỳ nếu cảnh quan độ thích nghi thay đổi (phân phối dữ liệu mới). Theo dõi sự ổn định của độ thích nghi tốt nhất.

```figure
swarm-stigmergy
```

## Xây dựng

`code/main.py` triển khai:

- `LMPSO` — PSO trên các tham số prompt số (nhiệt độ, trọng số top_k). "Việc tạo LLM" của mỗi hạt được mô phỏng như một hàm độ thích nghi dạng kịch bản. Chạy thuật toán trong 30 vòng lặp và hiển thị sự hội tụ g_best.
- `AMRO_S` — Định tuyến kiểu ACO. 3 tác nhân, 4 loại tác vụ, ma trận pheromone, 100 tác vụ được định tuyến. In ra phân phối (loại tác vụ → lựa chọn tác nhân) theo thời gian để hiển thị sự hình thành vệt.
- So sánh: định tuyến ngẫu nhiên vs định tuyến ACO trên cùng một luồng tác vụ. Đo lường chất lượng và độ trễ.

Chạy:

```
python3 code/main.py
```

Đầu ra mong đợi:
- LMPSO: Độ thích nghi g_best cải thiện từ ngẫu nhiên đến gần tối ưu sau 30 vòng lặp.
- AMRO-S: Bảng pheromone ổn định trên tác nhân đúng cho mỗi loại tác vụ; định tuyến ACO đánh bại ngẫu nhiên khoảng 30-40% về chất lượng và cũng giảm độ trễ (ít lần thử lại hơn).

## Sử dụng

`outputs/skill-swarm-optimizer.md` giúp lựa chọn giữa PSO, ACO, thuật toán di truyền và các bộ tối ưu hóa dựa trên gradient cho các vấn đề tối ưu hóa LLM / tác nhân.

## Triển khai

- **Bắt đầu nhỏ.** 10-20 hạt, 20-50 vòng lặp. Chỉ mở rộng quy mô nếu đường cong hội tụ cho thấy mức tăng rõ ràng.
- **Ghi nhật ký pheromone hoặc g_best mỗi vòng lặp.** Gỡ lỗi các bộ tối ưu hóa bầy đàn mà không có vệt theo dõi là một cực hình.
- **Cập nhật có kiểm soát chất lượng.** Đặc biệt đối với định tuyến ACO: các tác nhân nhanh-và-sai không được phép tích lũy pheromone.
- **Đặt lại sự suy giảm khi có sự thay đổi phân phối.** Khi phân phối đánh giá của bạn thay đổi, các pheromone cũ đã lỗi thời; hãy đặt lại hoặc tạm thời tăng gấp đôi tỷ lệ suy giảm.
- **Giới hạn chi phí mỗi vòng lặp.** Phát hành chỉ số chi phí-mỗi-vòng-lặp. PSO tốn 500 USD/vòng lặp mà chỉ tăng 0,5% là không thể triển khai.

## Bài tập

1. Chạy `code/main.py`. Quan sát sự hội tụ của LMPSO. Thay đổi kích thước quần thể 5, 10, 20, 50. Tại kích thước nào thì thời gian hội tụ bão hòa?
2. Triển khai thí nghiệm "trôi dạt thảm khốc": sau vòng lặp 30, thay đổi hàm độ thích nghi. PSO thích nghi nhanh như thế nào? Việc đặt lại `p_best` có giúp ích không?
3. Thêm cổng kiểm soát chất lượng vào AMRO-S: chỉ gửi pheromone vào các lần chạy có điểm đánh giá > 0,7. Điều này thay đổi sự hội tụ như thế nào so với phiên bản không có cổng?
4. Đọc LMPSO (arXiv:2504.09247). Ánh xạ "vận tốc như một prompt" của bài báo trở lại vận tốc số của bạn. Điều gì bị mất trong quá trình mô phỏng và điều gì được bảo toàn?
5. Đọc AMRO-S (arXiv:2603.12933). Triển khai "đường dẫn nhanh suy luận" tách biệt với cập nhật pheromone bất đồng bộ. Điều này thay đổi độ trễ hệ thống dưới tải trọng duy trì như thế nào?

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|----------------|------------------------|
| PSO | "Particle Swarm Optimization" | Kennedy-Eberhart 1995. Bộ tối ưu hóa không cần gradient dựa trên quần thể. |
| ACO | "Ant Colony Optimization" | Dorigo 1992. Tối ưu hóa đường đi/tuyến đường thông qua các vệt pheromone. |
| LMPSO | "PSO với tạo LLM" | arXiv:2504.09247. Vận tốc là một prompt; LLM tạo ra các ứng viên. |
| Model Swarms | "PSO trên trọng số chuyên gia" | arXiv:2410.11163. Cập nhật không cần gradient trên không gian con tham số mô hình. |
| AMRO-S | "ACO cho định tuyến tác nhân" | arXiv:2603.12933. Ma trận pheromone trên loại tác vụ × tác nhân. |
| p_best / g_best | "Tốt nhất cá nhân / toàn cục" | Các giải pháp tốt nhất của từng hạt và toàn bầy tìm thấy cho đến nay. |
| Pheromone | "Bộ nhớ định tuyến" | Cường độ trên một cạnh; suy giảm theo thời gian; gửi vào khi có chất lượng. |
| Cập nhật có kiểm soát chất lượng | "Chỉ học từ các lần chạy tốt" | Gửi pheromone có điều kiện dựa trên kiểm tra chất lượng. |
| Trôi dạt thảm khốc | "Thay đổi phân phối" | Cảnh quan độ thích nghi thay đổi; p_best và pheromone cũ trở nên lỗi thời. |

## Đọc thêm

- [Kennedy & Eberhart — Particle Swarm Optimization](https://ieeexplore.ieee.org/document/488968) — bài báo PSO năm 1995
- [Dorigo — Ant Colony Optimization](https://www.aco-metaheuristic.org/about.html) — nền tảng ACO năm 1992
- [LMPSO — Language Model Particle Swarm Optimization](https://arxiv.org/abs/2504.09247) — PSO cho đầu ra LLM có cấu trúc
- [Model Swarms — gradient-free LLM expert optimization](https://arxiv.org/abs/2410.11163) — PSO trên không gian con trọng số mô hình
- [AMRO-S — ant-colony multi-agent routing](https://arxiv.org/abs/2603.12933) — định tuyến dựa trên pheromone với cổng chất lượng