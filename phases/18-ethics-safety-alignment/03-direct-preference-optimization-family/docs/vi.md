# Họ thuật toán Direct Preference Optimization

> Rafailov và cộng sự (2023) đã chỉ ra rằng nghiệm tối ưu của RLHF có dạng đóng dựa trên dữ liệu ưu tiên (preference data), vì vậy bạn có thể bỏ qua mô hình phần thưởng (reward model) tường minh và tối ưu hóa trực tiếp chính sách (policy). Nhận định đó đã khai sinh ra một họ thuật toán — IPO, KTO, SimPO, ORPO, BPO — mỗi thuật toán khắc phục một chế độ lỗi (failure mode) của DPO. Đến năm 2026, các thuật toán căn chỉnh trực tiếp (direct alignment algorithms - DAA) được sử dụng trong nhiều đợt huấn luyện hậu kỳ (post-training) ở quy mô tiên phong hơn cả PPO. Tuy nhiên, đường cong tối ưu hóa quá mức (over-optimization) từ Bài 2 vẫn áp dụng: DAA không thoát khỏi định luật Goodhart, chúng chỉ chuyển dịch vị trí mà nó gây tác động.

**Type:** Học
**Languages:** Python (stdlib, bộ so sánh preference-loss biến thể six-variant)
**Prerequisites:** Giai đoạn 18 · 01 (InstructGPT), Giai đoạn 18 · 02 (Reward hacking), Giai đoạn 10 · 08 (DPO basics)
**Time:** ~75 phút

## Mục tiêu học tập

- Suy luận dạng đóng của DPO từ nghiệm tối ưu của RLHF-với-KL.
- Nêu rõ chế độ lỗi mà mỗi thuật toán IPO, KTO, SimPO, ORPO, BPO khắc phục trong DPO.
- Phân biệt "implicit reward gap" (khoảng cách phần thưởng ngầm) với "preference strength" (độ mạnh ưu tiên) và giải thích tại sao ánh xạ đồng nhất (identity mapping) của IPO lại quan trọng.
- Giải thích tại sao Rafailov và cộng sự (NeurIPS 2024) chứng minh rằng các DAA vẫn tối ưu hóa quá mức mặc dù không có RM tường minh.

## Vấn đề

Mục tiêu của RLHF (Bài 1):

```
max_pi E_{x,y~pi} [ r(x, y) ] - beta * KL(pi || pi_ref)
```

có một nghiệm tối ưu đã biết:

```
pi*(y|x) = (1/Z(x)) * pi_ref(y|x) * exp(r(x, y) / beta)
```

Vì vậy, phần thưởng được định nghĩa ngầm bởi tỷ lệ giữa chính sách tối ưu và chính sách tham chiếu:

```
r(x, y) = beta * log(pi*(y|x) / pi_ref(y|x)) + beta * log Z(x)
```

Thay thế điều này vào hàm khả năng ưu tiên Bradley-Terry, hàm phân hoạch `Z(x)` bị triệt tiêu vì nó chỉ phụ thuộc vào `x`. Những gì còn lại là một hàm mất mát chỉ dựa trên các tham số chính sách — không cần mô hình phần thưởng. Đó chính là DPO.

Điểm khó khăn: phép suy luận giả định rằng nghiệm tối ưu có thể đạt được, dữ liệu ưu tiên nằm trong phân phối (in-distribution), và chính sách tham chiếu là mỏ neo chế độ thực sự. Không điều nào trong số này hoàn toàn đúng. Mỗi thành viên trong họ thuật toán này đều khắc phục một giả định bị vi phạm khác nhau.

## Khái niệm

### DPO (Rafailov và cộng sự, 2023)

```
L_DPO = -log sigmoid(
  beta * log(pi(y_w | x) / pi_ref(y_w | x))
  - beta * log(pi(y_l | x) / pi_ref(y_l | x))
)
```

Những điều có thể sai sót:

- Khoảng cách phần thưởng ngầm `beta * (log(pi/pi_ref)_w - log(pi/pi_ref)_l)` là không bị chặn. Một ưu tiên nhỏ có thể tạo ra khoảng cách lớn tùy ý.
- Hàm mất mát đẩy log-prob của phản hồi được chọn và bị từ chối theo các hướng ngược nhau. Nó có thể đẩy log-prob tuyệt đối của phản hồi được chọn xuống thấp miễn là phản hồi bị từ chối giảm nhanh hơn. Đây là hiện tượng Phản hồi được chọn bị suy giảm (Degraded Chosen Response).
- Các ưu tiên nằm ngoài phân phối (cặp hiếm vs cặp hiếm) tạo ra các phần thưởng ngầm tùy ý.

### IPO (Azar và cộng sự, 2024)

Identity Preference Optimization thay thế log-sigmoid bằng một ánh xạ đồng nhất trên xác suất ưu tiên. Hàm mất mát trở thành sai số bình phương trên một mục tiêu bị chặn:

```
L_IPO = (log(pi(y_w | x) / pi_ref(y_w | x)) - log(pi(y_l | x) / pi_ref(y_l | x)) - 1/(2 beta))^2
```

Biên độ bị chặn bởi `1/(2 beta)`. Độ mạnh ưu tiên và khoảng cách phần thưởng ngầm tỷ lệ thuận với nhau. Không có hiện tượng bùng nổ giá trị.

### KTO (Ethayarajh và cộng sự, 2024)

Kahneman-Tversky Optimization loại bỏ hoàn toàn cấu trúc cặp. Với một đầu ra được gán nhãn đơn lẻ và tín hiệu nhị phân "đáng mong đợi" hoặc "không đáng mong đợi", nó ánh xạ tới một tiện ích lý thuyết triển vọng (prospect-theory utility):

```
v(x, y) = sigma(beta * log(pi(y|x) / pi_ref(y|x)) - z_ref)
```

với các trọng số khác nhau cho lợi nhuận và thua lỗ (tâm lý sợ thua lỗ - loss aversion). Lợi ích: bạn có thể sử dụng dữ liệu không theo cặp, vốn phong phú hơn nhiều.

### SimPO (Meng và cộng sự, 2024)

Simple Preference Optimization căn chỉnh tín hiệu huấn luyện với quá trình tạo văn bản. Loại bỏ hoàn toàn chính sách tham chiếu và chuẩn hóa log-likelihood theo độ dài:

```
L_SimPO = -log sigmoid(
  (beta / |y_w|) * log pi(y_w | x)
  - (beta / |y_l|) * log pi(y_l | x)
  - gamma
)
```

với biên độ `gamma` để ổn định. Việc chuẩn hóa độ dài loại bỏ động lực khai thác chế độ lỗi thiên kiến độ dài của DPO (`y_w` dài hơn tạo ra khoảng cách log-prob lớn hơn theo cấu trúc).

### ORPO (Hong và cộng sự, 2024)

Odds-Ratio Preference Optimization thêm một số hạng ưu tiên vào negative log-likelihood của SFT tiêu chuẩn:

```
L_ORPO = L_NLL(y_w) + lambda * L_OR
L_OR = -log sigmoid(log(odds(y_w) / odds(y_l)))
```

Không có chính sách tham chiếu — số hạng SFT đóng vai trò là bộ điều chuẩn (regularizer). Huấn luyện trong một giai đoạn duy nhất từ mô hình cơ sở đến mô hình đã căn chỉnh. Không cần checkpoint SFT riêng biệt.

### BPO (Đệ trình ICLR 2026, OpenReview id=b97EwMUWu7)

Xác định vấn đề Phản hồi được chọn bị suy giảm: DPO bảo toàn thứ tự `y_w > y_l` nhưng log-prob tuyệt đối của `y_w` có thể giảm. BPO thêm một dòng hiệu chỉnh đơn giản để phạt các chuyển động giảm xuống trên phản hồi được chọn. Báo cáo cho thấy độ chính xác tăng +10.1% trên Llama-3.1-8B-Instruct về suy luận toán học so với DPO.

### Kết quả phổ quát: DAA vẫn tối ưu hóa quá mức

Rafailov và cộng sự trong bài báo "Scaling Laws for Reward Model Overoptimization in Direct Alignment Algorithms" (NeurIPS 2024) đã huấn luyện các chính sách bằng DPO, IPO, SLiC trên nhiều tập dữ liệu với các ngân sách KL khác nhau. Các đường cong phần thưởng vàng-so-với-KL có cùng hình dạng đỉnh-và-sụp đổ như Gao và cộng sự đã mô tả. Các truy vấn phần thưởng ngầm sử dụng các mẫu nằm ngoài phân phối trong quá trình huấn luyện; điều chuẩn KL không ổn định được điều này.

Các DAA không thoát khỏi định luật Goodhart. Chúng thay đổi bề mặt nơi định luật này gây tác động từ "mô hình phần thưởng bị tối ưu hóa quá mức" sang "tỷ lệ chính sách tham chiếu bị tối ưu hóa quá mức". Giải pháp khắc phục phổ quát — dữ liệu tốt hơn, ensemble, dừng sớm (early stopping) — áp dụng cho cả hai.

### Lựa chọn giữa các thuật toán (2026)

- Nếu bạn có dữ liệu ưu tiên theo cặp lớn: DPO với beta bảo thủ, SimPO nếu thấy rõ thiên kiến độ dài.
- Nếu bạn có phản hồi nhị phân không theo cặp: KTO.
- Nếu bạn muốn quy trình một giai đoạn từ mô hình cơ sở: ORPO.
- Nếu bạn thấy log-prob của phản hồi được chọn bị suy giảm trong nhật ký DPO: BPO.
- Nếu độ mạnh ưu tiên thay đổi rộng và DPO đang bão hòa: IPO.

Mỗi phòng thí nghiệm đều chạy thử cả năm thuật toán trên một bộ dữ liệu và chọn người chiến thắng cho từng tác vụ. Không có lý do gì để nghiệm tối ưu cho suy luận toán học lại giống với suy luận cho an toàn.

```figure
dpo-margin
```

## Sử dụng

`code/main.py` so sánh sáu hàm mất mát (DPO, IPO, KTO, SimPO, ORPO, BPO) trên một tập dữ liệu ưu tiên mô phỏng, nơi độ mạnh ưu tiên thực sự thay đổi theo từng cặp. Mỗi hàm mất mát được tối ưu hóa dựa trên cùng một mẫu 500 cặp với một chính sách softmax nhỏ. Vẽ biểu đồ tỷ lệ thắng cuối cùng, độ lệch log-prob được chọn, và độ lan tỏa phần thưởng ngầm theo từng phương pháp.

## Triển khai

Bài học này tạo ra `outputs/skill-preference-loss-selector.md`. Dựa trên các thống kê tập dữ liệu (theo cặp vs không theo cặp, độ mạnh ưu tiên biến thiên vs đồng nhất, phân phối độ dài) và mục tiêu (một giai đoạn hoặc SFT-rồi-đến-ưu tiên), hãy đề xuất một hàm mất mát ưu tiên và báo cáo chế độ lỗi mà nó bảo vệ.

## Bài tập

1. Chạy `code/main.py`. Báo cáo mức giảm log-prob được chọn cuối cùng cho DPO và BPO. BPO sẽ giữ lại xác suất tuyệt đối được chọn cao hơn — hãy xác minh điều này.

2. Sửa đổi dữ liệu ưu tiên sao cho tất cả các cặp có độ mạnh bằng nhau. Phương pháp nào trong sáu phương pháp là mạnh mẽ nhất? Phương pháp nào bị suy giảm? Giải thích lợi thế của IPO trong trường hợp này.

3. Làm cho các phản hồi bị từ chối trung bình dài gấp 2 lần phản hồi được chọn. Mà không thay đổi bất cứ điều gì khác, hãy chứng minh bằng số liệu sự khai thác độ dài của DPO và cách khắc phục của SimPO.

4. Rafailov và cộng sự (NeurIPS 2024) tuyên bố các DAA tối ưu hóa quá mức. Hãy tái tạo một phiên bản điểm đơn lẻ: vẽ biểu đồ phân kỳ KL giữa phản hồi được chọn và bị từ chối, sau đó quan sát sự tối ưu hóa quá mức trong DPO ở beta lớn.

5. Đọc tóm tắt bài báo BPO (OpenReview b97EwMUWu7). Viết ra dòng hiệu chỉnh đơn giản mà BPO thêm vào DPO. Xác nhận lại với cách triển khai trong `code/main.py`.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|------------------------|
| DPO | "RLHF không cần mô hình phần thưởng" | Hàm mất mát suy ra từ nghiệm tối ưu RLHF dạng đóng; chỉ dựa trên tham số chính sách |
| Implicit reward | "tỷ lệ log" | `beta * log(pi(y\|x) / pi_ref(y\|x))` — phần thưởng ngầm định bởi DPO |
| IPO | "DPO bị chặn" | Thay thế log-sigmoid bằng ánh xạ đồng nhất; khoảng cách phần thưởng ngầm bị chặn bởi `1/(2 beta)` |
| KTO | "DPO không theo cặp" | Tiện ích lý thuyết triển vọng trên các nhãn đơn với tâm lý sợ thua lỗ |
| SimPO | "DPO không cần tham chiếu" | Log-likelihood chuẩn hóa độ dài + biên độ; không có chính sách tham chiếu |
| ORPO | "DPO một giai đoạn" | NLL + số hạng ưu tiên tỷ lệ chênh lệch (odds-ratio); huấn luyện từ mô hình cơ sở trong một lần chạy |
| BPO | "DPO bảo toàn phản hồi được chọn" | DPO cộng với hình phạt cho việc giảm log-prob tuyệt đối của phản hồi được chọn |
| Degraded Chosen | "phản hồi được chọn bị giảm" | DPO làm giảm log-prob của phản hồi được chọn miễn là phản hồi bị từ chối giảm nhanh hơn |
| DAA | "thuật toán căn chỉnh trực tiếp" | Bất kỳ phương pháp mất mát ưu tiên nào bỏ qua RM tường minh |

## Đọc thêm

- [Rafailov và cộng sự — Direct Preference Optimization (NeurIPS 2023, arXiv:2305.18290)](https://arxiv.org/abs/2305.18290)
- [Azar và cộng sự — A General Theoretical Paradigm to Understand Learning from Human Preferences (AISTATS 2024, arXiv:2310.12036)](https://arxiv.org/abs/2310.12036) — IPO
- [Ethayarajh và cộng sự — KTO: Model Alignment as Prospect Theoretic Optimization (arXiv:2402.01306)](https://arxiv.org/abs/2402.01306)
- [Meng, Xia, Chen — SimPO (NeurIPS 2024, arXiv:2405.14734)](https://arxiv.org/abs/2405.14734)
- [Hong, Lee, Thorne — ORPO (EMNLP 2024, arXiv:2403.07691)](https://arxiv.org/abs/2403.07691)
- [BPO — Behavior Preservation Optimization (ICLR 2026 OpenReview b97EwMUWu7)](https://openreview.net/forum?id=b97EwMUWu7)
- [Rafailov và cộng sự — Scaling Laws for RM Overoptimization in DAAs (NeurIPS 2024, arXiv:2406.02900)](https://arxiv.org/abs/2406.02900)