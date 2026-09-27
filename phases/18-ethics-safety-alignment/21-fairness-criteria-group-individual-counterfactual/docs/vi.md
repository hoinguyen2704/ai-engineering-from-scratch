# Các tiêu chí về tính công bằng — Nhóm, Cá nhân, Phản thực tế (Counterfactual)

> Văn học về tính công bằng được cấu trúc bởi ba nhóm chính. Công bằng nhóm (Group fairness): demographic parity, equalized odds, conditional use accuracy equality — các tỷ lệ bằng nhau trên các nhóm được bảo vệ khi xét trung bình. Công bằng cá nhân (Dwork et al. 2012): những cá nhân tương tự nhau nhận được các quyết định tương tự nhau; điều kiện Lipschitz trên bản đồ quyết định. Công bằng phản thực tế (Counterfactual fairness) (Kusner et al. 2017): một quyết định là công bằng với một cá nhân nếu nó không thay đổi khi các thuộc tính nhạy cảm được thay đổi theo hướng phản thực tế. Kết quả lý thuyết năm 2024 (NeurIPS 2024): tồn tại sự đánh đổi cố hữu giữa CF và độ chính xác; một phương pháp bất khả tri với mô hình (model-agnostic) có thể chuyển đổi một bộ dự báo tối ưu-nhưng-không-công-bằng thành một bộ dự báo CF với mức giảm độ chính xác bị chặn. Backtracking counterfactuals (arXiv:2401.13935, tháng 1 năm 2024): mô hình mới tránh yêu cầu can thiệp vào các thuộc tính được pháp luật bảo vệ. Hòa giải triết học (ICLR Blogposts 2024): với các đồ thị nhân quả, việc thỏa mãn một số thước đo công bằng nhóm nhất định sẽ kéo theo công bằng phản thực tế.

**Type:** Học tập
**Languages:** Python (stdlib, so sánh ba tiêu chí)
**Prerequisites:** Phase 18 · 20 (bias), Phase 02 (classical ML)
**Time:** ~60 phút

## Mục tiêu học tập

- Nêu được ba tiêu chí công bằng nhóm (demographic parity, equalized odds, conditional use accuracy equality) và một kết quả về tính bất khả thi.
- Mô tả công bằng cá nhân thông qua công thức Lipschitz của Dwork et al. 2012.
- Mô tả công bằng phản thực tế và sự phụ thuộc của nó vào đồ thị nhân quả.
- Giải thích backtracking counterfactuals và lý do tại sao chúng tránh được vấn đề can thiệp vào thuộc tính được bảo vệ.

## Vấn đề

Bài 20 nói về việc đo lường bias. Bài 21 nói về việc xác định tiêu chuẩn công bằng mà phép đo đó cần phục vụ. Ba nhóm tiêu chuẩn này đưa ra các tiêu chuẩn khác biệt về mặt cấu trúc — một mô hình có thể công bằng với nhóm nhưng không công bằng với cá nhân, hoặc công bằng phản thực tế nhưng không công bằng với nhóm. Việc chọn tiêu chuẩn là một quyết định chính sách; không có tiêu chuẩn nào là tối ưu cho mọi trường hợp.

## Khái niệm

### Công bằng nhóm (Group fairness)

- **Demographic parity.** P(Y=1 | A=a) = P(Y=1 | A=a') cho mọi nhóm. Tỷ lệ chấp nhận bằng nhau.
- **Equalized odds.** P(Y=1 | Y*=y, A=a) = P(Y=1 | Y*=y, A=a'). TPR và FPR bằng nhau giữa các nhóm.
- **Conditional use accuracy equality.** P(Y*=y | Y=y, A=a) = P(Y*=y | Y=y, A=a'). Giá trị dự báo bằng nhau giữa các nhóm.

Tính bất khả thi (Chouldechova, Kleinberg-Mullainathan-Raghavan 2017): ba tiêu chí này không thể thỏa mãn đồng thời khi các tỷ lệ cơ sở (base rates) không bằng nhau.

### Công bằng cá nhân (Individual fairness)

Dwork et al. 2012. Một bản đồ quyết định f là công bằng với cá nhân xét theo một thước đo tương đồng d đặc thù cho tác vụ nếu |f(x) - f(x')| <= L * d(x, x') với một hằng số Lipschitz L. Những cá nhân tương tự nhau nhận được các quyết định tương tự nhau.

Yêu cầu xác định d. Đây là câu hỏi về chính sách, không phải thống kê.

### Công bằng phản thực tế (Counterfactual fairness)

Kusner et al. 2017. Một quyết định là công bằng phản thực tế với cá nhân i nếu, theo một mô hình nhân quả của quần thể, quyết định đó không thay đổi khi các thuộc tính nhạy cảm của i được thay đổi theo hướng phản thực tế.

Yêu cầu một DAG nhân quả. DAG là một lựa chọn mô hình hóa. Công bằng phản thực tế chỉ có giá trị khi DAG đó hợp lý.

### Sự đánh đổi giữa CF và độ chính xác

Lý thuyết NeurIPS 2024: tồn tại sự đánh đổi cố hữu giữa công bằng phản thực tế và độ chính xác dự báo. Một phương pháp bất khả tri với mô hình có thể chuyển đổi một bộ dự báo tối ưu-nhưng-không-công-bằng thành một bộ dự báo CF, với chi phí độ chính xác bị chặn. Chi phí độ chính xác phụ thuộc vào độ lớn của hệ số thuộc tính nhạy cảm trong bộ dự báo tối ưu không công bằng.

### Backtracking counterfactuals

arXiv:2401.13935 (tháng 1 năm 2024). Các phản thực tế truyền thống yêu cầu can thiệp vào thuộc tính nhạy cảm — "quyết định có thay đổi không nếu người này có giới tính khác." Về mặt pháp lý, điều này có vấn đề: các thuộc tính được bảo vệ không thể bị can thiệp trong luật phân loại.

Backtracking counterfactuals đảo ngược hướng: thay vì can thiệp vào thuộc tính, hãy đặt câu hỏi tổ hợp các đặc trưng thực tế nào của cá nhân đã tạo ra kết quả phản thực tế đó. Điều này tránh được sự phản đối về mặt pháp lý.

### Hòa giải triết học

ICLR Blogposts 2024. Với một đồ thị nhân quả trong tay, việc thỏa mãn một số thước đo công bằng nhóm nhất định sẽ kéo theo công bằng phản thực tế. Ba nhóm này không trực giao; chúng là các khía cạnh khác nhau của cùng một cấu trúc nhân quả cơ bản.

Điều này không giải quyết được các định lý bất khả thi (tỷ lệ cơ sở không bằng nhau vẫn ngăn cản việc đạt được công bằng nhóm đồng thời). Nhưng nó cho thấy sự đối lập rõ ràng giữa "nhóm" và "cá nhân / phản thực tế" một phần là do không làm rõ mô hình nhân quả.

### Vị trí trong Phase 18

Bài 20 là đo lường bias. Bài 21 là định nghĩa công bằng. Bài 22 là quyền riêng tư (differential privacy). Bài 23 là đóng dấu bản quyền (watermarking). Đây là các bài học liên quan đến phân bổ, bổ sung cho các bài 7-11 liên quan đến lừa đảo.

```figure
an-fairness-trilemma
```

## Sử dụng

`code/main.py` xây dựng một tập dữ liệu phân loại nhị phân mẫu với một thuộc tính nhạy cảm và các tỷ lệ cơ sở không bằng nhau. Tính toán demographic parity, equalized odds, và conditional use accuracy equality trên một bộ phân loại đơn giản. Quan sát sự bất đồng giữa ba thước đo. Áp dụng tái trọng số (re-weighting) cho demographic parity và quan sát chi phí của nó đối với hai thước đo còn lại.

## Triển khai

Bài học này tạo ra `outputs/skill-fairness-criterion.md`. Với một tuyên bố hoặc chính sách về công bằng, xác định tiêu chí nào đang được yêu cầu, liệu mô hình có thể thỏa mãn các tiêu chí còn lại dưới các tỷ lệ cơ sở không bằng nhau đã nêu hay không, và tuyên bố đó phụ thuộc vào DAG nhân quả nào.

## Bài tập

1. Chạy `code/main.py`. Báo cáo ba thước đo nhóm trên dữ liệu mặc định. Áp dụng tái trọng số nhắm vào demographic parity và báo cáo lại.

2. Triển khai thước đo công bằng cá nhân của Dwork et al. 2012 sử dụng L2 trên các đặc trưng không nhạy cảm. Báo cáo có bao nhiêu cặp vi phạm điều kiện Lipschitz với hằng số L=1.

3. Đọc Kusner et al. 2017. Xây dựng một DAG nhân quả hai đặc trưng đơn giản cho việc chấm điểm hồ sơ và xác định điều kiện công bằng phản thực tế mà nó ngụ ý.

4. Bài báo về backtracking counterfactuals năm 2024 tránh can thiệp vào các thuộc tính được bảo vệ. Mô tả một tình huống mà điều này quan trọng đối với việc tuân thủ pháp luật.

5. Sự hòa giải của ICLR 2024 lập luận rằng công bằng nhóm và công bằng phản thực tế là các khía cạnh của cùng một cấu trúc. Chọn hai trong ba tiêu chí trong `code/main.py` và nêu giả định nhân quả sẽ làm cho chúng tương đương nhau.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|------------------------|
| Demographic parity | "tỷ lệ bằng nhau" | P(Y=1 \| A=a) bằng nhau giữa các nhóm |
| Equalized odds | "TPR/FPR bằng nhau" | Tỷ lệ dương tính thật và dương tính giả bằng nhau giữa các nhóm |
| Conditional use accuracy | "PPV/NPV bằng nhau" | Giá trị dự báo bằng nhau giữa các nhóm |
| Individual fairness | "Điều kiện Lipschitz" | Những cá nhân tương tự nhau nhận được quyết định tương tự nhau |
| Counterfactual fairness | "bất biến khi thay đổi nhân quả" | Quyết định không thay đổi khi thuộc tính phản thực tế bị thay đổi |
| Backtracking counterfactual | "giải thích qua thực tế" | Phản thực tế được suy luận ngược từ kết quả, không phải xuôi từ thuộc tính |
| Impossibility theorem | "ba tiêu chí xung đột" | Chouldechova / KMR 2017: các tiêu chí nhóm loại trừ lẫn nhau khi tỷ lệ cơ sở không bằng nhau |

## Đọc thêm

- [Dwork et al. — Fairness through Awareness (arXiv:1104.3913)](https://arxiv.org/abs/1104.3913) — công bằng cá nhân
- [Kusner, Loftus, Russell, Silva — Counterfactual Fairness (arXiv:1703.06856)](https://arxiv.org/abs/1703.06856) — công bằng phản thực tế
- [Chouldechova — Fair prediction with disparate impact (arXiv:1703.00056)](https://arxiv.org/abs/1703.00056) — tính bất khả thi
- [Backtracking Counterfactuals (arXiv:2401.13935)](https://arxiv.org/abs/2401.13935) — mô hình mới cho các can thiệp vào thuộc tính được bảo vệ