# Instruction-Following as Alignment Signal

> Mọi phê bình sau này về RLHF đều lập luận chống lại quy trình này. Trước khi bạn nghiên cứu cách áp lực tối ưu hóa làm sai lệch một proxy (đại diện), bạn phải nhìn thấy proxy đó. InstructGPT (Ouyang và cộng sự, 2022) đã định nghĩa kiến trúc tham chiếu: supervised fine-tuning trên các cặp hướng dẫn-phản hồi, một reward model được huấn luyện trên các bảng xếp hạng ưu tiên theo cặp, và PPO dựa trên reward model với hình phạt KL đối với policy SFT. Một mô hình InstructGPT 1.3B đã được ưu tiên hơn GPT-3 175B. Kết quả duy nhất đó là lý do tại sao mọi phòng thí nghiệm tiên phong vào năm 2026 vẫn vận hành quy trình hậu huấn luyện (post-training) theo hình thái RLHF.

**Type:** Learn
**Languages:** Python (stdlib, toy three-stage pipeline)
**Prerequisites:** Phase 10 · 06 (SFT), Phase 10 · 07 (RLHF), Phase 10 · 08 (DPO)
**Time:** ~45 phút

## Mục tiêu học tập

- Nêu tên ba giai đoạn của quy trình InstructGPT và hàm mất mát (loss) được sử dụng trong mỗi giai đoạn.
- Giải thích tại sao một mô hình 1.3B được tinh chỉnh theo hướng dẫn lại vượt qua GPT-3 175B gốc trong đánh giá ưu tiên của con người.
- Nêu rõ hình phạt KL trong giai đoạn 3 bảo vệ chống lại điều gì và tại sao việc loại bỏ nó lại dẫn đến hành vi tìm kiếm mode (mode-seeking).
- Mô tả "thuế căn chỉnh" (alignment tax) và phương pháp PPO-ptx mà Ouyang và cộng sự đã sử dụng để giảm thiểu nó.

## Vấn đề

Các mô hình ngôn ngữ tiền huấn luyện (pre-trained) chỉ hoàn thành văn bản. Chúng không trả lời câu hỏi. Hãy hỏi GPT-3 "viết một hàm Python đảo ngược một danh sách" và bạn thường nhận lại một prompt khác, vì phần lớn phân phối huấn luyện là văn bản web, vốn tiếp tục bằng nhiều văn bản web hơn. Mô hình đang làm đúng công việc của nó — nhưng công việc đó lại sai mục đích.

Proxy mà mọi phòng thí nghiệm nghiêm túc sử dụng để khắc phục điều này là ưu tiên của con người. Hai kết quả hoàn thành được gửi đến người đánh giá; người đánh giá chọn kết quả tốt hơn; một reward model học từ người đánh giá đó. Sau đó, một vòng lặp RL sẽ chuyển dịch policy về phía các đầu ra mà reward model chấm điểm cao. Đó là toàn bộ luận điểm của InstructGPT trong ba câu. Phần còn lại của bài báo là kỹ thuật.

## Khái niệm

### Giai đoạn 1: supervised fine-tuning (SFT)

Thu thập các cặp prompt-phản hồi trong đó phản hồi là những gì một con người có thiện chí sẽ viết. Ouyang và cộng sự đã sử dụng 13 nghìn prompt từ các người dán nhãn và OpenAI API. Tinh chỉnh mô hình cơ sở trên dữ liệu này với hàm mất mát cross-entropy tiêu chuẩn.

Những gì SFT mang lại: mô hình hiện trả lời các câu hỏi thay vì tiếp tục chúng. Những gì nó không mang lại: bất kỳ tín hiệu nào về việc người đánh giá ưu tiên câu trả lời nào khi có nhiều câu trả lời hợp lý.

### Giai đoạn 2: reward model (RM)

Đối với mỗi prompt, lấy mẫu K kết quả hoàn thành từ mô hình SFT. Một người dán nhãn xếp hạng chúng. Huấn luyện một reward model chấm điểm bất kỳ cặp prompt-phản hồi nào sao cho, đối với các cặp mà `y_w` được ưu tiên hơn `y_l`:

```
L_RM = -log sigmoid(r(x, y_w) - r(x, y_l))
```

Đây là hàm mất mát ưu tiên theo cặp Bradley-Terry. RM thường được khởi tạo từ mô hình SFT với phần LM head được thay thế bằng một scalar head.

Các reward model thường nhỏ: 6B là đủ cho InstructGPT 175B. Chúng cũng rất mong manh — phần 5 của bài báo chủ yếu nói về các hành vi "reward-hacking" xuất hiện ở quy mô nhỏ.

### Giai đoạn 3: PPO với hình phạt KL

Xác định mục tiêu:

```
J(pi) = E_{x~D, y~pi(.|x)} [ r(x, y) ] - beta * KL(pi(.|x) || pi_SFT(.|x))
```

Tối đa hóa bằng PPO. Thuật ngữ KL giữ cho `pi` không bị trôi quá xa so với policy SFT. Nếu không có nó, bộ tối ưu hóa sẽ tìm thấy các ví dụ đối nghịch (adversarial examples) — các chuỗi có điểm số cao theo RM vì RM chưa bao giờ thấy chúng, chứ không phải vì con người thực sự ưu tiên chúng.

Hệ số KL `beta` là siêu tham số RLHF quan trọng nhất. Quá thấp: xảy ra reward hacking. Quá cao: không có cải thiện so với SFT.

### Thuế căn chỉnh (Alignment tax)

Sau RLHF, mô hình được con người ưu tiên hơn nhưng lại suy giảm trên các benchmark tiêu chuẩn (SQuAD, HellaSwag, DROP). Ouyang và cộng sự gọi đây là thuế căn chỉnh và khắc phục nó bằng PPO-ptx: trộn các gradient tiền huấn luyện vào mục tiêu RL để mô hình không quên cách thực hiện các tác vụ hạ nguồn mà nó chưa bao giờ được thưởng.

```
J_ptx(pi) = J(pi) + gamma * E_{x~D_pretrain} [ log pi(x) ]
```

PPO-ptx đã trở thành tiêu chuẩn. Anthropic, DeepMind và Meta đều sử dụng một biến thể nào đó của nó.

### Kết quả

Một mô hình InstructGPT 1.3B (SFT + RM + PPO-ptx) được người dán nhãn ưu tiên hơn GPT-3 175B gốc khoảng 70% thời gian. Khoảng cách này nới rộng trên các prompt kiểm tra ẩn từ lưu lượng truy cập thực tế. Hai điều cần rút ra từ con số này:

1. Căn chỉnh là một trục khác với năng lực. Mô hình 175B có năng lực cao hơn; mô hình 1.3B có sự căn chỉnh tốt hơn; người dán nhãn ưu tiên mô hình đã được căn chỉnh.
2. Nền tảng năng lực được thiết lập bởi mô hình cơ sở. Bạn không thể RLHF một mô hình cơ sở để biết các sự kiện mà nó chưa bao giờ thấy.

### Tại sao đây là điểm tham chiếu cho Phase 18

Mọi phê bình trong các bài học sau này — reward hacking (Bài 2), DPO (Bài 3), sycophancy (Bài 4), CAI (Bài 5), sleeper agents (Bài 7), alignment faking (Bài 9) — đều lập luận chống lại một phần nào đó của quy trình này. Reward hacking tấn công giai đoạn 2. DPO hợp nhất giai đoạn 2 và 3. CAI thay thế người dán nhãn con người. Sycophancy cho thấy người dán nhãn là một tín hiệu thiên kiến. Alignment faking cho thấy policy có thể vượt qua giai đoạn 3 hoàn toàn. Bạn không thể theo dõi bất kỳ phê bình nào trong số này nếu không nắm vững quy trình này trong đầu trước.

```figure
al-instruct-pipeline
```

## Sử dụng

`code/main.py` mô phỏng ba giai đoạn trên dữ liệu ưu tiên đồ chơi. "Policy" cơ sở là một đồng xu thiên kiến trên các hành động {A, B, C}. Giai đoạn 1 SFT bắt chước các hành động của người dán nhãn trên 200 prompt. Giai đoạn 2 khớp một reward model Bradley-Terry từ 500 bảng xếp hạng theo cặp. Giai đoạn 3 chạy một bản cập nhật PPO đơn giản hóa với hình phạt KL đối với policy SFT. Bạn có thể quan sát phần thưởng tăng lên, sự phân kỳ KL tăng trưởng và policy trôi đi — và bạn có thể tắt thuật ngữ KL để thấy reward hacking xuất hiện trong vòng 50 bước cập nhật.

Những điều cần quan sát:

- Quỹ đạo phần thưởng với `beta = 0.1` so với `beta = 0.0`.
- KL(pi || pi_SFT) qua các bước huấn luyện.
- Phân phối hành động cuối cùng so với ưu tiên của người dán nhãn.

## Triển khai

Bài học này tạo ra `outputs/skill-instructgpt-explainer.md`. Với một mô tả quy trình RLHF hoặc tóm tắt bài báo, nó xác định giai đoạn nào trong ba giai đoạn đang được sửa đổi, hàm mất mát nào đang được sử dụng ở mỗi giai đoạn và liệu có hình phạt KL hoặc bộ điều chuẩn tương đương nào hiện diện hay không.

## Bài tập

1. Chạy `code/main.py`. Thiết lập `beta = 0.0` và báo cáo phân phối hành động sau 200 bước PPO. Giải thích hành vi tìm kiếm mode trong một đoạn văn.

2. Sửa đổi reward model để có thiên kiến +0.5 cho hành động B (một lỗi phần thưởng mô phỏng). Chạy PPO với `beta = 0.1`. Hình phạt KL có ngăn cản policy khai thác thiên kiến không? Tại `beta` nào thì việc khai thác trở nên rõ ràng?

3. Đọc Ouyang và cộng sự (arXiv:2203.02155) Hình 1. Tái tạo đường cong ưu tiên của người dán nhãn bằng cách chạy PPO trong 1, 5, 20, 100 bước và đo lường ưu tiên so với mô hình SFT.

4. Phần 4.3 của bài báo báo cáo rằng InstructGPT 1.3B đánh bại GPT-3 175B khoảng 70% thời gian. Tại sao tỷ lệ này lại cao hơn trên các prompt sản xuất ẩn so với các prompt của chính người dán nhãn?

5. Thay thế hàm mất mát PPO bằng DPO (Phase 10 · 08) trên cùng dữ liệu ưu tiên. So sánh sự trôi dạt policy cuối cùng (KL so với SFT) và phần thưởng cuối cùng. Phương pháp nào trôi dạt xa hơn ở mức phần thưởng tương đương?

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|------------------------|
| SFT | "instruction tuning" | Giai đoạn 1: tinh chỉnh cross-entropy trên các cặp prompt-phản hồi |
| Reward model | "the RM" | Scalar regressor trên (prompt, phản hồi) được huấn luyện với Bradley-Terry trên các nhãn theo cặp |
| Bradley-Terry | "pairwise preference loss" | -log sigmoid(r_w - r_l); giảm xếp hạng theo cặp thành phân loại nhị phân |
| KL penalty | "the regularizer" | `beta * KL(pi \|\| pi_SFT)` — giữ cho RL policy gần với neo SFT |
| PPO-ptx | "PPO with pretraining mix" | Thêm một phần log-likelihood tiền huấn luyện vào mục tiêu PPO để bù đắp thuế căn chỉnh |
| Alignment tax | "the RLHF regression" | Sự sụt giảm sau RLHF trên các benchmark tiêu chuẩn mà RLHF không nhắm tới |
| Labeler preference | "the ground truth" | Mẫu xếp hạng của con người; RM là một proxy thống kê cho điều này, không phải cho "giá trị con người" |

## Đọc thêm

- [Ouyang và cộng sự — Training language models to follow instructions with human feedback (arXiv:2203.02155)](https://arxiv.org/abs/2203.02155) — bài báo InstructGPT, nền tảng cho mọi quy trình RLHF sau này
- [Stiennon và cộng sự — Learning to summarize from human feedback (arXiv:2009.01325)](https://arxiv.org/abs/2009.01325) — tiền thân của RLHF cho tóm tắt văn bản
- [Christiano và cộng sự — Deep reinforcement learning from human preferences (arXiv:1706.03741)](https://arxiv.org/abs/1706.03741) — công thức RL dựa trên ưu tiên gốc
- [Bai và cộng sự — Training a Helpful and Harmless Assistant with RLHF (arXiv:2204.05862)](https://arxiv.org/abs/2204.05862) — phần mở rộng HH của Anthropic cho quy trình InstructGPT