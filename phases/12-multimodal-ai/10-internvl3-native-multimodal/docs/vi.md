# InternVL3: Native Multimodal Pretraining

> Mọi VLM mã nguồn mở trước InternVL3 đều tuân theo công thức ba bước giống nhau: lấy một LLM văn bản đã được huấn luyện trên hàng nghìn tỷ token văn bản, gắn thêm một bộ mã hóa thị giác (vision encoder), sau đó tinh chỉnh các điểm nối. Cách này hiệu quả nhưng để lại "nợ căn chỉnh" (alignment debt) — LLM văn bản đã tiêu tốn toàn bộ ngân sách tiền huấn luyện cho văn bản thuần túy và không hiểu các token thị giác một cách tự nhiên. Khi bạn thêm thị giác vào sau (post-hoc), LLM phải học lại cách liên kết đầu vào thị giác với khả năng suy luận văn bản của nó mà không được quên đi văn bản. InternVL3 (Zhu và cộng sự, tháng 4 năm 2025) bác bỏ phương pháp post-hoc: chỉ một lần tiền huấn luyện duy nhất, văn bản và đa phương thức được đan xen ngay từ bước đầu tiên. Kết quả đạt được ngang bằng với Gemini 2.5 Pro trên MMMU-Pro ở quy mô 78B tham số mã nguồn mở. Bài học này giải thích lý do cho việc tiền huấn luyện tự nhiên (native pretraining) và những thay đổi khi bạn thực hiện nó.

**Type:** Learn
**Languages:** Python (stdlib, training-corpus mixer)
**Prerequisites:** Phase 12 · 05, Phase 12 · 07 (recipes)
**Time:** ~120 phút

## Mục tiêu học tập

- Giải thích lý do tại sao việc huấn luyện VLM theo kiểu post-hoc lại tích tụ nợ căn chỉnh, trích dẫn ba triệu chứng có thể đo lường được (quên thảm họa, trôi dạt câu trả lời, thiếu nhất quán giữa thị giác và văn bản).
- Mô tả hỗn hợp dữ liệu tiền huấn luyện tự nhiên của InternVL3 và lý do tại sao tỷ lệ văn bản : đan xen : chú thích lại quan trọng.
- So sánh V2PE (mã hóa vị trí thị giác biến đổi) với M-RoPE của Qwen2-VL.
- Nêu tên các tối ưu hóa triển khai Visual Resolution Router (ViR) và Decoupled Vision-Language (DvD).

## Vấn đề

Huấn luyện VLM theo kiểu post-hoc là mặc định. LLaVA, BLIP-2, Qwen-VL, Idefics — tất cả đều lấy một LLM đã được tiền huấn luyện (Llama, Vicuna, Qwen, Mistral) và thêm thị giác vào. Các giai đoạn huấn luyện thường trông như sau:

1. LLM đóng băng + bộ mã hóa thị giác đóng băng + projector có thể huấn luyện, được huấn luyện trên các cặp chú thích để căn chỉnh embedding.
2. Mở đóng băng LLM, huấn luyện trên dữ liệu hướng dẫn (LLaVA-Instruct, ShareGPT4V).
3. Tinh chỉnh tùy chọn cho tác vụ cụ thể.

Ba triệu chứng của nợ căn chỉnh xuất hiện:

- Quên thảm họa (Catastrophic forgetting). VLM post-hoc quên đi các kỹ năng chỉ dành cho văn bản. Điểm GSM8K giảm 5-10 điểm. Điểm Hellaswag giảm. Các tác nhân văn bản thuần túy bị thoái hóa.
- Trôi dạt câu trả lời (Answer drift). Các cách diễn đạt nhỏ của cùng một câu hỏi thị giác nhận được các câu trả lời khác nhau. Bộ mã hóa thị giác kết nối với LLM bằng các liên kết yếu hơn so với chính các token của LLM.
- Thiếu nhất quán giữa thị giác và văn bản (Visual-text inconsistency). VLM có thể mô tả một hình ảnh chính xác và sau đó trả lời một câu hỏi mâu thuẫn với chính mô tả của nó. Các token thị giác không tham gia vào các kiểm tra tính nhất quán nội bộ của LLM giống như cách văn bản thực hiện.

Những triệu chứng này đã được ghi chép rõ ràng. Phần 4 của MM1.5 định lượng chúng. Các thử nghiệm cắt bỏ (ablations) của LLaVA-OneVision gợi ý về chúng. Tiền huấn luyện tự nhiên là câu trả lời.

## Khái niệm

### Tiền huấn luyện đa phương thức tự nhiên

InternVL3 huấn luyện từ đầu trên một kho dữ liệu đa phương thức tự nhiên ngay từ bước đầu tiên. Hỗn hợp bao gồm:

- 40% dữ liệu chỉ văn bản (FineWeb, Proof-Pile-2, v.v.)
- 35% dữ liệu văn bản-hình ảnh đan xen (OBELICS, kiểu MMC4)
- 20% dữ liệu cặp hình ảnh-chú thích
- 5% dữ liệu video-văn bản

Các token thị giác, token văn bản và các tương tác đa phương thức đều tham gia vào cùng một hàm mất mát (loss) từ bước gradient đầu tiên. Không có tiền huấn luyện căn chỉnh, không có giai đoạn đóng băng projector, không có sự quên thảm họa cần phục hồi.

Huấn luyện là một giai đoạn duy nhất cho mô hình cơ sở. Tinh chỉnh hướng dẫn diễn ra sau đó, nhưng mô hình cơ sở đã hiểu các token thị giác như những công dân hạng nhất.

### V2PE (mã hóa vị trí thị giác biến đổi)

Qwen2-VL sử dụng M-RoPE với phân bổ trục cố định. InternVL3 giới thiệu V2PE: mã hóa vị trí thay đổi theo loại phương thức (văn bản, hình ảnh, video) với khả năng mở rộng có thể học được. Trong thực tế:

- Token văn bản có vị trí 1D (chỉ số văn bản).
- Các mảng hình ảnh (patches) có vị trí 2D (hàng, cột).
- Các khung hình video có vị trí 3D (thời gian, hàng, cột).

Cả ba chia sẻ cùng một cơ sở tần số RoPE, nhưng việc phân bổ hidden-dim theo băng tần là một tham số được học thay vì một sự phân chia cố định. Sự tự do để đánh đổi giữa độ phân giải tần số không gian và thời gian trong quá trình tiền huấn luyện.

Tuyên bố thử nghiệm của V2PE: tăng 1-2 điểm trên các benchmark video so với M-RoPE ở cùng mức tính toán. Không phải là một cuộc cách mạng, nhưng sạch sẽ hơn.

### Visual Resolution Router (ViR)

Tối ưu hóa triển khai. Không phải tất cả hình ảnh đều cần mã hóa độ phân giải đầy đủ. Một bức ảnh với một vật thể ở chi tiết thấp sẽ lãng phí token khi được mã hóa ở độ phân giải gốc 1280px. ViR là một bộ phân loại nhỏ dự đoán độ phân giải tối thiểu cần thiết để trả lời câu hỏi, trước khi mã hóa.

Việc định tuyến có ba cấp độ: độ phân giải thấp (256 token), trung bình (576), cao (2048+). Đối với 60% truy vấn trong lưu lượng sản xuất, mức thấp hoặc trung bình là đủ. Hiệu quả ròng: thông lượng gấp 2-3 lần ở chất lượng tương đương.

### Triển khai Decoupled Vision-Language (DvD)

Khi bạn phục vụ một VLM lớn, bộ mã hóa thị giác chạy một lần cho mỗi hình ảnh nhưng LLM chạy tự hồi quy cho mỗi token đầu ra. Hai thành phần có các nút thắt cổ chai khác nhau (thị giác = băng thông bộ nhớ GPU cho conv + attention; LLM = KV cache). DvD tách chúng ra các GPU riêng biệt với luồng dữ liệu giữa chúng.

Đối với mô hình bộ mã hóa 8B + 400M, DvD tăng gấp đôi thông lượng trên mỗi node so với việc đặt chung.

### Chất lượng giai đoạn đơn so với đa giai đoạn

Tuyên bố benchmark chính của InternVL3: ở mức 78B tham số, ngang bằng với MMMU-Pro của Gemini 2.5 Pro. Ở mức 38B, ngang bằng với GPT-4o. Ở mức 8B, dẫn đầu bảng xếp hạng open-8B. Tất cả đều dựa trên công thức tiền huấn luyện + tinh chỉnh hướng dẫn một giai đoạn.

Giả thuyết nợ căn chỉnh có thể đo lường được: InternVL3-8B mất ít điểm benchmark văn bản (MMLU, GSM8K) hơn so với Qwen2.5-VL-7B trên mỗi đơn vị tăng điểm benchmark thị giác. Mô hình là một mô hình tổng quát hơn vì quá trình huấn luyện là một khối thống nhất, không phải hai phần.

### InternVL3.5 và InternVL-U

InternVL3.5 (tháng 8 năm 2025) mở rộng công thức. Vẫn là phương pháp tiền huấn luyện tự nhiên, nhiều dữ liệu hơn, nhiều tham số hơn. Những cải tiến trên MMMU là tăng dần.

InternVL-U (2026) bổ sung thế hệ thống nhất — đầu ra hình ảnh thông qua các đầu MMDiT trên cùng một backbone. Chữ "U" viết tắt cho "Understanding + generation" (Hiểu + tạo), theo đuổi các mô hình thống nhất kiểu Transfusion (Bài học 12.13). Cùng một backbone tiền huấn luyện tự nhiên hỗ trợ cả đầu hiểu và đầu tạo.

### Đánh đổi của tiền huấn luyện tự nhiên

Tiền huấn luyện tự nhiên không miễn phí:

- Tính toán. Huấn luyện một VLM mới từ đầu tốn kém như huấn luyện một LLM văn bản — hàng triệu giờ GPU. Thích ứng post-hoc tái sử dụng trọng số LLM hiện có, tiết kiệm phần lớn chi phí.
- Dữ liệu. Các kho dữ liệu văn bản-hình ảnh đan xen ở quy mô lớn rất hiếm. OBELICS có 141 triệu tài liệu; MMC4 có 571 triệu. Văn bản thuần túy có sẵn ở mức 15 nghìn tỷ token. Sự khan hiếm dữ liệu tiền huấn luyện đa phương thức là một ràng buộc cứng.
- Tái sử dụng Base-LLM. Tiền huấn luyện tự nhiên từ bỏ tùy chọn thay thế một LLM mới sau này. Post-hoc cho phép bạn đổi Llama-3.1 lấy Llama-4 bằng cách chỉ huấn luyện lại adapter.

Canh bạc mà InternVL3 thực hiện: nợ căn chỉnh tồi tệ hơn so với việc mất khả năng tái sử dụng. Các benchmark ủng hộ tuyên bố này. Chi phí sản xuất ngăn cản các phòng thí nghiệm tương lai sao chép một cách rẻ tiền. Các VLM post-hoc sẽ tiếp tục tồn tại vì chúng vẫn rẻ hơn cho hầu hết các dự án.

```figure
l5-native-pretrain
```

## Sử dụng

`code/main.py` là một công cụ trộn kho dữ liệu huấn luyện và mô phỏng bộ định tuyến ViR. Nó:

- Lấy một hỗn hợp kho dữ liệu mục tiêu (%văn bản, %đan xen, %chú thích, %video) và tính toán các bước dự kiến cho mỗi phương thức.
- Mô phỏng định tuyến ViR trên một lô truy vấn (phân phối: 50% chi tiết thấp, 30% trung bình, 20% chi tiết cao) và báo cáo số lượng token trung bình.
- Báo cáo ước tính thông lượng DvD dựa trên FLOPs của bộ mã hóa so với LLM.
- In ra bảng so sánh song song giữa tiền huấn luyện post-hoc và tự nhiên về tham số, tính toán, dữ liệu và các triệu chứng nợ căn chỉnh dự kiến.

## Triển khai

Bài học này tạo ra `outputs/skill-native-vs-posthoc-auditor.md`. Với một kế hoạch huấn luyện VLM được đề xuất, nó kiểm tra xem nên đi theo hướng tự nhiên hay post-hoc, gắn cờ rủi ro nợ căn chỉnh và đề xuất một hỗn hợp kho dữ liệu. Sử dụng nó khi bạn đang định cỡ một dự án open-VLM mới và cần chọn chiến lược huấn luyện.

## Bài tập

1. Ước tính sự chênh lệch tính toán giữa InternVL3-8B (tiền huấn luyện tự nhiên) và LLaVA-OneVision-7B (post-hoc). Tỷ lệ giờ GPU xấp xỉ là bao nhiêu? Điều gì giải thích cho khoảng cách này?

2. InternVL3 báo cáo 40% văn bản / 35% đan xen / 20% chú thích / 5% video. Nếu tác vụ mục tiêu của bạn nặng về video, hãy đề xuất một tỷ lệ mới và lập luận tại sao mô hình cơ sở vẫn cần dữ liệu văn bản và chú thích đáng kể.

3. Đọc phần 4 của MM1.5 về sự quên. Nêu tên chính xác benchmark mà tại đó việc huấn luyện post-hoc cho thấy sự thoái hóa lớn nhất. Sự thoái hóa đó tốn kém bao nhiêu?

4. ViR định tuyến 60% lưu lượng truy cập đến mã hóa độ phân giải thấp. Những loại truy vấn nào mà nó định tuyến sai (gửi đến độ phân giải thấp khi cần độ phân giải cao)? Đề xuất ba chế độ lỗi của bộ định tuyến.

5. DvD tách thị giác và LLM lên các GPU riêng biệt. Trong mô hình lưu lượng truy cập nào thì DvD làm giảm thông lượng thay vì giúp ích?

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|------------------------|
| Native multimodal pretraining | "Từ đầu cùng nhau" | Các token văn bản + hình ảnh + video tham gia vào hàm mất mát từ bước 1, không gắn thêm sau |
| Alignment debt | "Phạt post-hoc" | Sự thoái hóa có thể đo lường được trong các kỹ năng văn bản và tính nhất quán của câu trả lời do việc gắn thị giác vào một LLM đã đóng băng |
| V2PE | "Mã hóa vị trí thị giác biến đổi" | Phân bổ mã hóa vị trí có thể học được theo từng phương thức; người kế nhiệm M-RoPE của InternVL3 |
| ViR | "Bộ định tuyến độ phân giải" | Bộ phân loại nhỏ chọn độ phân giải tối thiểu cần thiết cho mỗi truy vấn trước khi mã hóa, tiết kiệm token suy luận |
| DvD | "Triển khai tách rời" | Bộ mã hóa thị giác trên một GPU, LLM trên một GPU khác, với luồng chuyển giao; tăng gấp đôi thông lượng cho các VLM lớn |
| InternVL-U | "Hiểu + tạo thống nhất" | Bản cập nhật năm 2026 bổ sung các đầu tạo hình ảnh vào backbone tiền huấn luyện tự nhiên |
| Interleaved corpus | "OBELICS / MMC4" | Các tài liệu có văn bản và hình ảnh theo thứ tự đọc tự nhiên; nguyên liệu thô cho tiền huấn luyện tự nhiên |

## Đọc thêm

- [Chen và cộng sự — InternVL 1 (arXiv:2312.14238)](https://arxiv.org/abs/2312.14238)
- [Zhu và cộng sự — InternVL3 (arXiv:2504.10479)](https://arxiv.org/abs/2504.10479)
- [InternVL3.5 (arXiv:2508.18265)](https://arxiv.org/abs/2508.18265)
- [InternVL-U (arXiv:2603.09877)](https://arxiv.org/abs/2603.09877)
- [Zhang và cộng sự — MM1.5 (arXiv:2409.20566)](https://arxiv.org/abs/2409.20566)