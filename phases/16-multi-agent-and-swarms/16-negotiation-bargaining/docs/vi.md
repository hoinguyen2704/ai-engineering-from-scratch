# Đàm phán và Thương lượng

> Các tác nhân (agent) đàm phán về tài nguyên, giá cả, phân bổ nhiệm vụ và các điều khoản. Bộ benchmark năm 2026 đã rất rõ ràng: NegotiationArena (arXiv:2402.05863) cho thấy các LLM có thể cải thiện lợi nhuận khoảng 20% thông qua thao túng nhân vật (ví dụ: "sự tuyệt vọng"); "Measuring Bargaining Abilities" (arXiv:2402.15813) chỉ ra rằng vai trò người mua khó hơn người bán và việc tăng quy mô (scale) không giúp ích gì — mô hình **OG-Narrator** của họ (bộ tạo đề xuất tất định + người kể chuyện LLM) đã đẩy tỷ lệ chốt giao dịch từ 26,67% lên 88,88%; Cuộc thi Đàm phán Tự trị Quy mô lớn (arXiv:2503.06416) đã thực hiện khoảng 180.000 cuộc đàm phán và phát hiện ra rằng các tác nhân **che giấu chuỗi suy nghĩ (chain-of-thought-concealing)** giành chiến thắng bằng cách ẩn đi lý luận của mình khỏi đối phương; Bhattacharya và cộng sự 2025 dựa trên các chỉ số của Dự án Đàm phán Harvard đã xếp hạng Llama-3 hiệu quả nhất, Claude-3 hung hăng nhất, GPT-4 công bằng nhất. Bài học này triển khai Contract Net Protocol (tiền thân của FIPA, Bài 02), kết nối một người mua/người bán theo phong cách LLM, thực hiện phân tách theo kiểu OG-Narrator và đo lường cách tỷ lệ chốt giao dịch thay đổi theo từng lựa chọn cấu trúc.

**Type:** Học + Xây dựng
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 02 (Di sản FIPA-ACL), Phase 16 · 09 (Mạng lưới Swarm song song)
**Time:** ~75 phút

## Vấn đề

Hai tác nhân cần thống nhất về một mức giá. Nếu để tự do với các prompt ngôn ngữ thuần túy, các LLM giai đoạn 2024-2026 chốt giao dịch với tỷ lệ thấp đáng ngạc nhiên (~27% trong các cuộc thương lượng được tham số hóa chặt chẽ trong arXiv:2402.15813). Việc tăng quy mô không giải quyết được vấn đề này: GPT-4 không có cấu trúc tốt hơn trong việc thương lượng so với GPT-3.5; nó chỉ giỏi hơn về *ngôn ngữ* thương lượng.

Vấn đề cốt lõi là các LLM gộp chung hai công việc — quyết định đề xuất và kể chuyện về đề xuất đó. OG-Narrator đã tách biệt chúng: một bộ tạo đề xuất tất định tính toán các bước đi bằng số; LLM chỉ đóng vai trò kể chuyện. Tỷ lệ chốt giao dịch tăng vọt lên ~89%.

Điều này phản ánh một phát hiện cổ điển trong hệ thống đa tác nhân: tách biệt cơ chế khỏi lớp giao tiếp sẽ mang lại chiến thắng. Contract Net Protocol (FIPA, 1996; Smith, 1980) là cơ chế thị trường nhiệm vụ tham chiếu. Hãy cắm một LLM vào vị trí kể chuyện và bạn sẽ có một thị trường nhiệm vụ hiện đại được vận hành bởi LLM.

## Khái niệm

### Contract Net, trong một đoạn văn

Contract Net Protocol của Smith năm 1980: một **quản lý (manager)** phát đi một **lời kêu gọi đề xuất (cfp)**; các **nhà thầu (bidders)** phản hồi bằng các thông điệp **propose** chứa các đề xuất của họ; người quản lý chọn người chiến thắng và gửi **accept-proposal** cho người thắng cuộc và **reject-proposal** cho những người thua cuộc. Người chiến thắng thực hiện công việc. Thông điệp tùy chọn: **refuse** (nhà thầu từ chối đề xuất). FIPA đã hệ thống hóa điều này thành giao thức tương tác `fipa-contract-net`.

### Tại sao OG-Narrator giành chiến thắng

"Measuring Bargaining Abilities of Language Models" (arXiv:2402.15813) đã quan sát thấy rằng:

- LLM thường phá vỡ các quy tắc thương lượng (đưa ra mức giá vô lý, phớt lờ ZOPA của bên kia).
- Chúng neo giá kém (chấp nhận các đề xuất đầu tiên tồi tệ; phản đề xuất ở các mức giá mang tính biểu tượng thay vì chiến lược).
- Chỉ riêng việc tăng quy mô không khắc phục được những điều này. Các mô hình lớn hơn tạo ra ngôn ngữ hợp lý hơn với sai số chiến lược tương tự.

Phân tách OG-Narrator:

```
           ┌──────────────────┐        ┌──────────────────┐
  state  → │ offer generator  │ price → │  LLM narrator    │ → message
           │  (deterministic) │        │  (writes the     │
           │                  │        │   human-style    │
           └──────────────────┘        │   accompaniment) │
                                       └──────────────────┘
```

Bộ tạo đề xuất là một chiến lược đàm phán cổ điển: mô hình đàm phán Rubinstein, chiến lược Zeuthen, hoặc một chiến lược ăn miếng trả trả miếng (tit-for-tat) đơn giản về giá. LLM đóng vai trò kể chuyện. Thông điệp chứa mức giá tất định và cách diễn đạt bằng ngôn ngữ tự nhiên.

Tỷ lệ chốt giao dịch tăng vọt vì:
- Giá cả nằm trong vùng đàm phán.
- Các điểm neo mang tính chiến lược, không phải cảm xúc.
- LLM làm tốt việc mà nó giỏi nhất: viết lách.

### Các phát hiện từ NegotiationArena

arXiv:2402.05863 cung cấp bộ benchmark chuẩn. Các phát hiện chính:

- LLM có thể cải thiện lợi nhuận khoảng 20% bằng cách áp dụng các nhân vật ("Tôi rất tuyệt vọng để bán món này trước thứ Sáu") — thao túng nhân vật là một chiến thuật thực sự.
- Các tác nhân công bằng/hợp tác bị khai thác bởi các tác nhân đối nghịch; việc phòng thủ đòi hỏi phải có tư thế phản kháng rõ ràng.
- Các cặp đối xứng hội tụ vào các kết quả không công bằng trong khoảng 40% các kịch bản benchmark.

Đây không phải là "LLM là những nhà đàm phán tồi." Mà là "LLM đàm phán quá giống con người, bao gồm cả những phần dễ bị khai thác."

### Che giấu chuỗi suy nghĩ (Chain-of-thought concealment)

Cuộc thi Đàm phán Tự trị Quy mô lớn (arXiv:2503.06416) đã thực hiện khoảng 180.000 cuộc đàm phán qua nhiều chiến lược LLM. Những người chiến thắng đã che giấu lý luận của họ khỏi đối phương:

- Nếu một tác nhân in "Tôi chỉ có thể xuống mức $75; my reservation price is $70" vào một bản nháp hiển thị công khai, đối thủ sẽ đọc được nó.
- Những người chiến thắng tính toán chiến lược một cách riêng tư; kênh đầu ra chỉ chứa đề xuất và phần kể chuyện tối thiểu cần thiết.

Đây là tiếng vang năm 2026 của lý thuyết trò chơi cổ điển (Aumann 1976 về tính hợp lý và thông tin): tiết lộ định giá riêng tư của bạn sẽ làm giảm lợi nhuận. LLM không tự nhận thức được điều này và vui vẻ gõ các mức giá dự phòng của chúng vào các dấu vết suy luận mà đối phương có thể nhìn thấy.

Bài học kỹ thuật: tách biệt ngữ cảnh bản nháp riêng tư khỏi ngữ cảnh thông điệp công khai. Đây không phải là tùy chọn.

### Bhattacharya và cộng sự 2025 — xếp hạng mô hình

Dựa trên các chỉ số của Dự án Đàm phán Harvard (đàm phán nguyên tắc, tôn trọng BATNA, có qua có lại về lợi ích):

- **Llama-3** hiệu quả nhất trong việc chốt các giao dịch (tỷ lệ chốt + lợi nhuận).
- **Claude-3** là nhà đàm phán hung hăng nhất (neo giá cao, nhượng bộ muộn).
- **GPT-4** là công bằng nhất (phương sai lợi nhuận nhỏ nhất giữa các cặp).

Đây là ảnh chụp nhanh năm 2025. Vấn đề không phải là mô hình nào thắng vào tháng 4 năm 2026 — mà là các mô hình cơ sở khác nhau có phong cách đàm phán bền bỉ. Các tập hợp không đồng nhất (Bài 15) bao gồm điều này như một nguồn đa dạng.

### Phân bổ nhiệm vụ thông qua Contract Net + LLM

Việc tái sử dụng Contract Net hiện đại cho đa tác nhân LLM:

1. Tác nhân quản lý phân tách nhiệm vụ thành các đơn vị.
2. Phát đi `cfp` với mô tả nhiệm vụ cho các tác nhân công nhân.
3. Mỗi công nhân trả về một đề xuất: `(price, eta, confidence)` trong đó giá có thể là token, đơn vị tính toán hoặc đô la.
4. Người quản lý chọn người chiến thắng (một hoặc nhiều, tùy thuộc vào nhiệm vụ) và trao giải.
5. Những công nhân bị từ chối được tự do đấu thầu các nhiệm vụ khác.

Cách này mở rộng tốt vượt quá 100 công nhân vì sự phối hợp là phát-và-phản hồi, không phải trò chuyện đồng bộ. Được sử dụng trong sản xuất: các mô hình điều phối của Microsoft Agent Framework, một số triển khai LangGraph.

### Đàm phán tương tác LLM-Stakeholders

NeurIPS 2024 (https://proceedings.neurips.cc/paper_files/paper/2024/file/984dd3db213db2d1454a163b65b84d08-Paper-Datasets_and_Benchmarks_Track.pdf) giới thiệu các trò chơi có thể chấm điểm nhiều bên với **điểm số bí mật** và **ngưỡng chấp nhận tối thiểu**. Mỗi bên liên quan có các tiện ích riêng tư; LLM phải suy luận chúng từ các thông điệp. Đây là sự tổng quát hóa của đàm phán hai bên thành hình thành liên minh N bên. Có liên quan đến các thị trường nhiệm vụ sản xuất với khả năng công nhân không đồng nhất.

### Quy tắc kể chuyện-so-với-cơ chế

Trên tất cả các benchmark đàm phán 2024-2026, quy tắc kỹ thuật nhất quán là:

> Hãy để LLM kể chuyện. Đừng để LLM tính toán đề xuất.

Nếu đề xuất cần là một con số (giá, ETA, số lượng), hãy tạo nó một cách tất định từ trạng thái đàm phán và để LLM tạo ra cách diễn đạt. Nếu đề xuất cần là một cấu trúc đề xuất (phân tách nhiệm vụ, phân công vai trò), hãy để LLM soạn thảo nó, nhưng xác thực nó dựa trên lược đồ và kiểm tra ràng buộc trước khi gửi.

```figure
a5-og-narrator
```

## Xây dựng

`code/main.py` triển khai:

- `ContractNetManager`, `ContractNetTask`, `Bid` — quản lý + nhà thầu, phát cfp, thu thập đề xuất, trao giải.
- `og_narrator_bargain(state, rng)` — người mua OG-Narrator: nhượng bộ tất định theo kiểu Zeuthen về phía điểm giữa.
- `seller_response(state, rng)` — chính sách phản đề xuất tất định của người bán (sự thật cấu trúc cho cả hai phong cách).
- `naive_llm_bargain(state, rng)` — mô phỏng một người thương lượng toàn LLM: chọn giá với phương sai cao, thường nằm ngoài ZOPA.
- Đo lường: tỷ lệ chốt giao dịch qua 1000 thử nghiệm với giá dự phòng mới được lấy mẫu cho mỗi thử nghiệm.

Chạy:

```
python3 code/main.py
```

Kết quả mong đợi: tỷ lệ chốt giao dịch của LLM ngây thơ ~65-75%; tỷ lệ chốt giao dịch của OG-Narrator ~85-95%; khoảng cách 15-25 điểm là lợi thế cấu trúc của việc phân tách tạo đề xuất khỏi kể chuyện. Cộng với một ví dụ phân bổ thị trường nhiệm vụ Contract Net với ba nhà thầu và một nhiệm vụ.

## Sử dụng

`outputs/skill-bargainer-designer.md` thiết kế một giao thức đàm phán: ai tạo đề xuất (tất định hoặc LLM), ai kể chuyện, cách các bản nháp riêng tư tách biệt khỏi các thông điệp công khai và cách tỷ lệ chốt giao dịch được giám sát.

## Triển khai

Danh sách kiểm tra đàm phán sản xuất:

- **Tách biệt bản nháp.** Trạng thái riêng tư không bao giờ được tiếp cận ngữ cảnh của đối phương. Điều này không thể thương lượng.
- **Tạo đề xuất tất định.** Giá cả, số lượng, ETA: hãy tính toán, đừng prompt.
- **Xác thực tất cả các đề xuất đến** dựa trên một lược đồ. Từ chối các đề xuất nằm ngoài ZOPA tại ranh giới giao thức.
- **Giới hạn vòng đàm phán.** Tối đa 3-5 vòng; leo thang lên người hòa giải khi bế tắc.
- **Đo lường tỷ lệ chốt và phương sai lợi nhuận** liên tục. Tỷ lệ chốt giảm là một triệu chứng — thường là do trôi prompt hoặc tấn công từ phía đối phương.
- **Ghi nhật ký tất cả các đề xuất bị từ chối** với lý do tất định. Đối với các quản lý Contract Net, các nhà thầu thua cuộc cần hiểu lý do tại sao.

## Bài tập

1. Chạy `code/main.py`. Xác nhận OG-Narrator đánh bại LLM ngây thơ về tỷ lệ chốt giao dịch. Chênh lệch bao nhiêu?
2. Triển khai **cải thiện lợi nhuận dựa trên nhân vật** (arXiv:2402.05863) — người mua áp dụng nhân vật "tuyệt vọng muốn mua trong tuần này" chỉ trong phần kể chuyện, bộ tạo đề xuất không đổi. Tỷ lệ chốt hoặc lợi nhuận có thay đổi không?
3. Triển khai **che giấu** chuỗi suy nghĩ: duy trì một chuỗi bản nháp riêng tư không được chuyển cho đối phương. Điều gì xảy ra nếu bạn vô tình làm rò rỉ nó (mô phỏng bằng cách hoán đổi các kênh)?
4. Mở rộng Contract Net thành đấu giá N-nhà thầu với giá dự trữ. Khi tất cả các giá thầu đều vượt quá giá dự trữ, người quản lý quyết định giữa giá thấp nhất và chất lượng cao nhất như thế nào? Bạn chọn quy tắc trao giải nào và tại sao?
5. Đọc Bhattacharya và cộng sự 2025 về các chỉ số của Dự án Đàm phán Harvard. Triển khai hai nhà thương lượng với các phong cách khác nhau (hung hăng vs công bằng). Đo lường phương sai lợi nhuận trong các cặp đối xứng và bất đối xứng.

## Thuật ngữ chính

| Thuật ngữ | Mọi người nói | Ý nghĩa thực sự |
|------|----------------|------------------------|
| Contract Net | "Thị trường nhiệm vụ" | Smith 1980, FIPA 1996. cfp + propose + accept/reject. Thị trường nhiệm vụ chuẩn. |
| ZOPA | "Vùng có thể thỏa thuận" | Sự chồng lấp giữa mức tối đa của người mua và mức tối thiểu của người bán. Các đề xuất nằm ngoài vùng này không thể chốt. |
| BATNA | "Giải pháp thay thế tốt nhất cho một thỏa thuận đàm phán" | Phương án dự phòng của bạn nếu giao dịch này thất bại. Thiết lập giá dự phòng của bạn. |
| OG-Narrator | "Bộ tạo đề xuất + người kể chuyện" | Phân tách: đề xuất tất định, kể chuyện bằng LLM. |
| Chiến lược Zeuthen | "Nhượng bộ giảm thiểu rủi ro" | Bộ tạo đề xuất cổ điển nhượng bộ dựa trên các giới hạn rủi ro. |
| Đàm phán Rubinstein | "Cân bằng đề xuất luân phiên" | Mô hình lý thuyết trò chơi cho đàm phán vô hạn với chiết khấu. |
| Che giấu CoT | "Ẩn lý luận của bạn" | Những người chiến thắng trong arXiv:2503.06416 giữ các bản nháp riêng tư; kênh công khai chỉ hiển thị đề xuất. |
| Thao túng nhân vật | "Tạo tư thế cảm xúc" | arXiv:2402.05863: ~20% lợi nhuận tăng thêm từ các nhân vật tuyệt vọng/khẩn cấp. |

## Đọc thêm

- [NegotiationArena](https://arxiv.org/abs/2402.05863) — bộ benchmark; các phát hiện về thao túng nhân vật và khai thác
- [Measuring Bargaining Abilities of Language Models](https://arxiv.org/abs/2402.15813) — OG-Narrator và kết quả người mua khó hơn người bán
- [Large-Scale Autonomous Negotiation Competition](https://arxiv.org/abs/2503.06416) — ~180k cuộc đàm phán; che giấu chuỗi suy nghĩ giành chiến thắng
- [LLM-Stakeholders Interactive Negotiation (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/984dd3db213db2d1454a163b65b84d08-Paper-Datasets_and_Benchmarks_Track.pdf) — các trò chơi có thể chấm điểm nhiều bên với các tiện ích bí mật
- [Smith 1980 — The Contract Net Protocol](https://ieeexplore.ieee.org/document/1675516) — cơ chế cổ điển, IEEE Transactions on Computers