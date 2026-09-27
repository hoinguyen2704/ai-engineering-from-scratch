# Capstone 05 — Tác nhân Nghiên cứu Tự hành (Lớp AI-Scientist)

> AI-Scientist-v2 của Sakana đã xuất bản các bài báo hoàn chỉnh. Agent Laboratory đã thực hiện các thí nghiệm. Allen AI đã chia sẻ các dấu vết (traces). Hình thái năm 2026 là tìm kiếm trên cây (tree search) theo kế hoạch-thực thi-xác minh cho các thí nghiệm, với ngân sách giới hạn, thực thi mã trong sandbox, trình viết LaTeX có phản hồi thị giác và một nhóm đánh giá tự động theo phong cách NeurIPS. Capstone này yêu cầu bạn xây dựng một hệ thống như vậy, chạy từ đầu đến cuối với chi phí dưới 30 đô la mỗi bài báo và vượt qua bài kiểm tra red team về thoát khỏi sandbox mà Sakana đã ghi lại.

**Type:** Capstone
**Languages:** Python (tác nhân + sandbox), LaTeX (đầu ra)
**Prerequisites:** Giai đoạn 2 (ML), Giai đoạn 3 (deep learning), Giai đoạn 7 (transformers), Giai đoạn 10 (LLMs từ đầu), Giai đoạn 14 (tác nhân), Giai đoạn 15 (tự hành), Giai đoạn 16 (đa tác nhân), Giai đoạn 18 (an toàn)
**Phases exercised:** P0 · P2 · P3 · P7 · P10 · P14 · P15 · P16 · P18
**Time:** 40 giờ

## Vấn đề

Các tác nhân nghiên cứu tự hành đã vượt qua một ngưỡng quan trọng vào năm 2026. AI-Scientist-v2 của Sakana AI đã được xuất bản trên Nature với các bài báo được tạo ra vượt qua vòng bình duyệt tại hội thảo. ShinkaEvolve (ICLR 2026) đã mở rộng hướng đi này sang việc phát triển các giả thuyết. Agent Laboratory của AMD đã cung cấp các dấu vết có thể tái lập. Các tác nhân này không phải là phép thuật — chúng là một vòng lặp kế hoạch-thực thi-xác minh chạy trên một cây các thí nghiệm ứng viên, với giới hạn chi phí, sandbox ràng buộc bằng seed và đánh giá tự động. Kỹ năng nằm ở vòng lặp, ngân sách và câu chuyện về an toàn.

Bạn sẽ học vòng lặp này bằng cách triển khai nó dựa trên một ý tưởng hạt giống trong một lĩnh vực hẹp (ví dụ: cắt tỉa độ thưa thớt của attention trên một transformer 100 triệu tham số). Giá trị không nằm ở việc khám phá ra điều gì đó mới trong lần chạy đầu tiên. Giá trị nằm ở cơ sở hạ tầng: tìm kiếm trên cây, sandbox thí nghiệm, vòng lặp viết-đánh giá, và báo cáo red-team. Nhóm Sakana đã ghi lại các lỗi thoát khỏi sandbox; tác nhân của bạn phải vượt qua cùng một bài kiểm tra red team đó.

## Khái niệm

Tác nhân là một thuật toán tìm kiếm trên cây theo kiểu best-first. Các nút là các đặc tả thí nghiệm: (giả thuyết, cấu hình, mã, kết quả mong đợi). Bước mở rộng đề xuất các nút con với những thay đổi nhỏ (đổi optimizer, thay đổi batch size, cắt tỉa một thành phần). Mỗi nút con chạy trong một sandbox mới với giới hạn tài nguyên nghiêm ngặt. Kết quả được phản hồi vào một hàm tính điểm để xếp hạng các nút theo (tính mới × chất lượng × ngân sách còn lại). Cây phát triển cho đến khi hết ngân sách, sau đó nhánh tốt nhất sẽ được viết thành bài báo.

Trình viết là đa phương thức. Nó tạo ra bản thảo LaTeX, biên dịch, hiển thị các hình ảnh và đưa tệp PDF đã hiển thị trở lại chế độ thị giác của Claude Opus 4.7 để phê bình về bố cục, độ rõ nét của hình ảnh và sự liên kết giữa tuyên bố và bằng chứng. Một nhóm gồm năm giám khảo LLM đưa ra điểm số theo phong cách NeurIPS (tính mới, độ chặt chẽ, độ rõ ràng, khả năng tái lập, tác động); nếu điểm trung bình giảm xuống dưới ngưỡng, bài báo sẽ quay lại trình viết kèm theo phê bình.

An toàn là yếu tố then chốt. Mỗi thí nghiệm chạy trong sandbox E2B hoặc Daytona không có kết nối mạng, giới hạn thời gian thực và giới hạn tài nguyên cố định. Bước tạo mã của tác nhân đi qua một lớp chính sách chặn các syscall có thể thoát khỏi sandbox. Báo cáo red-team tái lập bề mặt tấn công mà Sakana đã ghi lại (fork bomb, thoát khỏi hệ thống tệp, các lệnh gọi mạng do LLM viết).

## Kiến trúc

```
seed idea + domain
      |
      v
  literature search (Semantic Scholar + OpenAlex + FAISS cache)
      |
      v
  LangGraph plan-execute-verify tree
      |
      v
  +--- expand node ----+      per-node sandbox
  |                    |      (E2B / Daytona)
  v                    v      resource caps
  child_1           child_k   no network egress
  |                    |      deterministic seeds
  v                    v
  run experiment       run experiment
  |                    |
  v                    v
  score nodes by (novelty, quality, budget)
      |
      v
  best branch -> LaTeX writer
      |
      v
  compile + vision critique (Opus 4.7 vision)
      |
      v
  reviewer ensemble (5 LLM judges, NeurIPS rubric)
      |
      v
  paper.pdf + review.md + trace.json
```

## Stack

- Điều phối: LangGraph với checkpointing và các cổng phê duyệt của con người
- Tìm kiếm trên cây: best-first tùy chỉnh trên các nút thí nghiệm (kiểu AB-MCTS từ Sakana v2)
- Sandbox: E2B cho mỗi thí nghiệm, dự phòng Docker-in-Docker; giới hạn tài nguyên qua cgroups
- Tài liệu: Semantic Scholar Graph API + OpenAlex + bộ nhớ đệm FAISS cục bộ của các tóm tắt
- Trình viết: Mẫu LaTeX + Claude Opus 4.7 (chế độ thị giác) để phê bình hình ảnh và bố cục
- Người đánh giá: nhóm 5 giám khảo (Opus 4.7, GPT-5.4, Gemini 3 Pro, DeepSeek R1, Qwen3-Max) với tổng hợp có trọng số
- Khung thí nghiệm: PyTorch 2.5 cho các thí nghiệm vật lý, W&B để ghi nhật ký
- Khả năng quan sát: Langfuse cho các dấu vết tác nhân, ngân sách cứng 30 đô la mỗi bài báo

```figure
ce-experiment-tree
```

## Xây dựng

1. **Hạt giống và phạm vi lĩnh vực.** Chọn một ý tưởng hạt giống (ví dụ: "điều tra các mẫu thưa thớt trong bản đồ attention của các transformer dưới 1B tham số"). Xác định không gian tìm kiếm: mô hình, tập dữ liệu, ngân sách tính toán.

2. **Lướt qua tài liệu.** Truy vấn Semantic Scholar + OpenAlex cho 50 bài báo liên quan được trích dẫn nhiều nhất; lưu trữ tóm tắt cục bộ; tạo một bản tóm tắt lĩnh vực dài 1 trang.

3. **Dàn giáo cây.** Khởi tạo gốc với giả thuyết hạt giống. Triển khai `expand(node) -> children` với các đề xuất chỉnh sửa nhỏ (một thay đổi cấu hình mỗi nút con). Triển khai `score(node)` như một số hạng trọng số tính mới × chất lượng × ngân sách.

4. **Bao bọc Sandbox.** Mỗi thí nghiệm chạy `docker run --network=none --memory=8g --cpus=2 --pids-limit=256 --read-only` (hoặc chính sách E2B tương đương). Các seed được ghi vào sandbox; đầu ra được gắn ở chế độ chỉ đọc.

5. **Vòng lặp kế hoạch-thực thi-xác minh.** `plan` đề xuất các nút con. `execute` chạy sandbox, thu thập nhật ký và số liệu. `verify` chạy các kiểm tra đơn vị trên số liệu (loss có giảm không? việc cắt tỉa có cô lập được hiệu ứng không?). Các nút thất bại sẽ lưu lý do thất bại trên cây.

6. **Trình viết.** Sau khi hết ngân sách, chọn nhánh tốt nhất. Hiển thị hình ảnh bằng matplotlib. Tạo bản thảo LaTeX thông qua Claude Opus 4.7 với ngữ cảnh dấu vết của nhánh. Biên dịch. Đưa tệp PDF đã biên dịch trở lại Opus 4.7 vision để phê bình. Lặp lại.

7. **Nhóm đánh giá.** Năm giám khảo chấm điểm bản thảo dựa trên (tính mới, độ chặt chẽ, độ rõ ràng, khả năng tái lập, tác động) với các tiêu chí kiểu NeurIPS. Nếu trung bình < 4.0/5, quay lại trình viết kèm phê bình. Dừng lại sau 3 lần viết lại.

8. **Red team.** Xây dựng hoặc tích hợp một tập hợp các tác vụ đối kháng nhắm vào sandbox: fork bomb, nỗ lực exfiltration mạng, thoát khỏi hệ thống tệp, các ký tự đặc biệt shell do LLM viết. Xác nhận tất cả đều bị chặn. Viết báo cáo kết quả.

9. **Khả năng tái lập.** Mỗi bài báo đi kèm với JSON dấu vết tìm kiếm trên cây, các seed, liên kết chạy W&B, cấu hình sandbox và một README để tái lập từ đầu đến cuối.

## Sử dụng

```
$ ai-scientist run --seed "attention sparsity in sub-1B transformers" --budget 30
[lit]    50 papers, digest in 12s
[tree]   expanded 8 nodes, budget 12/30
[exec]   node #3 sparsity=top-8, loss=2.83 (best so far)
[exec]   node #6 sparsity=top-4, loss=3.12 (worse)
[exec]   ...
[tree]   chose branch rooted at node #3 (novelty 0.62, quality 0.81)
[write]  LaTeX draft v1 complete
[vision] critique: figure 2 legend too small, claim-evidence ok
[write]  draft v2 after 3 edits
[review] mean 4.2/5 (novelty 3.9, rigor 4.3, clarity 4.1, repro 4.5, impact 4.2)
[done]   paper.pdf + review.md + trace.json     $28.40 spent
```

## Giao hàng

`outputs/skill-ai-scientist.md` là sản phẩm bàn giao. Với một ý tưởng hạt giống + một lĩnh vực + ngân sách 30 đô la, nó chạy toàn bộ quy trình và tạo ra một bài báo có thể đánh giá được cùng với gói tái lập.

| Trọng số | Tiêu chí | Cách đo lường |
|:-:|---|---|
| 25 | Chất lượng bài báo | Đánh giá mù dựa trên tiêu chí so với các bài báo hội thảo đã xuất bản |
| 20 | Độ chặt chẽ thí nghiệm | Các baseline, seed, cắt tỉa; mọi tuyên bố đều được hỗ trợ bởi một ô trong bảng kết quả |
| 20 | Kỷ luật chi phí và tính toán | Giới hạn 30 đô la/bài báo, được theo dõi bởi Langfuse |
| 20 | An toàn | Vượt qua red team sandbox; chính sách mạng và công tắc ngắt được xác minh |
| 15 | Khả năng tái lập | Chạy lại bằng một lệnh với các seed giống hệt nhau để tái lập bài báo |
| **100** | | |

## Bài tập

1. Chạy quy trình với ba ý tưởng hạt giống khác nhau trong cùng một lĩnh vực. So sánh các phần nào của tìm kiếm trên cây bị trùng lặp. Xác định tính toán lãng phí do trùng lặp.

2. Thêm một cổng có sự tham gia của con người trước khi thực thi thí nghiệm cho các nút được ước tính trên 5 đô la. Đo lường tổng chi phí giảm bao nhiêu.

3. Thay thế nhóm đánh giá bằng một giám khảo duy nhất. Đo lường tỷ lệ chấp nhận sai trên một tập hợp các bài báo xấu đã biết.

4. Giới thiệu một bài kiểm tra red team về exfiltration mạng: tác nhân viết mã cố gắng `curl` một địa chỉ bên ngoài. Xác nhận chính sách `--network=none` chặn nó. Ghi lại nỗ lực này.

5. So sánh tìm kiếm trên cây của bạn với một baseline ngẫu nhiên phẳng (cùng ngân sách, không có chiến lược mở rộng). Báo cáo mức tăng tính mới × chất lượng.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|------------------------|
| Tìm kiếm trên cây | "Mở rộng kiểu AB-MCTS" | Khám phá best-first trên các nút thí nghiệm với điểm số tính mới×chất lượng×ngân sách |
| Sandbox | "Cô lập thí nghiệm" | Container không có mạng, giới hạn CPU/bộ nhớ, seed cố định, đầu vào chỉ đọc |
| Phê bình thị giác | "Render-then-read" | Biên dịch bài báo thành PDF, đưa PDF trở lại VLM để phê bình bố cục và tuyên bố-bằng chứng |
| Nhóm đánh giá | "Bình duyệt tự động" | Nhiều giám khảo LLM chấm điểm bài báo với tiêu chí NeurIPS; tổng hợp có trọng số để kiểm soát quy trình |
| Điểm tính mới | "Cái này có mới không?" | Heuristic phạt sự gần gũi với bộ nhớ đệm 50 bài báo tài liệu |
| Giới hạn chi phí | "$ budget" | Giới hạn cứng cho tổng chi tiêu mỗi bài báo; bộ đếm Langfuse + ước tính trước khi chạy |
| Red team | "Kiểm toán thoát sandbox" | Các tác vụ đối kháng sẽ thoát khỏi sandbox nếu chính sách bị sai |

## Đọc thêm

- [Kho lưu trữ Sakana AI-Scientist-v2](https://github.com/SakanaAI/AI-Scientist-v2) — tác nhân nghiên cứu sản xuất tham chiếu
- [Bài báo Sakana AI-Scientist-v1 (arXiv:2408.06292)](https://arxiv.org/abs/2408.06292) — phương pháp luận gốc
- [ShinkaEvolve (Sakana ICLR 2026)](https://sakana.ai) — mở rộng tiến hóa
- [Agent Laboratory (AMD)](https://github.com/SamuelSchmidgall/AgentLaboratory) — khung phòng thí nghiệm nghiên cứu đa vai trò
- [Tài liệu LangGraph](https://langchain-ai.github.io/langgraph/) — lớp điều phối tham chiếu
- [Semantic Scholar Graph API](https://api.semanticscholar.org/) — tìm kiếm tài liệu
- [E2B sandboxes](https://e2b.dev) — cô lập thí nghiệm tham chiếu
- [Hướng dẫn người đánh giá NeurIPS](https://neurips.cc/Conferences/2026/Reviewer-Guidelines) — tiêu chí mà nhóm đánh giá mã hóa