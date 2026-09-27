# Society of Mind và Multi-Agent Debate

> Tiền đề năm 1986 của Minsky — trí tuệ là một xã hội của các chuyên gia — cứ mỗi thập kỷ lại được tái khám phá. Năm 2023, Du và cộng sự đã biến nó thành một thuật toán cụ thể: nhiều instance LLM cùng đề xuất câu trả lời, đọc câu trả lời của nhau, phê bình và cập nhật. Qua N vòng, chúng hội tụ về một sự đồng thuận vượt trội hơn so với zero-shot CoT và reflection trên sáu tác vụ suy luận và kiểm chứng thực tế. Hai phát hiện quan trọng: cả **nhiều tác nhân (multiple agents)** và **nhiều vòng (multiple rounds)** đều đóng góp độc lập. Xã hội này vượt trội hơn so với độc thoại của một tác nhân duy nhất; sự trao đổi qua nhiều vòng vượt trội hơn so với bỏ phiếu một lần (one-shot voting).

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 04 (Primitive Model)
**Time:** ~60 minutes

## Vấn đề

Self-consistency — lấy mẫu một mô hình nhiều lần và chọn câu trả lời chiếm đa số — là cải tiến suy luận rẻ nhất mà bạn có thể áp dụng. Nó hiệu quả, nhưng nhanh chóng đạt ngưỡng bão hòa. Bạn có thể tăng gấp đôi số lượng mẫu mà không thấy thêm bước nhảy vọt đáng kể nào.

Debate (tranh luận) phá vỡ sự bão hòa đó. Thay vì N mẫu độc lập từ một mô hình, N tác nhân đọc suy luận của nhau và sửa đổi. Mối tương quan giữa các mẫu giảm xuống (chúng không còn là i.i.d. nữa), và điểm hội tụ thường là kết quả đúng trong khi bỏ phiếu i.i.d. lại sai một cách đầy tự tin.

## Khái niệm

### Thuật toán của Du và cộng sự (2023)

Từ arXiv:2305.14325 (ICML 2024):

1. Mỗi tác nhân trong số N tác nhân đưa ra câu trả lời ban đầu cho câu hỏi.
2. Với vòng r = 2..R: mỗi tác nhân được xem câu trả lời ở vòng r-1 của các tác nhân khác và được yêu cầu "xem xét những điều này, hãy đưa ra câu trả lời cập nhật của bạn."
3. Sau R vòng, thực hiện bỏ phiếu đa số cho các câu trả lời cuối cùng.

Bài báo thử nghiệm trên các benchmark MMLU, GSM8K, tiểu sử, MATH và kiểm chứng thực tế. Debate liên tục vượt qua CoT và Self-Reflection.

### Hai núm điều chỉnh độc lập

Các thử nghiệm cắt bỏ (ablations) từ cùng bài báo:

- **Chỉ số lượng tác nhân** (1 vòng, bỏ phiếu đa số của N) vượt trội hơn tác nhân đơn lẻ trên hầu hết các tác vụ, nhưng bị chững lại.
- **Chỉ số vòng** (1 tác nhân nhìn thấy suy luận trước đó của chính nó) hầu như không giúp ích — điểm yếu đã biết của reflection.
- **Cả hai kết hợp lại** tạo ra những bước nhảy vọt lớn. Sự trao đổi qua nhiều vòng giữa nhiều tác nhân thúc đẩy sự gia tăng hiệu suất.

### Tại sao nó hiệu quả

Hai cơ chế:

1. **Tiếp xúc với sự bất đồng.** Khi một tác nhân nhìn thấy chuỗi suy luận của tác nhân khác với kết luận khác biệt, nó buộc phải biện minh hoặc cập nhật. Dù thế nào, ngữ cảnh cho vòng r+1 cũng phong phú hơn vòng r.
2. **Giảm lỗi tương quan.** Trong self-consistency, tất cả các mẫu đều đến từ cùng một mô hình, vì vậy các lỗi tương quan với nhau — bạn lấy trung bình thành một câu trả lời sai một cách tự tin. Các mô hình khác nhau hoặc các seed khác nhau sẽ làm giảm tương quan. Các *quan điểm tranh luận* khác nhau còn làm giảm tương quan mạnh hơn nữa.

### Tranh luận không đồng nhất (Heterogeneous debate)

A-HMAD và các nghiên cứu tiếp theo sử dụng *các mô hình cơ sở khác nhau* cho các tác nhân khác nhau. Llama + Claude + GPT tranh luận giúp giảm thiểu sự sụp đổ do độc canh (Bài học 26) vì các lỗi tương quan của một họ mô hình không được chia sẻ bởi các họ khác.

Nhược điểm: một mô hình yếu tham gia tranh luận có thể kéo sự đồng thuận về phía câu trả lời sai của nó (xem "Should we be going MAD?", arXiv:2311.17371).

### NLSOM — phần mở rộng 129 tác nhân

Zhuge và cộng sự ("Mindstorms in Natural Language-Based Societies of Mind," arXiv:2305.17066) đã mở rộng ý tưởng này lên các xã hội 129 thành viên. Kết quả: sự chuyên môn hóa và tự tổ chức xuất hiện khi quy mô tăng lên, và hệ thống vượt trội hơn tác nhân đơn lẻ trên các tác vụ như trả lời câu hỏi bằng hình ảnh (visual question answering).

### Các chế độ thất bại

- **Sycophancy cascade (Hiệu ứng nịnh hót).** Tất cả các tác nhân đều nghe theo tác nhân nào nghe có vẻ tự tin nhất. Cuộc tranh luận sụp đổ theo tiếng nói lớn nhất. Việc nhắc nhở (prompting) cho các vai trò đối kháng ("một tác nhân phải lập luận cho vị trí ngược lại") sẽ giúp ích.
- **Topic drift (Lệch chủ đề).** Các cuộc tranh luận qua nhiều vòng bị lệch khỏi câu hỏi ban đầu. Cách giảm thiểu: chèn lại câu hỏi vào mỗi vòng.
- **Bùng nổ tính toán.** N tác nhân × R vòng = N·R cuộc gọi LLM, mỗi cuộc gọi với ngữ cảnh ngày càng tăng. Một cuộc tranh luận 5 tác nhân, 5 vòng là 25 cuộc gọi với ngữ cảnh ngày càng lớn. Chi phí cho mỗi câu hỏi có thể vượt quá 10 lần một cuộc gọi CoT đơn lẻ.

```figure
multi-agent-debate
```

## Xây dựng

`code/main.py` chạy một cuộc tranh luận 3 tác nhân × 3 vòng về một câu hỏi toán học, nơi mỗi tác nhân bắt đầu với một câu trả lời khác nhau (có thể sai). Các tác nhân được lập trình sẵn — mỗi tác nhân "cập nhật" bằng cách lấy trung bình câu trả lời của các tác nhân lân cận, có trọng số theo độ tự tin được lập trình. Sự hội tụ có thể thấy rõ trong nhật ký từng vòng.

Bản demo cho thấy hai hiệu ứng chính:

- Một vòng trao đổi duy nhất đưa các tác nhân đến gần câu trả lời đúng hơn.
- Các vòng bổ sung sau vòng 2 cho thấy lợi nhuận giảm dần (khớp với sự chững lại của Du và cộng sự).

Chạy:

```
python3 code/main.py
```

## Sử dụng

`outputs/skill-debate-configurator.md` cấu hình một cuộc tranh luận cho một tác vụ mới: số lượng tác nhân, số vòng, tính không đồng nhất (cùng mô hình so với hỗn hợp), phân công vai trò (đối xứng so với một đối kháng). Nó cũng ước tính chi phí token trước khi bạn chạy.

## Triển khai

Nếu bạn triển khai debate:

- **Giới hạn vòng ở mức 3.** Du và cộng sự cho thấy 3 vòng thu được hầu hết lợi ích. Nhiều hơn là chi phí, không phải chất lượng.
- **Giới hạn tác nhân ở mức 5.** Ngoài 5, sự phình to ngữ cảnh và chi phí sẽ chiếm ưu thế.
- **Mặc định không đồng nhất.** Ít nhất hai mô hình cơ sở khác nhau trong nhóm.
- **Vị trí đối kháng.** Một tác nhân được nhắc nhở để bất đồng bất kể điều gì. Phá vỡ sự nịnh hót.
- **Ghi nhật ký mọi vòng.** Các hệ thống tranh luận ẩn các vòng trung gian không thể được gỡ lỗi hoặc kiểm toán.

## Bài tập

1. Chạy `code/main.py`, sau đó đặt số vòng là 5 và quan sát lợi nhuận giảm dần. Tại vòng nào thì sự hội tụ bổ sung dừng lại?
2. Thêm tác nhân thứ tư với vai trò đối kháng: luôn bất đồng với đa số hiện tại. Điều này phá vỡ hay cải thiện sự hội tụ?
3. Vẽ (in) điểm số đồng thuận mỗi vòng (tỷ lệ tác nhân chọn câu trả lời đa số). Khi nào nó đạt 1.0 và điều đó có tương đương với "đúng" không?
4. Đọc các thử nghiệm cắt bỏ trong Phần 4 của Du và cộng sự. Tái tạo kết quả "chỉ tác nhân" so với "chỉ vòng" so với "cả hai" bằng mã này.
5. Đọc "Should we be going MAD?" (arXiv:2311.17371) và liệt kê hai biến thể tranh luận ngoài round-robin — ví dụ: do giám khảo dẫn dắt, chuỗi tranh luận, đối kháng.

## Thuật ngữ chính

| Thuật ngữ | Mọi người nói gì | Ý nghĩa thực sự |
|------|----------------|------------------------|
| Society of Mind | "Ý tưởng của Minsky" | Trí tuệ như các chuyên gia tương tác; khung năm 1986 hiện được vận hành thông qua tranh luận LLM. |
| Multi-agent debate | "Các tác nhân tranh luận" | N tác nhân đề xuất, phê bình lẫn nhau, sửa đổi qua R vòng, bỏ phiếu đa số. |
| Consensus | "Chúng đồng ý" | Không phải chân lý nhận thức — chỉ là tỷ lệ chọn câu trả lời đa số. Có thể sai một cách tự tin. |
| Rounds | "Các bước trao đổi" | Một vòng = mỗi tác nhân đọc các tác nhân khác và cập nhật một lần. |
| Heterogeneous debate | "Trộn các họ mô hình" | Sử dụng các mô hình cơ sở khác nhau để giảm tương quan lỗi. |
| Sycophancy cascade | "Mọi người đồng ý với kẻ lớn tiếng" | Thất bại tranh luận khi các tác nhân nghe theo tác nhân tự tin nhất bất kể tính đúng đắn. |
| NLSOM | "Xã hội 129 tác nhân" | Xã hội trí tuệ dựa trên ngôn ngữ tự nhiên; phiên bản mở rộng của Zhuge và cộng sự. |
| Correlated error | "Cùng mô hình, cùng lỗi" | Tại sao self-consistency bão hòa; tranh luận qua các quan điểm khác nhau sẽ làm giảm tương quan. |

## Đọc thêm

- [Du và cộng sự — Improving Factuality and Reasoning in Language Models through Multiagent Debate](https://arxiv.org/abs/2305.14325) — bài báo tham khảo, ICML 2024
- [Zhuge và cộng sự — Mindstorms in Natural Language-Based Societies of Mind](https://arxiv.org/abs/2305.17066) — NLSOM 129 tác nhân
- [Should we be going MAD? A Look at Multi-Agent Debate Strategies for LLMs](https://arxiv.org/abs/2311.17371) — benchmark các biến thể tranh luận
- [Trang dự án Debate](https://composable-models.github.io/llm_debate/) — mã nguồn, demo và chi tiết thử nghiệm của Du và cộng sự