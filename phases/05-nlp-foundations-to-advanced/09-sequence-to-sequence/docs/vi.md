# Các mô hình Sequence-to-Sequence

> Hai RNN đang đóng vai một phiên dịch viên. Điểm nghẽn mà chúng gặp phải chính là lý do tại sao cơ chế attention ra đời.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 08 (CNNs + RNNs cho văn bản), Phase 3 · 11 (Giới thiệu về PyTorch)
**Time:** ~75 phút

## Vấn đề

Phân loại (Classification) ánh xạ một chuỗi có độ dài biến thiên thành một nhãn duy nhất. Dịch thuật ánh xạ một chuỗi có độ dài biến thiên thành một chuỗi có độ dài biến thiên khác. Đầu vào và đầu ra nằm trong các từ vựng khác nhau, có thể là các ngôn ngữ khác nhau, và không có sự đảm bảo về độ dài tương đương.

Kiến trúc seq2seq (Sutskever, Vinyals, Le, 2014) đã giải quyết vấn đề này bằng một công thức đơn giản một cách có chủ đích. Hai RNN. Một cái đọc câu nguồn và tạo ra một vector ngữ cảnh (context vector) có kích thước cố định. Cái còn lại đọc vector đó và tạo ra câu đích theo từng token một. Vẫn là mã nguồn bạn đã viết cho bài 08, chỉ là được kết nối theo cách khác.

Nội dung này đáng để nghiên cứu vì hai lý do. Thứ nhất, điểm nghẽn của vector ngữ cảnh là thất bại mang tính sư phạm hữu ích nhất trong NLP. Nó thúc đẩy mọi thứ mà attention và transformer làm tốt. Thứ hai, công thức huấn luyện (teacher forcing, scheduled sampling, beam search khi suy luận) vẫn áp dụng cho mọi hệ thống tạo văn bản hiện đại, bao gồm cả LLM.

## Khái niệm

**Encoder.** Một RNN đọc câu nguồn. Trạng thái ẩn cuối cùng của nó là **vector ngữ cảnh** — một bản tóm tắt có kích thước cố định của toàn bộ đầu vào. Về lý thuyết, nó không làm mất thông tin gì ngoài nguồn.

**Decoder.** Một RNN khác được khởi tạo từ vector ngữ cảnh. Tại mỗi bước, nó lấy token được tạo ra trước đó làm đầu vào và tạo ra một phân phối trên từ vựng đích. Sử dụng lấy mẫu (sample) hoặc argmax để chọn token tiếp theo. Đưa nó ngược lại vào. Lặp lại cho đến khi token `<EOS>` được tạo ra hoặc đạt đến độ dài tối đa.

**Huấn luyện:** Hàm mất mát Cross-entropy tại mỗi bước của decoder, được cộng dồn trên toàn bộ chuỗi. Backprop qua thời gian (backprop through time) tiêu chuẩn qua cả hai mạng.

**Teacher forcing.** Trong quá trình huấn luyện, đầu vào của decoder tại bước `t` là token *đúng thực tế (ground-truth)* tại vị trí `t-1`, chứ không phải dự đoán trước đó của chính decoder. Điều này giúp ổn định quá trình huấn luyện; nếu không có nó, các sai lầm ban đầu sẽ tích tụ và mô hình không bao giờ học được. Khi suy luận, bạn buộc phải sử dụng các dự đoán của chính mô hình, vì vậy luôn có một khoảng cách phân phối giữa huấn luyện và suy luận. Khoảng cách đó được gọi là **exposure bias**.

**Điểm nghẽn.** Mọi thứ encoder học được về nguồn phải được ép vào một vector ngữ cảnh duy nhất đó. Các câu dài sẽ mất chi tiết. Các từ hiếm bị làm mờ. Việc sắp xếp lại (ví dụ: "chat noir" vs "black cat") phải được ghi nhớ thay vì tính toán.

Attention (bài 10) khắc phục điều này bằng cách cho phép decoder nhìn vào *mọi* trạng thái ẩn của encoder, thay vì chỉ trạng thái cuối cùng. Đó chính là toàn bộ ý tưởng cốt lõi.

```figure
lstm-gates
```

## Xây dựng

### Bước 1: một encoder

```python
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(src_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, src):
        e = self.embed(src)
        outputs, hidden = self.gru(e)
        return outputs, hidden
```

`outputs` có hình dạng `[batch, seq_len, hidden_dim]` — một trạng thái ẩn cho mỗi vị trí đầu vào. `hidden` có hình dạng `[1, batch, hidden_dim]` — bước cuối cùng. Bài 08 đã nói "pool qua các đầu ra để phân loại". Ở đây, chúng ta giữ lại trạng thái ẩn cuối cùng làm vector ngữ cảnh và bỏ qua các đầu ra theo từng bước.

### Bước 2: một decoder

```python
class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tgt_vocab_size)

    def forward(self, token, hidden):
        e = self.embed(token)
        out, hidden = self.gru(e, hidden)
        logits = self.fc(out)
        return logits, hidden
```

Decoder được gọi từng bước một. Đầu vào: một batch gồm các token đơn lẻ và trạng thái ẩn hiện tại. Đầu ra: các logit từ vựng cho token tiếp theo và trạng thái ẩn đã cập nhật.

### Bước 3: vòng lặp huấn luyện với teacher forcing

```python
def train_batch(encoder, decoder, src, tgt, bos_id, optimizer, teacher_forcing_ratio=0.9):
    optimizer.zero_grad()
    _, hidden = encoder(src)
    batch_size, tgt_len = tgt.shape
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    loss = 0.0
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for t in range(tgt_len):
        logits, hidden = decoder(input_token, hidden)
        step_loss = loss_fn(logits.squeeze(1), tgt[:, t])
        loss += step_loss
        use_teacher = torch.rand(1).item() < teacher_forcing_ratio
        if use_teacher:
            input_token = tgt[:, t].unsqueeze(1)
        else:
            input_token = logits.argmax(dim=-1)

    loss.backward()
    optimizer.step()
    return loss.item() / tgt_len
```

Có hai tham số cần lưu ý. `ignore_index=0` bỏ qua mất mát trên các token padding. `teacher_forcing_ratio` là xác suất sử dụng token đúng thay vì dự đoán của mô hình tại mỗi bước. Bắt đầu ở mức 1.0 (teacher forcing hoàn toàn) và giảm dần xuống ~0.5 trong quá trình huấn luyện để thu hẹp khoảng cách exposure-bias.

### Bước 4: vòng lặp suy luận (greedy)

```python
@torch.no_grad()
def greedy_decode(encoder, decoder, src, bos_id, eos_id, max_len=50):
    _, hidden = encoder(src)
    batch_size = src.shape[0]
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    output_ids = []
    for _ in range(max_len):
        logits, hidden = decoder(input_token, hidden)
        next_token = logits.argmax(dim=-1)
        output_ids.append(next_token)
        input_token = next_token
        if (next_token == eos_id).all():
            break
    return torch.cat(output_ids, dim=1)
```

Greedy decoding chọn token có xác suất cao nhất tại mỗi bước. Nó có thể đi chệch hướng: một khi bạn đã chọn một token, bạn không thể rút lại. **Beam search** giữ lại top-`k` các chuỗi con tiềm năng và chọn chuỗi hoàn chỉnh có điểm số cao nhất ở cuối. Beam width từ 3-5 là tiêu chuẩn.

### Bước 5: minh chứng cho điểm nghẽn

Huấn luyện mô hình trên một tác vụ sao chép đơn giản: nguồn `[a, b, c, d, e]`, đích `[a, b, c, d, e]`. Tăng độ dài chuỗi. Quan sát độ chính xác.

```
seq_len=5   copy accuracy: 98%
seq_len=10  copy accuracy: 91%
seq_len=20  copy accuracy: 62%
seq_len=40  copy accuracy: 23%
```

Một trạng thái ẩn GRU duy nhất không thể ghi nhớ không mất mát một đầu vào dài 40 token. Thông tin vẫn tồn tại ở mỗi bước của encoder, nhưng decoder chỉ nhìn thấy trạng thái cuối cùng. Attention khắc phục trực tiếp điều này.

## Sử dụng

PyTorch có các template seq2seq dựa trên `nn.Transformer` và `nn.LSTM`. Thư viện `transformers` của Hugging Face cung cấp các mô hình encoder-decoder hoàn chỉnh (BART, T5, mBART, NLLB) được huấn luyện trên hàng tỷ token.

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("facebook/bart-base")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")

src = tok("Translate this to French: Hello, how are you?", return_tensors="pt")
out = model.generate(**src, max_new_tokens=50, num_beams=4)
print(tok.decode(out[0], skip_special_tokens=True))
```

Các mô hình encoder-decoder hiện đại đã thay thế RNN bằng transformer. Hình thái cấp cao (encoder, decoder, tạo token từng bước một) giống hệt với bài báo seq2seq năm 2014. Cơ chế bên trong mỗi khối là khác nhau.

### Khi nào vẫn nên dùng seq2seq dựa trên RNN

Gần như không bao giờ, đối với các dự án mới. Các ngoại lệ cụ thể:

- Dịch thuật trực tuyến (streaming) nơi bạn tiêu thụ đầu vào từng token một với bộ nhớ giới hạn.
- Tạo văn bản trên thiết bị (on-device) nơi chi phí bộ nhớ của transformer là quá lớn.
- Sư phạm. Hiểu về điểm nghẽn encoder-decoder là con đường nhanh nhất để hiểu tại sao transformer chiến thắng.

### Exposure bias và các biện pháp giảm thiểu

- **Scheduled sampling.** Giảm dần tỷ lệ teacher forcing trong quá trình huấn luyện để mô hình học cách phục hồi từ những sai lầm của chính nó.
- **Minimum risk training.** Huấn luyện dựa trên điểm BLEU cấp câu thay vì cross-entropy cấp token. Gần hơn với những gì bạn thực sự muốn.
- **Reinforcement learning fine-tuning.** Thưởng cho trình tạo chuỗi bằng một chỉ số. Được sử dụng trong RLHF của LLM hiện đại.

Cả ba phương pháp này vẫn áp dụng cho việc tạo văn bản dựa trên transformer.

## Triển khai

Lưu dưới dạng `outputs/prompt-seq2seq-design.md`:

```markdown
---
name: seq2seq-design
description: Design a sequence-to-sequence pipeline for a given task.
phase: 5
lesson: 09
---

Given a task (translation, summarization, paraphrase, question rewrite), output:

1. Architecture. Pretrained transformer encoder-decoder (BART, T5, mBART, NLLB) is the default. RNN-based seq2seq only for specific constraints.
2. Starting checkpoint. Name it (`facebook/bart-base`, `google/flan-t5-base`, `facebook/nllb-200-distilled-600M`). Match the checkpoint to task and language coverage.
3. Decoding strategy. Greedy for deterministic output, beam search (width 4-5) for quality, sampling with temperature for diversity. One sentence justification.
4. One failure mode to verify before shipping. Exposure bias manifests as generation drift on longer outputs; sample 20 outputs at the 90th-percentile length and eyeball.

Refuse to recommend training a seq2seq from scratch for under a million parallel examples. Flag any pipeline that uses greedy decoding for user-facing content as fragile (greedy repeats and loops).
```

## Bài tập

1. **Dễ.** Triển khai tác vụ sao chép đơn giản. Huấn luyện một GRU seq2seq trên các cặp đầu vào-đầu ra nơi đích bằng nguồn. Đo độ chính xác ở độ dài 5, 10, 20. Tái hiện lại điểm nghẽn.
2. **Trung bình.** Thêm beam search decoding với beam width là 3. Đo BLEU trên một tập dữ liệu song song nhỏ so với greedy. Ghi lại những trường hợp beam search thắng thế (thường là các token cuối) và những nơi nó không tạo ra sự khác biệt.
3. **Khó.** Fine-tune `facebook/bart-base` trên tập dữ liệu diễn giải (paraphrase) gồm 10k cặp. So sánh đầu ra beam-4 của mô hình đã fine-tune với mô hình cơ sở trên các đầu vào chưa từng thấy. Báo cáo điểm BLEU và chọn 10 ví dụ định tính.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| Encoder | Input RNN | Đọc nguồn. Tạo ra các trạng thái ẩn theo từng bước và một vector ngữ cảnh cuối cùng. |
| Decoder | Output RNN | Khởi tạo từ vector ngữ cảnh. Tạo các token đích từng cái một. |
| Context vector | Bản tóm tắt | Trạng thái ẩn cuối cùng của encoder. Kích thước cố định. Điểm nghẽn mà attention giải quyết. |
| Teacher forcing | Sử dụng token đúng | Đưa token đúng thực tế trước đó vào tại thời điểm huấn luyện. Giúp ổn định việc học. |
| Exposure bias | Khoảng cách train/test | Mô hình được huấn luyện trên token đúng nên không bao giờ luyện tập việc phục hồi từ sai lầm của chính nó. |
| Beam search | Giải mã tốt hơn | Giữ lại top-k chuỗi con tiềm năng tại mỗi bước thay vì chọn greedy. |

## Đọc thêm

- [Sutskever, Vinyals, Le (2014). Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) — bài báo seq2seq gốc. Bốn trang.
- [Cho et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) — giới thiệu GRU và khung encoder-decoder.
- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — bài báo về attention. Hãy đọc ngay sau bài học này.
- [Hướng dẫn PyTorch NLP from Scratch](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) — mã nguồn seq2seq + attention có thể xây dựng được.