# CNNs và RNNs cho văn bản

> Các phép tích chập (convolutions) học các n-gram. Các mạng hồi quy (recurrences) ghi nhớ. Cả hai đều đã bị thay thế bởi cơ chế attention. Tuy nhiên, chúng vẫn quan trọng trên các phần cứng bị giới hạn tài nguyên.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 11 (PyTorch Intro), Phase 5 · 03 (Word Embeddings), Phase 4 · 02 (Convolutions from Scratch)
**Time:** ~75 phút

## Vấn đề

TF-IDF và Word2Vec tạo ra các vector phẳng bỏ qua thứ tự từ. Một bộ phân loại được xây dựng trên chúng không thể phân biệt được `dog bites man` với `man bites dog`. Thứ tự từ đôi khi mang lại tín hiệu quan trọng.

Hai họ kiến trúc đã lấp đầy khoảng trống đó trước khi các transformer xuất hiện.

**Mạng tích chập cho văn bản (TextCNN).** Áp dụng các phép tích chập 1D trên các chuỗi word embedding. Một bộ lọc (filter) có độ rộng 3 là một bộ phát hiện trigram có thể học được: nó bao phủ ba từ và xuất ra một điểm số. Xếp chồng các độ rộng khác nhau (2, 3, 4, 5) để phát hiện các mẫu đa quy mô. Max-pool về một biểu diễn có kích thước cố định. Phẳng, song song, nhanh.

**Mạng hồi quy (RNN, LSTM, GRU).** Xử lý các token từng cái một, duy trì một trạng thái ẩn (hidden state) mang thông tin về phía trước. Tuần tự, có khả năng ghi nhớ, độ dài đầu vào linh hoạt. Thống trị mô hình hóa chuỗi từ năm 2014 đến 2017, sau đó attention xuất hiện.

Bài học này xây dựng cả hai, sau đó chỉ ra những hạn chế đã thúc đẩy sự ra đời của attention.

## Khái niệm

**TextCNN** (Kim, 2014). Các token được nhúng (embedded). Một phép tích chập 1D với độ rộng `k` trượt bộ lọc qua các `k`-gram liên tiếp của các embedding, tạo ra một bản đồ đặc trưng (feature map). Global max-pooling trên bản đồ đó chọn ra kích hoạt mạnh nhất. Ghép (concatenate) các đầu ra max-pooled từ nhiều độ rộng bộ lọc khác nhau. Đưa vào một đầu phân loại (classifier head).

Tại sao nó hiệu quả. Một bộ lọc là một n-gram có thể học được. Max-pooling có tính bất biến vị trí, vì vậy "not good" sẽ kích hoạt cùng một đặc trưng dù nó nằm ở đầu hay giữa câu đánh giá. Ba độ rộng bộ lọc với 100 bộ lọc mỗi loại mang lại cho bạn 300 bộ phát hiện n-gram đã học. Quá trình huấn luyện diễn ra song song; không có sự phụ thuộc tuần tự.

**RNN.** Tại mỗi bước thời gian `t`, trạng thái ẩn `h_t = f(W * x_t + U * h_{t-1} + b)`. Chia sẻ `W`, `U`, `b` qua thời gian. Trạng thái ẩn tại thời điểm `T` là bản tóm tắt của toàn bộ tiền tố. Để phân loại, thực hiện pooling trên `h_1 ... h_T` (max, mean, hoặc last).

Các RNN đơn thuần gặp vấn đề triệt tiêu gradient (vanishing gradients). **LSTM** bổ sung các cổng (gates) quyết định những gì cần quên, lưu trữ và xuất ra, giúp ổn định gradient qua các chuỗi dài. **GRU** đơn giản hóa LSTM thành hai cổng; hiệu suất tương đương với ít tham số hơn.

**Bidirectional RNNs** chạy một RNN theo chiều xuôi và một RNN theo chiều ngược, sau đó ghép các trạng thái ẩn lại với nhau. Biểu diễn của mỗi token sẽ thấy được cả ngữ cảnh bên trái và bên phải. Rất cần thiết cho các tác vụ gắn nhãn.

```figure
rnn-unroll
```

## Xây dựng

### Bước 1: TextCNN trong PyTorch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_classes, filter_widths=(2, 3, 4), n_filters=64, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, n_filters, kernel_size=k)
            for k in filter_widths
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(x))
            p = F.max_pool1d(c, c.size(2)).squeeze(2)
            pooled.append(p)
        h = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(h))
```

`transpose(1, 2)` định hình lại `[batch, seq_len, embed_dim]` thành `[batch, embed_dim, seq_len]` vì `nn.Conv1d` coi trục giữa là các kênh (channels). Đầu ra sau pooling có kích thước cố định bất kể độ dài đầu vào.

### Bước 2: Bộ phân loại LSTM

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes, bidirectional=True, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        factor = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * factor, n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids)
        out, _ = self.lstm(x)
        pooled = out.max(dim=1).values
        return self.fc(self.dropout(pooled))
```

Max-pool trên toàn bộ chuỗi, thay vì chỉ lấy trạng thái cuối cùng. Đối với phân loại, max-pooling thường tốt hơn việc lấy trạng thái ẩn cuối cùng vì thông tin ở cuối một chuỗi dài có xu hướng lấn át trạng thái cuối.

### Bước 3: Demo về triệt tiêu gradient (trực giác)

Một RNN đơn thuần không có cổng không thể học được các phụ thuộc tầm xa. Hãy xem xét một tác vụ đơn giản: dự đoán xem token `A` có xuất hiện ở bất kỳ đâu trong chuỗi hay không. Nếu `A` nằm ở vị trí 1 và chuỗi dài 100 token, gradient từ hàm mất mát phải truyền ngược qua 99 phép nhân của trọng số hồi quy. Nếu trọng số nhỏ hơn 1, gradient sẽ triệt tiêu. Nếu lớn hơn 1, nó sẽ bùng nổ.

```python
def vanishing_gradient_sim(seq_len, recurrent_weight=0.9):
    import math
    return math.pow(recurrent_weight, seq_len)


# At weight=0.9 over 100 steps:
#   0.9 ^ 100 ≈ 2.7e-5
# The gradient from step 100 to step 1 is effectively zero.
```

LSTM khắc phục điều này bằng một **cell state** chạy xuyên suốt mạng với các tương tác cộng (cổng quên điều chỉnh nó theo phép nhân, nhưng gradient vẫn chảy dọc theo "đường cao tốc"). GRU thực hiện điều tương tự với ít tham số hơn. Cả hai đều giúp bạn huấn luyện ổn định qua các chuỗi dài hơn 100 bước.

### Bước 4: Tại sao điều này vẫn chưa đủ

Ba vấn đề vẫn tồn tại ngay cả với LSTM.

1. **Nút thắt tuần tự.** Huấn luyện một RNN trên chuỗi dài 1000 yêu cầu 1000 bước lan truyền tiến/lùi nối tiếp. Không thể song song hóa theo thời gian.
2. **Vector ngữ cảnh kích thước cố định trong thiết lập encoder-decoder.** Decoder chỉ thấy trạng thái ẩn cuối cùng của encoder, vốn đã bị nén từ toàn bộ đầu vào. Các đầu vào dài sẽ mất chi tiết. Bài học 09 đề cập trực tiếp vấn đề này.
3. **Giới hạn độ chính xác với các phụ thuộc xa.** LSTM vượt trội hơn RNN đơn thuần nhưng vẫn gặp khó khăn trong việc truyền tải thông tin cụ thể qua hơn 200 bước.

Attention đã giải quyết cả ba vấn đề trên. Transformers đã loại bỏ hoàn toàn tính hồi quy. Bài học 10 là bước ngoặt.

## Sử dụng

`nn.LSTM`, `nn.GRU`, và `nn.Conv1d` của PyTorch đã sẵn sàng cho sản xuất. Mã huấn luyện là tiêu chuẩn.

Hugging Face cung cấp các embedding đã được huấn luyện sẵn mà bạn có thể cắm vào làm lớp đầu vào:

```python
from transformers import AutoModel

encoder = AutoModel.from_pretrained("bert-base-uncased")
for param in encoder.parameters():
    param.requires_grad = False


class BertCNN(nn.Module):
    def __init__(self, n_classes, filter_widths=(2, 3, 4), n_filters=64):
        super().__init__()
        self.encoder = encoder
        self.convs = nn.ModuleList([nn.Conv1d(768, n_filters, kernel_size=k) for k in filter_widths])
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = out.transpose(1, 2)
        pooled = [F.max_pool1d(F.relu(conv(x)), kernel_size=conv(x).size(2)).squeeze(2) for conv in self.convs]
        return self.fc(torch.cat(pooled, dim=1))
```

Danh sách kiểm tra khi nào nên sử dụng:

- **Edge / suy luận trên thiết bị.** TextCNN với GloVe embedding nhỏ hơn 10-100 lần so với transformer. Nếu mục tiêu triển khai của bạn là điện thoại, đây là lựa chọn phù hợp.
- **Phân loại trực tuyến / streaming.** RNN xử lý từng token một; transformer cần toàn bộ chuỗi. Đối với văn bản đến theo thời gian thực, LSTM vẫn thắng thế.
- **Mô hình nhỏ cho baseline.** Lặp lại nhanh trên một tác vụ mới. Huấn luyện TextCNN trong 5 phút trên CPU.
- **Gắn nhãn chuỗi với dữ liệu hạn chế.** BiLSTM-CRF (bài học 06) vẫn là kiến trúc NER cấp sản xuất cho 1k-10k câu được gắn nhãn.

Mọi thứ khác hãy dùng transformer.

## Triển khai

Lưu dưới dạng `outputs/prompt-text-encoder-picker.md`:

```markdown
---
name: text-encoder-picker
description: Pick a text encoder architecture for a given constraint set.
phase: 5
lesson: 08
---

Given constraints (task, data volume, latency budget, deploy target, compute budget), output:

1. Encoder architecture: TextCNN, BiLSTM, BiLSTM-CRF, transformer fine-tune, or "use a pretrained transformer as a frozen encoder + small head".
2. Embedding input: random init, GloVe / fastText frozen, or contextualized transformer embeddings.
3. Training recipe in 5 lines: optimizer, learning rate, batch size, epochs, regularization.
4. One monitoring signal. For RNN/CNN models: attention mechanism absence means they miss long-range deps; check per-length accuracy. For transformers: fine-tuning collapse if LR too high; check train loss.

Refuse to recommend fine-tuning a transformer when data is under ~500 labeled examples without showing that a TextCNN / BiLSTM baseline has plateaued. Flag edge deployment as needing architecture-before-everything.
```

## Bài tập

1. **Dễ.** Huấn luyện một TextCNN trên tập dữ liệu đồ chơi 3 lớp (bạn tự tạo dữ liệu). Xác minh rằng các độ rộng bộ lọc (2, 3, 4) trung bình cho kết quả F1 tốt hơn so với một độ rộng duy nhất (3).
2. **Trung bình.** Triển khai max-pool, mean-pool, và last-state pooling cho bộ phân loại LSTM. So sánh trên một tập dữ liệu nhỏ; ghi lại phương pháp pooling nào thắng và đưa ra giả thuyết tại sao.
3. **Khó.** Xây dựng một bộ gắn nhãn BiLSTM-CRF NER (kết hợp bài học 06 và bài này). Huấn luyện trên CoNLL-2003. So sánh với baseline CRF đơn thuần từ bài học 06 và với một mô hình BERT đã fine-tune. Báo cáo thời gian huấn luyện, bộ nhớ và F1.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| TextCNN | CNN cho văn bản | Chồng các phép tích chập 1D trên word embedding với global max-pool. Kim (2014). |
| RNN | Mạng hồi quy | Trạng thái ẩn được cập nhật tại mỗi bước thời gian: `h_t = f(W x_t + U h_{t-1})`. |
| LSTM | Gated RNN | Thêm các cổng input / forget / output + một cell state. Huấn luyện ổn định qua các chuỗi dài. |
| GRU | LSTM đơn giản hơn | Hai cổng thay vì ba. Độ chính xác tương đương, ít tham số hơn. |
| Bidirectional | Cả hai chiều | RNN xuôi + ngược được ghép lại. Mỗi token thấy được cả hai phía của ngữ cảnh. |
| Vanishing gradient | Tín hiệu huấn luyện bị mất | Việc nhân lặp đi lặp lại với các trọng số <1 trong RNN đơn thuần khiến gradient ở các bước đầu gần như bằng 0. |

## Đọc thêm

- [Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) — bài báo về TextCNN. Tám trang. Dễ đọc.
- [Hochreiter, S. và Schmidhuber, J. (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) — bài báo về LSTM. Rõ ràng một cách bất ngờ.
- [Olah, C. (2015). Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) — các sơ đồ giúp mọi người tiếp cận LSTM dễ dàng hơn.