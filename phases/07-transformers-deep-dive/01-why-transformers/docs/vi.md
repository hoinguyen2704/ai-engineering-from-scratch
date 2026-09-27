# Tại sao lại là Transformers — Những vấn đề của RNN

> RNN xử lý từng token một. Transformers xử lý tất cả các token cùng một lúc. Đặt cược kiến trúc đơn lẻ đó đã thay đổi mọi đường cong mở rộng (scaling curve) trong deep learning sau năm 2017.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 3 (Deep Learning Core), Phase 5 · 09 (Sequence-to-Sequence), Phase 5 · 10 (Attention Mechanism)
**Time:** ~45 phút

## Vấn đề

Trước năm 2017, mọi mô hình chuỗi hiện đại nhất trên thế giới — ngôn ngữ, dịch thuật, giọng nói — đều là mạng thần kinh tái phát (recurrent neural network). LSTMs và GRUs đã thống trị các benchmark dịch thuật tương đương với ImageNet trong nửa thập kỷ. Chúng là công cụ duy nhất mà mọi người có.

Chúng có ba điểm yếu chí mạng. Tính toán tuần tự đồng nghĩa với việc bạn không thể song song hóa theo trục thời gian: token `t+1` cần hidden state từ token `t`. Một chuỗi 1.024 token đồng nghĩa với 1.024 bước nối tiếp trên một GPU có khả năng thực hiện 1.000.000 phép tính dấu phẩy động mỗi chu kỳ. Thời gian thực tế để huấn luyện (wall-clock time) tăng tuyến tính theo độ dài chuỗi trên phần cứng được thiết kế cho tính toán song song.

Vanishing gradients (triệt tiêu đạo hàm) đồng nghĩa với việc thông tin từ 50 token trước đó đã bị nén qua 50 hàm phi tuyến tính. Các đơn vị tái phát có cổng (LSTM, GRU) làm giảm bớt sự suy giảm này nhưng không bao giờ loại bỏ được nó. Các phụ thuộc tầm xa (long-range dependencies) — "cuốn sách tôi đọc mùa hè năm ngoái trên chuyến bay đến Kyoto là…" — thường xuyên thất bại.

Hidden states có độ rộng cố định đồng nghĩa với việc encoder nén toàn bộ chuỗi nguồn vào một vector duy nhất trước khi decoder nhìn thấy bất cứ thứ gì. Không quan trọng nguồn là 5 token hay 500; nút thắt cổ chai (bottleneck) đều có cùng hình dạng.

Bài báo năm 2017 "Attention Is All You Need" đã đề xuất một điều cấp tiến: loại bỏ hoàn toàn tính tái phát. Hãy để mọi vị trí chú ý (attend) đến mọi vị trí khác một cách song song. Huấn luyện trong một phép nhân ma trận lớn thay vì 1.024 phép nhân tuần tự.

Kết quả là nó thống trị mọi phương thức (modality) tính đến năm 2026. Ngôn ngữ (GPT-5, Claude 4, Llama 4), thị giác (ViT, DINOv2, SAM 3), âm thanh (Whisper), sinh học (AlphaFold 3), robot (RT-2). Cùng một khối, đầu vào khác nhau.

## Khái niệm

![RNN sequential compute vs Transformer parallel attention](../assets/rnn-vs-transformer.svg)

**Tính tái phát là một nút thắt cổ chai.** Một RNN tính toán `h_t = f(h_{t-1}, x_t)`. Mỗi bước phụ thuộc vào bước trước đó. Bạn không thể tính `h_5` trước `h_4`. Trên các GPU hiện đại với hơn 10.000 lõi song song, điều này lãng phí 99% silicon trên một chuỗi dài.

**Attention là một cơ chế phát sóng (broadcast).** Self-attention tính toán `output_i = sum_j(a_ij * v_j)` cho mọi cặp `(i, j)` cùng một lúc. Toàn bộ ma trận attention N×N được lấp đầy trong một phép nhân ma trận theo lô (batched matmul). Không bước nào phụ thuộc vào bước khác. GPU rất ưa thích điều này.

**Tốc độ tăng không phải là một hằng số.** Đó là sự khác biệt giữa độ sâu tuần tự `O(N)` và độ sâu tuần tự `O(1)`. Trong thực tế, các transformer huấn luyện nhanh hơn 5–10 lần mỗi epoch trên phần cứng tương đương ở N=512, và khoảng cách này nới rộng theo độ dài chuỗi cho đến khi bạn chạm vào bức tường bộ nhớ `O(N²)` của attention (điều mà Flash Attention sau đó đã khắc phục — xem Bài 12).

**Chi phí của transformer.** Bộ nhớ attention tăng theo `O(N²)`. Với ngữ cảnh 2K, không vấn đề gì. Với ngữ cảnh 128K, bạn cần các cửa sổ trượt (sliding windows), ngoại suy RoPE, phân mảnh Flash Attention, hoặc các biến thể attention tuyến tính. Tính tái phát là `O(N)` về cả thời gian và bộ nhớ; transformer đánh đổi thời gian lấy bộ nhớ và sau đó giành lại thời gian thông qua tính song song.

**Sự thay đổi về inductive bias.** RNN giả định tính cục bộ và tính gần đây (recency). Transformers không giả định gì cả — mọi cặp đều là ứng viên cho attention. Đó là lý do tại sao transformers cần nhiều dữ liệu hơn để huấn luyện tốt nhưng lại mở rộng tốt hơn khi đã có đủ dữ liệu. Chinchilla (2022) đã chính thức hóa điều này: với đủ token, một transformer luôn đánh bại một RNN có cùng số lượng tham số.

```figure
rnn-vs-parallel
```

## Xây dựng

Không có mạng thần kinh nào ở đây — chúng ta mô phỏng nút thắt cổ chai cốt lõi bằng số học để bạn cảm nhận được khoảng cách trên máy tính xách tay của mình.

### Bước 1: đo độ sâu tuần tự

Xem `code/main.py`. Chúng ta xây dựng hai hàm. Một hàm mã hóa chuỗi dưới dạng chuỗi các phép cộng (tuần tự, giống RNN). Một hàm mã hóa nó dưới dạng phép rút gọn song song (phát sóng, giống attention). Cùng một toán học, đồ thị phụ thuộc khác nhau.

```python
def rnn_style(xs):
    h = 0.0
    for x in xs:
        h = 0.9 * h + x   # can't parallelize: h depends on previous h
    return h

def attention_style(xs):
    return sum(xs) / len(xs)  # every x is independent
```

Chúng ta đo thời gian cả hai trên các chuỗi lên tới 100.000 phần tử. Phiên bản RNN là O(N) và là một pipeline CPU đơn lẻ. Ngay cả trong Python thuần, kiểu rút gọn theo phong cách attention vẫn đánh bại nó ở độ dài ≥ 1.000 vì `sum()` của Python được triển khai bằng C và lặp mà không có chi phí thông dịch (interpreter overhead) cho mỗi bước.

### Bước 2: đếm các phép tính lý thuyết

Cả hai thuật toán đều thực hiện N phép cộng. Sự khác biệt nằm ở *độ sâu phụ thuộc*: bao nhiêu phép tính phải xảy ra tuần tự trước khi phép tiếp theo có thể bắt đầu. Độ sâu RNN = N. Độ sâu Attention = log(N) với phép rút gọn cây, hoặc 1 với phép quét song song. Độ sâu, không phải số lượng phép tính, mới quyết định thời gian GPU.

### Bước 3: mở rộng thực nghiệm trên các chuỗi dài

Chúng ta in một bảng thời gian giúp nhìn thấy rõ khoảng cách O(N). Trên máy Mac năm 2026, các chuỗi dưới 1.000 phần tử quá nhanh để đo lường. Các chuỗi 100.000 phần tử cho thấy một quá trình quét tuyến tính rõ ràng. Hãy mở rộng điều đó lên một transformer 16.384 token với 12 lớp tương đương LSTM và bạn sẽ thấy tại sao thời gian huấn luyện thực tế lại là một rào cản vào năm 2016.

## Sử dụng

Khi nào vẫn nên chọn RNN vào năm 2026:

| Tình huống | Lựa chọn |
|-----------|------|
| Suy luận trực tuyến (streaming inference), từng token một, bộ nhớ hằng số | RNN hoặc mô hình không gian trạng thái (Mamba, RWKV) |
| Chuỗi rất dài (>1M token) nơi bộ nhớ attention bùng nổ | Linear attention, Mamba 2, Hyena |
| Thiết bị biên không có bộ tăng tốc matmul | RNN tách biệt theo chiều sâu (depthwise-separable) vẫn thắng về FLOPs/watt |
| Bất kỳ trường hợp nào khác (huấn luyện, suy luận theo lô, ngữ cảnh lên đến 128K) | Transformer |

Các mô hình không gian trạng thái (SSMs) như Mamba về cơ bản là các RNN với tham số hóa có cấu trúc mang lại cho chúng những ưu điểm của cả hai: bộ nhớ quét `O(N)`, huấn luyện song song thông qua quét chọn lọc. Chúng khôi phục 90% chất lượng của transformer với khả năng mở rộng ngữ cảnh dài tốt hơn. Vào năm 2026, hầu hết các phòng thí nghiệm tiên phong đều huấn luyện các mô hình lai SSM+transformer (ví dụ: Jamba, Samba) — tính tái phát không chết, nó là một thành phần.

## Triển khai

Xem `outputs/skill-architecture-picker.md`. Kỹ năng này chọn một kiến trúc cho một bài toán chuỗi mới dựa trên các ràng buộc về độ dài, thông lượng và ngân sách huấn luyện. Bạn nên luôn từ chối đề xuất một RNN thuần túy cho các đợt huấn luyện trên 1B token mà không nêu rõ sự đánh đổi.

## Bài tập

1. **Dễ.** Lấy `rnn_style` từ `code/main.py` và thay thế hidden state vô hướng bằng một vector hidden state độ dài 64. Đo lại. Chi phí tuần tự tăng bao nhiêu với chiều của hidden state?
2. **Trung bình.** Triển khai phép cộng tiền tố song song (Hillis-Steele scan) bằng Python thuần. Xác minh rằng nó tạo ra kết quả số học giống như quét tuần tự trên độ dài 1024. Đếm độ sâu.
3. **Khó.** Chuyển đổi phép rút gọn kiểu attention sang PyTorch trên GPU. Đo thời gian cả hai khi bạn quét độ dài chuỗi từ 64 đến 65.536. Vẽ biểu đồ và giải thích hình dạng đường cong.

## Thuật ngữ chính

| Thuật ngữ | Mọi người nói | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| Recurrence | "RNN là tuần tự" | Tính toán trong đó bước `t` phụ thuộc vào bước `t-1`, buộc thực thi tuần tự dọc theo trục thời gian. |
| Serial depth | "Đồ thị sâu bao nhiêu" | Chuỗi dài nhất của các phép tính phụ thuộc; giới hạn thời gian thực tế ngay cả trên phần cứng vô hạn. |
| Attention | "Để các token nhìn nhau" | Tổng có trọng số `sum_j a_ij v_j` trong đó `a_ij` đến từ điểm số tương đồng giữa các vị trí i và j. |
| Context window | "Mô hình nhìn thấy bao nhiêu" | Số lượng vị trí mà một lớp attention có thể nhận làm đầu vào; chi phí bộ nhớ bậc hai tăng theo đây. |
| Inductive bias | "Các giả định được đưa vào kiến trúc" | Tiền đề về dữ liệu trông như thế nào; CNN giả định tính bất biến tịnh tiến, RNN giả định tính gần đây. |
| State-space model | "RNN với đại số đằng sau" | Tính tái phát được tham số hóa để huấn luyện song song thông qua các ma trận không gian trạng thái có cấu trúc. |
| Quadratic bottleneck | "Tại sao ngữ cảnh tốn kém" | Bộ nhớ attention = `O(N²)` theo độ dài chuỗi; Flash Attention ẩn đi các hằng số, không phải sự mở rộng. |

## Đọc thêm

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) — bài báo đã khai tử tính tái phát trong NLP chính thống.
- [Bahdanau, Cho, Bengio (2014). Neural MT by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — nơi attention ra đời, được gắn vào một RNN.
- [Hochreiter, Schmidhuber (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) — bài báo LSTM gốc, để lưu hồ sơ.
- [Gu, Dao (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) — câu trả lời tái phát hiện đại cho transformers.