# Xây dựng Tokenizer từ đầu

> Bài 01 đã cung cấp cho bạn một món đồ chơi. Bài học này sẽ trao cho bạn một vũ khí.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 10, Lesson 01 (Tokenizers: BPE, WordPiece, SentencePiece)
**Time:** ~90 phút

## Mục tiêu học tập

- Xây dựng một BPE tokenizer cấp độ sản xuất (production-grade) có khả năng xử lý Unicode, chuẩn hóa khoảng trắng và các special token.
- Triển khai cơ chế dự phòng byte-level (byte-level fallback) để tokenizer có thể mã hóa bất kỳ đầu vào nào (bao gồm emoji, CJK và mã nguồn) mà không gặp phải các unknown token.
- Thêm các mẫu regex tiền xử lý (pre-tokenization) để tách văn bản tại các ranh giới từ trước khi áp dụng các quy tắc gộp BPE.
- Huấn luyện một tokenizer tùy chỉnh trên một tập dữ liệu (corpus) và đánh giá tỷ lệ nén của nó so với tiktoken trên văn bản đa ngôn ngữ.

## Vấn đề

Tokenizer BPE từ Bài 01 hoạt động tốt với văn bản tiếng Anh. Bây giờ hãy thử với tiếng Nhật. Hoặc emoji. Hoặc mã nguồn Python với sự pha trộn giữa tab và khoảng trắng.

Nó sẽ bị lỗi.

Không phải vì BPE sai, mà vì cách triển khai chưa hoàn thiện. Một tokenizer cấp độ sản xuất phải xử lý các byte thô ở bất kỳ mã hóa nào, chuẩn hóa Unicode trước khi tách, quản lý các special token không bao giờ được gộp, kết hợp tiền xử lý với tách subword, và tất cả phải thực hiện đủ nhanh để không trở thành nút thắt cổ chai trong quy trình huấn luyện xử lý 15 nghìn tỷ token.

Tokenizer của GPT-2 có 50.257 token. Llama 3 có 128.256. GPT-4 có khoảng 100.000. Đây không phải là những con số nhỏ. Các bảng gộp (merge tables) đằng sau những từ vựng đó được huấn luyện trên hàng trăm gigabyte văn bản, và các cơ chế xung quanh -- chuẩn hóa, tiền xử lý, chèn special token, định dạng chat template -- là thứ phân biệt một tokenizer chỉ xử lý được "hello world" với một tokenizer xử lý được toàn bộ internet.

Bạn sẽ xây dựng chính những cơ chế đó.

## Khái niệm

### Quy trình đầy đủ (The Full Pipeline)

Một tokenizer cấp độ sản xuất không chỉ là một thuật toán. Đó là một quy trình gồm năm giai đoạn, mỗi giai đoạn giải quyết một vấn đề khác nhau.

```mermaid
graph LR
    A[Raw Text] --> B[Normalize]
    B --> C[Pre-Tokenize]
    C --> D[BPE Merge]
    D --> E[Special Tokens]
    E --> F[Token IDs]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#1a1a2e,stroke:#e94560,color:#fff
    style C fill:#1a1a2e,stroke:#e94560,color:#fff
    style D fill:#1a1a2e,stroke:#e94560,color:#fff
    style E fill:#1a1a2e,stroke:#e94560,color:#fff
    style F fill:#1a1a2e,stroke:#e94560,color:#fff
```

Mỗi giai đoạn có một nhiệm vụ cụ thể:

| Giai đoạn | Chức năng | Tại sao quan trọng |
|-----------|-----------|-------------------|
| Chuẩn hóa (Normalize) | NFKC Unicode, tùy chọn viết thường, tùy chọn loại bỏ dấu | Chữ ghép "fi" (U+FB01) trở thành "fi" (hai ký tự). Nếu không, cùng một từ sẽ nhận các token khác nhau. |
| Tiền xử lý (Pre-Tokenize) | Tách văn bản thành các đoạn trước khi dùng BPE | Ngăn BPE gộp qua ranh giới từ. "the cat" không bao giờ nên tạo ra token "e c". |
| Gộp BPE (BPE Merge) | Áp dụng các quy tắc gộp đã học vào chuỗi byte | Cốt lõi của việc nén. Biến byte thô thành các subword token. |
| Special Tokens | Chèn [BOS], [EOS], [PAD], các đánh dấu chat template | Các token này có ID cố định. Chúng không bao giờ tham gia vào quá trình gộp BPE. Mô hình cần chúng để hiểu cấu trúc. |
| Ánh xạ ID (ID Mapping) | Chuyển đổi chuỗi token thành ID số nguyên | Mô hình nhìn thấy số nguyên, không phải chuỗi. |

### Byte-Level BPE

Tokenizer của Bài 01 hoạt động trên các byte UTF-8. Đó là hướng đi đúng. Nhưng chúng ta đã bỏ qua một điều quan trọng: điều gì xảy ra khi các byte đó không phải là UTF-8 hợp lệ?

Byte-level BPE giải quyết vấn đề này bằng cách coi mọi giá trị byte có thể (0-255) là một token hợp lệ. Từ vựng cơ sở của bạn chính xác là 256 mục. Bất kỳ tệp nào -- văn bản, nhị phân, bị hỏng -- đều có thể được token hóa mà không tạo ra unknown token.

GPT-2 đã thêm một thủ thuật: ánh xạ mỗi byte sang một ký tự Unicode có thể in được để từ vựng vẫn dễ đọc đối với con người. Byte 0x20 (khoảng trắng) trở thành ký tự "G" trong ánh xạ của họ. Điều này chỉ mang tính thẩm mỹ. Thuật toán không quan tâm.

Sức mạnh thực sự: byte-level BPE xử lý mọi ngôn ngữ trên trái đất. Các ký tự tiếng Trung là 3 byte UTF-8 mỗi ký tự. Tiếng Nhật có thể là 3-4 byte. Tiếng Ả Rập, Devanagari, emoji -- tất cả chỉ là các chuỗi byte. Thuật toán BPE tìm các mẫu trong các chuỗi byte này giống hệt cách nó tìm các mẫu trong các byte ASCII tiếng Anh.

### Tiền xử lý (Pre-Tokenization)

Trước khi BPE chạm vào văn bản của bạn, bạn cần tách nó thành các đoạn. Điều này ngăn thuật toán gộp tạo ra các token vượt qua ranh giới từ.

GPT-2 sử dụng một mẫu regex để tách văn bản:

```
'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

Mẫu này tách các từ viết tắt ("don't" trở thành "don" + "'t"), các từ có khoảng trắng dẫn đầu tùy chọn, số, dấu câu và khoảng trắng. Khoảng trắng dẫn đầu được giữ lại gắn liền với từ -- vì vậy "the cat" trở thành [" the", " cat"], không phải ["the", " ", "cat"].

Llama sử dụng SentencePiece, bỏ qua hoàn toàn regex. Nó coi luồng byte thô là một chuỗi dài và để thuật toán BPE tự tìm ranh giới. Cách này đơn giản hơn nhưng cho phép BPE tự do hơn trong việc tạo ra các token xuyên từ.

Sự lựa chọn này rất quan trọng. Regex của GPT-2 ngăn tokenizer học rằng "the" ở cuối một từ và "the" ở đầu từ tiếp theo nên được gộp lại. SentencePiece cho phép điều đó, đôi khi tạo ra khả năng nén hiệu quả hơn nhưng các token ít mang tính diễn giải hơn.

### Special Tokens

Mọi tokenizer cấp độ sản xuất đều dành riêng các ID token cho các đánh dấu cấu trúc:

| Token | Mục đích | Được sử dụng bởi |
|-------|----------|------------------|
| `[BOS]` / `<s>` | Bắt đầu chuỗi | Llama 3, GPT |
| `[EOS]` / `</s>` | Kết thúc chuỗi | Tất cả các mô hình |
| `[PAD]` | Padding để căn chỉnh batch | BERT, T5 |
| `[UNK]` | Unknown token (byte-level BPE loại bỏ cái này) | BERT, WordPiece |
| `<\|im_start\|>` | Bắt đầu ranh giới tin nhắn chat | ChatGPT, Qwen |
| `<\|im_end\|>` | Kết thúc ranh giới tin nhắn chat | ChatGPT, Qwen |
| `<\|user\|>` | Đánh dấu lượt người dùng | Llama 3 |
| `<\|assistant\|>` | Đánh dấu lượt trợ lý | Llama 3 |

Các special token không bao giờ bị tách bởi BPE. Chúng được khớp chính xác trước khi thuật toán gộp chạy, được thay thế bằng ID cố định của chúng, và văn bản xung quanh được token hóa bình thường.

### Chat Templates

Đây là nơi hầu hết mọi người cảm thấy bối rối và hầu hết các triển khai bị lỗi.

Khi bạn gửi tin nhắn đến một mô hình chat, API chấp nhận một danh sách các tin nhắn:

```
[
  {"role": "system", "content": "You are helpful."},
  {"role": "user", "content": "Hello"},
  {"role": "assistant", "content": "Hi there!"}
]
```

Mô hình không nhìn thấy JSON. Nó nhìn thấy một chuỗi token phẳng. Chat template chuyển đổi các tin nhắn thành chuỗi phẳng đó bằng cách sử dụng các special token. Mỗi mô hình thực hiện việc này theo cách khác nhau:

```
Llama 3:
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are helpful.<|eot_id|><|start_header_id|>user<|end_header_id|>

Hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Hi there!<|eot_id|>

ChatGPT:
<|im_start|>system
You are helpful.<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
Hi there!<|im_end|>
```

Nếu làm sai template, mô hình sẽ tạo ra rác. Nó đã được huấn luyện trên một định dạng chính xác. Bất kỳ sai lệch nào -- thiếu dòng mới, token bị tráo đổi, thừa khoảng trắng -- đều khiến đầu vào nằm ngoài phân phối huấn luyện.

### Tốc độ

Python quá chậm cho việc token hóa trong sản xuất.

tiktoken (OpenAI) được viết bằng Rust với các binding cho Python. HuggingFace tokenizers cũng là Rust. SentencePiece là C++. Những công cụ này đạt tốc độ nhanh hơn 10-100 lần so với Python thuần.

Để hình dung: việc token hóa 15 nghìn tỷ token cho quá trình tiền huấn luyện Llama 3 với tốc độ 1 triệu token mỗi giây (Python nhanh) sẽ mất 174 ngày. Với 100 triệu token mỗi giây (Rust), nó chỉ mất 1,7 ngày.

Bạn đang xây dựng bằng Python để hiểu thuật toán. Trong sản xuất, bạn sẽ sử dụng một triển khai đã biên dịch và chỉ tương tác với wrapper Python.

```figure
weight-tying
```

## Xây dựng

### Bước 1: Mã hóa Byte-Level

Nền tảng. Chuyển đổi bất kỳ chuỗi nào thành một chuỗi byte, ánh xạ mỗi byte sang một ký tự có thể in được để hiển thị, và đảo ngược quy trình.

```python
def bytes_to_tokens(text):
    return list(text.encode("utf-8"))

def tokens_to_text(token_bytes):
    return bytes(token_bytes).decode("utf-8", errors="replace")
```

Kiểm tra trên văn bản đa ngôn ngữ để xem số lượng byte:

```python
texts = [
    ("English", "hello"),
    ("Chinese", "你好"),
    ("Emoji", "🔥"),
    ("Mixed", "hello你好🔥"),
]

for label, text in texts:
    b = bytes_to_tokens(text)
    print(f"{label}: {len(text)} chars -> {len(b)} bytes -> {b}")
```

"hello" là 5 byte. "你好" là 6 byte (3 byte mỗi ký tự). Emoji ngọn lửa là 4 byte. Byte-level tokenizer không quan tâm đó là ngôn ngữ gì. Byte là byte.

### Bước 2: Pre-Tokenizer với Regex

Tách văn bản thành các đoạn bằng mẫu regex của GPT-2. Mỗi đoạn được token hóa độc lập bởi BPE.

```python
import re

try:
    import regex
    GPT2_PATTERN = regex.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
except ImportError:
    GPT2_PATTERN = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"""
    )

def pre_tokenize(text):
    return [match.group() for match in GPT2_PATTERN.finditer(text)]
```

Mô-đun `regex` hỗ trợ các escape thuộc tính Unicode (`\p{L}` cho chữ cái, `\p{N}` cho số). Mô-đun `re` của thư viện chuẩn thì không, vì vậy chúng ta quay lại sử dụng các lớp ký tự ASCII. Đối với các tokenizer đa ngôn ngữ cấp độ sản xuất, hãy cài đặt `regex`.

Hãy thử:

```python
print(pre_tokenize("Hello, world! Don't stop."))
# [' Hello', ',', ' world', '!', " Don", "'t", ' stop', '.']
```

Khoảng trắng dẫn đầu vẫn gắn liền với từ. Các từ viết tắt tách tại dấu nháy đơn. Dấu câu trở thành một đoạn riêng. BPE sẽ không bao giờ gộp các token qua các ranh giới này.

### Bước 3: BPE trên các chuỗi Byte

Thuật toán cốt lõi từ Bài 01, nhưng bây giờ hoạt động trên các đoạn đã được tiền xử lý độc lập.

```python
from collections import Counter

def get_byte_pairs(chunks):
    pairs = Counter()
    for chunk in chunks:
        byte_seq = list(chunk.encode("utf-8"))
        for i in range(len(byte_seq) - 1):
            pairs[(byte_seq[i], byte_seq[i + 1])] += 1
    return pairs

def apply_merge(byte_seq, pair, new_id):
    merged = []
    i = 0
    while i < len(byte_seq):
        if i < len(byte_seq) - 1 and byte_seq[i] == pair[0] and byte_seq[i + 1] == pair[1]:
            merged.append(new_id)
            i += 2
        else:
            merged.append(byte_seq[i])
            i += 1
    return merged
```

### Bước 4: Xử lý Special Token

Các special token cần khớp chính xác và ID cố định. Chúng bỏ qua hoàn toàn BPE.

```python
class SpecialTokenHandler:
    def __init__(self):
        self.special_tokens = {}
        self.pattern = None

    def add_token(self, token_str, token_id):
        self.special_tokens[token_str] = token_id
        escaped = [re.escape(t) for t in sorted(self.special_tokens.keys(), key=len, reverse=True)]
        self.pattern = re.compile("|".join(escaped))

    def split_with_specials(self, text):
        if not self.pattern:
            return [(text, False)]
        parts = []
        last_end = 0
        for match in self.pattern.finditer(text):
            if match.start() > last_end:
                parts.append((text[last_end:match.start()], False))
            parts.append((match.group(), True))
            last_end = match.end()
        if last_end < len(text):
            parts.append((text[last_end:], False))
        return parts
```

### Bước 5: Lớp Tokenizer đầy đủ

Kết hợp mọi thứ lại với nhau: chuẩn hóa, tách theo special token, tiền xử lý, gộp BPE, ánh xạ sang ID.

```python
import unicodedata

class ProductionTokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.special_handler = SpecialTokenHandler()
        self.next_id = 256

    def normalize(self, text):
        return unicodedata.normalize("NFKC", text)

    def train(self, text, num_merges):
        text = self.normalize(text)
        chunks = pre_tokenize(text)
        chunk_bytes = [list(chunk.encode("utf-8")) for chunk in chunks]

        for i in range(num_merges):
            pairs = Counter()
            for seq in chunk_bytes:
                for j in range(len(seq) - 1):
                    pairs[(seq[j], seq[j + 1])] += 1
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            new_id = self.next_id
            self.next_id += 1
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            chunk_bytes = [apply_merge(seq, best, new_id) for seq in chunk_bytes]

    def add_special_token(self, token_str):
        token_id = self.next_id
        self.next_id += 1
        self.special_handler.add_token(token_str, token_id)
        self.vocab[token_id] = token_str.encode("utf-8")
        return token_id

    def encode(self, text):
        text = self.normalize(text)
        parts = self.special_handler.split_with_specials(text)
        all_ids = []
        for part_text, is_special in parts:
            if is_special:
                all_ids.append(self.special_handler.special_tokens[part_text])
            else:
                for chunk in pre_tokenize(part_text):
                    byte_seq = list(chunk.encode("utf-8"))
                    for pair, new_id in self.merges.items():
                        byte_seq = apply_merge(byte_seq, pair, new_id)
                    all_ids.extend(byte_seq)
        return all_ids

    def decode(self, ids):
        byte_parts = []
        for token_id in ids:
            if token_id in self.vocab:
                byte_parts.append(self.vocab[token_id])
        return b"".join(byte_parts).decode("utf-8", errors="replace")

    def vocab_size(self):
        return len(self.vocab)
```

### Bước 6: Kiểm tra đa ngôn ngữ

Bài kiểm tra thực sự. Thử nghiệm với tiếng Anh, tiếng Trung, emoji và mã nguồn.

```python
corpus = (
    "The quick brown fox jumps over the lazy dog. "
    "The quick brown fox runs through the forest. "
    "Machine learning models process natural language. "
    "Deep learning transforms how we build software. "
    "def train(model, data): return model.fit(data) "
    "def predict(model, x): return model(x) "
)

tok = ProductionTokenizer()
tok.train(corpus, num_merges=50)

bos = tok.add_special_token("<|begin|>")
eos = tok.add_special_token("<|end|>")

test_texts = [
    "The quick brown fox.",
    "你好世界",
    "Hello 🌍 World",
    "def foo(x): return x + 1",
    f"<|begin|>Hello<|end|>",
]

for text in test_texts:
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    print(f"Input:   {text}")
    print(f"Tokens:  {len(ids)} ids")
    print(f"Decoded: {decoded}")
    print()
```

Các ký tự tiếng Trung tạo ra 3 byte mỗi ký tự. Emoji tạo ra 4 byte. Không có cái nào làm hỏng tokenizer. Không có cái nào tạo ra unknown token. Đó là sức mạnh của byte-level BPE.

## Sử dụng

### So sánh các Tokenizer thực tế

Tải các tokenizer thực tế từ Llama 3, GPT-4 và Mistral. Xem cách mỗi loại xử lý cùng một đoạn văn bản đa ngôn ngữ.

```python
import tiktoken

gpt4_enc = tiktoken.get_encoding("cl100k_base")

test_paragraph = "Machine learning is powerful. 机器学习很强大。 L'apprentissage automatique est puissant. 🤖💪"

tokens = gpt4_enc.encode(test_paragraph)
pieces = [gpt4_enc.decode([t]) for t in tokens]
print(f"GPT-4 ({len(tokens)} tokens): {pieces}")
```

```python
from transformers import AutoTokenizer

llama_tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
mistral_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

for name, tok in [("Llama 3", llama_tok), ("Mistral", mistral_tok)]:
    tokens = tok.encode(test_paragraph)
    pieces = tok.convert_ids_to_tokens(tokens)
    print(f"{name} ({len(tokens)} tokens): {pieces[:20]}...")
```

Bạn sẽ thấy số lượng token khác nhau cho cùng một văn bản. Llama 3 với từ vựng 128K tích cực hơn trong việc gộp các mẫu phổ biến. GPT-4 với 100K nằm ở giữa. Mistral với 32K tạo ra nhiều token hơn nhưng có lớp embedding nhỏ hơn.

Sự đánh đổi luôn giống nhau: từ vựng lớn hơn có nghĩa là chuỗi ngắn hơn nhưng nhiều tham số hơn.

## Triển khai

Bài học này tạo ra một prompt để xây dựng và gỡ lỗi các tokenizer cấp độ sản xuất. Xem `outputs/prompt-tokenizer-builder.md`.

## Bài tập

1. **Dễ:** Thêm phương thức `get_token_bytes(id)` hiển thị các byte thô cho bất kỳ ID token nào. Sử dụng nó để kiểm tra xem các token gộp phổ biến nhất của bạn thực sự đại diện cho cái gì.
2. **Trung bình:** Triển khai pre-tokenizer kiểu Llama, tách theo khoảng trắng và chữ số nhưng giữ lại các khoảng trắng dẫn đầu. So sánh từ vựng của nó với cách tiếp cận regex của GPT-2 trên cùng một tập dữ liệu.
3. **Khó:** Thêm phương thức chat template nhận danh sách các tin nhắn `{"role": ..., "content": ...}` và tạo ra chuỗi token chính xác cho định dạng chat của Llama 3. Kiểm tra nó so với triển khai của HuggingFace.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực sự |
|-----------|----------------------|------------------|
| Byte-level BPE | "Tokenizer hoạt động trên byte" | BPE với từ vựng cơ sở gồm 256 giá trị byte -- xử lý mọi đầu vào mà không có unknown token |
| Pre-tokenization | "Tách trước khi BPE" | Tách bằng regex hoặc quy tắc để ngăn BPE gộp qua ranh giới từ |
| NFKC normalization | "Dọn dẹp Unicode" | Phân tách chính tắc theo sau bởi kết hợp tương thích -- chữ ghép "fi" trở thành "fi", "A" fullwidth trở thành "A" |
| Chat template | "Cách tin nhắn trở thành token" | Định dạng chính xác để chuyển đổi danh sách tin nhắn role/content thành chuỗi token phẳng -- phụ thuộc vào mô hình và phải khớp với định dạng huấn luyện |
| Special tokens | "Token điều khiển" | ID token dành riêng bỏ qua BPE -- [BOS], [EOS], [PAD], các đánh dấu chat -- được khớp chính xác trước khi gộp |
| Fertility | "Token trên mỗi từ" | Tỷ lệ token đầu ra so với từ đầu vào -- 1.3 cho tiếng Anh trong GPT-4, 2-3 cho tiếng Hàn, cao hơn nghĩa là lãng phí ngữ cảnh |
| tiktoken | "Tokenizer của OpenAI" | Triển khai BPE bằng Rust với các binding Python -- nhanh hơn 10-100 lần so với Python thuần |
| Merge table | "Từ vựng" | Danh sách thứ tự các cặp byte được gộp trong quá trình huấn luyện -- đây CHÍNH LÀ kiến thức đã học của tokenizer |

## Đọc thêm

- [Mã nguồn OpenAI tiktoken](https://github.com/openai/tiktoken) -- Triển khai BPE bằng Rust được sử dụng bởi GPT-3.5/4
- [HuggingFace tokenizers](https://github.com/huggingface/tokenizers) -- Thư viện tokenizer Rust hỗ trợ BPE, WordPiece, Unigram
- [Bài báo Llama 3 (Meta, 2024)](https://arxiv.org/abs/2407.21783) -- chi tiết về từ vựng 128K và huấn luyện tokenizer
- [SentencePiece (Kudo & Richardson, 2018)](https://arxiv.org/abs/1808.06226) -- token hóa không phụ thuộc ngôn ngữ
- [Mã nguồn tokenizer GPT-2](https://github.com/openai/gpt-2/blob/master/src/encoder.py) -- ánh xạ byte-sang-Unicode gốc