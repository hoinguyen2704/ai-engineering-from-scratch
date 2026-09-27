# Subword Tokenization — BPE, WordPiece, Unigram, SentencePiece

> Các tokenizer cấp từ (word tokenizer) thường gặp khó khăn với những từ chưa từng thấy. Các tokenizer cấp ký tự (character tokenizer) lại làm tăng độ dài chuỗi quá mức. Các tokenizer cấp dưới từ (subword tokenizer) giải quyết vấn đề này bằng cách kết hợp cả hai. Mọi LLM hiện đại đều sử dụng một trong số chúng.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (Text Processing), Phase 5 · 04 (GloVe / FastText / Subword)
**Time:** ~60 minutes

## Vấn đề

Từ điển của bạn có 50.000 từ. Người dùng nhập "untokenizable". Tokenizer của bạn trả về `[UNK]`. Mô hình lúc này không có tín hiệu nào về từ đó. Tệ hơn: tài liệu ở phân vị thứ 90 trong tập dữ liệu của bạn có 40 từ hiếm, nghĩa là mất đi 40 bit thông tin trên mỗi tài liệu.

Subword tokenization giải quyết vấn đề này. Các từ phổ biến vẫn là một token duy nhất. Các từ hiếm được phân tách thành các mảnh có nghĩa: `untokenizable` → `un`, `token`, `izable`. Dữ liệu huấn luyện bao phủ mọi thứ vì bất kỳ chuỗi nào cuối cùng cũng là một chuỗi các byte.

Mọi LLM tiên phong vào năm 2026 đều sử dụng một trong ba thuật toán (BPE, Unigram, WordPiece), được đóng gói trong một trong ba thư viện (tiktoken, SentencePiece, HF Tokenizers). Bạn không thể phát hành một mô hình ngôn ngữ mà không chọn một trong số đó.

## Khái niệm

![BPE vs Unigram vs WordPiece, character-by-character](../assets/subword-tokenization.svg)

**BPE (Byte-Pair Encoding).** Bắt đầu với từ điển cấp ký tự. Đếm mọi cặp liền kề. Hợp nhất cặp xuất hiện thường xuyên nhất thành một token mới. Lặp lại cho đến khi đạt kích thước từ điển mục tiêu. Thuật toán thống trị: GPT-2/3/4, Llama, Gemma, Qwen2, Mistral.

**Byte-level BPE.** Thuật toán tương tự nhưng thực hiện trên các byte thô (256 token cơ sở) thay vì các ký tự Unicode. Đảm bảo không có token `[UNK]` — bất kỳ chuỗi byte nào cũng có thể mã hóa được. GPT-2 sử dụng 50.257 token (256 byte + 50.000 lần hợp nhất + 1 token đặc biệt).

**Unigram.** Bắt đầu với một từ điển khổng lồ. Gán cho mỗi token một xác suất unigram. Lặp lại việc loại bỏ các token mà việc xóa chúng làm tăng log-likelihood của tập dữ liệu ít nhất. Có tính xác suất khi suy luận (inference): có thể lấy mẫu các cách token hóa (hữu ích cho việc tăng cường dữ liệu thông qua subword regularization). Được sử dụng bởi T5, mBART, ALBERT, XLNet, Gemma.

**WordPiece.** Hợp nhất các cặp tối đa hóa likelihood của tập dữ liệu huấn luyện thay vì tần suất thô. Được sử dụng bởi BERT, DistilBERT, ELECTRA.

**SentencePiece vs tiktoken.** SentencePiece là thư viện *huấn luyện* từ điển (BPE hoặc Unigram) trực tiếp trên văn bản Unicode thô, mã hóa khoảng trắng thành `▁`. tiktoken là *encoder* nhanh của OpenAI cho các từ điển đã được xây dựng sẵn; nó không thực hiện huấn luyện.

Quy tắc chung:

- **Huấn luyện từ điển mới:** SentencePiece (đa ngôn ngữ, không cần tiền xử lý token) hoặc HF Tokenizers.
- **Suy luận nhanh với từ điển GPT:** tiktoken (cl100k_base, o200k_base).
- **Cả hai:** HF Tokenizers — một thư viện duy nhất, hỗ trợ cả huấn luyện và phục vụ (serving).

```figure
bpe-merge
```

## Xây dựng

### Bước 1: BPE từ đầu

Xem `code/main.py`. Vòng lặp:

```python
def train_bpe(corpus, num_merges):
    vocab = {tuple(word) + ("</w>",): count for word, count in corpus.items()}
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for symbols, freq in vocab.items():
            for a, b in zip(symbols, symbols[1:]):
                pairs[(a, b)] += freq
        if not pairs:
            break
        best = pairs.most_common(1)[0][0]
        merges.append(best)
        vocab = apply_merge(vocab, best)
    return merges
```

Ba sự thật mà thuật toán mã hóa. `</w>` đánh dấu kết thúc từ để "low" (hậu tố) và "lower" (tiền tố) vẫn khác biệt. Trọng số tần suất giúp các cặp có tần suất cao thắng thế sớm. Danh sách hợp nhất được sắp xếp — suy luận áp dụng các lần hợp nhất theo thứ tự huấn luyện.

### Bước 2: encode với các lần hợp nhất đã học

```python
def encode_bpe(word, merges):
    symbols = list(word) + ["</w>"]
    for a, b in merges:
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == a and symbols[i + 1] == b:
                symbols = symbols[:i] + [a + b] + symbols[i + 2:]
            else:
                i += 1
    return symbols
```

Độ phức tạp ngây thơ O(n·|merges|). Các triển khai thực tế (tiktoken, HF Tokenizers) sử dụng tra cứu thứ hạng hợp nhất với hàng đợi ưu tiên và chạy trong thời gian gần như tuyến tính.

### Bước 3: SentencePiece trong thực tế

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="my_tokenizer",
    vocab_size=8000,
    model_type="bpe",          # or "unigram"
    character_coverage=0.9995, # lower for CJK (e.g. 0.9995 for English, 0.995 for Japanese)
    normalization_rule_name="nmt_nfkc",
)

sp = spm.SentencePieceProcessor(model_file="my_tokenizer.model")
print(sp.encode("untokenizable", out_type=str))
# ['▁un', 'token', 'izable']
```

Lưu ý: không cần tiền xử lý token, khoảng trắng được mã hóa thành `▁`, `character_coverage` kiểm soát mức độ quyết liệt trong việc bảo tồn các ký tự hiếm so với việc ánh xạ chúng sang `<unk>`.

### Bước 4: tiktoken cho các từ điển tương thích với OpenAI

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
print(enc.encode("untokenizable"))        # [127340, 101028]
print(len(enc.encode("Hello, world!")))   # 4
```

Chỉ mã hóa. Nhanh (backend Rust). Khớp chính xác với cách token hóa của GPT-4/5 để đếm byte, ước tính chi phí, lập ngân sách cửa sổ ngữ cảnh.

## Những cạm bẫy vẫn tồn tại vào năm 2026

- **Tokenizer drift.** Huấn luyện trên từ điển A, triển khai với từ điển B. ID token khác nhau; mô hình xuất ra rác. Hãy kiểm tra hash `tokenizer.json` trong CI.
- **Sự mơ hồ về khoảng trắng.** BPE "hello" và " hello" tạo ra các token khác nhau. Luôn chỉ định rõ `add_special_tokens` và `add_prefix_space`.
- **Thiếu hụt huấn luyện đa ngôn ngữ.** Các tập dữ liệu nặng về tiếng Anh tạo ra các từ điển chia nhỏ các hệ chữ không phải Latin thành nhiều hơn 5-10 lần số token. Cùng một prompt tốn kém hơn 5-10 lần trong tiếng Nhật/Ả Rập trên GPT-3.5. o200k_base đã khắc phục một phần điều này.
- **Chia tách Emoji.** Một emoji đơn lẻ có thể chiếm 5 token. Hãy kiểm tra cách xử lý emoji khi lập ngân sách ngữ cảnh.

## Sử dụng

Stack năm 2026:

| Tình huống | Lựa chọn |
|-----------|------|
| Huấn luyện mô hình đơn ngữ từ đầu | HF Tokenizers (BPE) |
| Huấn luyện mô hình đa ngôn ngữ | SentencePiece (Unigram, `character_coverage=0.9995`) |
| Phục vụ API tương thích OpenAI | tiktoken (`o200k_base` cho GPT-4+) |
| Từ điển chuyên ngành (code, toán, protein) | Huấn luyện BPE tùy chỉnh trên tập dữ liệu chuyên ngành, hợp nhất với từ điển cơ sở |
| Suy luận tại biên (edge), mô hình nhỏ | Unigram (từ điển nhỏ hơn hoạt động tốt hơn) |

Kích thước từ điển là một quyết định về quy mô, không phải là hằng số. Phỏng đoán thô: 32k cho tham số <1B, 50-100k cho 1-10B, 200k+ cho đa ngôn ngữ/tiên phong.

## Triển khai

Lưu dưới dạng `outputs/skill-bpe-vs-wordpiece.md`:

```markdown
---
name: tokenizer-picker
description: Pick tokenizer algorithm, vocab size, library for a given corpus and deployment target.
version: 1.0.0
phase: 5
lesson: 19
tags: [nlp, tokenization]
---

Given a corpus (size, languages, domain) and deployment target (training from scratch / fine-tuning / API-compatible inference), output:

1. Algorithm. BPE, Unigram, or WordPiece. One-sentence reason.
2. Library. SentencePiece, HF Tokenizers, or tiktoken. Reason.
3. Vocab size. Rounded to nearest 1k. Reason tied to model size and language coverage.
4. Coverage settings. `character_coverage`, `byte_fallback`, special-token list.
5. Validation plan. Average tokens-per-word on held-out set, OOV rate, compression ratio, round-trip decode equality.

Refuse to train a character-coverage <0.995 tokenizer on corpora with rare-script content. Refuse to ship a vocab without a frozen `tokenizer.json` hash check in CI. Flag any monolingual tokenizer under 16k vocab as likely under-spec.
```

## Bài tập

1. **Dễ.** Huấn luyện BPE với 500 lần hợp nhất trên tập dữ liệu nhỏ của `code/main.py`. Encode ba từ chưa từng thấy. Có bao nhiêu từ tạo ra chính xác 1 token so với >1 token?
2. **Trung bình.** So sánh số lượng token trên 100 câu Wikipedia tiếng Anh giữa `cl100k_base`, `o200k_base` và một SentencePiece BPE bạn huấn luyện với vocab=32k. Báo cáo tỷ lệ nén của mỗi loại.
3. **Khó.** Huấn luyện cùng một tập dữ liệu với BPE, Unigram và WordPiece. Đo lường độ chính xác hạ nguồn khi sử dụng từng loại trên một bộ phân loại cảm xúc nhỏ. Liệu sự lựa chọn này có làm thay đổi kết quả F1 hơn 1 điểm không?

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| BPE | Byte-Pair Encoding | Hợp nhất tham lam các cặp ký tự thường gặp nhất cho đến khi đạt kích thước từ điển mục tiêu. |
| Byte-level BPE | Không bao giờ có token lạ | BPE trên 256 byte thô; GPT-2 / Llama sử dụng cách này. |
| Unigram | Tokenizer xác suất | Loại bỏ từ một tập ứng viên lớn bằng log-likelihood; được T5, Gemma sử dụng. |
| SentencePiece | Loại có khoảng trắng | Thư viện huấn luyện BPE/Unigram trên văn bản thô; khoảng trắng được mã hóa thành `▁`. |
| tiktoken | Loại nhanh | Encoder BPE dựa trên Rust của OpenAI cho các từ điển có sẵn. Không huấn luyện. |
| Merge list | Các con số ma thuật | Danh sách thứ tự các lần hợp nhất `(a, b) → ab`; suy luận áp dụng theo thứ tự. |
| Character coverage | Bao nhiêu là quá hiếm? | Tỷ lệ các ký tự trong tập huấn luyện mà tokenizer phải bao phủ; thường là ~0.9995. |

## Đọc thêm

- [Sennrich, Haddow, Birch (2015). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — bài báo về BPE.
- [Kudo (2018). Subword Regularization with Unigram Language Model](https://arxiv.org/abs/1804.10959) — bài báo về Unigram.
- [Kudo, Richardson (2018). SentencePiece: A simple and language independent subword tokenizer](https://arxiv.org/abs/1808.06226) — thư viện.
- [Hugging Face — Summary of the tokenizers](https://huggingface.co/docs/transformers/tokenizer_summary) — tài liệu tham khảo tóm tắt.
- [OpenAI tiktoken repo](https://github.com/openai/tiktoken) — cookbook + danh sách mã hóa.