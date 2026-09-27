# Xử lý văn bản — Tokenization, Stemming, Lemmatization

> Ngôn ngữ là liên tục. Các mô hình là rời rạc. Tiền xử lý chính là cầu nối.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 2 · 14 (Naive Bayes)
**Time:** ~45 phút

## Vấn đề

Một mô hình không thể đọc "The cats were running." Nó chỉ đọc các số nguyên.

Mọi hệ thống NLP đều bắt đầu với ba câu hỏi giống nhau: Từ bắt đầu ở đâu? Gốc của từ là gì? Làm thế nào để coi "run", "running", "ran" là cùng một thứ khi cần thiết, và là những thứ khác nhau khi không cần?

Nếu thực hiện sai bước tokenization, mô hình sẽ học từ dữ liệu rác. Nếu tokenizer của bạn coi `don't` là một token nhưng `do n't` lại là hai, phân phối huấn luyện sẽ bị chia tách. Nếu stemmer của bạn gộp `organization` và `organ` thành cùng một gốc, mô hình chủ đề (topic modeling) sẽ thất bại. Nếu lemmatizer cần ngữ cảnh từ loại (part-of-speech) mà bạn không cung cấp, các động từ sẽ bị xử lý như danh từ.

Bài học này xây dựng ba bước tiền xử lý từ đầu, sau đó chỉ ra cách NLTK và spaCy thực hiện công việc tương tự để bạn thấy được các đánh đổi.

## Khái niệm

Ba thao tác. Mỗi thao tác có một nhiệm vụ và một chế độ lỗi.

**Tokenization** chia một chuỗi thành các token. "Token" là một khái niệm cố tình mơ hồ vì độ chi tiết phù hợp phụ thuộc vào tác vụ. Cấp độ từ cho NLP cổ điển. Cấp độ subword cho các Transformer. Cấp độ ký tự cho các ngôn ngữ không có khoảng trắng.

**Stemming** cắt bỏ các hậu tố bằng các quy tắc. Nhanh, mạnh bạo, máy móc. `running -> run`. `organization -> organ`. Cái thứ hai chính là chế độ lỗi.

**Lemmatization** đưa một từ về dạng từ điển của nó bằng cách sử dụng kiến thức ngữ pháp. Chậm hơn, chính xác, cần bảng tra cứu hoặc bộ phân tích hình thái. `ran -> run` (cần biết "ran" là thì quá khứ của "run"). `better -> good` (cần biết các dạng so sánh).

Quy tắc ngón tay cái: Hãy dùng Stemming khi tốc độ là ưu tiên và bạn có thể chấp nhận nhiễu (lập chỉ mục tìm kiếm, phân loại thô). Hãy dùng Lemmatization khi ý nghĩa là quan trọng (trả lời câu hỏi, tìm kiếm ngữ nghĩa, bất cứ thứ gì người dùng sẽ đọc).

```figure
edit-distance
```

## Xây dựng

### Bước 1: Regex word tokenizer

Tokenizer đơn giản và hữu ích nhất là tách dựa trên các ký tự không phải chữ-số trong khi vẫn giữ dấu câu như các token riêng biệt. Không hoàn hảo, không phải cuối cùng, nhưng nó chạy chỉ trong một dòng.

```python
import re

def tokenize(text):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]", text)
```

Ba mẫu theo thứ tự ưu tiên. Các từ có dấu nháy đơn bên trong tùy chọn (`don't`, `it's`). Các số thuần túy. Bất kỳ ký tự đơn lẻ nào không phải khoảng trắng và không phải chữ-số (dấu câu).

```python
>>> tokenize("The cats weren't running at 3pm.")
['The', 'cats', "weren't", 'running', 'at', '3', 'pm', '.']
```

Các chế độ lỗi cần lưu ý: `3pm` bị tách thành `['3', 'pm']` vì chúng ta xen kẽ giữa các chuỗi chữ cái và chuỗi chữ số. Đủ tốt cho hầu hết các tác vụ. URL, email, hashtag đều bị hỏng. Đối với môi trường production, hãy thêm các mẫu trước các mẫu chung.

### Bước 2: Porter stemmer (chỉ bước 1a)

Thuật toán Porter đầy đủ có năm giai đoạn quy tắc. Chỉ riêng bước 1a đã bao phủ các hậu tố tiếng Anh thường gặp nhất và dạy cho bạn về mô hình này.

```python
def stem_step_1a(word):
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s") and len(word) > 1:
        return word[:-1]
    return word
```

```python
>>> [stem_step_1a(w) for w in ["caresses", "ponies", "caress", "cats"]]
['caress', 'poni', 'caress', 'cat']
```

Đọc các quy tắc từ trên xuống dưới. Quy tắc `ies -> i` là lý do tại sao ra `ponies -> poni`, chứ không phải `pony`. Porter thực tế có bước 1b để sửa lỗi này. Các quy tắc cạnh tranh nhau. Quy tắc nào đứng trước sẽ thắng. Thứ tự quan trọng hơn bất kỳ quy tắc đơn lẻ nào.

### Bước 3: Lemmatizer dựa trên tra cứu

Lemmatization thực thụ cần hình thái học. Một phiên bản giảng dạy dễ hiểu sử dụng bảng lemma nhỏ và một phương án dự phòng.

```python
LEMMA_TABLE = {
    ("running", "VERB"): "run",
    ("ran", "VERB"): "run",
    ("runs", "VERB"): "run",
    ("better", "ADJ"): "good",
    ("best", "ADJ"): "good",
    ("cats", "NOUN"): "cat",
    ("cat", "NOUN"): "cat",
    ("were", "VERB"): "be",
    ("was", "VERB"): "be",
    ("is", "VERB"): "be",
}

def lemmatize(word, pos):
    key = (word.lower(), pos)
    if key in LEMMA_TABLE:
        return LEMMA_TABLE[key]
    if pos == "VERB" and word.endswith("ing"):
        return word[:-3]
    if pos == "NOUN" and word.endswith("s"):
        return word[:-1]
    return word.lower()
```

```python
>>> lemmatize("running", "VERB")
'run'
>>> lemmatize("cats", "NOUN")
'cat'
>>> lemmatize("better", "ADJ")
'good'
>>> lemmatize("watched", "VERB")
'watched'
```

Trường hợp cuối cùng là điểm mấu chốt. `watched` không có trong bảng của chúng ta và phương án dự phòng chỉ xử lý `ing`. Lemmatization thực tế bao phủ `ed`, động từ bất quy tắc, tính từ so sánh, số nhiều có thay đổi âm thanh (`children -> child`). Đây là lý do tại sao các hệ thống production sử dụng WordNet, bộ morphologizer của spaCy, hoặc một bộ phân tích hình thái đầy đủ.

### Bước 4: Kết nối chúng lại

```python
def preprocess(text, pos_tagger=None):
    tokens = tokenize(text)
    stems = [stem_step_1a(t.lower()) for t in tokens]
    tags = pos_tagger(tokens) if pos_tagger else [(t, "NOUN") for t in tokens]
    lemmas = [lemmatize(word, pos) for word, pos in tags]
    return {"tokens": tokens, "stems": stems, "lemmas": lemmas}
```

Mảnh ghép còn thiếu là một bộ POS tagger. Phase 5 · 07 (POS Tagging) sẽ xây dựng một bộ như vậy. Hiện tại, hãy mặc định mọi thứ là `NOUN` và thừa nhận hạn chế này.

## Sử dụng

NLTK và spaCy cung cấp các phiên bản dùng cho production. Mỗi loại chỉ cần vài dòng code.

### NLTK

```python
import nltk
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger_eng")

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

text = "The cats were running."
tokens = word_tokenize(text)
stems = [PorterStemmer().stem(t) for t in tokens]
lemmatizer = WordNetLemmatizer()
tagged = pos_tag(tokens)


def nltk_pos_to_wordnet(tag):
    if tag.startswith("V"):
        return "v"
    if tag.startswith("J"):
        return "a"
    if tag.startswith("R"):
        return "r"
    return "n"


lemmas = [lemmatizer.lemmatize(t, nltk_pos_to_wordnet(tag)) for t, tag in tagged]
```

`word_tokenize` xử lý các từ viết tắt, Unicode, các trường hợp biên mà regex của bạn bỏ lỡ. `PorterStemmer` chạy cả năm giai đoạn. `WordNetLemmatizer` cần POS tag được chuyển đổi từ lược đồ Penn Treebank của NLTK sang tập viết tắt của WordNet. Việc kết nối chuyển đổi ở trên là phần mà hầu hết các hướng dẫn đều bỏ qua.

### spaCy

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running.")

for token in doc:
    print(token.text, token.lemma_, token.pos_)
```

```
The      the     DET
cats     cat     NOUN
were     be      AUX
running  run     VERB
.        .       PUNCT
```

spaCy ẩn toàn bộ pipeline đằng sau `nlp(text)`. Tokenization, POS tagging, và lemmatization đều được thực hiện. Nhanh hơn NLTK ở quy mô lớn. Chính xác hơn ngay khi cài đặt. Đánh đổi là bạn không thể dễ dàng thay thế các thành phần riêng lẻ.

### Khi nào chọn cái nào

| Tình huống | Chọn |
|-----------|------|
| Giảng dạy, nghiên cứu, thay thế thành phần | NLTK |
| Production, đa ngôn ngữ, ưu tiên tốc độ | spaCy |
| Transformer pipeline (dù sao bạn cũng sẽ tokenize bằng tokenizer của mô hình) | Sử dụng `tokenizers` / `transformers` và bỏ qua tiền xử lý cổ điển |

### Hai chế độ lỗi mà không ai cảnh báo bạn

Hầu hết các hướng dẫn đều dạy thuật toán rồi dừng lại. Hai điều sau đây sẽ gây khó khăn cho một pipeline tiền xử lý thực tế, và chúng hầu như không bao giờ được đề cập.

**Sự trôi dạt về khả năng tái lập (Reproducibility drift).** NLTK và spaCy thay đổi hành vi tokenization và lemmatizer giữa các phiên bản. Những gì tạo ra `['do', "n't"]` trong spaCy 2.x có thể tạo ra `["don't"]` trong 3.x. Mô hình của bạn đã được huấn luyện trên một phân phối. Quá trình suy luận (inference) hiện chạy trên một phân phối khác. Độ chính xác giảm dần một cách âm thầm mà không ai biết tại sao. Hãy ghim (pin) các phiên bản thư viện trong `requirements.txt`. Viết một bài kiểm tra hồi quy tiền xử lý để cố định kết quả tokenization mong đợi của 20 câu mẫu. Chạy nó mỗi khi nâng cấp.

**Sự không khớp giữa huấn luyện / suy luận.** Huấn luyện với tiền xử lý mạnh (viết thường, loại bỏ stopword, stemming), triển khai trên dữ liệu người dùng thô, và chứng kiến hiệu suất sụt giảm. Đây là lỗi NLP phổ biến nhất trong production. Nếu bạn tiền xử lý trong quá trình huấn luyện, bạn phải chạy chính xác hàm đó trong quá trình suy luận. Hãy đóng gói tiền xử lý như một hàm bên trong gói mô hình, không phải là một ô trong notebook mà nhóm vận hành phải viết lại.

## Triển khai

Một prompt có thể tái sử dụng giúp các kỹ sư chọn chiến lược tiền xử lý mà không cần đọc ba cuốn sách giáo khoa.

Lưu dưới dạng `outputs/prompt-preprocessing-advisor.md`:

```markdown
---
name: preprocessing-advisor
description: Recommends a tokenization, stemming, and lemmatization setup for an NLP task.
phase: 5
lesson: 01
---

You advise on classical NLP preprocessing. Given a task description, you output:

1. Tokenization choice (regex, NLTK word_tokenize, spaCy, or transformer tokenizer). Explain why.
2. Whether to stem, lemmatize, both, or neither. Explain why.
3. Specific library calls. Name the functions. Quote the POS-tag translation if NLTK is involved.
4. One failure mode the user should test for.

Refuse to recommend stemming for user-visible text. Refuse to recommend lemmatization without POS tags. Flag non-English input as needing a different pipeline.
```

## Bài tập

1. **Dễ.** Mở rộng `tokenize` để giữ các URL dưới dạng các token đơn lẻ. Kiểm tra: `tokenize("Visit https://example.com today.")` sẽ tạo ra một token URL duy nhất.
2. **Trung bình.** Triển khai bước 1b của Porter. Nếu một từ chứa nguyên âm và kết thúc bằng `ed` hoặc `ing`, hãy loại bỏ nó. Xử lý quy tắc phụ âm kép (`hopping -> hop`, không phải `hopp`).
3. **Khó.** Xây dựng một lemmatizer sử dụng WordNet làm bảng tra cứu nhưng quay lại sử dụng Porter stemmer của bạn khi WordNet không có mục nhập. Đo lường độ chính xác trên một tập dữ liệu đã được gắn nhãn so với WordNet thuần và Porter thuần.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| Token | Một từ | Bất kỳ đơn vị nào mà mô hình tiêu thụ. Có thể là từ, subword, ký tự, hoặc byte. |
| Stem | Gốc của từ | Kết quả của việc cắt hậu tố dựa trên quy tắc. Không phải lúc nào cũng là một từ thực sự. |
| Lemma | Dạng từ điển | Dạng bạn sẽ tra cứu. Cần ngữ cảnh ngữ pháp để tính toán chính xác. |
| POS tag | Từ loại | Danh mục như NOUN, VERB, ADJ. Cần thiết để lemmatize chính xác. |
| Morphology | Quy tắc hình thái từ | Cách một từ thay đổi dạng dựa trên thì, số, cách. Lemmatization phụ thuộc vào nó. |

## Đọc thêm

- [Porter, M. F. (1980). An algorithm for suffix stripping](https://tartarus.org/martin/PorterStemmer/def.txt) — bài báo gốc, năm trang, vẫn là lời giải thích rõ ràng nhất.
- [spaCy 101 — linguistic features](https://spacy.io/usage/linguistic-features) — cách một pipeline thực tế được kết nối.
- [NLTK book, chapter 3](https://www.nltk.org/book/ch03.html) — các trường hợp biên của tokenization mà bạn chưa từng nghĩ tới.