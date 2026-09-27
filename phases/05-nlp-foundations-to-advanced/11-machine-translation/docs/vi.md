# Machine Translation

> Dịch thuật là tác vụ đã nuôi sống nghiên cứu NLP trong ba mươi năm qua và vẫn đang tiếp tục như vậy.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 10 (Attention Mechanism), Phase 5 · 04 (GloVe, FastText, Subword)
**Time:** ~75 phút

## Vấn đề

Một mô hình đọc một câu bằng ngôn ngữ này và tạo ra một câu bằng ngôn ngữ khác. Độ dài thay đổi. Thứ tự từ thay đổi. Một số từ nguồn ánh xạ tới nhiều từ đích và ngược lại. Các thành ngữ không thể dịch theo kiểu một-một. "I miss you" trong tiếng Pháp là "tu me manques" — nghĩa đen là "bạn đang thiếu đối với tôi". Không có sự căn chỉnh (alignment) ở cấp độ từ nào có thể tồn tại được với điều đó.

Dịch máy là tác vụ đã buộc NLP phải phát minh ra encoder-decoder, attention, transformer và cuối cùng là toàn bộ mô hình LLM. Mỗi bước tiến đều đạt được vì chất lượng dịch thuật có thể đo lường được và khoảng cách giữa con người và máy móc vẫn còn rất lớn.

Bài học này bỏ qua phần lịch sử và dạy về quy trình làm việc của năm 2026: pretrained multilingual encoder-decoder (NLLB-200 hoặc mBART), subword tokenization, beam search, đánh giá BLEU và chrF, cùng với một vài chế độ lỗi (failure modes) vẫn thường xuất hiện trong môi trường production mà chưa được xử lý.

## Khái niệm

![MT pipeline: tokenize → encode → decode with attention → detokenize](../assets/mt-pipeline.svg)

MT hiện đại là một transformer encoder-decoder được huấn luyện trên văn bản song ngữ (parallel text). Encoder đọc văn bản nguồn theo cách token hóa của ngôn ngữ đó. Decoder tạo ra văn bản đích, từng subword một, sử dụng đầu ra của encoder thông qua cross-attention (bài học 10). Quá trình giải mã (decoding) sử dụng beam search để tránh cái bẫy của greedy-decoding. Đầu ra được detokenized, detruecased và chấm điểm dựa trên một bản tham chiếu.

Ba lựa chọn vận hành quyết định chất lượng MT trong thực tế.

- **Tokenizer.** SentencePiece BPE được huấn luyện trên một tập dữ liệu đa ngôn ngữ. Từ vựng dùng chung giữa các ngôn ngữ là thứ cho phép các cặp zero-shot trong NLLB.
- **Kích thước mô hình.** NLLB-200 distilled 600M có thể chạy trên laptop. NLLB-200 3.3B là mặc định cho production. 54.5B là giới hạn cho nghiên cứu.
- **Giải mã (Decoding).** Beam width 4-5 cho nội dung thông thường. Length penalty để tránh đầu ra quá ngắn. Constrained decoding khi bạn cần sự nhất quán về thuật ngữ.

```figure
seq2seq-alignment
```

## Xây dựng

### Bước 1: gọi một pretrained MT

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = "facebook/nllb-200-distilled-600M"
tok = AutoTokenizer.from_pretrained(model_id, src_lang="eng_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

src = "The cats are running."
inputs = tok(src, return_tensors="pt")

out = model.generate(
    **inputs,
    forced_bos_token_id=tok.convert_tokens_to_ids("fra_Latn"),
    num_beams=5,
    length_penalty=1.0,
    max_new_tokens=64,
)
print(tok.batch_decode(out, skip_special_tokens=True)[0])
```

```text
Les chats courent.
```

Có ba điều quan trọng ở đây. `src_lang` cho tokenizer biết script và cách phân đoạn nào cần áp dụng. `forced_bos_token_id` cho decoder biết ngôn ngữ nào cần tạo ra. Cả hai đều là các thủ thuật đặc thù của NLLB; mBART và M2M-100 sử dụng các quy ước riêng của chúng và chúng không thể thay thế cho nhau.

### Bước 2: BLEU và chrF

BLEU đo lường sự trùng lặp n-gram giữa đầu ra và bản tham chiếu. Bốn kích thước n-gram tham chiếu (1-4), trung bình nhân của độ chính xác (precisions), brevity penalty cho đầu ra quá ngắn. Điểm số nằm trong khoảng [0, 100]. Được sử dụng phổ biến. Khó diễn giải: 30 BLEU là "có thể sử dụng"; 40 là "tốt"; 50 là "xuất sắc"; sự khác biệt dưới 1 BLEU chỉ là nhiễu.

chrF đo lường F-score ở cấp độ ký tự. Nhạy bén hơn với các ngôn ngữ giàu hình thái (morphologically rich) nơi BLEU đếm thiếu các kết quả khớp. Thường được báo cáo cùng với BLEU.

```python
import sacrebleu

hypotheses = ["Les chats courent."]
references = [["Les chats courent."]]

bleu = sacrebleu.corpus_bleu(hypotheses, references)
chrf = sacrebleu.corpus_chrf(hypotheses, references)
print(f"BLEU: {bleu.score:.1f}  chrF: {chrf.score:.1f}")
```

Luôn sử dụng `sacrebleu`. Nó chuẩn hóa việc token hóa để các điểm số có thể so sánh được giữa các bài báo. Việc tự viết hàm tính BLEU là nguyên nhân dẫn đến các kết quả benchmark gây hiểu lầm.

### Hệ thống phân cấp đánh giá ba tầng (2026)

Đánh giá MT hiện đại sử dụng ba nhóm chỉ số bổ trợ. Hãy sử dụng ít nhất hai nhóm.

- **Heuristic** (BLEU, chrF). Nhanh, dựa trên tham chiếu, dễ diễn giải, không nhạy với cách diễn đạt lại (paraphrase). Sử dụng để so sánh kế thừa và phát hiện hồi quy.
- **Learned** (COMET, BLEURT, BERTScore). Các mô hình thần kinh được huấn luyện dựa trên đánh giá của con người; so sánh sự tương đồng về ngữ nghĩa của bản dịch với nguồn và tham chiếu. COMET có mối liên hệ cao nhất với nghiên cứu MT từ năm 2023 và là mặc định cho production năm 2026 khi chất lượng là ưu tiên hàng đầu.
- **LLM-as-judge** (không cần tham chiếu). Yêu cầu một mô hình lớn chấm điểm các bản dịch dựa trên độ trôi chảy, độ đầy đủ, tông giọng, sự phù hợp văn hóa. GPT-4-as-judge khớp với đánh giá của con người khoảng 80% khi rubric được thiết kế tốt. Sử dụng cho nội dung mở nơi không có tham chiếu.

Stack thực tế năm 2026: `sacrebleu` cho BLEU và chrF, `unbabel-comet` cho COMET, và một LLM được prompt để lấy tín hiệu cuối cùng từ con người. Hãy hiệu chuẩn mọi chỉ số dựa trên 50-100 ví dụ được con người dán nhãn trước khi tin tưởng vào dữ liệu production.

Các chỉ số không cần tham chiếu (COMET-QE, BLEURT-QE, LLM-as-judge) cho phép bạn đánh giá bản dịch mà không cần tham chiếu, điều này quan trọng đối với các cặp ngôn ngữ hiếm (long-tail) nơi không có bản dịch tham chiếu.

### Bước 3: những lỗi thường gặp trong production

Quy trình làm việc ở trên sẽ dịch trôi chảy 80% thời gian và thất bại âm thầm trong 20% còn lại. Các chế độ lỗi được đặt tên:

- **Hallucination.** Mô hình tự tạo ra nội dung không có trong nguồn. Phổ biến trong các lĩnh vực từ vựng lạ. Triệu chứng: đầu ra trôi chảy nhưng khẳng định các sự kiện mà nguồn không đề cập. Cách giảm thiểu: constrained decoding trên các thuật ngữ chuyên ngành, con người kiểm duyệt nội dung được quản lý, giám sát đầu ra dài hơn nhiều so với đầu vào.
- **Off-target generation.** Mô hình dịch sang sai ngôn ngữ. NLLB khá dễ mắc lỗi này trên các cặp ngôn ngữ hiếm. Cách giảm thiểu: xác minh `forced_bos_token_id` và luôn giải mã với một mô hình kiểm tra ID ngôn ngữ trên đầu ra.
- **Terminology drift.** "Sign up" trở thành "s'inscrire" trong tài liệu 1 và "créer un compte" trong tài liệu 2. Đối với văn bản UI và các chuỗi hiển thị cho người dùng, sự nhất quán quan trọng hơn chất lượng thô. Cách giảm thiểu: giải mã có ràng buộc thuật ngữ (glossary-constrained decoding) hoặc từ điển hậu chỉnh sửa.
- **Formality mismatch.** "Tu" vs "vous" trong tiếng Pháp, các cấp độ lịch sự trong tiếng Nhật. Mô hình chọn hình thức phổ biến nhất trong quá trình huấn luyện. Đối với nội dung hướng tới khách hàng, điều này thường sai. Cách giảm thiểu: prompt tiền tố với một token hình thức nếu mô hình hỗ trợ, hoặc fine-tune một mô hình nhỏ trên các tập dữ liệu chỉ có văn phong trang trọng.
- **Length explosion on short input.** Các câu đầu vào rất ngắn thường tạo ra bản dịch quá dài vì hình phạt độ dài (length penalty) giảm mạnh dưới ~5 token nguồn. Cách giảm thiểu: giới hạn độ dài tối đa cứng tỷ lệ thuận với độ dài nguồn.

### Bước 4: fine-tuning cho một lĩnh vực

Các mô hình pretrained là những mô hình tổng quát. Dịch thuật pháp lý, y tế hoặc hội thoại trong game được hưởng lợi đáng kể từ việc fine-tune trên dữ liệu song ngữ chuyên ngành. Công thức không có gì lạ:

```python
from transformers import Trainer, TrainingArguments
from datasets import Dataset

pairs = [
    {"src": "The defendant pleaded guilty.", "tgt": "L'accusé a plaidé coupable."},
]

ds = Dataset.from_list(pairs)


def preprocess(ex):
    return tok(
        ex["src"],
        text_target=ex["tgt"],
        truncation=True,
        max_length=128,
        padding="max_length",
    )


ds = ds.map(preprocess, remove_columns=["src", "tgt"])

args = TrainingArguments(output_dir="out", per_device_train_batch_size=4, num_train_epochs=3, learning_rate=3e-5)
Trainer(model=model, args=args, train_dataset=ds).train()
```

Vài nghìn ví dụ song ngữ chất lượng cao tốt hơn hàng trăm nghìn ví dụ nhiễu được thu thập từ web. Chất lượng dữ liệu huấn luyện là đòn bẩy lớn nhất trong production.

## Sử dụng

Stack production năm 2026 cho MT:

| Trường hợp sử dụng | Điểm bắt đầu khuyến nghị |
|---------|---------------------------|
| Bất kỳ ngôn ngữ nào, 200 ngôn ngữ | `facebook/nllb-200-distilled-600M` (laptop) hoặc `nllb-200-3.3B` (production) |
| Tập trung vào tiếng Anh, chất lượng cao, 50 ngôn ngữ | `facebook/mbart-large-50-many-to-many-mmt` |
| Chạy nhanh, inference rẻ, Anh-Pháp/Đức/Tây Ban Nha | Helsinki-NLP / Marian models |
| Độ trễ thấp, phía trình duyệt | ONNX-quantized Marian (~50 MB) |
| Chất lượng tối đa, sẵn sàng chi trả | GPT-4 / Claude / Gemini với các prompt dịch thuật |

LLM hiện đã vượt qua các mô hình MT chuyên dụng trên một số cặp ngôn ngữ tính đến năm 2026, đặc biệt là về nội dung thành ngữ và ngữ cảnh dài. Sự đánh đổi là chi phí mỗi token và độ trễ. Hãy chọn LLM khi độ dài ngữ cảnh, sự nhất quán về phong cách hoặc thích ứng lĩnh vực thông qua prompt quan trọng hơn thông lượng (throughput).

## Triển khai

Lưu dưới dạng `outputs/skill-mt-evaluator.md`:

```markdown
---
name: mt-evaluator
description: Evaluate a machine translation output for shipping.
version: 1.0.0
phase: 5
lesson: 11
tags: [nlp, translation, evaluation]
---

Given a source text and a candidate translation, output:

1. Automatic score estimate. BLEU and chrF ranges you would expect. State whether a reference is available.
2. Five-point human-verifiable check list: (a) content preservation (no hallucinations), (b) correct language, (c) register / formality match, (d) terminology consistency with glossary if provided, (e) no truncation or length explosion.
3. One domain-specific issue to probe. E.g., for legal: named entities and statute citations. For medical: drug names and dosages. For UI: placeholder variables `{name}`.
4. Confidence flag. "Ship" / "Ship with review" / "Do not ship". Tie to the severity of issues found in step 2.

Refuse to ship a translation without a language-ID check on output. Refuse to evaluate without a reference unless the user explicitly opts in to reference-free scoring (COMET-QE, BLEURT-QE). Flag any content over 1000 tokens as likely needing chunked translation.
```

## Bài tập

1. **Dễ.** Dịch một đoạn văn tiếng Anh 5 câu sang tiếng Pháp và ngược lại sang tiếng Anh bằng `nllb-200-distilled-600M`. Đo lường mức độ gần gũi của vòng lặp so với bản gốc. Bạn sẽ thấy sự bảo toàn ngữ nghĩa với sự thay đổi trong lựa chọn từ ngữ.
2. **Trung bình.** Triển khai kiểm tra ID ngôn ngữ trên đầu ra bản dịch bằng `fasttext lid.176` hoặc `langdetect`. Tích hợp vào lệnh gọi MT để các thế hệ sai ngôn ngữ bị bắt trước khi trả về.
3. **Khó.** Fine-tune `nllb-200-distilled-600M` trên tập dữ liệu chuyên ngành gồm 5.000 cặp câu do bạn chọn. Đo lường BLEU trên tập dữ liệu giữ lại (held-out set) trước và sau khi fine-tune. Báo cáo loại câu nào được cải thiện và loại nào bị giảm sút.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| BLEU | Điểm dịch thuật | Độ chính xác n-gram với hình phạt độ dài. [0, 100]. |
| chrF | F-score ký tự | F-score cấp độ ký tự. Nhạy hơn cho các ngôn ngữ giàu hình thái. |
| NMT | MT thần kinh | Transformer encoder-decoder được huấn luyện trên văn bản song ngữ. Mặc định từ 2017+. |
| NLLB | No Language Left Behind | Họ mô hình MT 200 ngôn ngữ của Meta. |
| Constrained decoding | Đầu ra có kiểm soát | Ép buộc các token hoặc n-gram cụ thể xuất hiện / không xuất hiện trong đầu ra. |
| Hallucination | Nội dung tự tạo | Đầu ra của mô hình không được hỗ trợ bởi nguồn. |

## Đọc thêm

- [Costa-jussà et al. (2022). No Language Left Behind: Scaling Human-Centered Machine Translation](https://arxiv.org/abs/2207.04672) — bài báo về NLLB.
- [Post (2018). A Call for Clarity in Reporting BLEU Scores](https://aclanthology.org/W18-6319/) — tại sao `sacrebleu` là cách duy nhất đúng để báo cáo điểm BLEU.
- [Popović (2015). chrF: character n-gram F-score for automatic MT evaluation](https://aclanthology.org/W15-3049/) — bài báo về chrF.
- [Hugging Face MT guide](https://huggingface.co/docs/transformers/tasks/translation) — hướng dẫn thực hành fine-tuning.