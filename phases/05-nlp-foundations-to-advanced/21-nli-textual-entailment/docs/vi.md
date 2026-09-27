# Natural Language Inference — Textual Entailment

> "t entails h" có nghĩa là một người đọc t sẽ kết luận h là đúng. NLI là tác vụ dự đoán mối quan hệ entailment (kéo theo) / contradiction (mâu thuẫn) / neutral (trung lập). Nghe có vẻ nhàm chán nhưng lại là nền tảng quan trọng trong môi trường production.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 05 (Sentiment Analysis), Phase 5 · 13 (Question Answering)
**Time:** ~60 minutes

## The Problem

Bạn đã xây dựng một bộ tóm tắt văn bản (summarizer). Nó tạo ra một bản tóm tắt. Làm thế nào để bạn biết bản tóm tắt đó không chứa thông tin bịa đặt (hallucination)?

Bạn đã xây dựng một chatbot. Nó trả lời "có". Làm thế nào để bạn biết câu trả lời đó được hỗ trợ bởi đoạn văn bản đã truy xuất (retrieved passage)?

Bạn cần phân loại 10.000 bài báo theo chủ đề. Bạn không có nhãn huấn luyện. Bạn có thể tái sử dụng một mô hình có sẵn không?

Cả ba vấn đề trên đều quy về Natural Language Inference. NLI đặt câu hỏi: với một tiền đề `t` và một giả thuyết `h`, liệu `h` có được kéo theo bởi `t`, hay là mâu thuẫn, hoặc trung lập (không liên quan)?

- **Kiểm tra Hallucination:** `t` = tài liệu nguồn, `h` = khẳng định trong tóm tắt. Nếu không phải entailment = hallucination.
- **Grounded QA:** `t` = đoạn văn bản truy xuất, `h` = câu trả lời được tạo ra. Nếu không phải entailment = thông tin bịa đặt.
- **Zero-shot classification:** `t` = tài liệu, `h` = nhãn được diễn đạt thành lời ("Đây là về thể thao"). Nếu là entailment = nhãn được dự đoán.

Một tác vụ, ba ứng dụng trong production. Đây là lý do tại sao mọi framework đánh giá RAG đều tích hợp sẵn một mô hình NLI bên dưới.

## The Concept

![NLI: three-way classification, premise vs hypothesis](../assets/nli.svg)

**Ba loại nhãn.**

- **Entailment.** `t` → `h`. "Con mèo đang ở trên tấm thảm" kéo theo "Có một con mèo".
- **Contradiction.** `t` → ¬`h`. "Con mèo đang ở trên tấm thảm" mâu thuẫn với "Không có con mèo nào cả".
- **Neutral.** Không có suy luận nào theo cả hai hướng. "Con mèo đang ở trên tấm thảm" là trung lập với "Con mèo đang đói".

**Không phải là logic hình thức.** NLI là suy luận ngôn ngữ *tự nhiên* — những gì một người đọc bình thường sẽ suy luận, không phải logic chặt chẽ. "John dắt chó đi dạo" kéo theo "John có một con chó" trong NLI, nhưng logic bậc nhất (first-order logic) chỉ thừa nhận điều đó nếu bạn tiên đề hóa quyền sở hữu.

**Các tập dữ liệu.**

- **SNLI** (2015). 570k cặp câu được con người gán nhãn, sử dụng chú thích ảnh làm tiền đề. Phạm vi hẹp.
- **MultiNLI** (2017). 433k cặp câu thuộc 10 thể loại khác nhau. Tập dữ liệu huấn luyện tiêu chuẩn vào năm 2026.
- **ANLI** (2019). Adversarial NLI. Con người viết các ví dụ được thiết kế đặc biệt để đánh bại các mô hình hiện có. Khó hơn.
- **DocNLI, ConTRoL** (2020–21). Tiền đề có độ dài cấp tài liệu. Kiểm tra khả năng suy luận đa chặng (multi-hop) và suy luận tầm xa.

**Kiến trúc.** Một transformer encoder (BERT, RoBERTa, DeBERTa) đọc `[CLS] premise [SEP] hypothesis [SEP]`. Đại diện `[CLS]` được đưa vào một lớp softmax 3 chiều. Huấn luyện trên MNLI, đánh giá trên các benchmark độc lập, đạt độ chính xác 90%+ trên các cặp dữ liệu cùng phân phối.

**Zero-shot thông qua NLI.** Với một tài liệu và các nhãn ứng viên, hãy biến mỗi nhãn thành một giả thuyết ("Văn bản này nói về thể thao"). Tính xác suất entailment cho mỗi nhãn. Chọn nhãn có xác suất cao nhất. Đây là cơ chế đằng sau pipeline `zero-shot-classification` của Hugging Face.

```figure
nli-router
```

## Build It

### Step 1: chạy một mô hình NLI đã được huấn luyện trước

```python
from transformers import pipeline

nli = pipeline("text-classification",
               model="facebook/bart-large-mnli",
               top_k=None)  # return all labels; replaces deprecated return_all_scores=True

premise = "The cat is sleeping on the couch."
hypothesis = "There is a cat in the room."

result = nli({"text": premise, "text_pair": hypothesis})[0]
print(result)
# [{'label': 'entailment', 'score': 0.97},
#  {'label': 'neutral', 'score': 0.02},
#  {'label': 'contradiction', 'score': 0.01}]
```

Đối với NLI trong production, `facebook/bart-large-mnli` và `microsoft/deberta-v3-large-mnli` là các lựa chọn mặc định mã nguồn mở. DeBERTa-v3 đang đứng đầu các bảng xếp hạng.

### Step 2: zero-shot classification

```python
zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "The stock market rallied after the central bank cut interest rates."
labels = ["finance", "sports", "politics", "technology"]

result = zs(text, candidate_labels=labels)
print(result)
# {'labels': ['finance', 'politics', 'technology', 'sports'],
#  'scores': [0.92, 0.05, 0.02, 0.01]}
```

Template mặc định là "This example is about {label}.". Tùy chỉnh với `hypothesis_template`. Không cần dữ liệu huấn luyện. Không cần fine-tuning. Hoạt động ngay lập tức.

### Step 3: kiểm tra độ trung thực (faithfulness) cho RAG

```python
def is_faithful(answer, context, threshold=0.5):
    result = nli({"text": context, "text_pair": answer})[0]
    entail = next(s for s in result if s["label"] == "entailment")
    return entail["score"] > threshold
```

Đây là cốt lõi của tính faithfulness trong RAGAS. Chia câu trả lời được tạo ra thành các khẳng định nguyên tử (atomic claims). Kiểm tra từng khẳng định dựa trên ngữ cảnh đã truy xuất. Báo cáo tỷ lệ các khẳng định được kéo theo (entail).

### Step 4: tự xây dựng bộ phân loại NLI (khái niệm)

Xem `code/main.py` để biết một ví dụ đơn giản chỉ dùng thư viện chuẩn: tiền đề và giả thuyết được so sánh thông qua sự trùng lặp từ vựng + phát hiện phủ định. Không cạnh tranh được với các mô hình transformer — nhưng nó cho thấy hình thái của tác vụ: đầu vào là hai văn bản, đầu ra là nhãn 3 chiều, hàm mất mát = cross-entropy trên `{entail, contradict, neutral}`.

## Pitfalls

- **Phím tắt chỉ dựa trên giả thuyết.** Các mô hình có thể dự đoán nhãn chỉ từ giả thuyết với độ chính xác ~60% trên SNLI vì các từ "not", "nobody", "never" tương quan với mâu thuẫn. Đây là baseline mạnh để phát hiện sự rò rỉ nhãn.
- **Heuristic trùng lặp từ vựng.** Heuristic chuỗi con ("mọi chuỗi con đều được kéo theo") vượt qua SNLI nhưng thất bại với HANS/ANLI. Hãy sử dụng các benchmark đối nghịch (adversarial).
- **Suy giảm hiệu năng với tài liệu dài.** Các mô hình NLI cấp câu đơn giảm 20+ điểm F1 trên các tiền đề dài. Hãy sử dụng các mô hình được huấn luyện với DocNLI cho ngữ cảnh dài.
- **Độ nhạy của template zero-shot.** "This example is about {label}" so với "{label}" so với "The topic is {label}" có thể làm thay đổi độ chính xác tới 10+ điểm. Hãy tinh chỉnh template.
- **Lệch miền dữ liệu.** MNLI huấn luyện trên tiếng Anh tổng quát. Văn bản pháp lý, y tế và khoa học cần các mô hình NLI chuyên biệt (ví dụ: SciNLI, MedNLI).

## Use It

Stack năm 2026:

| Use case | Model |
|---------|-------|
| NLI đa mục đích | `microsoft/deberta-v3-large-mnli` |
| Nhanh / edge | `cross-encoder/nli-deberta-v3-base` |
| Zero-shot classification (nhẹ) | `facebook/bart-large-mnli` |
| NLI cấp tài liệu | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` |
| Đa ngôn ngữ | `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` |
| Phát hiện hallucination trong RAG | Lớp NLI bên trong RAGAS / DeepEval |

Meta-pattern năm 2026: NLI là "băng dính" của việc hiểu văn bản. Bất cứ khi nào bạn cần biết "A có hỗ trợ B không?" hoặc "A có mâu thuẫn với B không?" — hãy tìm đến NLI trước khi gọi một LLM khác.

## Ship It

Lưu dưới dạng `outputs/skill-nli-picker.md`:

```markdown
---
name: nli-picker
description: Pick an NLI model, label template, and evaluation setup for a classification / faithfulness / zero-shot task.
version: 1.0.0
phase: 5
lesson: 21
tags: [nlp, nli, zero-shot]
---

Given a use case (faithfulness check, zero-shot classification, document-level inference), output:

1. Model. Named NLI checkpoint. Reason tied to domain, length, language.
2. Template (if zero-shot). Verbalization pattern. Example.
3. Threshold. Entailment cutoff for the decision rule. Reason based on calibration.
4. Evaluation. Accuracy on held-out labeled set, hypothesis-only baseline, adversarial subset.

Refuse to ship zero-shot classification without a 100-example labeled sanity check. Refuse to use a sentence-level NLI model on document-length premises. Flag any claim that NLI solves hallucination — it reduces it; it does not eliminate it.
```

## Exercises

1. **Dễ.** Chạy `facebook/bart-large-mnli` trên 20 bộ ba (tiền đề, giả thuyết, nhãn) tự tạo bao gồm cả ba lớp. Đo lường độ chính xác. Thêm các bẫy "heuristic chuỗi con" đối nghịch ("I did not eat the cake" so với "I ate the cake") và xem liệu nó có bị lỗi không.
2. **Trung bình.** So sánh template zero-shot `"This text is about {label}"` với `"The topic is {label}"` và `"{label}"` trên 100 tiêu đề tin tức AG News. Báo cáo sự thay đổi độ chính xác.
3. **Khó.** Xây dựng bộ kiểm tra độ trung thực RAG: phân tách khẳng định nguyên tử + NLI cho mỗi khẳng định. Đánh giá trên 50 câu trả lời do RAG tạo ra với ngữ cảnh chuẩn (gold context). Đo lường tỷ lệ dương tính giả và âm tính giả so với nhãn thủ công.

## Key Terms

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| NLI | Natural Language Inference | Phân loại 3 chiều mối quan hệ tiền đề-giả thuyết. |
| RTE | Recognizing Textual Entailment | Tên cũ của NLI; cùng một tác vụ. |
| Entailment | "t implies h" | Một người đọc bình thường sẽ kết luận h là đúng nếu có t. |
| Contradiction | "t rules out h" | Một người đọc bình thường sẽ kết luận h là sai nếu có t. |
| Neutral | "undecided" | Không có suy luận nào từ t đến h theo cả hai hướng. |
| Zero-shot classification | NLI as classifier | Diễn đạt nhãn thành giả thuyết, chọn entailment cao nhất. |
| Faithfulness | Is the answer supported? | NLI trên cặp (ngữ cảnh truy xuất, câu trả lời được tạo). |

## Further Reading

- [Bowman et al. (2015). A large annotated corpus for learning natural language inference](https://arxiv.org/abs/1508.05326) — SNLI.
- [Williams, Nangia, Bowman (2017). A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference](https://arxiv.org/abs/1704.05426) — MultiNLI.
- [Nie et al. (2019). Adversarial NLI](https://arxiv.org/abs/1910.14599) — benchmark ANLI.
- [Yin, Hay, Roth (2019). Benchmarking Zero-shot Text Classification](https://arxiv.org/abs/1909.00161) — NLI-as-classifier.
- [He et al. (2021). DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654) — mô hình NLI chủ lực năm 2026.