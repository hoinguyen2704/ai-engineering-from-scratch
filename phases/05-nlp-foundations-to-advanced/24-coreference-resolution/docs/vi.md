# Coreference Resolution

> "She called him. He did not answer. The doctor was at lunch." Ba tham chiếu đến hai người và không ai được gọi tên. Coreference resolution (giải quyết tham chiếu) giúp xác định ai là ai.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 06 (NER), Phase 5 · 07 (POS & Parsing)
**Time:** ~60 minutes

## The Problem

Trích xuất mọi đề cập đến Apple Inc. từ một bài báo dài 300 từ. Việc này rất dễ khi bài báo viết "Apple". Nhưng sẽ rất khó khi nó viết "the company", "they", "Cupertino's technology giant", hoặc "Jobs's firm". Nếu không giải quyết được các đề cập này về cùng một thực thể, pipeline NER của bạn sẽ bỏ lỡ 60-80% các đề cập.

Coreference resolution liên kết mọi biểu thức đề cập đến cùng một thực thể trong thế giới thực thành một cụm (cluster). Đây là chất keo kết nối giữa NLP bề mặt (NER, parsing) và ngữ nghĩa chuyên sâu (IE, QA, tóm tắt, KG).

Tại sao nó quan trọng trong năm 2026:

- Tóm tắt: "The CEO announced..." so với "Tim Cook announced..." — bản tóm tắt nên nêu tên CEO.
- Trả lời câu hỏi: "Who did she call?" đòi hỏi phải giải quyết được "she" là ai.
- Trích xuất thông tin: một knowledge graph có "PER1 founded Apple" và "Jobs founded Apple" là các mục riêng biệt là sai.
- IE đa tài liệu: việc hợp nhất các đề cập giữa các bài báo về cùng một sự kiện được gọi là cross-document coreference.

## The Concept

![Coreference clustering: mentions → entities](../assets/coref.svg)

**Nhiệm vụ.** Đầu vào: một tài liệu. Đầu ra: một cụm các đề cập (spans) trong đó mỗi cụm đề cập đến một thực thể.

**Các loại đề cập (Mention types).**

- **Named entity.** "Tim Cook"
- **Nominal.** "the CEO", "the company"
- **Pronominal.** "he", "she", "they", "it"
- **Appositive.** "Tim Cook, Apple's CEO,"

**Các kiến trúc (Architectures).**

1. **Rule-based (Hobbs, 1978).** Giải quyết đại từ dựa trên cây cú pháp sử dụng các quy tắc ngữ pháp. Là baseline tốt. Đáng ngạc nhiên là rất khó để vượt qua trong việc xử lý đại từ.
2. **Mention-pair classifier.** Với mỗi cặp đề cập (m_i, m_j), dự đoán xem chúng có corefer (cùng tham chiếu) hay không. Gom cụm bằng bao đóng bắc cầu (transitive closure). Đây là tiêu chuẩn trước năm 2016.
3. **Mention-ranking.** Với mỗi đề cập, xếp hạng các tiền đề (antecedents) ứng viên (bao gồm cả "không có tiền đề"). Chọn kết quả cao nhất.
4. **Span-based end-to-end (Lee et al., 2017).** Sử dụng Transformer encoder. Liệt kê tất cả các span ứng viên lên đến một giới hạn độ dài. Dự đoán điểm số đề cập. Dự đoán xác suất tiền đề cho mỗi span. Gom cụm theo kiểu tham lam (greedy). Đây là mặc định hiện đại.
5. **Generative (2024+).** Prompt cho LLM: "List every pronoun in this text and its antecedent." Hoạt động tốt với các trường hợp dễ, nhưng gặp khó khăn với tài liệu dài và các tham chiếu hiếm.

**Các chỉ số đánh giá.** Năm chỉ số tiêu chuẩn (MUC, B³, CEAF, BLANC, LEA) vì không có chỉ số đơn lẻ nào nắm bắt được chất lượng gom cụm. Báo cáo trung bình của ba chỉ số đầu tiên dưới dạng CoNLL F1. Trạng thái tốt nhất (SOTA) năm 2026 trên CoNLL-2012: ~83 F1.

**Các trường hợp khó đã biết.**

- Các mô tả xác định (definite descriptions) đề cập đến các thực thể đã được giới thiệu từ nhiều trang trước.
- Bridging anaphora ("the wheels" → một chiếc xe đã được đề cập trước đó).
- Zero anaphora trong các ngôn ngữ như tiếng Trung và tiếng Nhật.
- Cataphora (đại từ đứng trước thực thể): "When **she** walked in, Mary smiled."

```figure
coref-links
```

## Build It

### Step 1: pretrained neural coreference (AllenNLP / spaCy-experimental)

```python
import spacy
nlp = spacy.load("en_coreference_web_trf")   # experimental model
doc = nlp("Apple announced new products. The company said they would ship soon.")
for cluster in doc._.coref_clusters:
    print(cluster, "->", [m.text for m in cluster])
```

Trên một tài liệu dài hơn, bạn sẽ nhận được kết quả như:
- Cluster 1: [Apple, The company, they]
- Cluster 2: [new products]

### Step 2: rule-based pronoun resolver (teaching)

Xem `code/main.py` để biết cách triển khai chỉ sử dụng stdlib:

1. Trích xuất các đề cập: thực thể có tên (các span viết hoa), đại từ (tra cứu từ điển), các mô tả xác định ("the X").
2. Với mỗi đại từ, nhìn vào K đề cập trước đó và chấm điểm dựa trên:
   - Sự phù hợp về giới tính/số lượng (heuristic)
   - Độ gần gũi (càng gần càng tốt)
   - Vai trò cú pháp (ưu tiên chủ ngữ)
3. Liên kết với tiền đề có điểm số cao nhất.

Cách này không cạnh tranh được với các mô hình neural, nhưng nó cho thấy không gian tìm kiếm và các quyết định mà một mô hình end-to-end phải thực hiện.

### Step 3: using LLMs for coreference

```python
prompt = f"""Text: {text}

List every pronoun and noun phrase that refers to a person or company.
Cluster them by what they refer to. Output JSON:
[{{"entity": "Apple", "mentions": ["Apple", "the company", "it"]}}, ...]
"""
```

Có hai chế độ lỗi cần lưu ý. Thứ nhất, LLM hợp nhất quá mức ("him" và "her" cùng chỉ hai người khác nhau). Thứ hai, LLM âm thầm bỏ qua các đề cập trong tài liệu dài. Luôn kiểm tra lại bằng cách kiểm tra span-offset.

### Step 4: evaluation

Script conll-2012 tiêu chuẩn tính toán MUC, B³, CEAF-φ4 và báo cáo giá trị trung bình. Đối với đánh giá nội bộ, hãy bắt đầu với độ chính xác (precision) và độ thu hồi (recall) ở cấp độ span trên tập kiểm tra đã được gán nhãn, sau đó thêm F1 cho việc liên kết đề cập.

## Pitfalls

- **Singleton explosion.** Một số hệ thống báo cáo mọi đề cập là một cụm riêng biệt. B³ khá khoan dung, nhưng MUC sẽ phạt lỗi này. Luôn kiểm tra cả ba chỉ số.
- **Pronouns in long context.** Hiệu suất giảm ~15 F1 trên các tài liệu dài hơn 2.000 token. Hãy chia nhỏ (chunk) cẩn thận.
- **Gender assumptions.** Các quy tắc giới tính được mã hóa cứng sẽ thất bại với các tham chiếu phi nhị nguyên, tổ chức, động vật. Hãy sử dụng các mô hình đã học hoặc chấm điểm trung tính.
- **LLM drift on long docs.** Một lệnh gọi API đơn lẻ không thể gom cụm các đề cập một cách đáng tin cậy qua hơn 50 đoạn văn. Hãy sử dụng kỹ thuật sliding-window + merge.

## Use It

Stack công nghệ năm 2026:

| Tình huống | Lựa chọn |
|-----------|------|
| Tiếng Anh, tài liệu đơn lẻ | `en_coreference_web_trf` (spaCy-experimental) hoặc AllenNLP neural coref |
| Đa ngôn ngữ | SpanBERT / XLM-R được huấn luyện trên OntoNotes hoặc Multilingual CoNLL |
| Cross-document event coref | Các mô hình end-to-end chuyên dụng (SOTA 2025–26) |
| Quick LLM baseline | GPT-4o / Claude với prompt coref đầu ra có cấu trúc |
| Hệ thống hội thoại sản xuất | Rule-based fallback + neural primary + đánh giá thủ công cho các slot quan trọng |

Mô hình tích hợp được triển khai năm 2026: chạy NER trước, chạy coref, sau đó hợp nhất các cụm coref vào các thực thể NER. Các tác vụ hạ nguồn sẽ thấy một thực thể trên mỗi cụm, thay vì một thực thể trên mỗi đề cập.

## Ship It

Lưu dưới dạng `outputs/skill-coref-picker.md`:

```markdown
---
name: coref-picker
description: Pick a coreference approach, evaluation plan, and integration strategy.
version: 1.0.0
phase: 5
lesson: 24
tags: [nlp, coref, information-extraction]
---

Given a use case (single-doc / multi-doc, domain, language), output:

1. Approach. Rule-based / neural span-based / LLM-prompted / hybrid. One-sentence reason.
2. Model. Named checkpoint if neural.
3. Integration. Order of operations: tokenize → NER → coref → downstream task.
4. Evaluation. CoNLL F1 (MUC + B³ + CEAF-φ4 average) on held-out set + manual cluster review on 20 documents.

Refuse LLM-only coref for documents over 2,000 tokens without sliding-window merge. Refuse any pipeline that runs coref without a mention-level precision-recall report. Flag gender-heuristic systems deployed in demographically diverse text.
```

## Exercises

1. **Easy.** Chạy bộ giải quyết dựa trên quy tắc trong `code/main.py` trên 5 đoạn văn tự soạn. Đo lường độ chính xác của việc liên kết đề cập so với ground truth.
2. **Medium.** Sử dụng mô hình neural coref đã được huấn luyện trước trên một bài báo. So sánh các cụm với chú thích thủ công của riêng bạn. Nó đã thất bại ở đâu?
3. **Hard.** Xây dựng một pipeline NER có tăng cường coref: NER trước, sau đó hợp nhất thông qua các cụm coref. Đo lường sự cải thiện về độ bao phủ thực thể so với chỉ dùng NER trên 100 bài báo.

## Key Terms

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| Mention | Một tham chiếu | Một đoạn văn bản đề cập đến một thực thể (tên, đại từ, cụm danh từ). |
| Antecedent | Cái mà "nó" đề cập đến | Đề cập xuất hiện trước mà đề cập sau liên kết tới. |
| Cluster | Các đề cập của thực thể | Tập hợp các đề cập cùng chỉ về một thực thể trong thế giới thực. |
| Anaphora | Tham chiếu ngược | Đề cập sau chỉ về đề cập trước ("he" → "John"). |
| Cataphora | Tham chiếu xuôi | Đề cập trước chỉ về đề cập sau ("When he arrived, John..."). |
| Bridging | Tham chiếu ngầm | "I bought a car. The wheels were bad." (bánh xe của CHIẾC XE đó.) |
| CoNLL F1 | Con số trên bảng xếp hạng | Trung bình của các điểm F1 MUC, B³, CEAF-φ4. |

## Further Reading

- [Jurafsky & Martin, SLP3 Ch. 26 — Coreference Resolution and Entity Linking](https://web.stanford.edu/~jurafsky/slp3/26.pdf) — chương sách giáo khoa kinh điển.
- [Lee et al. (2017). End-to-end Neural Coreference Resolution](https://arxiv.org/abs/1707.07045) — giải pháp end-to-end dựa trên span.
- [Joshi et al. (2020). SpanBERT](https://arxiv.org/abs/1907.10529) — pretraining giúp cải thiện coref.
- [Pradhan et al. (2012). CoNLL-2012 Shared Task](https://aclanthology.org/W12-4501/) — bộ benchmark tiêu chuẩn.
- [Hobbs (1978). Resolving Pronoun References](https://www.sciencedirect.com/science/article/pii/0024384178900064) — kinh điển về phương pháp dựa trên quy tắc.