# Trích xuất Quan hệ & Xây dựng Đồ thị Tri thức (Knowledge Graph)

> NER tìm thấy các thực thể. Entity linking neo chúng lại. Trích xuất quan hệ tìm các cạnh kết nối giữa chúng. Một đồ thị tri thức là tổng hòa của các nút, các cạnh và nguồn gốc của chúng.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 06 (NER), Phase 5 · 25 (Entity Linking)
**Time:** ~60 phút

## Vấn đề

Một nhà phân tích đọc: "Tim Cook trở thành CEO của Apple vào năm 2011." Bốn sự thật:

- `(Tim Cook, role, CEO)`
- `(Tim Cook, employer, Apple)`
- `(Tim Cook, start_date, 2011)`
- `(Apple, type, Organization)`

Trích xuất Quan hệ (Relation Extraction - RE) biến văn bản tự do thành các bộ ba (triples) có cấu trúc `(subject, relation, object)`. Tổng hợp trên toàn bộ kho ngữ liệu và bạn sẽ có một đồ thị tri thức. Tổng hợp và truy vấn, bạn sẽ có một nền tảng suy luận cho RAG, phân tích hoặc kiểm toán tuân thủ.

Vấn đề của năm 2026: Các LLM trích xuất quan hệ một cách đầy nhiệt tình. Quá nhiệt tình. Chúng tạo ra các bộ ba ảo tưởng (hallucination) mà văn bản nguồn không hỗ trợ. Nếu không có nguồn gốc (provenance), bạn không thể phân biệt được bộ ba thực tế với hư cấu hợp lý. Câu trả lời của năm 2026 là các pipeline neo-và-xác-minh (anchor-and-verify) theo phong cách AEVS.

## Khái niệm

![Text → triples → knowledge graph](../assets/relation-extraction.svg)

**Dạng bộ ba (Triple form).** `(subject_entity, relation_type, object_entity)`. Các quan hệ đến từ một ontology đóng (các thuộc tính Wikidata, FIBO, UMLS) hoặc một tập mở (phong cách OpenIE, bất cứ thứ gì).

**Ba phương pháp trích xuất.**

1. **Dựa trên quy tắc / mẫu (Rule / pattern-based).** Các mẫu Hearst: "X như là Y" → `(Y, isA, X)`. Cộng với regex thủ công. Cứng nhắc, chính xác, có thể giải thích được.
2. **Bộ phân loại có giám sát (Supervised classifier).** Với hai thực thể được đề cập trong một câu, dự đoán quan hệ từ một tập hợp cố định. Được huấn luyện trên TACRED, ACE, KBP. Tiêu chuẩn giai đoạn 2015–2022.
3. **LLM tạo sinh (Generative LLM).** Ra lệnh cho mô hình xuất ra các bộ ba. Hoạt động ngay lập tức. Cần nguồn gốc, nếu không sẽ tạo ra những thứ rác rưởi trông có vẻ hợp lý.

**AEVS (Anchor-Extraction-Verification-Supplement, 2026).** Khung giảm thiểu ảo tưởng hiện tại:

- **Anchor (Neo).** Xác định mọi span thực thể và span cụm từ quan hệ với vị trí chính xác.
- **Extract (Trích xuất).** Tạo các bộ ba được liên kết với các span neo.
- **Verify (Xác minh).** Đối chiếu từng phần tử của bộ ba với văn bản nguồn; loại bỏ bất kỳ thứ gì không được hỗ trợ.
- **Supplement (Bổ sung).** Một lượt kiểm tra độ bao phủ để đảm bảo không có span nào đã neo bị bỏ sót.

Ảo tưởng giảm mạnh. Đòi hỏi nhiều tài nguyên tính toán hơn nhưng có thể kiểm toán được.

**Sự đánh đổi giữa mở và đóng.**

- **Ontology đóng.** Danh sách thuộc tính cố định (ví dụ: hơn 11.000 thuộc tính của Wikidata). Có thể dự đoán. Có thể truy vấn. Khó để tự tạo mới.
- **Open IE.** Bất kỳ cụm động từ nào cũng trở thành một quan hệ. Độ bao phủ (recall) cao. Độ chính xác (precision) thấp. Khó truy vấn.

Các đồ thị tri thức trong sản xuất thường kết hợp cả hai: Open IE để khám phá, sau đó chuẩn hóa các quan hệ về một ontology đóng trước khi hợp nhất vào đồ thị chính.

```figure
relation-triples
```

## Xây dựng

### Bước 1: trích xuất dựa trên mẫu

```python
PATTERNS = [
    (r"(?P<s>[A-Z]\w+) (?:is|was) (?:a|an|the) (?P<o>[A-Z]?\w+)", "isA"),
    (r"(?P<s>[A-Z]\w+) (?:is|was) born in (?P<o>\w+)", "bornIn"),
    (r"(?P<s>[A-Z]\w+) works? (?:at|for) (?P<o>[A-Z]\w+)", "worksAt"),
    (r"(?P<s>[A-Z]\w+) founded (?P<o>[A-Z]\w+)", "founded"),
]
```

Xem `code/main.py` để biết trình trích xuất mẫu đầy đủ. Các mẫu Hearst vẫn được sử dụng trong các pipeline chuyên biệt theo miền vì chúng có thể gỡ lỗi được.

### Bước 2: phân loại quan hệ có giám sát

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSequenceClassification.from_pretrained("Babelscape/rebel-large")

text = "Tim Cook was born in Alabama. He later became CEO of Apple."
encoded = tok(text, return_tensors="pt", truncation=True)
output = model.generate(**encoded, max_length=200)
triples = tok.batch_decode(output, skip_special_tokens=False)
```

REBEL là một trình trích xuất quan hệ seq2seq: đầu vào là văn bản, đầu ra là các bộ ba, đã ở dạng ID thuộc tính Wikidata. Được tinh chỉnh trên dữ liệu giám sát từ xa (distant-supervision). Đây là baseline trọng số mở tiêu chuẩn.

### Bước 3: trích xuất bằng LLM với neo (anchoring)

```python
prompt = f"""Extract (subject, relation, object) triples from the text.
For each triple, include the exact character span in the source text.

Text: {text}

Output JSON:
[{{"subject": {{"text": "...", "span": [start, end]}},
   "relation": "...",
   "object": {{"text": "...", "span": [start, end]}}}}, ...]

Only include triples fully supported by the text. No inference beyond what is stated.
"""
```

Xác minh mọi span được trả về so với nguồn. Loại bỏ bất kỳ thứ gì mà `text[start:end] != triple_entity`. Đây là bước "xác minh" của AEVS ở dạng tối giản.

### Bước 4: chuẩn hóa về một ontology đóng

```python
RELATION_MAP = {
    "is the CEO of": "P169",       # "chief executive officer"
    "was born in":   "P19",         # "place of birth"
    "founded":        "P112",       # "founded by" (inverted subject/object)
    "works at":       "P108",       # "employer"
}


def canonicalize(relation):
    rel_low = relation.lower().strip()
    if rel_low in RELATION_MAP:
        return RELATION_MAP[rel_low]
    return None   # drop unmapped open relations or route to manual review
```

Chuẩn hóa thường chiếm 60-80% khối lượng công việc kỹ thuật. Hãy dự trù thời gian cho nó.

### Bước 5: xây dựng một đồ thị nhỏ và truy vấn

```python
triples = extract(text)
graph = {}
for s, r, o in triples:
    graph.setdefault(s, []).append((r, o))


def neighbors(node, relation=None):
    return [(r, o) for r, o in graph.get(node, []) if relation is None or r == relation]


print(neighbors("Tim Cook", relation="P108"))    # -> [(P108, Apple)]
```

Đây là nguyên tử của mọi hệ thống RAG-over-KG. Mở rộng nó với các kho lưu trữ bộ ba RDF (Blazegraph, Virtuoso), đồ thị thuộc tính (Neo4j) hoặc các kho lưu trữ đồ thị tăng cường vector.

## Các cạm bẫy

- **Coreference trước RE.** "Anh ấy đã thành lập Apple" — RE cần biết "anh ấy" là ai. Hãy chạy coref trước (bài học 24).
- **Chuẩn hóa thực thể.** "Apple Inc" và "Apple" phải được phân giải về cùng một nút. Entity linking trước (bài học 25).
- **Bộ ba ảo tưởng.** LLM xuất ra các bộ ba mà văn bản không hỗ trợ. Hãy thực thi việc xác minh span.
- **Sự trôi dạt chuẩn hóa quan hệ.** Các quan hệ Open IE không nhất quán ("được sinh ra tại", "đến từ", "là người bản địa của"). Hãy quy về các ID chuẩn nếu không đồ thị sẽ không thể truy vấn được.
- **Lỗi thời gian.** "Tim Cook là CEO của Apple" — đúng bây giờ, sai vào năm 2005. Nhiều quan hệ bị giới hạn bởi thời gian. Sử dụng các bộ định tính (`P580` thời gian bắt đầu, `P582` thời gian kết thúc trong Wikidata).
- **Sai lệch miền.** REBEL được huấn luyện trên Wikipedia. Văn bản pháp lý, y tế và khoa học thường cần các mô hình RE được tinh chỉnh theo miền.

## Sử dụng

Stack công nghệ năm 2026:

| Tình huống | Lựa chọn |
|-----------|------|
| Sản xuất nhanh, miền chung | REBEL hoặc LlamaPred với chuẩn hóa Wikidata |
| Chuyên biệt miền (y sinh, pháp lý) | Tinh chỉnh miền kiểu SciREX + ontology tùy chỉnh |
| LLM prompt, đầu ra được kiểm toán | Pipeline AEVS: neo → trích xuất → xác minh → bổ sung |
| IE tin tức khối lượng lớn | Kết hợp mẫu + giám sát |
| Xây dựng KG từ đầu | Open IE + lượt chuẩn hóa thủ công |
| KG thời gian | Trích xuất với các bộ định tính (thời gian bắt đầu/kết thúc, thời điểm) |

Mô hình tích hợp: NER → coref → entity linking → trích xuất quan hệ → ánh xạ ontology → nạp vào đồ thị. Mỗi giai đoạn là một cổng kiểm soát chất lượng tiềm năng.

## Triển khai

Lưu dưới dạng `outputs/skill-re-designer.md`:

```markdown
---
name: re-designer
description: Design a relation extraction pipeline with provenance and canonicalization.
version: 1.0.0
phase: 5
lesson: 26
tags: [nlp, relation-extraction, knowledge-graph]
---

Given a corpus (domain, language, volume) and downstream use (KG-RAG, analytics, compliance), output:

1. Extractor. Pattern-based / supervised / LLM / AEVS hybrid. Reason tied to precision vs recall target.
2. Ontology. Closed property list (Wikidata / domain) or open IE with canonicalization pass.
3. Provenance. Every triple carries source char-span + doc id. Non-negotiable for audit.
4. Merge strategy. Canonical entity id + relation id + temporal qualifiers; dedup policy.
5. Evaluation. Precision / recall on 200 hand-labelled triples + hallucination-rate on LLM-extracted sample.

Refuse any LLM-based RE pipeline without span verification (source provenance). Refuse open-IE output flowing into a production graph without canonicalization. Flag pipelines with no temporal qualifier on time-bounded relations (employer, spouse, position).
```

## Bài tập

1. **Dễ.** Chạy trình trích xuất mẫu trong `code/main.py` trên 5 câu tin tức. Kiểm tra thủ công độ chính xác.
2. **Trung bình.** Sử dụng REBEL (hoặc một LLM nhỏ) trên cùng các câu đó. So sánh các bộ ba. Trình trích xuất nào có độ chính xác cao hơn? Độ bao phủ cao hơn?
3. **Khó.** Xây dựng pipeline AEVS: trích xuất bằng LLM + xác minh span so với nguồn. Đo tỷ lệ ảo tưởng trước và sau bước xác minh trên 50 câu kiểu Wikipedia.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| Triple | Chủ thể-quan hệ-đối tượng | `(s, r, o)` tuple là đơn vị nguyên tử của một KG. |
| Open IE | Trích xuất bất cứ thứ gì | Các cụm từ quan hệ từ vựng mở; độ bao phủ cao, độ chính xác thấp. |
| Closed ontology | Lược đồ cố định | Tập hợp giới hạn các loại quan hệ (Wikidata, UMLS, FIBO). |
| Canonicalization | Chuẩn hóa mọi thứ | Ánh xạ tên/quan hệ bề mặt về các ID chuẩn. |
| AEVS | Trích xuất có căn cứ | Pipeline Neo-Trích xuất-Xác minh-Bổ sung (2026). |
| Provenance | Liên kết nguồn sự thật | Mỗi bộ ba mang theo một ID tài liệu + char-span đến nguồn của nó. |
| Distant supervision | Nhãn giá rẻ | Căn chỉnh văn bản với một KG hiện có để tạo dữ liệu huấn luyện. |

## Đọc thêm

- [Mintz et al. (2009). Distant supervision for relation extraction without labeled data](https://www.aclweb.org/anthology/P09-1113.pdf) — bài báo về giám sát từ xa.
- [Huguet Cabot, Navigli (2021). REBEL: Relation Extraction By End-to-end Language generation](https://aclanthology.org/2021.findings-emnlp.204.pdf) — công cụ RE seq2seq.
- [Wadden et al. (2019). Entity, Relation, and Event Extraction with Contextualized Span Representations (DyGIE++)](https://arxiv.org/abs/1909.03546) — IE kết hợp.
- [AEVS — Anchor-Extraction-Verification-Supplement framework](https://www.mdpi.com/2073-431X/15/3/178) — thiết kế giảm thiểu ảo tưởng năm 2026.
- [Wikidata SPARQL tutorial](https://www.wikidata.org/wiki/Wikidata:SPARQL_tutorial) — truy vấn đồ thị chuẩn.