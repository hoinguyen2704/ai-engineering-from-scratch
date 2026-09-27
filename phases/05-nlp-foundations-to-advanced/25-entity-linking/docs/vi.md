# Entity Linking & Disambiguation

> NER tìm thấy "Paris." Entity linking quyết định: Paris, Pháp? Paris Hilton? Paris, Texas? Paris (hoàng tử thành Troy)? Nếu không có linking, knowledge graph của bạn sẽ vẫn mơ hồ.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 06 (NER), Phase 5 · 24 (Coreference Resolution)
**Time:** ~60 phút

## Vấn đề

Một câu viết: "Jordan beat the press." NER gắn nhãn "Jordan" là PERSON. Tốt. Nhưng *Jordan nào*?

- Michael Jordan (bóng rổ)?
- Michael B. Jordan (diễn viên)?
- Michael I. Jordan (giáo sư ML tại Berkeley — vâng, sự nhầm lẫn này là có thật trong các bài báo ML)?
- Jordan (quốc gia)?
- Jordan (tên riêng tiếng Do Thái)?

Entity linking (EL) giải quyết từng mention bằng cách ánh xạ tới một mục duy nhất trong knowledge base: Wikidata, Wikipedia, DBpedia hoặc KB chuyên biệt của bạn. Gồm hai tác vụ con:

1. **Candidate generation (Tạo ứng viên).** Với "Jordan", những mục nào trong KB là hợp lý?
2. **Disambiguation (Khử mơ hồ).** Dựa vào ngữ cảnh, ứng viên nào là đúng?

Cả hai bước đều có thể học được và đều có benchmark. Pipeline kết hợp đã ổn định trong một thập kỷ — điều thay đổi chính là chất lượng của bộ khử mơ hồ.

## Khái niệm

![Entity linking pipeline: mention → candidates → disambiguated entity](../assets/entity-linking.svg)

**Candidate generation.** Với dạng bề mặt của mention ("Jordan"), hãy tra cứu các ứng viên trong alias index. Từ điển alias của Wikipedia bao phủ hầu hết các named entity: "JFK" → John F. Kennedy, Jacqueline Kennedy, sân bay JFK, JFK (phim). Một index điển hình trả về 10-30 ứng viên cho mỗi mention.

**Disambiguation: ba phương pháp.**

1. **Prior + context (Milne & Witten, 2008).** `P(entity | mention) × context-similarity(entity, text)`. Hoạt động tốt, nhanh, không cần huấn luyện.
2. **Embedding-based (ESS / REL / Blink).** Encode mention + ngữ cảnh. Encode mô tả của từng ứng viên. Chọn giá trị cosine lớn nhất. Đây là mặc định cho giai đoạn 2020-2024.
3. **Generative (GENRE, 2021; LLM-based, 2023+).** Giải mã tên chuẩn của entity theo từng token. Bị ràng buộc bởi một trie các tên entity hợp lệ để đảm bảo đầu ra chắc chắn là một KB id hợp lệ.

**End-to-end vs pipeline.** Các mô hình hiện đại (ELQ, BLINK, ExtEnD, GENRE) chạy NER + candidate generation + disambiguation trong một lượt. Các hệ thống pipeline vẫn chiếm ưu thế trong sản xuất vì bạn có thể thay thế các thành phần.

### Hai phép đo lường

- **Mention recall (candidate gen).** Tỷ lệ các mention chuẩn mà entity đúng xuất hiện trong danh sách ứng viên. Đây là ngưỡng tối thiểu cho toàn bộ pipeline.
- **Disambiguation accuracy / F1.** Với các ứng viên đúng, tần suất top-1 là chính xác.

Luôn báo cáo cả hai. Một hệ thống có độ chính xác khử mơ hồ 99% trên 80% candidate recall thì pipeline thực tế chỉ đạt 80%.

```figure
gx-entity-linking
```

## Xây dựng

### Bước 1: xây dựng alias index từ các Wikipedia redirects

```python
alias_to_entities = {
    "jordan": ["Q41421 (Michael Jordan)", "Q810 (Jordan, country)", "Q254110 (Michael B. Jordan)"],
    "paris":  ["Q90 (Paris, France)", "Q663094 (Paris, Texas)", "Q55411 (Paris Hilton)"],
    "apple":  ["Q312 (Apple Inc.)", "Q89 (apple, fruit)"],
}
```

Dữ liệu alias Wikipedia: ~18 triệu cặp (alias, entity). Tải xuống từ các bản dump của Wikidata. Lưu trữ dưới dạng inverted index.

### Bước 2: khử mơ hồ dựa trên ngữ cảnh

```python
def disambiguate(mention, context, alias_index, entity_desc):
    candidates = alias_index.get(mention.lower(), [])
    if not candidates:
        return None, 0.0
    context_words = set(tokenize(context))
    best, best_score = None, -1
    for entity_id in candidates:
        desc_words = set(tokenize(entity_desc[entity_id]))
        union = len(context_words | desc_words)
        score = len(context_words & desc_words) / union if union else 0.0
        if score > best_score:
            best, best_score = entity_id, score
    return best, best_score
```

Jaccard overlap chỉ là đồ chơi. Hãy thay thế bằng cosine similarity trên các embedding (xem `code/main.py` bước 2 cho phiên bản transformer).

### Bước 3: dựa trên embedding (phong cách BLINK)

```python
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_mention(text, mention_span):
    start, end = mention_span
    marked = f"{text[:start]} [MENTION] {text[start:end]} [/MENTION] {text[end:]}"
    return encoder.encode([marked], normalize_embeddings=True)[0]

def embed_entity(entity_id, description):
    return encoder.encode([f"{entity_id}: {description}"], normalize_embeddings=True)[0]
```

Tại thời điểm index, embed mọi KB entity một lần. Tại thời điểm truy vấn, embed mention + ngữ cảnh một lần, tính dot-product với tập ứng viên, chọn giá trị lớn nhất.

### Bước 4: generative entity linking (khái niệm)

GENRE giải mã tiêu đề Wikipedia của entity theo từng ký tự. Constrained decoding (xem bài 20) đảm bảo chỉ các tiêu đề hợp lệ mới được xuất ra. Tích hợp chặt chẽ với trie hỗ trợ bởi KB. Hậu duệ hiện đại là REL-GEN và LLM-prompted EL với cấu trúc đầu ra.

```python
prompt = f"""Text: {text}
Mention: {mention}
List the best Wikipedia title for this mention.
Respond with JSON: {{"title": "..."}}"""
```

Kết hợp với whitelist (Outlines `choice`), đây là pipeline EL đơn giản nhất để triển khai vào năm 2026.

### Bước 5: đánh giá trên AIDA-CoNLL

AIDA-CoNLL là benchmark EL tiêu chuẩn: 1.393 bài báo Reuters, 34k mention, các entity Wikipedia. Báo cáo độ chính xác in-KB (`P@1`) và tỷ lệ phát hiện NIL (out-of-KB).

## Các cạm bẫy

- **Xử lý NIL.** Một số mention không có trong KB (entity mới nổi, người ít tên tuổi). Hệ thống phải dự đoán NIL thay vì đoán sai entity. Được đo lường riêng biệt.
- **Lỗi biên mention.** NER ở phía trước bỏ lỡ các span một phần ("Bank of America" chỉ được gắn nhãn là "Bank"). EL recall sẽ giảm.
- **Thiên kiến phổ biến (Popularity bias).** Các hệ thống đã huấn luyện thường dự đoán quá mức các entity phổ biến. Một mention "Michael I. Jordan" trong bài báo ML thường bị liên kết nhầm với Michael Jordan bóng rổ.
- **Cross-lingual EL.** Ánh xạ các mention trong văn bản tiếng Trung sang các entity Wikipedia tiếng Anh. Yêu cầu một bộ encoder đa ngôn ngữ hoặc một bước dịch thuật.
- **KB lỗi thời.** Các công ty, sự kiện, con người mới không có trong bản dump Wikipedia năm ngoái. Các pipeline sản xuất cần một vòng lặp làm mới.

## Sử dụng

Stack năm 2026:

| Tình huống | Lựa chọn |
|-----------|------|
| Tiếng Anh đa mục đích + Wikipedia | BLINK hoặc REL |
| Đa ngôn ngữ, KB = Wikipedia | mGENRE |
| Thân thiện với LLM, ít mention/ngày | Prompt Claude/GPT-4 với danh sách ứng viên + JSON ràng buộc |
| KB chuyên biệt (y tế, pháp lý) | BERT tùy chỉnh với truy xuất nhận thức KB + fine-tune trên tập dữ liệu kiểu AIDA |
| Độ trễ cực thấp | Chỉ dùng exact-match prior (baseline Milne-Witten) |
| Nghiên cứu SOTA | GENRE / ExtEnD / generative LLM-EL |

Mô hình sản xuất triển khai năm 2026: NER → coref → EL trên mỗi mention → gộp các cụm thành một entity chuẩn cho mỗi cụm. Đầu ra: một KB id cho mỗi entity trong tài liệu, không phải một cho mỗi mention.

## Triển khai

Lưu dưới dạng `outputs/skill-entity-linker.md`:

```markdown
---
name: entity-linker
description: Design an entity linking pipeline — KB, candidate generator, disambiguator, evaluation.
version: 1.0.0
phase: 5
lesson: 25
tags: [nlp, entity-linking, knowledge-graph]
---

Given a use case (domain KB, language, volume, latency budget), output:

1. Knowledge base. Wikidata / Wikipedia / custom KB. Version date. Refresh cadence.
2. Candidate generator. Alias-index, embedding, or hybrid. Target mention recall @ K.
3. Disambiguator. Prior + context, embedding-based, generative, or LLM-prompted.
4. NIL strategy. Threshold on top score, classifier, or explicit NIL candidate.
5. Evaluation. Mention recall @ 30, top-1 accuracy, NIL-detection F1 on held-out set.

Refuse any EL pipeline without a mention-recall baseline (you cannot evaluate a disambiguator without knowing candidate gen surfaced the right entity). Refuse any pipeline using LLM-prompted EL without constrained output to valid KB ids. Flag systems where popularity bias affects minority entities (e.g. name-clashes) without domain fine-tuning.
```

## Bài tập

1. **Dễ.** Triển khai bộ khử mơ hồ prior+context trong `code/main.py` trên 10 mention mơ hồ (Paris, Jordan, Apple). Gán nhãn thủ công entity đúng. Đo lường độ chính xác.
2. **Trung bình.** Encode 50 mention mơ hồ bằng sentence transformer. Embed mô tả của từng ứng viên. So sánh việc khử mơ hồ dựa trên embedding với Jaccard context overlap.
3. **Khó.** Xây dựng một KB chuyên biệt 1k-entity (ví dụ: nhân viên + sản phẩm trong công ty của bạn). Triển khai NER + EL end-to-end. Đo lường precision và recall trên 100 câu kiểm thử.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| Entity linking (EL) | Liên kết tới Wikipedia | Ánh xạ một mention tới một mục KB duy nhất. |
| Candidate generation | Nó có thể là ai? | Trả về danh sách rút gọn các mục KB hợp lý cho một mention. |
| Disambiguation | Chọn cái đúng | Chấm điểm các ứng viên bằng ngữ cảnh, chọn người chiến thắng. |
| Alias index | Bảng tra cứu | Ánh xạ từ dạng bề mặt → các entity ứng viên. |
| NIL | Không có trong KB | Dự đoán rõ ràng rằng không có mục KB nào khớp. |
| KB | Knowledge base | Wikidata, Wikipedia, DBpedia hoặc KB chuyên biệt của bạn. |
| AIDA-CoNLL | Benchmark | 1.393 bài báo Reuters với các liên kết entity chuẩn. |

## Đọc thêm

- [Milne, Witten (2008). Learning to Link with Wikipedia](https://www.cs.waikato.ac.nz/~ihw/papers/08-DM-IHW-LearningToLinkWithWikipedia.pdf) — phương pháp prior+context nền tảng.
- [Wu et al. (2020). Zero-shot Entity Linking with Dense Entity Retrieval (BLINK)](https://arxiv.org/abs/1911.03814) — công cụ dựa trên embedding.
- [De Cao et al. (2021). Autoregressive Entity Retrieval (GENRE)](https://arxiv.org/abs/2010.00904) — EL tạo sinh với constrained decoding.
- [Hoffart et al. (2011). Robust Disambiguation of Named Entities in Text (AIDA)](https://www.aclweb.org/anthology/D11-1072.pdf) — bài báo benchmark.
- [REL: An Entity Linker Standing on the Shoulders of Giants (2020)](https://arxiv.org/abs/2006.01969) — stack sản xuất mở.