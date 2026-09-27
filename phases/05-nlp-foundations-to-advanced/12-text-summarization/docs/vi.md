# Tóm tắt văn bản (Text Summarization)

> Các hệ thống trích xuất (extractive) cho bạn biết tài liệu đã nói gì. Các hệ thống tóm tắt trừu tượng (abstractive) cho bạn biết tác giả muốn truyền tải điều gì. Nhiệm vụ khác nhau, cạm bẫy cũng khác nhau.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 11 (Machine Translation)
**Time:** ~75 phút

## Vấn đề

Một bài báo dài 2.000 từ xuất hiện trong nguồn cấp dữ liệu của bạn. Bạn cần 120 từ để nắm bắt nội dung đó. Bạn có thể chọn ba câu quan trọng nhất từ bài báo (trích xuất) hoặc viết lại nội dung bằng ngôn ngữ của riêng mình (trừu tượng). Cả hai đều được gọi là tóm tắt, nhưng chúng là những vấn đề hoàn toàn khác nhau.

Tóm tắt trích xuất là một bài toán xếp hạng. Chấm điểm từng câu, sau đó trả về top-`k`. Đầu ra luôn đúng ngữ pháp vì nó được lấy nguyên văn. Rủi ro là bỏ lỡ nội dung được phân bổ rải rác khắp bài báo.

Tóm tắt trừu tượng là một bài toán tạo văn bản (generation). Một Transformer tạo ra văn bản mới dựa trên đầu vào. Đầu ra trôi chảy và cô đọng nhưng có thể "ảo giác" (hallucinate) các sự kiện không có trong nguồn. Rủi ro là sự bịa đặt đầy tự tin.

Bài học này sẽ xây dựng cả hai phương pháp, cùng với các chế độ lỗi đặc trưng của từng loại.

## Khái niệm

![Extractive TextRank vs abstractive transformer](../assets/summarization.svg)

**Trích xuất (Extractive):** Coi bài báo như một đồ thị, trong đó các nút là các câu và các cạnh là độ tương đồng. Chạy PageRank (hoặc thuật toán tương tự) trên đồ thị để chấm điểm các câu dựa trên mức độ liên kết của chúng với các câu còn lại. Các câu có điểm cao nhất chính là bản tóm tắt. Cách triển khai kinh điển là **TextRank** (Mihalcea và Tarau, 2004).

**Trừu tượng (Abstractive):** Tinh chỉnh (fine-tune) một Transformer encoder-decoder (BART, T5, Pegasus) trên các cặp tài liệu-tóm tắt. Tại thời điểm suy luận (inference), mô hình đọc tài liệu và tạo bản tóm tắt từng token một thông qua cross-attention. Đặc biệt, Pegasus sử dụng mục tiêu tiền huấn luyện "gap-sentence" (câu bị khuyết), giúp nó trở nên xuất sắc trong việc tóm tắt mà không cần tinh chỉnh quá nhiều.

Đánh giá bằng **ROUGE** (Recall-Oriented Understudy for Gisting Evaluation). ROUGE-1 và ROUGE-2 chấm điểm sự trùng lặp unigram và bigram. ROUGE-L chấm điểm chuỗi con chung dài nhất (longest common subsequence). Điểm càng cao càng tốt, nhưng 40 ROUGE-L được coi là "tốt" và 50 là "xuất sắc". Mọi bài báo đều báo cáo cả ba chỉ số này. Hãy sử dụng gói `rouge-score`.

```figure
summarize-collapse
```

## Xây dựng

### Bước 1: TextRank (trích xuất)

```python
import math
import re
from collections import Counter


def sentence_split(text):
    return re.split(r"(?<=[.!?])\s+", text.strip())


def similarity(s1, s2):
    w1 = Counter(s1.lower().split())
    w2 = Counter(s2.lower().split())
    intersection = sum((w1 & w2).values())
    denom = math.log(len(w1) + 1) + math.log(len(w2) + 1)
    if denom == 0:
        return 0.0
    return intersection / denom


def textrank(text, top_k=3, damping=0.85, iterations=50, epsilon=1e-4):
    sentences = sentence_split(text)
    n = len(sentences)
    if n <= top_k:
        return sentences

    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                sim[i][j] = similarity(sentences[i], sentences[j])

    scores = [1.0] * n
    for _ in range(iterations):
        new_scores = [1 - damping] * n
        for i in range(n):
            total_out = sum(sim[i]) or 1e-9
            for j in range(n):
                if sim[i][j] > 0:
                    new_scores[j] += damping * sim[i][j] / total_out * scores[i]
        if max(abs(s - ns) for s, ns in zip(scores, new_scores)) < epsilon:
            scores = new_scores
            break
        scores = new_scores

    ranked = sorted(range(n), key=lambda k: scores[k], reverse=True)[:top_k]
    ranked.sort()
    return [sentences[i] for i in ranked]
```

Có hai điều đáng lưu ý. Hàm tương đồng sử dụng độ trùng lặp từ vựng đã chuẩn hóa log, đây là biến thể TextRank gốc. Cosine của các vector TF-IDF cũng hoạt động hiệu quả. Hệ số giảm chấn (damping factor) 0.85 và số lần lặp là các giá trị mặc định của PageRank.

### Bước 2: Tóm tắt trừu tượng với BART

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

article = """(long news article text)"""

summary = summarizer(article, max_length=120, min_length=60, do_sample=False)
print(summary[0]["summary_text"])
```

BART-large-CNN đã được tinh chỉnh trên tập dữ liệu CNN/DailyMail. Nó tạo ra các bản tóm tắt theo phong cách tin tức ngay lập tức. Đối với các lĩnh vực khác (bài báo khoa học, hội thoại, pháp lý), hãy sử dụng checkpoint Pegasus tương ứng hoặc tinh chỉnh trên dữ liệu mục tiêu của bạn.

### Bước 3: Đánh giá ROUGE

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
scores = scorer.score(reference_summary, generated_summary)
print({k: round(v.fmeasure, 3) for k, v in scores.items()})
```

Luôn sử dụng stemming (cắt gốc từ). Nếu không, "running" và "run" sẽ được tính là các từ khác nhau và ROUGE sẽ đếm thiếu.

### Vượt ra ngoài ROUGE (Đánh giá tóm tắt năm 2026)

ROUGE đã là chỉ số tóm tắt thống trị trong hai mươi năm qua và nó không còn đủ vào năm 2026. Một phân tích tổng hợp quy mô lớn về các bài báo NLG cho thấy:

- **BERTScore** (độ tương đồng embedding theo ngữ cảnh) đã chiếm ưu thế từ năm 2023 và hiện được báo cáo cùng với ROUGE trong hầu hết các bài báo tóm tắt.
- **BARTScore** coi việc đánh giá là một bài toán tạo văn bản: chấm điểm bản tóm tắt dựa trên xác suất mà một mô hình BART tiền huấn luyện gán cho nó dựa trên nguồn.
- **MoverScore** (Earth Mover's Distance trên các embedding theo ngữ cảnh) đã đạt vị trí hàng đầu trong các benchmark tóm tắt năm 2025 vì nó nắm bắt sự trùng lặp ngữ nghĩa tốt hơn ROUGE.
- **FactCC** và **QA-based faithfulness** rất phổ biến trong giai đoạn 2021-2023, hiện nay thường được thay thế bằng **G-Eval** (một chuỗi prompt GPT-4 chấm điểm tính mạch lạc, tính nhất quán, tính trôi chảy, tính liên quan với lập luận chain-of-thought).
- **G-Eval** và các phương pháp LLM-judge tương tự khớp với đánh giá của con người khoảng 80% khi các tiêu chí được thiết kế tốt.

Khuyến nghị sản xuất: báo cáo ROUGE-L để so sánh kế thừa, BERTScore cho độ trùng lặp ngữ nghĩa, G-Eval cho tính mạch lạc và tính xác thực. Hiệu chỉnh dựa trên 50-100 bản tóm tắt được con người gắn nhãn.

### Bước 4: Vấn đề tính xác thực (factuality)

Các bản tóm tắt trừu tượng dễ bị ảo giác. Các bản tóm tắt trích xuất có rủi ro ảo giác thấp hơn nhiều vì đầu ra được lấy nguyên văn từ nguồn, mặc dù chúng vẫn có thể gây hiểu lầm nếu các câu nguồn bị tách khỏi ngữ cảnh, lỗi thời hoặc trích dẫn sai thứ tự. Đây là lý do lớn nhất khiến các hệ thống sản xuất vẫn ưu tiên các phương pháp trích xuất cho nội dung liên quan đến tuân thủ.

Các loại ảo giác cần lưu ý:

- **Entity swap (Hoán đổi thực thể):** Nguồn nói "John Smith." Tóm tắt nói "John Brown."
- **Number drift (Sai lệch số liệu):** Nguồn nói "25.000." Tóm tắt nói "25 triệu."
- **Polarity flip (Đảo ngược cực tính):** Nguồn nói "từ chối lời đề nghị." Tóm tắt nói "chấp nhận lời đề nghị."
- **Fact invention (Bịa đặt sự kiện):** Nguồn không đề cập đến CEO. Tóm tắt nói CEO đã phê duyệt.

Các phương pháp đánh giá hiệu quả:

- **FactCC:** Một bộ phân loại nhị phân được huấn luyện dựa trên sự kéo theo (entailment) giữa câu nguồn và câu tóm tắt. Dự đoán là thực tế/không thực tế.
- **QA-based factuality:** Đặt các câu hỏi cho mô hình QA mà câu trả lời nằm trong nguồn. Nếu bản tóm tắt hỗ trợ các câu trả lời khác, hãy gắn cờ.
- **Entity-level F1:** So sánh các thực thể được đặt tên (named entities) trong nguồn và tóm tắt. Các thực thể chỉ xuất hiện trong tóm tắt là đáng ngờ.

Đối với bất kỳ nội dung nào hướng tới người dùng mà tính xác thực quan trọng (tin tức, y tế, pháp lý, tài chính), trích xuất là lựa chọn mặc định an toàn hơn. Tóm tắt trừu tượng cần một bước kiểm tra tính xác thực trong quy trình.

## Sử dụng

Stack công nghệ năm 2026:

| Trường hợp sử dụng | Khuyến nghị |
|---------|-------------|
| Tin tức, tóm tắt 3-5 câu, tiếng Anh | `facebook/bart-large-cnn` |
| Bài báo khoa học | `google/pegasus-pubmed` hoặc T5 đã tinh chỉnh |
| Đa tài liệu, dạng dài | Bất kỳ LLM nào có context 32k+, sử dụng prompt |
| Tóm tắt hội thoại | `philschmid/bart-large-cnn-samsum` |
| Trích xuất, rủi ro ảo giác thấp | TextRank hoặc LSA / LexRank của `sumy` |

Các LLM với context dài thường vượt qua các mô hình chuyên biệt vào năm 2026 khi tài nguyên tính toán không phải là rào cản. Sự đánh đổi nằm ở chi phí và khả năng tái lập; các mô hình chuyên biệt cho kết quả nhất quán hơn.

## Triển khai

Lưu dưới dạng `outputs/skill-summary-picker.md`:

```markdown
---
name: summary-picker
description: Pick extractive or abstractive, named library, factuality check.
version: 1.0.0
phase: 5
lesson: 12
tags: [nlp, summarization]
---

Given a task (document type, compliance requirement, length, compute budget), output:

1. Approach. Extractive or abstractive. Explain in one sentence why.
2. Starting model / library. Name it. `sumy.TextRankSummarizer`, `facebook/bart-large-cnn`, `google/pegasus-pubmed`, or an LLM prompt.
3. Evaluation plan. ROUGE-1, ROUGE-2, ROUGE-L (use rouge-score with stemming). Plus factuality check if abstractive.
4. One failure mode to probe. Entity swap is the most common in abstractive news summarization; flag samples where source entities do not appear in summary.

Refuse abstractive summarization for medical, legal, financial, or regulated content without a factuality gate. Flag input over the model's context window as needing chunked map-reduce summarization (not just truncation).
```

## Bài tập

1. **Dễ.** Chạy TextRank trên 5 bài báo. So sánh 3 câu hàng đầu với bản tóm tắt tham chiếu. Đo lường ROUGE-L. Bạn sẽ thấy ROUGE-L đạt 30-45 trên các bài báo kiểu CNN/DailyMail.
2. **Trung bình.** Triển khai tính xác thực ở cấp độ thực thể: trích xuất các thực thể được đặt tên từ nguồn và tóm tắt (sử dụng spaCy), tính toán recall của các thực thể nguồn trong tóm tắt và precision của các thực thể tóm tắt so với nguồn. Precision cao và recall thấp nghĩa là an toàn nhưng ngắn gọn; precision thấp nghĩa là có thực thể bị ảo giác.
3. **Khó.** So sánh BART-large-CNN với một LLM (Claude hoặc GPT-4) trên 50 bài báo CNN/DailyMail. Báo cáo ROUGE-L, tính xác thực (theo F1 thực thể) và chi phí cho mỗi bản tóm tắt. Ghi lại trường hợp nào mỗi mô hình thắng thế.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| Extractive | Chọn câu | Trả về các câu nguyên văn từ nguồn. Không bao giờ ảo giác. |
| Abstractive | Viết lại | Tạo văn bản mới dựa trên nguồn. Có thể ảo giác. |
| ROUGE | Chỉ số tóm tắt | Độ trùng lặp N-gram / LCS giữa đầu ra hệ thống và tham chiếu. |
| TextRank | Trích xuất dựa trên đồ thị | PageRank trên đồ thị tương đồng câu. |
| Factuality | Có đúng không | Liệu các khẳng định trong tóm tắt có được nguồn hỗ trợ không. |
| Hallucination | Nội dung bịa đặt | Nội dung trong tóm tắt mà nguồn không hỗ trợ. |

## Đọc thêm

- [Mihalcea và Tarau (2004). TextRank: Bringing Order into Texts](https://aclanthology.org/W04-3252/) — bài báo kinh điển về trích xuất.
- [Lewis et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training](https://arxiv.org/abs/1910.13461) — bài báo về BART.
- [Zhang et al. (2019). PEGASUS: Pre-training with Extracted Gap-sentences](https://arxiv.org/abs/1912.08777) — Pegasus và mục tiêu gap-sentence.
- [Lin (2004). ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/) — bài báo về ROUGE.
- [Maynez et al. (2020). On Faithfulness and Factuality in Abstractive Summarization](https://arxiv.org/abs/2005.00661) — bài báo về bối cảnh tính xác thực.