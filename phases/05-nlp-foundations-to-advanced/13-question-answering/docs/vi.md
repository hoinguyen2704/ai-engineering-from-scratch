# Hệ thống Hỏi đáp (Question Answering Systems)

> Ba hệ thống đã định hình nên QA hiện đại. Extractive tìm kiếm các đoạn văn bản (spans). Retrieval-augmented (RAG) dựa trên các tài liệu để cung cấp thông tin. Generative tạo ra câu trả lời. Mọi trợ lý AI hiện đại đều là sự kết hợp của cả ba.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 11 (Machine Translation), Phase 5 · 10 (Attention Mechanism)
**Time:** ~75 phút

## Vấn đề

Người dùng nhập "Khi nào chiếc iPhone đầu tiên ra mắt?" và mong đợi câu trả lời là "Ngày 29 tháng 6 năm 2007." Không phải "Lịch sử của Apple rất dài và đa dạng." Cũng không phải chỉ mỗi con số "2007" đứng tách biệt. Đó phải là một câu trả lời trực tiếp, có căn cứ và chính xác.

Ba kiến trúc đã thống trị lĩnh vực QA trong thập kỷ qua.

- **Extractive QA.** Với một câu hỏi và một đoạn văn bản được biết là chứa câu trả lời, hãy tìm chỉ số bắt đầu và kết thúc của đoạn văn bản chứa câu trả lời đó. SQuAD là benchmark chuẩn mực.
- **Open-domain QA.** Đoạn văn bản không được cung cấp sẵn. Trước tiên, hãy truy xuất đoạn văn bản liên quan, sau đó trích xuất hoặc tạo câu trả lời. Đây là nền tảng của mọi pipeline RAG ngày nay.
- **Generative / Closed-book QA.** Một mô hình ngôn ngữ lớn (LLM) trả lời dựa trên bộ nhớ tham số của nó. Không có bước truy xuất. Tốc độ suy luận nhanh nhất, nhưng độ tin cậy về dữ kiện thấp nhất.

Xu hướng năm 2026 là mô hình lai: truy xuất một vài đoạn văn bản tốt nhất, sau đó yêu cầu mô hình generative trả lời dựa trên các đoạn văn bản đó. Đó chính là RAG, và bài học 14 sẽ đi sâu vào phần truy xuất. Bài học này tập trung xây dựng phần QA.

## Khái niệm

![QA architectures: extractive, retrieval-augmented, generative](../assets/qa.svg)

**Extractive.** Mã hóa câu hỏi và đoạn văn bản cùng nhau bằng một transformer (họ BERT). Huấn luyện hai "đầu" (heads) để dự đoán chỉ số token bắt đầu và kết thúc của câu trả lời. Hàm mất mát (loss) là cross-entropy trên các vị trí hợp lệ. Đầu ra là một đoạn văn bản từ tài liệu gốc. Không bao giờ bị "ảo giác" (hallucination) (theo thiết kế), không bao giờ xử lý các câu hỏi mà đoạn văn bản không thể trả lời (theo thiết kế).

**Retrieval-augmented (RAG).** Gồm hai giai đoạn. Đầu tiên, một bộ truy xuất (retriever) tìm ra `k` đoạn văn bản hàng đầu từ một kho dữ liệu. Thứ hai, một bộ đọc (reader - extractive hoặc generative) tạo ra câu trả lời bằng cách sử dụng các đoạn văn bản đó. Việc tách biệt giữa retriever và reader cho phép mỗi phần được huấn luyện và đánh giá độc lập. RAG hiện đại thường thêm một bộ reranker ở giữa.

**Generative.** Một LLM chỉ có bộ giải mã (decoder-only) (GPT, Claude, Llama) trả lời dựa trên các trọng số đã học. Không có bước truy xuất. Rất xuất sắc với kiến thức phổ thông, nhưng cực kỳ tệ với các dữ kiện hiếm hoặc mới. Tỷ lệ ảo giác tỷ lệ nghịch với tần suất xuất hiện của dữ kiện trong dữ liệu tiền huấn luyện.

```figure
qa-span
```

## Xây dựng

### Bước 1: Extractive QA với mô hình tiền huấn luyện

```python
from transformers import pipeline

qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

passage = (
    "Apple Inc. released the first iPhone on June 29, 2007. "
    "The device was announced by Steve Jobs at Macworld in January 2007."
)
question = "When was the first iPhone released?"

answer = qa(question=question, context=passage)
print(answer)
```

```python
{'score': 0.98, 'start': 57, 'end': 70, 'answer': 'June 29, 2007'}
```

`deepset/roberta-base-squad2` được huấn luyện trên SQuAD 2.0, bao gồm cả các câu hỏi không thể trả lời. Theo mặc định, pipeline `question-answering` trả về đoạn văn bản có điểm số cao nhất ngay cả khi điểm số "null" của mô hình thắng — nó *không* tự động trả về câu trả lời trống. Để có hành vi "không có câu trả lời" rõ ràng, hãy truyền `handle_impossible_answer=True` vào lệnh gọi pipeline: khi đó pipeline chỉ trả về câu trả lời trống khi điểm số null vượt qua mọi điểm số của các đoạn văn bản. Luôn kiểm tra trường `score` trong mọi trường hợp.

### Bước 2: Pipeline truy xuất tăng cường (phác thảo)

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

corpus = [
    "Apple Inc. released the first iPhone on June 29, 2007.",
    "Macworld 2007 featured the iPhone announcement by Steve Jobs.",
    "Android launched in 2008 as Google's mobile operating system.",
    "The first iPod was released in 2001.",
]
corpus_embeddings = encoder.encode(corpus, normalize_embeddings=True)


def retrieve(question, top_k=2):
    q_emb = encoder.encode([question], normalize_embeddings=True)
    sims = (corpus_embeddings @ q_emb.T).squeeze()
    order = np.argsort(-sims)[:top_k]
    return [corpus[i] for i in order]


def answer(question):
    passages = retrieve(question, top_k=2)
    combined = " ".join(passages)
    return qa(question=question, context=combined)


print(answer("When was the first iPhone released?"))
```

Pipeline hai giai đoạn. Dense retriever (Sentence-BERT) tìm các đoạn văn bản liên quan bằng độ tương đồng ngữ nghĩa. Extractive reader (RoBERTa-SQuAD) trích xuất đoạn câu trả lời từ các đoạn văn bản hàng đầu đã được kết hợp. Hoạt động tốt trên các kho dữ liệu nhỏ. Đối với kho dữ liệu hàng triệu tài liệu, hãy sử dụng FAISS hoặc cơ sở dữ liệu vector.

### Bước 3: Generative với RAG

```python
def rag_generate(question, llm):
    passages = retrieve(question, top_k=3)
    prompt = f"""Context:
{chr(10).join('- ' + p for p in passages)}

Question: {question}

Answer using only the context above. If the context does not contain the answer, say "I don't know."
"""
    return llm(prompt)
```

Mẫu prompt (prompt pattern) rất quan trọng. Việc yêu cầu mô hình một cách rõ ràng phải dựa trên ngữ cảnh và trả về "Tôi không biết" khi ngữ cảnh không đủ sẽ giúp giảm tỷ lệ ảo giác từ 40-60% so với việc prompt thông thường. Các mẫu phức tạp hơn sẽ thêm trích dẫn, điểm tin cậy và trích xuất có cấu trúc.

### Bước 4: Đánh giá phản ánh thực tế

SQuAD sử dụng **Exact Match (EM)** và **token-level F1**. EM là sự khớp chính xác sau khi chuẩn hóa (viết thường, bỏ dấu câu, loại bỏ mạo từ) — dự đoán khớp hoàn toàn hoặc nhận 0 điểm. F1 được tính dựa trên sự chồng lấp token giữa dự đoán và tham chiếu, cho phép tính điểm một phần. Cả hai đều đánh giá thấp các cách diễn đạt khác nhau: "June 29, 2007" so với "June 29th, 2007" thường nhận 0 điểm EM (số thứ tự làm hỏng quá trình chuẩn hóa) nhưng vẫn nhận được điểm F1 đáng kể từ các token chồng lấp.

Đối với QA trong môi trường sản xuất:

- **Độ chính xác của câu trả lời (Answer accuracy)** (được đánh giá bởi LLM hoặc con người, vì các chỉ số tự động không nắm bắt được sự tương đương về ngữ nghĩa).
- **Độ chính xác của trích dẫn (Citation accuracy).** Đoạn văn bản được trích dẫn có thực sự hỗ trợ câu trả lời không? Rất dễ kiểm tra tự động bằng cách so khớp chuỗi giữa các trích dẫn được tạo và các đoạn văn bản đã truy xuất.
- **Hiệu chuẩn từ chối (Refusal calibration).** Khi câu trả lời không nằm trong các đoạn văn bản được truy xuất, hệ thống có trả lời đúng là "Tôi không biết" không? Hãy đo lường tỷ lệ tự tin sai (false confidence rate).
- **Độ nhớ lại của truy xuất (Retrieval recall).** Trước khi đánh giá reader, hãy đo lường xem retriever có đưa được đoạn văn bản đúng vào top-`k` hay không. Một reader không thể sửa chữa một đoạn văn bản bị thiếu.

### RAGAS: Khung đánh giá sản xuất năm 2026

`RAGAS` được xây dựng chuyên biệt cho các hệ thống RAG và là tiêu chuẩn mặc định vào năm 2026. Nó chấm điểm bốn khía cạnh mà không cần tham chiếu vàng (gold references):

- **Faithfulness (Độ trung thực).** Mỗi khẳng định trong câu trả lời có đến từ ngữ cảnh được truy xuất không? Được đo bằng sự kéo theo dựa trên NLI (NLI-based entailment). Đây là chỉ số ảo giác chính của bạn.
- **Answer relevance (Độ liên quan của câu trả lời).** Câu trả lời có giải quyết được câu hỏi không? Được đo bằng cách tạo các câu hỏi giả định từ câu trả lời và so sánh với câu hỏi gốc.
- **Context precision (Độ chính xác của ngữ cảnh).** Trong số các đoạn văn bản được truy xuất, bao nhiêu phần trăm thực sự liên quan? Độ chính xác thấp = nhiễu trong prompt.
- **Context recall (Độ nhớ lại của ngữ cảnh).** Tập hợp được truy xuất có chứa tất cả thông tin cần thiết không? Độ nhớ lại thấp = reader không thể thành công.

Việc chấm điểm không cần tham chiếu cho phép bạn đánh giá trên lưu lượng truy cập thực tế mà không cần các câu trả lời vàng được biên soạn sẵn. Hãy sử dụng LLM-as-judge cho các câu hỏi mở nơi các chỉ số khớp chính xác trở nên vô dụng.

`pip install ragas`. Kết nối retriever + reader của bạn. Nhận bốn giá trị vô hướng cho mỗi truy vấn. Cảnh báo khi có sự suy giảm hiệu năng.

## Sử dụng

Stack công nghệ năm 2026.

| Trường hợp sử dụng | Khuyến nghị |
|---------|-------------|
| Có sẵn đoạn văn bản, tìm đoạn câu trả lời | `deepset/roberta-base-squad2` |
| Trên một kho dữ liệu cố định, không chấp nhận closed-book | RAG: dense retriever + LLM reader |
| Thời gian thực trên kho tài liệu lớn | RAG với hybrid (BM25 + dense) retriever + reranker (bài học 14) |
| Conversational QA (câu hỏi tiếp nối) | LLM với lịch sử hội thoại + RAG cho mỗi lượt |
| Lĩnh vực đòi hỏi tính chính xác cao, có quy định | Extractive trên kho tài liệu có thẩm quyền; không bao giờ dùng generative đơn thuần |

Extractive QA không còn là xu hướng vào năm 2026 vì RAG với LLM xử lý được nhiều trường hợp hơn. Tuy nhiên, nó vẫn được sử dụng trong các bối cảnh yêu cầu trích dẫn nguyên văn: nghiên cứu pháp lý, tuân thủ quy định, công cụ kiểm toán.

## Triển khai

Lưu dưới dạng `outputs/skill-qa-architect.md`:

```markdown
---
name: qa-architect
description: Choose QA architecture, retrieval strategy, and evaluation plan.
version: 1.0.0
phase: 5
lesson: 13
tags: [nlp, qa, rag]
---

Given requirements (corpus size, question type, factuality constraint, latency budget), output:

1. Architecture. Extractive, RAG with extractive reader, RAG with generative reader, or closed-book LLM. One-sentence reason.
2. Retriever. None, BM25, dense (name the encoder), or hybrid.
3. Reader. SQuAD-tuned model, LLM by name, or "domain-fine-tuned DistilBERT."
4. Evaluation. EM + F1 for extractive benchmarks; answer accuracy + citation accuracy + refusal calibration for production. Name what you are measuring and how you are measuring it.

Refuse closed-book LLM answers for regulatory or compliance-sensitive questions. Refuse any QA system without a retrieval-recall baseline (you cannot evaluate the reader without knowing the retriever surfaced the right passage). Flag questions that require multi-hop reasoning as needing specialized multi-hop retrievers like HotpotQA-trained systems.
```

## Bài tập

1. **Dễ.** Thiết lập pipeline extractive SQuAD ở trên với 10 đoạn văn bản Wikipedia. Tự soạn 10 câu hỏi. Đo lường tỷ lệ câu trả lời đúng. Bạn sẽ thấy 7-9 câu đúng nếu đoạn văn bản và câu hỏi rõ ràng.
2. **Trung bình.** Thêm một bộ phân loại từ chối (refusal classifier). Khi điểm truy xuất cao nhất dưới một ngưỡng (ví dụ: 0.3 cosine), hãy trả về "Tôi không biết" thay vì gọi reader. Điều chỉnh ngưỡng trên một tập dữ liệu giữ lại (held-out set).
3. **Khó.** Xây dựng một pipeline RAG trên kho dữ liệu 10.000 tài liệu tùy chọn. Triển khai truy xuất lai (hybrid - BM25 + dense) với RRF fusion (xem bài học 14). Đo lường độ chính xác của câu trả lời khi có và không có bước hybrid. Ghi lại loại câu hỏi nào được hưởng lợi nhiều nhất.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| Extractive QA | Tìm đoạn câu trả lời | Dự đoán chỉ số bắt đầu và kết thúc của câu trả lời trong một đoạn văn bản cho trước. |
| Open-domain QA | QA trên một kho dữ liệu | Không có đoạn văn bản cho trước; phải truy xuất rồi mới trả lời. |
| RAG | Truy xuất rồi tạo | Retrieval-augmented generation. Pipeline gồm Retriever + Reader. |
| SQuAD | Benchmark chuẩn mực | Stanford Question Answering Dataset. Sử dụng chỉ số EM + F1. |
| Hallucination | Câu trả lời bịa đặt | Đầu ra của reader không được hỗ trợ bởi ngữ cảnh đã truy xuất. |
| Refusal calibration | Biết khi nào nên dừng | Hệ thống trả lời đúng "Tôi không biết" khi không thể trả lời. |

## Đọc thêm

- [Rajpurkar et al. (2016). SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250) — bài báo về benchmark.
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) — DPR, bộ truy xuất dense chuẩn mực cho QA.
- [Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — bài báo đặt tên cho RAG.
- [Gao et al. (2023). Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) — khảo sát toàn diện về RAG.