# Đánh giá LLM — RAGAS, DeepEval, G-Eval

> Exact-match và F1 bỏ lỡ sự tương đương về mặt ngữ nghĩa. Đánh giá thủ công không thể mở rộng. LLM-as-judge là câu trả lời cho môi trường production — với đủ sự hiệu chuẩn để tin tưởng vào con số.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 13 (Question Answering), Phase 5 · 14 (Information Retrieval)
**Time:** ~75 phút

## Vấn đề

Hệ thống RAG của bạn trả lời: "June 29th, 2007."
Đáp án tham chiếu chuẩn là: "June 29, 2007."
Exact Match cho điểm 0. F1 cho điểm ~75%. Con người sẽ cho 100%.

Bây giờ hãy nhân với 10.000 trường hợp kiểm thử. Nhân tiếp với mỗi thay đổi đối với retriever, chunking, prompt hoặc model. Bạn cần một bộ đánh giá hiểu được ý nghĩa, chạy với chi phí thấp ở quy mô lớn, không nói dối về các lỗi hồi quy (regressions) và làm nổi bật được các kiểu lỗi (failure modes) chính xác.

Năm 2026 có ba framework làm chủ vấn đề này.

- **RAGAS.** Retrieval-Augmented Generation ASsessment. Bốn chỉ số RAG (faithfulness, answer-relevance, context-precision, context-recall) với backend NLI + LLM-judge. Được hỗ trợ bởi nghiên cứu, nhẹ.
- **DeepEval.** Pytest cho LLM. G-Eval, hoàn thành tác vụ, ảo giác (hallucination), các chỉ số về thiên kiến (bias). Native cho CI/CD.
- **G-Eval.** Một phương pháp (và một chỉ số của DeepEval): LLM-as-judge với chain-of-thought, tiêu chí tùy chỉnh, điểm số 0-1.

Cả ba đều dựa vào LLM-as-judge. Bài học này xây dựng trực giác về phương pháp này và lớp tin cậy xung quanh nó.

## Khái niệm

![Four evaluation dimensions, LLM-as-judge architecture](../assets/llm-evaluation.svg)

**LLM-as-judge.** Thay thế một chỉ số tĩnh bằng một LLM chấm điểm các đầu ra dựa trên một rubric. Với `(query, context, answer)`, hãy prompt một LLM đóng vai giám khảo: "Chấm điểm 0-1 về độ trung thực (faithfulness)." Trả về điểm số.

Tại sao nó hiệu quả: LLM xấp xỉ khả năng phán đoán của con người với chi phí cực thấp. GPT-4o-mini ở mức ~$0.003 per scored case enables 1000-sample regression eval runs for under $5.

Tại sao nó thất bại âm thầm:

1. **Thiên kiến của giám khảo (Judge bias).** Giám khảo thích các câu trả lời dài hơn, các câu trả lời từ cùng dòng model của chúng, các câu trả lời khớp với phong cách của prompt.
2. **Lỗi phân tích JSON.** JSON lỗi → điểm NaN → bị loại bỏ âm thầm khỏi tổng hợp. Người dùng RAGAS biết nỗi đau này. Hãy chặn bằng try/except + chế độ lỗi rõ ràng.
3. **Sự trôi dạt qua các phiên bản model (Drift).** Nâng cấp model giám khảo sẽ thay đổi mọi chỉ số. Hãy đóng băng model giám khảo + phiên bản.

**Bộ tứ RAG.**

| Chỉ số | Câu hỏi | Backend |
|--------|----------|---------|
| Faithfulness | Mỗi tuyên bố trong câu trả lời có đến từ ngữ cảnh được truy xuất không? | NLI-based entailment |
| Answer relevance | Câu trả lời có giải quyết câu hỏi không? | Tạo các câu hỏi giả định từ câu trả lời; so sánh với câu hỏi thực |
| Context precision | Trong số các chunk được truy xuất, bao nhiêu phần trăm là liên quan? | LLM-judge |
| Context recall | Việc truy xuất có trả về mọi thứ cần thiết không? | LLM-judge so với đáp án chuẩn |

**G-Eval.** Xác định một tiêu chí tùy chỉnh: "Câu trả lời có trích dẫn nguồn chính xác không?" Framework tự động mở rộng thành các bước đánh giá chain-of-thought, sau đó chấm điểm 0-1. Tốt cho các khía cạnh chất lượng chuyên biệt mà RAGAS không bao phủ.

**Hiệu chuẩn (Calibration).** Đừng bao giờ tin vào điểm số thô của giám khảo cho đến khi bạn có sự tương quan với các nhãn của con người. Chạy 100 ví dụ được dán nhãn thủ công. Vẽ biểu đồ giám khảo so với con người. Tính Spearman rho. Nếu rho < 0.7, rubric giám khảo của bạn cần được cải thiện.

```figure
n5-judge-gauge
```

## Xây dựng

### Bước 1: faithfulness với NLI (phong cách RAGAS)

```python
from typing import Callable
from transformers import pipeline

nli = pipeline("text-classification",
               model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
               top_k=None)

# `llm` is any callable: prompt str -> generated str.
# Example: llm = lambda p: client.messages.create(model="claude-haiku-4-5", ...).content[0].text
LLM = Callable[[str], str]


def atomic_claims(answer: str, llm: LLM) -> list[str]:
    prompt = f"""Break this answer into simple factual claims (one per line):
{answer}
"""
    return llm(prompt).splitlines()


def faithfulness(answer: str, context: str, llm: LLM) -> float:
    claims = atomic_claims(answer, llm)
    if not claims:
        return 0.0
    supported = 0
    for claim in claims:
        result = nli({"text": context, "text_pair": claim})[0]
        entail = next((s for s in result if s["label"] == "entailment"), None)
        if entail and entail["score"] > 0.5:
            supported += 1
    return supported / len(claims)
```

Phân tách câu trả lời thành các tuyên bố nguyên tử (atomic claims). NLI-check từng tuyên bố so với ngữ cảnh được truy xuất. Faithfulness = tỷ lệ được hỗ trợ.

### Bước 2: answer relevance

```python
import numpy as np
from sentence_transformers import SentenceTransformer

# encoder: any model implementing .encode(texts, normalize_embeddings=True) -> ndarray
# e.g., encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")

def answer_relevance(question: str, answer: str, encoder, llm: LLM, n: int = 3) -> float:
    prompt = f"Write {n} questions this answer could be the answer to:\n{answer}"
    generated = [line for line in llm(prompt).splitlines() if line.strip()][:n]
    if not generated:
        return 0.0
    q_emb = np.asarray(encoder.encode([question], normalize_embeddings=True)[0])
    g_embs = np.asarray(encoder.encode(generated, normalize_embeddings=True))
    sims = [float(q_emb @ g_emb) for g_emb in g_embs]
    return sum(sims) / len(sims)
```

Nếu câu trả lời ngụ ý các câu hỏi khác với câu hỏi được đặt ra, độ liên quan sẽ giảm.

### Bước 3: G-Eval custom metric

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase

metric = GEval(
    name="Correctness",
    criteria="The answer should be factually accurate and match the expected output.",
    evaluation_steps=[
        "Read the expected output.",
        "Read the actual output.",
        "List factual claims in the actual output.",
        "For each claim, mark supported or unsupported by the expected output.",
        "Return score = fraction supported.",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
)

test = LLMTestCase(input="When was the first iPhone released?",
                   actual_output="June 29th, 2007.",
                   expected_output="June 29, 2007.")
metric.measure(test)
print(metric.score, metric.reason)
```

Các bước đánh giá chính là rubric. Các bước rõ ràng ổn định hơn các prompt "chấm điểm 0-1" ngầm định.

### Bước 4: CI gate

```python
import deepeval
from deepeval.metrics import FaithfulnessMetric, ContextualRelevancyMetric


def test_rag_system():
    cases = load_regression_cases()
    faith = FaithfulnessMetric(threshold=0.85)
    rel = ContextualRelevancyMetric(threshold=0.7)
    for case in cases:
        faith.measure(case)
        assert faith.score >= 0.85, f"faithfulness regression on {case.id}"
        rel.measure(case)
        assert rel.score >= 0.7, f"relevancy regression on {case.id}"
```

Đóng gói dưới dạng file pytest. Chạy trên mỗi PR. Chặn merge nếu có hồi quy.

### Bước 5: toy eval từ đầu

Xem `code/main.py`. Các phép xấp xỉ chỉ dùng thư viện chuẩn (stdlib) về faithfulness (sự chồng lấp của các tuyên bố trong câu trả lời với ngữ cảnh) và relevance (sự chồng lấp của các token câu trả lời với token câu hỏi). Không dùng cho production. Chỉ để thấy hình thái.

## Các cạm bẫy

- **Không hiệu chuẩn.** Một giám khảo có tương quan 0.3 với nhãn con người chỉ là nhiễu. Yêu cầu chạy hiệu chuẩn trước khi release.
- **Tự đánh giá (Self-evaluation).** Sử dụng cùng một LLM để tạo và đánh giá sẽ làm tăng điểm số lên 10-20%. Hãy sử dụng một dòng model khác cho giám khảo.
- **Thiên kiến vị trí trong đánh giá cặp (Pairwise judging).** Giám khảo thích lựa chọn đầu tiên được trình bày. Luôn xáo trộn thứ tự và chạy cả hai chiều.
- **Tổng hợp thô che giấu lỗi.** Điểm trung bình 0.85 thường che giấu 5% lỗi thảm họa. Luôn kiểm tra phân vị dưới cùng (bottom quantile).
- **Sự hư hỏng của tập dữ liệu vàng (Golden dataset rot).** Các tập đánh giá không được quản lý phiên bản sẽ trôi dạt theo thời gian, phá vỡ so sánh theo chiều dọc. Gắn thẻ tập dữ liệu với mỗi thay đổi.
- **Chi phí LLM.** Ở quy mô lớn, các cuộc gọi giám khảo chiếm ưu thế về chi phí. Sử dụng model rẻ nhất đáp ứng ngưỡng hiệu chuẩn. GPT-4o-mini, Claude Haiku, Mistral-small.

## Sử dụng

Stack năm 2026:

| Trường hợp sử dụng | Framework |
|---------|-----------|
| Giám sát chất lượng RAG | RAGAS (4 chỉ số) |
| Cổng hồi quy CI/CD | DeepEval + pytest |
| Tiêu chí miền tùy chỉnh | G-Eval trong DeepEval |
| Giám sát lưu lượng trực tuyến | RAGAS với chế độ không cần tham chiếu |
| Kiểm tra thủ công (Human-in-the-loop) | LangSmith hoặc Phoenix với UI chú thích |
| Red-teaming / đánh giá an toàn | Promptfoo + DeepEval |

Stack điển hình: RAGAS để giám sát, DeepEval cho CI, G-Eval cho các khía cạnh mới. Chạy cả ba; chúng bất đồng một cách hữu ích.

## Triển khai

Lưu dưới dạng `outputs/skill-eval-architect.md`:

```markdown
---
name: eval-architect
description: Design an LLM evaluation plan with calibrated judge and CI gates.
version: 1.0.0
phase: 5
lesson: 27
tags: [nlp, evaluation, rag]
---

Given a use case (RAG / agent / generative task), output:

1. Metrics. Faithfulness / relevance / context-precision / context-recall + any custom G-Eval metrics with criteria.
2. Judge model. Named model + version, rationale for cost vs accuracy.
3. Calibration. Hand-labeled set size, target Spearman rho vs human > 0.7.
4. Dataset versioning. Tag strategy, change log, stratification.
5. CI gate. Thresholds per metric, regression-window logic, bottom-quantile alert.

Refuse to rely on a judge untested against ≥50 human-labeled examples. Refuse self-evaluation (same model generates + judges). Refuse aggregate-only reporting without bottom-10% surfacing. Flag any pipeline where judge upgrade lands without parallel baseline eval.
```

## Bài tập

1. **Dễ.** Sử dụng RAGAS trên 10 ví dụ RAG với các ảo giác đã biết. Xác minh chỉ số faithfulness bắt được từng lỗi một.
2. **Trung bình.** Dán nhãn thủ công 50 câu trả lời QA từ 0-1 về độ chính xác. Chấm điểm với G-Eval. Đo Spearman rho giữa giám khảo và con người.
3. **Khó.** Xây dựng một cổng CI pytest với DeepEval. Cố tình làm hỏng retriever. Xác minh cổng chặn được lỗi. Thêm cảnh báo phân vị dưới cùng thông qua kiểm tra ngưỡng trên 10% thấp nhất.

## Thuật ngữ chính

| Thuật ngữ | Mọi người nói | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| LLM-as-judge | Chấm điểm bằng LLM | Prompt một model giám khảo để chấm điểm đầu ra 0-1 dựa trên rubric. |
| RAGAS | Thư viện chỉ số RAG | Framework đánh giá mã nguồn mở với 4 chỉ số RAG không cần tham chiếu. |
| Faithfulness | Câu trả lời có căn cứ không? | Tỷ lệ các tuyên bố trong câu trả lời được hỗ trợ bởi ngữ cảnh truy xuất. |
| Context precision | Các chunk truy xuất có liên quan không? | Tỷ lệ các chunk top-K thực sự quan trọng. |
| Context recall | Truy xuất có tìm thấy mọi thứ không? | Tỷ lệ các tuyên bố trong đáp án chuẩn được hỗ trợ bởi các chunk truy xuất. |
| G-Eval | Giám khảo LLM tùy chỉnh | Rubric + các bước đánh giá chain-of-thought + điểm số 0-1. |
| Calibration | Tin tưởng nhưng phải xác minh | Tương quan Spearman giữa điểm giám khảo và điểm con người. |

## Đọc thêm

- [Es et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217) — bài báo RAGAS.
- [Liu et al. (2023). G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment](https://arxiv.org/abs/2303.16634) — bài báo G-Eval.
- [Tài liệu DeepEval](https://deepeval.com/docs/metrics-introduction) — stack production mở.
- [Zheng et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) — các thiên kiến, hiệu chuẩn, giới hạn.
- [MLflow GenAI Scorer](https://mlflow.org/blog/third-party-scorers) — framework thống nhất tích hợp RAGAS, DeepEval, Phoenix.