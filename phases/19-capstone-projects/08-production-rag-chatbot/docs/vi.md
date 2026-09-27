# Capstone 08 — Chatbot RAG trong môi trường sản xuất cho các lĩnh vực được kiểm soát chặt chẽ

> Harvey, Glean, Mendable và LlamaCloud đều vận hành cùng một mô hình sản xuất vào năm 2026. Ingest dữ liệu bằng docling hoặc Unstructured và ColPali cho các tài liệu trực quan. Sử dụng Hybrid search. Re-rank với bge-reranker-v2-gemma. Tổng hợp câu trả lời với Claude Sonnet 4.7 sử dụng prompt caching với tỷ lệ hit rate đạt 60-80%. Bảo vệ bằng Llama Guard 4 và NeMo Guardrails. Giám sát bằng Langfuse và Phoenix. Đánh giá bằng RAGAS trên bộ golden set gồm 200 câu hỏi. Hãy xây dựng một hệ thống trong lĩnh vực được kiểm soát (pháp lý, lâm sàng, bảo hiểm), và bài tập capstone này yêu cầu bạn phải vượt qua bộ golden set, kiểm thử red team và bảng điều khiển drift.

**Type:** Capstone
**Languages:** Python (pipeline + API), TypeScript (chat UI)
**Prerequisites:** Phase 5 (NLP), Phase 7 (transformers), Phase 11 (LLM engineering), Phase 12 (multimodal), Phase 17 (infrastructure), Phase 18 (safety)
**Phases exercised:** P5 · P7 · P11 · P12 · P17 · P18
**Time:** 30 giờ

## Vấn đề

RAG trong các lĩnh vực được kiểm soát (hợp đồng pháp lý, giao thức thử nghiệm lâm sàng, chính sách bảo hiểm) là mô hình sản xuất phổ biến nhất năm 2026 vì ROI rõ ràng và rủi ro cụ thể. Harvey (Allen & Overy) đã xây dựng nó cho ngành luật. Mendable cung cấp phiên bản cho tài liệu nhà phát triển. Glean bao phủ tìm kiếm doanh nghiệp. Mô hình chung là: ingest dữ liệu độ trung thực cao, truy xuất hybrid với rerank, tổng hợp với việc thực thi trích dẫn và prompt caching, bảo vệ bằng nhiều lớp an toàn và giám sát drift liên tục.

Phần khó không nằm ở model. Đó là sự tuân thủ theo thẩm quyền (HIPAA, GDPR, SOC2), khả năng kiểm toán ở cấp độ trích dẫn, kiểm soát chi phí (prompt caching giúp giảm 60-90% chi phí khi hit rate cao), phát hiện ảo tưởng (hallucination) thông qua độ trung thực của RAGAS, và phát hiện drift khi tài liệu nguồn được cập nhật mà index chưa kịp đồng bộ. Capstone này yêu cầu bạn triển khai tất cả trên một bộ golden set gồm 200 câu hỏi cùng với bộ kiểm thử red team.

## Khái niệm

Pipeline có hai phía. **Ingestion**: docling hoặc Unstructured phân tích các tài liệu có cấu trúc; ColPali xử lý các tài liệu giàu hình ảnh; các chunk được tóm tắt, gắn thẻ và gắn nhãn quyền truy cập dựa trên vai trò. Vector được đưa vào pgvector + pgvectorscale (dưới 50 triệu vector) hoặc Qdrant Cloud; sparse BM25 chạy song song. **Conversation**: LangGraph xử lý bộ nhớ và hội thoại đa lượt; mỗi truy vấn thực hiện truy xuất hybrid, rerank với bge-reranker-v2-gemma-2b, tổng hợp với Claude Sonnet 4.7 (đã prompt-cached), truyền đầu ra qua Llama Guard 4 và NeMo Guardrails, và đưa ra phản hồi có kèm trích dẫn.

Stack đánh giá có bốn lớp. **Golden set** (200 cặp Q/A có nhãn kèm trích dẫn) để kiểm tra độ chính xác. **Red team** (jailbreak, nỗ lực trích xuất PII, câu hỏi ngoài phạm vi) để kiểm tra an toàn. **RAGAS** để tự động đánh giá độ trung thực / mức độ liên quan của câu trả lời / độ chính xác của ngữ cảnh theo từng lượt. **Drift dashboard** (Arize Phoenix) theo dõi chất lượng truy xuất và điểm số ảo tưởng hàng tuần.

Prompt caching là đòn bẩy chi phí. Claude 4.5+ và GPT-5+ hỗ trợ caching system prompt + ngữ cảnh đã truy xuất. Với hit rate 60-80%, chi phí mỗi truy vấn giảm 3-5 lần. Pipeline phải được thiết kế cho các tiền tố ổn định (system prompt + ngữ cảnh đã rerank trước) để đạt được hit rate cao.

## Kiến trúc

```
documents (contracts, protocols, policies)
      |
      v
docling / Unstructured parse + ColPali for visuals
      |
      v
chunks + summaries + role-labels + jurisdiction tags
      |
      v
pgvector + pgvectorscale  +  BM25 (Tantivy)
      |
query + role + jurisdiction
      |
      v
LangGraph conversational agent
   +--- retrieve (hybrid)
   +--- filter by role + jurisdiction
   +--- rerank (bge-reranker-v2-gemma-2b or Voyage rerank-2)
   +--- synthesize (Claude Sonnet 4.7, prompt cached)
   +--- guard (Llama Guard 4 + NeMo Guardrails + Presidio output PII scrub)
   +--- cite + return
      |
      v
eval:
  RAGAS faithfulness / answer_relevance / context_precision (online)
  Langfuse annotation queue (sampled)
  Arize Phoenix drift (weekly)
  red team suite (pre-release)
```

## Stack

- Ingestion: Unstructured.io hoặc docling cho tài liệu có cấu trúc; ColPali cho PDF giàu hình ảnh
- Vector DB: pgvector + pgvectorscale cho dưới 50 triệu vector; Qdrant Cloud cho quy mô lớn hơn
- Sparse: Tantivy BM25 với trọng số trường
- Orchestration: LlamaIndex Workflows (ingestion) + LangGraph (conversation)
- Re-ranker: bge-reranker-v2-gemma-2b tự host hoặc Voyage rerank-2 được host
- LLM: Claude Sonnet 4.7 với prompt caching; fallback Llama 3.3 70B tự host
- Eval: RAGAS 0.2 online, DeepEval cho các bộ kiểm thử ảo tưởng và jailbreak
- Observability: Langfuse tự host với hàng đợi chú thích; Arize Phoenix cho drift
- Guardrails: Llama Guard 4 phân loại đầu vào/đầu ra, chính sách NeMo Guardrails v0.12, Presidio để xóa PII
- Compliance: nhãn quyền truy cập dựa trên vai trò cho các chunk; thẻ thẩm quyền cho GDPR/HIPAA

```figure
canary-rollout
```

## Xây dựng

1. **Ingestion.** Phân tích corpus của bạn (1000-10000 tài liệu cho một bản build nghiêm túc) với Unstructured hoặc docling. Đối với các trang được quét / nhiều hình ảnh, hãy chuyển qua ColPali. Tạo các chunk kèm tóm tắt, nhãn vai trò, thẻ thẩm quyền.

2. **Index.** Dense embedding (Voyage-3 hoặc Nomic-embed-v2) vào pgvector + pgvectorscale. Index phụ BM25 qua Tantivy. Bộ lọc vai trò và thẩm quyền dưới dạng payload.

3. **Hybrid retrieve.** Lọc theo vai trò + thẩm quyền trước; sau đó chạy song song dense + BM25; hợp nhất với reciprocal rank fusion; top-20 gửi đến reranker; top-5 gửi đến synth.

4. **Tổng hợp với prompt caching.** System prompt + các chính sách tĩnh trong header cache; ngữ cảnh đã rerank làm phần mở rộng cache; câu hỏi người dùng làm hậu tố không cache. Mục tiêu đạt hit rate 60-80% ở trạng thái ổn định.

5. **Guardrails.** Llama Guard 4 cho đầu vào; NeMo Guardrails chặn các câu hỏi ngoài phạm vi hoặc chủ đề bị cấm; Presidio xóa PII vô tình xuất hiện trong đầu ra; thực thi trích dẫn sau bộ lọc.

6. **Golden set.** 200 cặp Q/A được dán nhãn bởi chuyên gia lĩnh vực với (câu trả lời, trích dẫn). Chấm điểm agent dựa trên khớp trích dẫn chính xác, độ chính xác của câu trả lời, độ trung thực (RAGAS).

7. **Red team.** 50 prompt đối kháng: jailbreak (PAIR, TAP), nỗ lực trích xuất PII, ngoài phạm vi, rò rỉ chéo thẩm quyền. Chấm điểm pass/fail và mức độ nghiêm trọng.

8. **Drift dashboard.** Arize Phoenix theo dõi chất lượng truy xuất (nDCG, độ trung thực của trích dẫn) hàng tuần. Cảnh báo khi giảm 5%.

9. **Báo cáo chi phí.** Langfuse: hit rate của prompt-caching, số token mỗi truy vấn, phân tích $/truy vấn theo từng giai đoạn.

## Sử dụng

```
$ chat --role=analyst --jurisdiction=GDPR
> what is the data-retention obligation for EU user profiles under our contract?
[retrieve]  hybrid top-20 filtered to GDPR + analyst-role
[rerank]    top-5 kept
[synth]     claude-sonnet-4.7, cache hit 74%, 0.8s
answer:
  The contract (Section 12.4, Master Services Agreement dated 2024-03-11)
  obligates EU user profile deletion within 30 days of termination per GDPR
  Article 17. The DPA amendment (DPA-v2.1, Section 5) extends this to 14 days
  for "restricted" category data.
  citations: [MSA-2024-03-11 s12.4, DPA-v2.1 s5]
```

## Triển khai

`outputs/skill-production-rag.md` mô tả sản phẩm bàn giao. Một chatbot trong lĩnh vực được kiểm soát được triển khai với các nhãn tuân thủ, đã vượt qua các tiêu chí đánh giá và được giám sát drift trực tiếp.

| Trọng số | Tiêu chí | Cách đo lường |
|:-:|---|---|
| 25 | Độ trung thực RAGAS + mức độ liên quan | Điểm online trên bộ golden set (200 Q/A) |
| 20 | Độ chính xác của trích dẫn | Tỷ lệ câu trả lời có nguồn xác thực |
| 20 | Độ bao phủ của Guardrail | Tỷ lệ pass của Llama Guard 4 + kết quả bộ kiểm thử jailbreak |
| 20 | Kỹ thuật chi phí / độ trễ | Hit rate của prompt-cache, độ trễ p95, $/truy vấn |
| 15 | Bảng điều khiển giám sát drift | Bảng điều khiển Phoenix trực tiếp với xu hướng chất lượng truy xuất hàng tuần |
| **100** | | |

## Bài tập

1. Xây dựng một phần corpus thứ hai dưới một thẩm quyền khác (ví dụ: HIPAA cùng với GDPR). Chứng minh bộ lọc vai trò + thẩm quyền ngăn chặn rò rỉ chéo trên 20 câu hỏi thăm dò chéo thẩm quyền.

2. Đo lường hit rate của prompt-cache trong một tuần lưu lượng sản xuất. Xác định những truy vấn nào làm hỏng tiền tố cache. Tái cấu trúc.

3. Thêm bộ nhớ đa lượt với bộ đệm tóm tắt 10k-token. Đo lường xem độ trung thực có giảm khi hội thoại kéo dài hay không.

4. Thay thế Claude Sonnet 4.7 bằng Llama 3.3 70B tự host. Đo lường sự thay đổi về $/truy vấn và độ trung thực.

5. Thêm chế độ "không chắc chắn": nếu điểm rerank cao nhất dưới một ngưỡng, agent sẽ nói "Tôi không có trích dẫn tin cậy" thay vì trả lời. Đo lường mức độ giảm tự tin sai lệch.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|------------------------|
| Prompt caching | "Cached system + context" | Tính năng của Claude/OpenAI: các token tiền tố được cache giảm giá 60-90% khi hit |
| RAGAS | "RAG evaluator" | Chấm điểm tự động về độ trung thực, mức độ liên quan, độ chính xác ngữ cảnh |
| Golden set | "Labeled eval" | 200+ cặp Q/A được chuyên gia dán nhãn kèm trích dẫn; sự thật gốc |
| Jurisdiction tag | "Compliance label" | Phạm vi GDPR/HIPAA/SOC2 gắn vào các chunk; được thực thi bởi bộ lọc truy xuất |
| Citation faithfulness | "Grounded answer rate" | Tỷ lệ các khẳng định được hỗ trợ bởi các đoạn nguồn có thể truy xuất |
| Drift | "Retrieval quality decay" | Thay đổi hàng tuần trong nDCG hoặc điểm trích dẫn; ngưỡng cảnh báo 5% |
| Red team | "Adversarial eval" | Kiểm thử jailbreak trước khi phát hành, trích xuất PII, thăm dò ngoài phạm vi |

## Đọc thêm

- [Harvey AI](https://www.harvey.ai) — tham chiếu stack sản xuất pháp lý
- [Glean enterprise search](https://www.glean.com) — tham chiếu RAG ở quy mô doanh nghiệp
- [Mendable documentation](https://mendable.ai) — tham chiếu RAG tài liệu nhà phát triển
- [LlamaCloud Parse + Index](https://docs.cloud.llamaindex.ai/llamaparse/getting_started) — ingest được quản lý
- [Anthropic prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — tham chiếu đòn bẩy chi phí
- [RAGAS 0.2 documentation](https://docs.ragas.io/) — framework đánh giá RAG chuẩn
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) — tham chiếu khả năng quan sát drift
- [Llama Guard 4](https://www.llama.com/docs/model-cards-and-prompt-formats/llama-guard-4/) — bộ phân loại an toàn năm 2026
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/) — framework chính sách rail