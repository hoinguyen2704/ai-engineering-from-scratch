# Capstone 04 — Multimodal Document QA (Vision-First PDF, Tables, Charts)

> Biên giới của document-QA năm 2026 đã chuyển dịch từ OCR-then-text sang vision-first late interaction. ColPali, ColQwen2.5 và ColQwen3-omni xử lý mỗi trang PDF như một hình ảnh, nhúng nó bằng multi-vector late interaction và cho phép truy vấn (query) tương tác trực tiếp với các patch. Trên các báo cáo tài chính 10-K, tài liệu khoa học và ghi chú viết tay, mô hình này vượt xa phương pháp OCR-first. Hãy xây dựng pipeline end-to-end trên 10.000 trang và công bố kết quả so sánh song song với OCR-then-text.

**Type:** Capstone
**Languages:** Python (pipeline), TypeScript (viewer UI)
**Prerequisites:** Phase 4 (computer vision), Phase 5 (NLP), Phase 7 (transformers), Phase 11 (LLM engineering), Phase 12 (multimodal), Phase 17 (infrastructure)
**Phases exercised:** P4 · P5 · P7 · P11 · P12 · P17
**Time:** 30 hours

## Problem

Các doanh nghiệp đang sở hữu những tệp PDF mà các pipeline OCR thông thường xử lý rất kém: các bản scan 10-K với bảng biểu bị xoay, tài liệu khoa học dày đặc công thức, biểu đồ chỉ có ý nghĩa khi ở dạng hình ảnh, và các ghi chú viết tay. Xử lý những tài liệu này theo hướng text-first đồng nghĩa với việc mất đi một nửa lượng thông tin. Câu trả lời của năm 2026 là late-interaction multi-vector retrieval trên hình ảnh trang gốc. ColPali (Illuin Tech) đã giới thiệu phương pháp này; ColQwen2.5-v0.2 và ColQwen3-omni đã nâng cao độ chính xác. Trên ViDoRe v3, vision-first retrieval đạt điểm số cao hơn đáng kể so với OCR-then-text — và khoảng cách này càng nới rộng đối với biểu đồ, bảng biểu và chữ viết tay.

Sự đánh đổi nằm ở lưu trữ và độ trễ. Một embedding ColQwen là khoảng 2048 patch vector mỗi trang, thay vì một vector 1024-dim duy nhất. Dung lượng lưu trữ thô sẽ tăng vọt. DocPruner (2026) mang lại khả năng cắt tỉa 50% mà không làm giảm độ chính xác đáng kể. Bạn sẽ index 10.000 trang, đo lường ViDoRe v3 nDCG@5, phục vụ câu trả lời dưới 2 giây và so sánh trực tiếp với baseline OCR-then-text.

## Concept

Late interaction có nghĩa là mỗi token truy vấn sẽ tính điểm với mỗi token patch, và điểm số tối đa cho mỗi token truy vấn sẽ được cộng lại. Bạn có được sự khớp nối chi tiết mà không cần một vector pooled duy nhất. Một multi-vector index (Vespa, Qdrant multi-vector, hoặc AstraDB) lưu trữ các embedding theo từng patch và chạy MaxSim tại thời điểm truy vấn.

Bộ trả lời là một vision-language model nhận truy vấn cộng với top-k trang được truy xuất dưới dạng hình ảnh và viết câu trả lời kèm theo các bằng chứng (bounding box hoặc tham chiếu trang). Qwen3-VL-30B, Gemini 2.5 Pro và InternVL3 là những lựa chọn tiên phong của năm 2026. Đối với các phương trình và ký hiệu khoa học, một phương án dự phòng OCR (Nougat, dots.ocr) được ghép nối như một kênh văn bản tùy chọn.

Đánh giá là một ma trận hai chiều. Một trục: loại nội dung (đoạn văn bản thuần, bảng biểu dày đặc, biểu đồ cột/đường, ghi chú viết tay, phương trình). Trục kia: phương pháp truy xuất (vision-first late interaction vs OCR-then-text vs hybrid). Mỗi ô nhận được nDCG@5 và độ chính xác của câu trả lời. Báo cáo là sản phẩm cuối cùng cần bàn giao.

## Architecture

```
PDFs -> page renderer (PyMuPDF, 180 DPI)
           |
           v
  ColQwen2.5-v0.2 embed (multi-vector per page, ~2048 patches)
           |
           +------> DocPruner 50% compression
           |
           v
   multi-vector index (Vespa or Qdrant multi-vector)
           |
query ----+----> retrieve top-k pages (MaxSim)
           |
           v
  VLM answerer: Qwen3-VL-30B | Gemini 2.5 Pro | InternVL3
    inputs: query + top-k page images + optional OCR text
           |
           v
  answer with cited page numbers + evidence regions
           |
           v
  Streamlit / Next.js viewer: highlighted boxes on source page
```

## Stack

- Page rendering: PyMuPDF (fitz) tại 180 DPI, portrait-normalized
- Late-interaction model: ColQwen2.5-v0.2 hoặc ColQwen3-omni (vidore team trên Hugging Face)
- Index: Vespa với multi-vector field, hoặc Qdrant multi-vector, hoặc AstraDB với MaxSim
- Pruning: Chính sách DocPruner 2026 (giữ lại các patch có phương sai cao, nén 50% với độ chính xác giảm < 0.5%)
- OCR fallback (phương trình / bảng biểu dày đặc): dots.ocr hoặc Nougat
- VLM answerer: Qwen3-VL-30B self-hosted hoặc Gemini 2.5 Pro hosted; InternVL3 làm fallback
- Evaluation: ViDoRe v3 benchmark, M3DocVQA cho suy luận đa trang
- Viewer UI: Next.js 15 với canvas overlay cho các vùng bằng chứng

```figure
ce-late-interaction
```

## Build It

1. **Ingest.** Xử lý một tập hợp 10.000 trang PDF bao gồm 10-K, tài liệu khoa học và tài liệu scan. Render mỗi trang thành file PNG 1536x2048. Lưu trữ `{doc_id, page_num, image_path}`.

2. **Embed.** Chạy ColQwen2.5-v0.2 trên mỗi hình ảnh trang. Output shape khoảng 2048 patch embedding với dim 128. Áp dụng DocPruner để giữ lại một nửa số patch có tín hiệu cao nhất. Ghi vào Vespa multi-vector field hoặc Qdrant multi-vector.

3. **Query.** Với mỗi truy vấn đến, nhúng bằng query tower (token-level embeddings). Chạy MaxSim so với index: với mỗi token truy vấn, lấy dot-product tối đa trên các patch embedding của trang, sau đó cộng lại. Trả về top-k trang.

4. **Synthesize.** Gọi Qwen3-VL-30B với truy vấn và top-5 hình ảnh trang. Prompt: "Trả lời chỉ sử dụng các trang được cung cấp. Trích dẫn mỗi khẳng định bằng (doc_id, page) và đặt tên cho vùng (hình ảnh, bảng, đoạn văn)."

5. **Evidence regions.** Hậu xử lý câu trả lời để trích xuất các vùng được trích dẫn. Nếu VLM xuất ra bounding box (Qwen3-VL có hỗ trợ), hãy render chúng dưới dạng overlay trong viewer.

6. **OCR fallback.** Đối với các trang được xác định là dày đặc phương trình (dựa trên heuristic về phương sai hình ảnh), chạy Nougat hoặc dots.ocr và truyền văn bản OCR như một kênh bổ sung bên cạnh hình ảnh.

7. **Eval.** Chạy ViDoRe v3 (retrieval nDCG@5) và M3DocVQA (độ chính xác QA đa trang). Đồng thời chạy pipeline OCR-then-text trên cùng tập dữ liệu với cùng bộ tổng hợp. Tạo ra ma trận loại nội dung × phương pháp.

8. **UI.** Prototype bằng Streamlit trước; sau đó là viewer sản xuất bằng Next.js 15 với overlay vùng bằng chứng theo từng trang.

## Use It

```
$ doc-qa ask "what was the 2024 operating margin change for segment EMEA?"
[retrieve]   top-5 pages in 320ms (ColQwen2.5, MaxSim, Vespa)
[synth]      qwen3-vl-30b, 1.4s, cited (form-10k-2024, p. 88) + (..., p. 92)
answer:
  EMEA operating margin moved from 18.2% to 16.8%, a 140bp decline.
  cited: 10-K-2024.pdf p.88 (Table 4, Segment Operating Margin)
         10-K-2024.pdf p.92 (MD&A, Operating Performance)
[viewer]     open with highlighted bounding boxes overlaid on p.88 Table 4
```

## Ship It

`outputs/skill-doc-qa.md` mô tả sản phẩm bàn giao: một hệ thống multimodal document QA vision-first được tinh chỉnh cho một tập dữ liệu cụ thể và được đánh giá so với baseline OCR-then-text trên ViDoRe v3.

| Weight | Criterion | How it is measured |
|:-:|---|---|
| 25 | ViDoRe v3 / M3DocVQA accuracy | Benchmark numbers vs OCR-text baseline and published leaderboard |
| 20 | Evidence-region grounding | Fraction of cited regions that actually contain the answer span |
| 20 | Storage and latency engineering | DocPruner compression ratio, index p95, answer p95 |
| 20 | Multi-page reasoning | Accuracy on a hand-labeled 100-question multi-page set |
| 15 | Source-inspection UX | Viewer clarity, overlay fidelity, side-by-side comparison tools |
| **100** | | |

## Exercises

1. Đo lường ColQwen2.5-v0.2 so với ColQwen3-omni trên cùng tập dữ liệu. Những trang nào một mô hình làm đúng còn mô hình kia làm sai? Thêm thẻ "content class" vào index để định tuyến theo loại.

2. Cắt tỉa embedding mạnh tay (75%, 90%). Tìm điểm giới hạn nén (compression cliff): điểm mà tại đó ViDoRe nDCG@5 giảm xuống dưới baseline OCR.

3. Xây dựng mô hình hybrid: chạy OCR-then-text và ColQwen song song, hợp nhất bằng RRF, rerank bằng cross-encoder. Mô hình hybrid có vượt trội hơn khi đứng riêng lẻ không? Nó giúp ích nhiều nhất ở đâu?

4. Thay thế Qwen3-VL-30B bằng một VLM nhỏ hơn (Qwen2.5-VL-7B). Đo lường đường cong độ chính xác trên mỗi đơn vị chi phí.

5. Thêm hỗ trợ ghi chú viết tay. Render tập dữ liệu viết tay, nhúng bằng ColQwen, đo lường khả năng truy xuất. So sánh với pipeline OCR viết tay.

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|------------------------|
| Late interaction | "ColPali-style retrieval" | Query tokens score against page patches independently; MaxSim aggregates |
| Multi-vector | "Per-patch embedding" | Each document has many vectors, not one pooled vector |
| MaxSim | "Late-interaction scoring" | For every query token, take max similarity over document vectors; sum |
| DocPruner | "Patch compression" | 2026 pruning that keeps 50% of patches with negligible accuracy loss |
| ViDoRe v3 | "Document-retrieval benchmark" | The 2026 standard for measuring visual-document retrieval |
| Evidence region | "Cited bounding box" | A bbox on the source page that localizes the answer span |
| OCR fallback | "Equation channel" | Text pipeline used alongside vision for equation- or table-heavy pages |

## Further Reading

- [ColPali (Illuin Tech) repository](https://github.com/illuin-tech/colpali) — reference late-interaction doc retrieval
- [ColPali paper (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449) — the foundational method paper
- [ColQwen family on Hugging Face](https://huggingface.co/vidore) — production-ready checkpoints
- [M3DocRAG (Adobe)](https://arxiv.org/abs/2411.04952) — multi-page multimodal RAG baseline
- [Vespa multi-vector tutorial](https://docs.vespa.ai/en/colpali.html) — reference serving stack
- [Qdrant multi-vector support](https://qdrant.tech/documentation/concepts/vectors/#multivectors) — alternate index
- [AstraDB multi-vector](https://docs.datastax.com/en/astra-db-serverless/databases/vector-search.html) — alternate managed index
- [Nougat OCR](https://github.com/facebookresearch/nougat) — equation-capable OCR fallback