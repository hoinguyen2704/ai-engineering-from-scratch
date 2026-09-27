#!/usr/bin/env python3
"""Dịch tài liệu bài học (docs/en.md) và câu hỏi trắc nghiệm (quiz.json) sang Tiếng Việt.

Bảo toàn 100%:
  - Giữ nguyên các file gốc tiếng Anh (en.md, quiz.json).
  - Cấu trúc kỹ thuật trong Markdown: code blocks, công thức toán ($...$, $$...$$),
    mermaid/figure, links, images, metadata headers.
  - Cấu trúc dữ liệu trong quiz.json: chỉ mục đáp án đúng ("correct"), phân loại ("stage"),
    và thứ tự các lựa chọn ("options").

Hỗ trợ các provider dịch:
  - gemini:    Dùng Google Gemini API (GEMINI_API_KEY hoặc LLM_API_KEY)
  - openai:    Dùng OpenAI (OPENAI_API_KEY hoặc LLM_API_KEY) — mặc định gpt-4o-mini hoặc gpt-4o
  - anthropic: Dùng Claude (ANTHROPIC_API_KEY hoặc LLM_API_KEY)
  - ollama:    Dùng model local (Ollama chạy tại http://localhost:11434, mặc định qwen2.5:7b hoặc llama3.2)
  - nllb:      Dùng open-source model Meta NLLB-200 (vie_Latn) chạy offline qua HuggingFace transformers
  - echo:      Chế độ kiểm thử (không gọi mạng, kiểm tra cấu trúc và đường dẫn)
"""

import argparse
import hashlib
import json
import os
import re
import sys
import urllib.request
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PHASES = ROOT / "phases"
OUT_ROOT = ROOT / "i18n" / "vi"
CACHE_FILE = ROOT / ".translate-cache-vi.json"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_catalog import LESSON_DIR_RE, PHASE_DIR_RE  # noqa: E402

# Bộ lọc bảo vệ cú pháp Markdown, Code, Math, Links
INLINE_CODE = re.compile(r"`[^`\n]+`")
INLINE_MATH = re.compile(r"(?<!\\)\$[^$\n]+?(?<!\\)\$")
IMAGE = re.compile(r"!\[[^\]]*\]\([^)]+\)")
LINK = re.compile(r"(?<!!)\[[^\]]+\]\([^)]+\)")
BOLD = re.compile(r"\*\*[^*\n]+\*\*|__[^_\n]+__")
BARE_URL = re.compile(r"https?://[^\s)]+")

PROTECT = [
    re.compile(r"```.*?\n.*?```", re.S),
    re.compile(r"~~~.*?\n.*?~~~", re.S),
    re.compile(r"\$\$.*?\$\$", re.S),
    INLINE_CODE, INLINE_MATH, IMAGE, BARE_URL,
]

SENTINEL = "⟦PROTECT_{}⟧"
SENT_RE = re.compile(r"[⟦\[]PROTECT_?(\d+)[⟧\]]")

SYSTEM_PROMPT_DOCS = """You are a professional senior AI engineer translating a machine learning engineering curriculum into natural, technical Vietnamese.
Translate the Markdown text from English into Vietnamese.

CRITICAL RULES:
1. Preserve every placeholder token of the form ⟦PROTECT_<number>⟧ EXACTLY, unchanged, in its original position. Never drop, translate, or move them.
2. Preserve Markdown structure exactly: headings (#), lists, tables, bold/italic markers, blockquotes (>).
3. Do NOT translate:
   - Technical terms, libraries, frameworks, architectures, and model IDs: PyTorch, NumPy, Transformer, Attention, LoRA, Word2Vec, Skip-gram, CBOW, softmax, ReLU, Adam, GPT, BERT, Claude, MCP, Ollama, CUDA, MPS, etc.
   - Metadata labels: **Type:**, **Languages:**, **Prerequisites:**, **Time:**.
4. Output ONLY the translated Markdown content. Do not add any greeting, preamble, or wrapping fences."""

SYSTEM_PROMPT_QUIZ = """You are a professional senior AI engineer translating machine learning multiple-choice quizzes into natural, technical Vietnamese.

CRITICAL RULES:
1. Return ONLY valid JSON matching the exact input structure. Do not add any markdown fences, preamble, or explanation.
2. Translate "question", each option in "options", and "explanation" into natural, technical Vietnamese.
3. Translate "title" if present. Keep "lesson" directory slug unchanged.
4. Keep technical terms intact (PyTorch, Transformer, LoRA, dot product, Attention, rank, embedding, backpropagation, etc.).
5. ABSOLUTE REQUIREMENT: Keep "stage" and "correct" fields EXACTLY unchanged! Never change the integer index in "correct"! Never alter the number or order of choices in "options"!"""


def protect(text, patterns=PROTECT):
    store = []

    def stash(m):
        store.append(m.group(0))
        return SENTINEL.format(len(store) - 1)

    for pat in patterns:
        text = pat.sub(stash, text)
    return text, store


def restore(text, store):
    for i in range(len(store) - 1, -1, -1):
        text = text.replace(f"⟦PROTECT_{i}⟧", store[i])
        text = text.replace(f"[[PROTECT_{i}]]", store[i])
        text = text.replace(f"[PROTECT_{i}]", store[i])
        text = text.replace(f"⟦PROTECT{i}⟧", store[i])

    def replace_match(m):
        idx = int(m.group(1))
        if 0 <= idx < len(store):
            return store[idx]
        return m.group(0)

    text = SENT_RE.sub(replace_match, text)
    return text


def source_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def clean_json_output(text: str) -> str:
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        return match.group(1).strip()
    return text


# ─── Các Provider dịch thuật ──────────────────────────────────────────────────

def call_gemini(system: str, text: str, is_json: bool = False) -> str:
    import time
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("LLM_API_KEY")
    if not api_key:
        raise ValueError("Vui lòng thiết lập biến môi trường GEMINI_API_KEY hoặc LLM_API_KEY.")

    preferred_model = os.environ.get("GEMINI_MODEL")
    candidate_models = [preferred_model] if preferred_model else [
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
        "gemini-flash-latest",
    ]

    gen_config = {"temperature": 0.1, "maxOutputTokens": 8192}
    if is_json:
        gen_config["responseMimeType"] = "application/json"

    payload = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"parts": [{"text": text}]}],
        "generationConfig": gen_config
    }
    encoded_payload = json.dumps(payload).encode("utf-8")

    last_error = None
    for model in candidate_models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        for attempt in range(4):
            req = urllib.request.Request(
                url,
                data=encoded_payload,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    return data["candidates"][0]["content"]["parts"][0]["text"]
            except urllib.error.HTTPError as e:
                last_error = e
                if e.code == 404:
                    break
                if e.code == 429 and attempt < 3:
                    wait_sec = 15 * (attempt + 1)
                    print(f"  [API 429 Rate Limit] Đang đợi {wait_sec}s rồi tự động thử lại...", file=sys.stderr)
                    time.sleep(wait_sec)
                    continue
                if e.code == 503 and attempt < 3:
                    time.sleep(5 * (attempt + 1))
                    continue
                break
            except Exception as e:
                last_error = e
                break

    raise RuntimeError(f"Lỗi khi gọi Gemini API với các model {candidate_models}: {last_error}")


def call_openai(system: str, text: str, is_json: bool = False) -> str:
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")
    if not api_key:
        raise ValueError("Vui lòng thiết lập biến môi trường OPENAI_API_KEY hoặc LLM_API_KEY.")
    model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
    url = "https://api.openai.com/v1/chat/completions"
    payload = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": text}
        ]
    }
    if is_json:
        payload["response_format"] = {"type": "json_object"}

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]


def call_anthropic(system: str, text: str, is_json: bool = False) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("LLM_API_KEY")
    if not api_key:
        raise ValueError("Vui lòng thiết lập biến môi trường ANTHROPIC_API_KEY hoặc LLM_API_KEY.")
    model = os.environ.get("LLM_MODEL", "claude-3-5-sonnet-latest")
    url = "https://api.anthropic.com/v1/messages"
    payload = {
        "model": model,
        "max_tokens": 8192,
        "system": system,
        "messages": [{"role": "user", "content": text}]
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01"
        },
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return "".join(b["text"] for b in data["content"] if b["type"] == "text")


def call_ollama(system: str, text: str, is_json: bool = False) -> str:
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    model = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
    url = f"{host.rstrip('/')}/api/generate"
    prompt = f"{system}\n\n[Content to Translate]:\n{text}"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2}
    }
    if is_json:
        payload["format"] = "json"

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["response"]
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Không thể kết nối tới Ollama tại {host}. Đảm bảo Ollama đang chạy (`ollama serve`). Chi tiết: {e}"
        )


def call_nllb(text: str) -> str:
    from translate_lessons import nllb_translate_doc
    return nllb_translate_doc(text, "vie_Latn")


def translate_markdown_content(text: str, provider: str) -> str:
    if provider == "echo":
        return text

    if provider == "nllb":
        return call_nllb(text)

    protected, store = protect(text)
    if provider == "gemini":
        translated = call_gemini(SYSTEM_PROMPT_DOCS, protected)
    elif provider == "openai":
        translated = call_openai(SYSTEM_PROMPT_DOCS, protected)
    elif provider == "anthropic":
        translated = call_anthropic(SYSTEM_PROMPT_DOCS, protected)
    elif provider == "ollama":
        translated = call_ollama(SYSTEM_PROMPT_DOCS, protected)
    else:
        raise ValueError(f"Provider không hỗ trợ: {provider}")

    restored = restore(translated, store)
    unresolved = SENT_RE.findall(restored)
    if unresolved:
        print(f"  ⚠️ Cảnh báo: Còn {len(unresolved)} placeholder chưa được khôi phục.", file=sys.stderr)

    return restored


def translate_quiz_content(quiz_data: dict, provider: str) -> dict:
    if provider == "echo":
        return quiz_data

    src_json_str = json.dumps(quiz_data, ensure_ascii=False, indent=2)

    if provider == "gemini":
        translated_str = call_gemini(SYSTEM_PROMPT_QUIZ, src_json_str, is_json=True)
    elif provider == "openai":
        translated_str = call_openai(SYSTEM_PROMPT_QUIZ, src_json_str, is_json=True)
    elif provider == "anthropic":
        translated_str = call_anthropic(SYSTEM_PROMPT_QUIZ, src_json_str)
    elif provider == "ollama":
        translated_str = call_ollama(SYSTEM_PROMPT_QUIZ, src_json_str, is_json=True)
    elif provider == "nllb":
        translated_quiz = json.loads(json.dumps(quiz_data))
        if "title" in translated_quiz:
            translated_quiz["title"] = translate_markdown_content(translated_quiz["title"], "nllb")
        for q in translated_quiz.get("questions", []):
            q["question"] = translate_markdown_content(q["question"], "nllb")
            q["options"] = [translate_markdown_content(opt, "nllb") for opt in q.get("options", [])]
            if "explanation" in q:
                q["explanation"] = translate_markdown_content(q["explanation"], "nllb")
        return translated_quiz
    else:
        raise ValueError(f"Provider không hỗ trợ: {provider}")

    cleaned = clean_json_output(translated_str)
    parsed = json.loads(cleaned)

    orig_questions = quiz_data.get("questions", [])
    trans_questions = parsed.get("questions", [])

    if len(orig_questions) != len(trans_questions):
        print(f"  ⚠️ Cảnh báo: Số câu hỏi dịch ({len(trans_questions)}) không khớp với bản gốc ({len(orig_questions)}). Khôi phục.", file=sys.stderr)
        return quiz_data

    for orig_q, trans_q in zip(orig_questions, trans_questions):
        trans_q["correct"] = orig_q["correct"]
        trans_q["stage"] = orig_q.get("stage", "check")
        if len(trans_q.get("options", [])) != len(orig_q.get("options", [])):
            print(f"  ⚠️ Cảnh báo: Số lựa chọn trong câu hỏi không khớp. Giữ options gốc.", file=sys.stderr)
            trans_q["options"] = orig_q["options"]

    if "lesson" in quiz_data:
        parsed["lesson"] = quiz_data["lesson"]

    return parsed


# ─── Thu thập danh sách mục tiêu dịch ──────────────────────────────────────────

def get_all_targets(include_docs=True, include_quiz=True, include_outputs=False):
    targets = []
    for phase in sorted(PHASES.iterdir()):
        if not (phase.is_dir() and PHASE_DIR_RE.match(phase.name)):
            continue
        for lesson in sorted(phase.iterdir()):
            if not (lesson.is_dir() and LESSON_DIR_RE.match(lesson.name)):
                continue

            # 1. Tài liệu chính: docs/en.md -> docs/vi.md
            doc = lesson / "docs" / "en.md"
            if include_docs and doc.is_file():
                targets.append({
                    "type": "doc",
                    "phase": phase.name,
                    "lesson": lesson.name,
                    "src": doc,
                    "in_place": doc.parent / "vi.md",
                    "i18n": OUT_ROOT / doc.relative_to(ROOT).parent / "vi.md",
                })

            # 2. Câu hỏi trắc nghiệm: quiz.json -> quiz.vi.json
            quiz = lesson / "quiz.json"
            if include_quiz and quiz.is_file():
                targets.append({
                    "type": "quiz",
                    "phase": phase.name,
                    "lesson": lesson.name,
                    "src": quiz,
                    "in_place": lesson / "quiz.vi.json",
                    "i18n": OUT_ROOT / quiz.relative_to(ROOT).parent / "quiz.vi.json",
                })

            # 3. Sản phẩm đầu ra: outputs/*.md -> outputs/*.vi.md
            if include_outputs:
                outputs_dir = lesson / "outputs"
                if outputs_dir.is_dir():
                    for out_md in sorted(outputs_dir.glob("*.md")):
                        if out_md.name.endswith(".vi.md") or out_md.name.endswith("_vi.md"):
                            continue
                        targets.append({
                            "type": "output",
                            "phase": phase.name,
                            "lesson": lesson.name,
                            "src": out_md,
                            "in_place": out_md.parent / f"{out_md.stem}.vi.md",
                            "i18n": OUT_ROOT / out_md.relative_to(ROOT).parent / f"{out_md.stem}.vi.md",
                        })
    return targets


def main():
    import time
    parser = argparse.ArgumentParser(
        description="Dịch tài liệu bài học và quiz sang Tiếng Việt (giữ nguyên file gốc, bảo toàn code và đáp án)."
    )
    parser.add_argument("--all", action="store_true", help="Dịch tất cả 20 phases (toàn bộ 523 bài học)")
    parser.add_argument("--phase", help="Lọc theo tên giai đoạn (VD: 01-math-foundations hoặc 01)")
    parser.add_argument("--only", help="Lọc chính xác 1 bài học (VD: phases/01-math-foundations/01-linear-algebra-intuition)")

    # Bộ lọc loại file dịch
    parser.add_argument("--quiz-only", action="store_true", help="Chỉ dịch file trắc nghiệm (quiz.json -> quiz.vi.json)")
    parser.add_argument("--docs-only", action="store_true", help="Chỉ dịch tài liệu (docs/en.md -> docs/vi.md)")
    parser.add_argument("--outputs", action="store_true", help="Dịch cả các file trong outputs/*.md -> *.vi.md")

    parser.add_argument(
        "--provider",
        choices=["gemini", "openai", "anthropic", "ollama", "nllb", "echo"],
        default=os.environ.get("TRANSLATE_PROVIDER", "gemini"),
        help="Công cụ dịch (mặc định: gemini, hoặc openai, anthropic, ollama, nllb, echo)"
    )
    parser.add_argument(
        "--dest",
        choices=["in-place", "i18n", "both"],
        default="in-place",
        help="Nơi lưu file dịch: 'in-place' (ngay cạnh file gốc), 'i18n' (i18n/vi/...), hoặc 'both' (cả hai)"
    )
    parser.add_argument("--delay", type=float, default=1.5, help="Thời gian nghỉ giữa các yêu cầu (giây, mặc định: 1.5)")
    parser.add_argument("--dry-run", action="store_true", help="Chỉ hiển thị danh sách các file sẽ dịch, không gọi API")
    parser.add_argument("--force", action="store_true", help="Bỏ qua cache sha256 và dịch lại tất cả")

    args = parser.parse_args()

    if not (args.all or args.phase or args.only):
        print("⚠️  Vui lòng chọn phạm vi dịch:", file=sys.stderr)
        print("   --all            : Dịch tất cả các phases")
        print("   --quiz-only --all: Chỉ dịch tất cả quiz.json sang quiz.vi.json")
        print("   --phase <name>   : Dịch 1 phase cụ thể (VD: --phase 01-math-foundations)")
        print("   --only <path>    : Dịch 1 bài học duy nhất")
        print("\nVí dụ dịch quiz toàn bộ giáo trình: python3 scripts/translate_to_vi.py --all --quiz-only")
        return 1

    include_docs = not args.quiz_only
    include_quiz = not args.docs_only
    include_outputs = args.outputs

    # Tải cache
    cache = {}
    if CACHE_FILE.is_file():
        try:
            cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            cache = {}

    def save_cache():
        CACHE_FILE.write_text(json.dumps(cache, indent=2, ensure_ascii=False), encoding="utf-8")

    all_targets = get_all_targets(
        include_docs=include_docs,
        include_quiz=include_quiz,
        include_outputs=include_outputs
    )

    matched_targets = []
    for item in all_targets:
        rel = str(item["src"].relative_to(ROOT))
        if args.only:
            target = args.only.strip("/")
            if not (rel == target or rel.startswith(target + "/")):
                continue
        if args.phase:
            phase_clean = args.phase.strip("/")
            if f"/{phase_clean}/" not in f"/{rel}" and not any(part.startswith(phase_clean) for part in item["src"].parts):
                continue
        matched_targets.append(item)

    if not matched_targets:
        print("❌ Không tìm thấy mục tiêu dịch nào phù hợp với bộ lọc đã chọn.", file=sys.stderr)
        return 1

    total = len(matched_targets)
    print(f"============================================================")
    print(f"🚀 Bắt đầu dịch {total} mục sang Tiếng Việt")
    print(f"   - Provider: {args.provider}")
    print(f"   - Đích đến: {args.dest}")
    print(f"   - Docs: {'BẬT' if include_docs else 'TẮT'} | Quiz: {'BẬT' if include_quiz else 'TẮT'} | Outputs: {'BẬT' if include_outputs else 'TẮT'}")
    print(f"   - Giữ nguyên các file gốc: CÓ (bảo toàn en.md & quiz.json)")
    print(f"============================================================\n")

    translated_count = 0
    skipped_count = 0
    start_time = time.time()
    current_phase = None

    for idx, item in enumerate(matched_targets, 1):
        rel = str(item["src"].relative_to(ROOT))
        phase_name = item["phase"]
        lesson_name = item["lesson"]
        target_type = item["type"]

        if phase_name != current_phase:
            current_phase = phase_name
            print(f"\n📂 [{phase_name}]")

        src_text = item["src"].read_text(encoding="utf-8")
        current_hash = source_hash(src_text)

        # Xác định file đích
        out_targets = []
        if args.dest in ("in-place", "both"):
            out_targets.append(item["in_place"])
        if args.dest in ("i18n", "both"):
            out_targets.append(item["i18n"])

        # Kiểm tra cache
        if not args.force and cache.get(rel) == current_hash and all(t.is_file() for t in out_targets):
            skipped_count += 1
            type_label = "[DOC]" if target_type == "doc" else ("[QUIZ]" if target_type == "quiz" else "[OUT]")
            print(f"  [{idx}/{total}] ⏩ Bỏ qua {type_label}: {lesson_name} (đã có)")
            continue

        if args.dry_run:
            type_label = "[DOC]" if target_type == "doc" else ("[QUIZ]" if target_type == "quiz" else "[OUT]")
            print(f"  [{idx}/{total}] [DRY-RUN] Sẽ dịch {type_label}: {lesson_name}")
            for t in out_targets:
                print(f"         -> {t.relative_to(ROOT)}")
            translated_count += 1
            continue

        type_label = "[DOC]" if target_type == "doc" else ("[QUIZ]" if target_type == "quiz" else "[OUT]")
        print(f"  [{idx}/{total}] ⏳ Đang dịch {type_label}: {lesson_name}...", end="", flush=True)
        t0 = time.time()

        try:
            if target_type == "quiz":
                quiz_data = json.loads(src_text)
                translated_quiz = translate_quiz_content(quiz_data, args.provider)
                out_content = json.dumps(translated_quiz, indent=2, ensure_ascii=False) + "\n"
            else:
                out_content = translate_markdown_content(src_text, args.provider)
        except Exception as e:
            print(f" ❌ Lỗi: {e}", file=sys.stderr)
            continue

        for target_path in out_targets:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_text(out_content, encoding="utf-8")

        cache[rel] = current_hash
        save_cache()
        translated_count += 1
        elapsed = time.time() - t0
        print(f" ✓ ({elapsed:.1f}s)")

        if args.delay > 0 and idx < total:
            time.sleep(args.delay)

    total_time = time.time() - start_time
    print(f"\n============================================================")
    print(f"🎉 Hoàn tất!")
    print(f"   - Đã dịch mới: {translated_count} mục")
    print(f"   - Giữ nguyên (đã có sẵn/cache): {skipped_count} mục")
    print(f"   - Tổng thời gian: {total_time/60:.1f} phút")
    print(f"============================================================\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
