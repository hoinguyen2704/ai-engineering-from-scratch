#!/usr/bin/env python3
"""Script làm bài kiểm tra trắc nghiệm tương tác trực tiếp trên Terminal.

Hỗ trợ:
  - Tự động ưu tiên đọc quiz.vi.json (tiếng Việt), nếu chưa có sẽ dùng quiz.json (tiếng Anh).
  - Giao diện dòng lệnh đẹp mắt, có màu sắc, chờ bạn gõ đáp án (A, B, C, D).
  - Chấm điểm ngay lập tức, giải thích chi tiết vì sao đúng/sai.
  - Tổng kết điểm thi cuối bài.

Cách dùng:
    # 1. Chạy theo đường dẫn bài học:
    python3 scripts/quiz.py phases/00-setup-and-tooling/01-dev-environment

    # 2. Chạy nhanh theo số Phase và số Bài:
    python3 scripts/quiz.py 0 1       # Phase 0, Bài 1
    python3 scripts/quiz.py 1 1       # Phase 1, Bài 1

    # 3. Chạy chế độ chọn danh sách (Interactive menu):
    python3 scripts/quiz.py
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PHASES = ROOT / "phases"

# Màu sắc ANSI terminal
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def find_quiz_file(arg1=None, arg2=None):
    # Nếu không truyền đối số, cho người dùng chọn phase & lesson
    if arg1 is None:
        phases = sorted([p for p in PHASES.iterdir() if p.is_dir() and p.name[0].isdigit()])
        print(f"\n{BOLD}{CYAN}=== DANH SÁCH CÁC GIAI ĐOẠN (PHASES) ==={RESET}")
        for p in phases:
            print(f"  {p.name}")
        phase_input = input(f"\n{YELLOW}Nhập tên hoặc số Phase muốn kiểm tra (VD: 0 hoặc 01-math-foundations): {RESET}").strip()
        if not phase_input:
            return None
        return find_quiz_file(phase_input)

    # Nếu truyền 2 số: phase và lesson (VD: 0 1 hoặc 1 1)
    if arg2 is not None:
        p_num = str(arg1).zfill(2)
        l_num = str(arg2).zfill(2)
        matched_phase = None
        for p in PHASES.iterdir():
            if p.is_dir() and p.name.startswith(p_num):
                matched_phase = p
                break
        if not matched_phase:
            print(f"{RED}❌ Không tìm thấy Phase có số thứ tự {arg1}{RESET}")
            return None

        matched_lesson = None
        for l in matched_phase.iterdir():
            if l.is_dir() and l.name.startswith(l_num):
                matched_lesson = l
                break
        if not matched_lesson:
            print(f"{RED}❌ Không tìm thấy Bài có số thứ tự {arg2} trong {matched_phase.name}{RESET}")
            return None
        target_dir = matched_lesson
    else:
        path = Path(arg1)
        if not path.is_absolute():
            path = ROOT / path

        if path.is_file():
            return path

        if path.is_dir():
            target_dir = path
        else:
            # Thử tìm theo từ khóa hoặc số phase
            query = str(arg1).strip().lower()
            matches = []
            for p in sorted(PHASES.iterdir()):
                if not p.is_dir():
                    continue
                if query == p.name.split("-")[0].lstrip("0") or query in p.name.lower():
                    matches.append(p)

            if len(matches) == 1:
                phase_dir = matches[0]
                lessons = sorted([l for l in phase_dir.iterdir() if l.is_dir() and l.name[0].isdigit()])
                print(f"\n{BOLD}{CYAN}=== CÁC BÀI HỌC TRONG {phase_dir.name} ==={RESET}")
                for l in lessons:
                    print(f"  {l.name}")
                l_input = input(f"\n{YELLOW}Nhập tên hoặc số bài muốn kiểm tra: {RESET}").strip()
                if not l_input:
                    return None
                return find_quiz_file(phase_dir.name, l_input)
            elif len(matches) > 1:
                print(f"{YELLOW}Có nhiều Phase khớp với từ khóa '{arg1}':{RESET}")
                for m in matches:
                    print(f"  - {m.name}")
                return None
            else:
                print(f"{RED}❌ Không tìm thấy bài học nào tại: {arg1}{RESET}")
                return None

    # Tìm quiz trong target_dir (ưu tiên quiz.vi.json)
    vi_quiz = target_dir / "quiz.vi.json"
    en_quiz = target_dir / "quiz.json"

    if vi_quiz.is_file():
        return vi_quiz
    elif en_quiz.is_file():
        return en_quiz
    else:
        print(f"{RED}❌ Không tìm thấy quiz.vi.json hoặc quiz.json trong thư mục: {target_dir}{RESET}")
        return None


def run_quiz(quiz_path: Path):
    with open(quiz_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = data.get("questions", [])
    if not questions:
        print(f"{RED}❌ File quiz không chứa câu hỏi nào!{RESET}")
        return

    is_vietnamese = quiz_path.name.endswith(".vi.json")
    lesson_name = quiz_path.parent.name
    phase_name = quiz_path.parent.parent.name

    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}{CYAN}📝 BÀI KIỂM TRA TRẮC NGHIỆM AI ENGINEERING{RESET}")
    print(f"   - Phase:  {phase_name}")
    print(f"   - Bài:    {lesson_name}")
    print(f"   - Ngôn ngữ: {'Tiếng Việt (quiz.vi.json)' if is_vietnamese else 'Tiếng Anh (quiz.json)'}")
    print(f"   - Số câu hỏi: {len(questions)}")
    print(f"{BOLD}{'='*60}{RESET}\n")

    letters = ["A", "B", "C", "D", "E", "F"]
    correct_count = 0

    for idx, q in enumerate(questions, 1):
        stage = q.get("stage", "check")
        prompt_text = q.get("question", "")
        options = q.get("options", [])
        correct_idx = q.get("correct", 0)
        explanation = q.get("explanation", "")

        print(f"{BOLD}Câu {idx}/{len(questions)} [{stage.upper()}]:{RESET} {prompt_text}\n")

        for opt_idx, opt_text in enumerate(options):
            lbl = letters[opt_idx] if opt_idx < len(letters) else str(opt_idx)
            print(f"   {CYAN}{lbl}){RESET} {opt_text}")

        # Nhận đáp án từ người dùng
        valid_keys = [letters[i] for i in range(len(options))]
        user_choice = None

        while True:
            try:
                ans = input(f"\n{YELLOW}👉 Lựa chọn của bạn ({'/'.join(valid_keys)}): {RESET}").strip().upper()
            except (KeyboardInterrupt, EOFError):
                print(f"\n\n{YELLOW}Đã dừng bài kiểm tra.{RESET}")
                return

            if ans in valid_keys:
                user_choice = letters.index(ans)
                break
            print(f"{RED}Vui lòng chỉ nhập một trong các ký tự: {', '.join(valid_keys)}{RESET}")

        # Kiểm tra kết quả
        correct_letter = letters[correct_idx]
        if user_choice == correct_idx:
            correct_count += 1
            print(f"\n{GREEN}{BOLD}✅ CHÍNH XÁC!{RESET}")
        else:
            print(f"\n{RED}{BOLD}❌ CHƯA ĐÚNG!{RESET} Đáp án đúng là: {GREEN}{BOLD}{correct_letter}{RESET}")

        if explanation:
            print(f"{DIM}💡 Giải thích:{RESET} {explanation}")

        print(f"\n{'-'*60}\n")

    # Tổng kết
    total = len(questions)
    percentage = (correct_count / total) * 100
    print(f"{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}📊 KẾT QUẢ CUỐI CÙNG:{RESET}")
    print(f"   - Số câu trả lời đúng: {BOLD}{correct_count}/{total}{RESET} ({percentage:.1f}%)")

    if percentage >= 70:
        print(f"   - Đánh giá: {GREEN}{BOLD}🎉 XUẤT SẮC! BẠN ĐÃ ĐẠT BÀI NÀY!{RESET}")
    else:
        print(f"   - Đánh giá: {YELLOW}{BOLD}⚠️ Bạn nên đọc lại tài liệu docs/vi.md để củng cố kiến thức.{RESET}")
    print(f"{BOLD}{'='*60}{RESET}\n")


def main():
    if len(sys.argv) == 1:
        quiz_file = find_quiz_file()
    elif len(sys.argv) == 2:
        quiz_file = find_quiz_file(sys.argv[1])
    elif len(sys.argv) >= 3:
        quiz_file = find_quiz_file(sys.argv[1], sys.argv[2])

    if quiz_file:
        run_quiz(quiz_file)


if __name__ == "__main__":
    main()

