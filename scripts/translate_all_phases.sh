#!/usr/bin/env bash
# ==============================================================================
# Script dịch toàn bộ 20 Phases (523 bài học & quiz) của AI Engineering from Scratch
# sang Tiếng Việt.
#
# Đặc điểm:
#   1. Giữ nguyên 100% các file gốc tiếng Anh (docs/en.md và quiz.json).
#   2. Tự động nhận diện API Key (OpenAI `sk-...`, Gemini `AIzaSy...`, hoặc Anthropic `sk-ant-...`).
#   3. Tự động lưu cache (SHA-256) từng bài, có thể dừng lại (Ctrl+C) và chạy lại
#      bất kỳ lúc nào mà không lo bị dịch lặp.
#   4. Dùng 'caffeinate' trên macOS để ngăn máy tính đi vào chế độ ngủ (sleep)
#      khi đang chạy bản dịch dài.
# ==============================================================================

set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"

# Kiểm tra Python virtualenv
if [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
else
    echo "❌ Không tìm thấy Python 3!"
    exit 1
fi

PROVIDER="gemini"

# Kiểm tra các biến môi trường API Key có sẵn
if [ -n "$OPENAI_API_KEY" ]; then
    PROVIDER="openai"
elif [ -n "$ANTHROPIC_API_KEY" ]; then
    PROVIDER="anthropic"
elif [ -n "$GEMINI_API_KEY" ]; then
    PROVIDER="gemini"
elif [ -n "$LLM_API_KEY" ]; then
    if [[ "$LLM_API_KEY" == sk-ant-* ]]; then
        export ANTHROPIC_API_KEY="$LLM_API_KEY"
        PROVIDER="anthropic"
    elif [[ "$LLM_API_KEY" == sk-* ]]; then
        export OPENAI_API_KEY="$LLM_API_KEY"
        PROVIDER="openai"
    else
        export GEMINI_API_KEY="$LLM_API_KEY"
        PROVIDER="gemini"
    fi
else
    echo "⚠️  Chưa phát hiện biến môi trường API Key."
    echo -n "Nhập API Key của bạn (OpenAI 'sk-...', Gemini 'AIza...', Anthropic 'sk-ant...'): "
    read -r KEY_INPUT
    if [ -z "$KEY_INPUT" ]; then
        echo "❌ API Key không được để trống!"
        exit 1
    fi

    # Tự động nhận diện nhà cung cấp dựa trên định dạng key
    if [[ "$KEY_INPUT" == sk-ant-* ]]; then
        export ANTHROPIC_API_KEY="$KEY_INPUT"
        PROVIDER="anthropic"
        echo "🔑 Nhận diện: Anthropic Claude API Key"
    elif [[ "$KEY_INPUT" == sk-* ]]; then
        export OPENAI_API_KEY="$KEY_INPUT"
        PROVIDER="openai"
        echo "🔑 Nhận diện: OpenAI API Key (sẽ dùng mô hình gpt-4o-mini)"
    else
        export GEMINI_API_KEY="$KEY_INPUT"
        PROVIDER="gemini"
        echo "🔑 Nhận diện: Google Gemini API Key"
    fi
fi

echo "============================================================"
echo "🌍 Chuẩn bị dịch 20 Phases (bài học & quiz) sang Tiếng Việt"
echo "   - Nhà cung cấp: $PROVIDER"
echo "============================================================"

# Kiểm tra lệnh caffeinate trên macOS để ngăn sleep
PREVENT_SLEEP=""
if command -v caffeinate >/dev/null 2>&1; then
    echo "☕ Đã kích hoạt chế độ caffeinate (ngăn máy tính tự ngủ khi đang dịch)."
    PREVENT_SLEEP="caffeinate -i"
fi

$PREVENT_SLEEP $PYTHON scripts/translate_to_vi.py --all --provider "$PROVIDER" "$@"
