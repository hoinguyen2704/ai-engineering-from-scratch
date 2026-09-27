# Dialogue State Tracking

> "Tôi muốn một nhà hàng giá rẻ ở phía bắc... thực ra hãy đổi thành mức trung bình... và thêm món Ý." Ba lượt hội thoại, ba lần cập nhật trạng thái. DST giữ cho từ điển slot-value đồng bộ để việc đặt chỗ hoạt động chính xác.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 17 (Chatbots), Phase 5 · 20 (Structured Outputs)
**Time:** ~75 phút

## Vấn đề

Trong một hệ thống hội thoại hướng tác vụ (task-oriented), mục tiêu của người dùng được mã hóa dưới dạng tập hợp các cặp slot-value: `{cuisine: italian, area: north, price: moderate}`. Mỗi lượt hội thoại của người dùng có thể thêm, thay đổi hoặc xóa một slot. Hệ thống phải đọc toàn bộ cuộc hội thoại và xuất ra trạng thái hiện tại một cách chính xác.

Chỉ cần sai một slot, hệ thống sẽ đặt nhầm nhà hàng, đặt sai chuyến bay hoặc tính phí sai thẻ. DST là bản lề kết nối giữa những gì người dùng nói và những gì backend thực thi.

Tại sao nó vẫn quan trọng vào năm 2026 bất chấp sự phát triển của LLM:

- Các lĩnh vực nhạy cảm về tuân thủ (ngân hàng, y tế, đặt vé máy bay) yêu cầu các giá trị slot mang tính xác định (deterministic), không phải là tạo văn bản tự do.
- Các tác nhân sử dụng công cụ (tool-use agents) vẫn cần phân giải slot trước khi gọi API.
- Việc sửa lỗi qua nhiều lượt hội thoại khó hơn vẻ ngoài của nó: "thực ra không phải, hãy đổi thành thứ Năm."

Pipeline hiện đại: các khái niệm DST cổ điển + bộ trích xuất LLM + các guardrail đầu ra có cấu trúc.

## Khái niệm

![DST: dialog history → slot-value state](../assets/dst.svg)

**Cấu trúc tác vụ.** Một schema định nghĩa các domain (nhà hàng, khách sạn, taxi) và các slot của chúng (ẩm thực, khu vực, giá cả, số người). Mỗi slot có thể trống, được điền bằng một giá trị từ tập hợp đóng (giá: {rẻ, trung bình, đắt}), hoặc một giá trị tự do (tên: "The Copper Kettle").

**Hai cách tiếp cận DST.**

- **Phân loại (Classification).** Với mỗi cặp (slot, candidate_value), dự đoán có/không. Phù hợp cho các slot có từ vựng đóng. Tiêu chuẩn trước năm 2020.
- **Tạo văn bản (Generation).** Dựa trên hội thoại, tạo ra các giá trị slot dưới dạng văn bản tự do. Phù hợp cho các slot có từ vựng mở. Đây là mặc định hiện đại.

**Chỉ số đo lường.** Joint Goal Accuracy (JGA) — tỷ lệ các lượt hội thoại mà *mọi* slot đều đúng. Được ăn cả, ngã về không. Bảng xếp hạng MultiWOZ 2.4 đạt đỉnh khoảng 83% vào năm 2026.

**Các kiến trúc.**

1. **Dựa trên quy tắc (Rule-based - slot regex + từ khóa).** Baseline mạnh cho các domain hẹp. Dễ debug.
2. **TripPy / BERT-DST.** Tạo văn bản dựa trên cơ chế copy với BERT encoding. Tiêu chuẩn trước thời LLM.
3. **LDST (LLaMA + LoRA).** LLM được tinh chỉnh hướng dẫn (instruction-tuned) với prompting theo domain-slot. Đạt chất lượng ngang ngửa ChatGPT trên MultiWOZ 2.4.
4. **Không cần Ontology (2024–26).** Bỏ qua schema; tạo tên slot và giá trị trực tiếp. Xử lý được các domain mở.
5. **Prompt + đầu ra có cấu trúc (2024–26).** LLM với Pydantic schema + giải mã có ràng buộc (constrained decoding). Chỉ 5 dòng code, sẵn sàng cho production.

### Các dạng lỗi cổ điển

- **Tham chiếu chéo giữa các lượt (Co-reference).** "Hãy giữ lựa chọn đầu tiên." Cần phân giải xem đó là lựa chọn nào.
- **Ghi đè vs thêm mới.** Người dùng nói "thêm món Ý." Bạn thay thế loại ẩm thực hay thêm vào?
- **Xác nhận ngầm định.** "OK tuyệt" — điều đó có nghĩa là đã chấp nhận đặt chỗ được đề xuất không?
- **Sửa lỗi.** "Thực ra hãy đổi thành 7 giờ tối." Phải cập nhật thời gian mà không xóa các slot khác.
- **Tham chiếu đến câu nói trước đó của hệ thống.** "Vâng, cái đó." "Cái đó" là cái nào?

```figure
n5-slot-tracker
```

## Xây dựng

### Bước 1: bộ trích xuất slot dựa trên quy tắc

Xem `code/main.py`. Regex + từ điển đồng nghĩa bao phủ 70% các câu lệnh chuẩn trong các domain hẹp:

```python
CUISINE_SYNONYMS = {
    "italian": ["italian", "pasta", "pizza", "italy"],
    "chinese": ["chinese", "chow mein", "noodles"],
}


def extract_cuisine(utterance):
    for canonical, synonyms in CUISINE_SYNONYMS.items():
        if any(syn in utterance.lower() for syn in synonyms):
            return canonical
    return None
```

Dễ hỏng nếu nằm ngoài từ vựng chuẩn. Hoạt động tốt cho các xác nhận slot mang tính xác định.

### Bước 2: vòng lặp cập nhật trạng thái

```python
def update_state(state, utterance):
    new_state = dict(state)
    for slot, extractor in SLOT_EXTRACTORS.items():
        value = extractor(utterance)
        if value is not None:
            new_state[slot] = value
    for slot in NEGATION_CLEARS:
        if is_negated(utterance, slot):
            new_state[slot] = None
    return new_state
```

Ba bất biến:

- Không bao giờ reset một slot mà người dùng không chạm vào.
- Phủ định rõ ràng ("quên món ẩm thực đi") phải xóa slot đó.
- Sửa lỗi của người dùng ("thực ra...") phải ghi đè, không được thêm mới.

### Bước 3: DST điều khiển bởi LLM với đầu ra có cấu trúc

```python
from pydantic import BaseModel
from typing import Literal, Optional
import instructor

class RestaurantState(BaseModel):
    cuisine: Optional[Literal["italian", "chinese", "indian", "thai", "any"]] = None
    area: Optional[Literal["north", "south", "east", "west", "center"]] = None
    price: Optional[Literal["cheap", "moderate", "expensive"]] = None
    people: Optional[int] = None
    day: Optional[str] = None


def llm_dst(history, llm):
    prompt = f"""You track the slot values of a restaurant booking across turns.
Dialogue so far:
{render(history)}

Update the state based on the latest user turn. Output only the JSON state."""
    return llm(prompt, response_model=RestaurantState)
```

Instructor + Pydantic đảm bảo một đối tượng trạng thái hợp lệ. Không regex, không lệch schema, không ảo tưởng (hallucination) các slot.

### Bước 4: đánh giá JGA

```python
def joint_goal_accuracy(predicted_states, gold_states):
    correct = sum(1 for p, g in zip(predicted_states, gold_states) if p == g)
    return correct / len(predicted_states)
```

Hiệu chỉnh: hệ thống dự đoán đúng TẤT CẢ các slot trong bao nhiêu phần trăm lượt hội thoại? Đối với MultiWOZ 2.4, các hệ thống hàng đầu năm 2026 đạt: 80-83%. Hệ thống in-domain của bạn nên vượt qua con số đó trên từ vựng hẹp của bạn, nếu không thì baseline LLM sẽ đánh bại bạn.

### Bước 5: xử lý sửa lỗi

```python
CORRECTION_CUES = {"actually", "no wait", "on second thought", "change that to"}


def is_correction(utterance):
    return any(cue in utterance.lower() for cue in CORRECTION_CUES)
```

Khi phát hiện sửa lỗi, hãy ghi đè slot được cập nhật gần nhất thay vì thêm mới. Rất khó để làm đúng nếu không có sự trợ giúp của LLM. Mô hình hiện đại: luôn để LLM tạo lại toàn bộ trạng thái từ lịch sử thay vì cập nhật tăng dần — cách này tự nhiên xử lý được các lỗi sửa đổi.

## Các cạm bẫy

- **Chi phí tạo lại toàn bộ lịch sử.** Để LLM tạo lại trạng thái mỗi lượt tốn O(n²) token. Hãy giới hạn lịch sử hoặc tóm tắt các lượt cũ.
- **Schema drift.** Thêm các slot mới sau khi đã huấn luyện sẽ làm hỏng dữ liệu cũ. Hãy đánh phiên bản cho schema của bạn.
- **Phân biệt chữ hoa/thường.** "Italian" vs "italian" vs "ITALIAN" — hãy chuẩn hóa mọi nơi.
- **Kế thừa ngầm định.** Nếu người dùng đã chỉ định "cho 4 người" trước đó, một yêu cầu mới cho thời gian khác không được xóa thông tin số người. Luôn truyền toàn bộ lịch sử.
- **Tự do vs tập hợp đóng.** Tên, thời gian và địa chỉ cần các slot tự do; ẩm thực và khu vực là tập hợp đóng. Hãy kết hợp cả hai trong schema.

## Sử dụng

Stack công nghệ năm 2026:

| Tình huống | Cách tiếp cận |
|-----------|----------|
| Domain hẹp (một hoặc hai ý định) | Dựa trên quy tắc + regex |
| Domain rộng, có sẵn dữ liệu gán nhãn | LDST (LLaMA + LoRA trên dữ liệu kiểu MultiWOZ) |
| Domain rộng, không có nhãn, sẵn sàng cho prod | LLM + Instructor + Pydantic schema |
| Giọng nói / âm thanh | ASR + bộ chuẩn hóa + LLM-DST |
| Luồng đặt chỗ đa domain | LLM hướng dẫn bởi schema với các model Pydantic theo domain |
| Nhạy cảm về tuân thủ | Ưu tiên dựa trên quy tắc, fallback bằng LLM với luồng xác nhận |

## Triển khai

Lưu dưới dạng `outputs/skill-dst-designer.md`:

```markdown
---
name: dst-designer
description: Design a dialogue state tracker — schema, extractor, update policy, evaluation.
version: 1.0.0
phase: 5
lesson: 29
tags: [nlp, dialogue, task-oriented]
---

Given a use case (domain, languages, vocab openness, compliance needs), output:

1. Schema. Domain list, slots per domain, open vs closed vocabulary per slot.
2. Extractor. Rule-based / seq2seq / LLM-with-Pydantic. Reason.
3. Update policy. Regenerate-whole-state / incremental; correction handling; negation handling.
4. Evaluation. Joint Goal Accuracy on a held-out dialogue set, slot-level precision/recall, confusion on the hardest slot.
5. Confirmation flow. When to explicitly ask the user to confirm (destructive actions, low-confidence extractions).

Refuse LLM-only DST for compliance-sensitive slots without a rule-based secondary check. Refuse any DST that cannot roll back a slot on user correction. Flag schemas without version tags.
```

## Bài tập

1. **Dễ.** Xây dựng bộ theo dõi trạng thái dựa trên quy tắc trong `code/main.py` cho 3 slot (ẩm thực, khu vực, giá cả). Kiểm thử trên 10 cuộc hội thoại tự soạn. Đo lường JGA.
2. **Trung bình.** Sử dụng cùng tập dữ liệu với Instructor + Pydantic + một LLM nhỏ. So sánh JGA. Kiểm tra các lượt hội thoại khó nhất.
3. **Khó.** Triển khai cả hai và định tuyến: ưu tiên dựa trên quy tắc, fallback sang LLM khi quy tắc đưa ra <2 slot với độ tin cậy thấp. Đo lường JGA kết hợp và chi phí suy luận mỗi lượt.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|-----------------------|
| DST | Dialogue state tracking | Duy trì từ điển slot-value qua các lượt hội thoại. |
| Slot | Đơn vị ý định người dùng | Tham số được đặt tên mà backend cần (ẩm thực, ngày tháng). |
| Domain | Khu vực tác vụ | Nhà hàng, khách sạn, taxi — các tập hợp slot. |
| JGA | Joint Goal Accuracy | Tỷ lệ các lượt hội thoại mà mọi slot đều đúng. Được ăn cả, ngã về không. |
| MultiWOZ | Benchmark | Tập dữ liệu Multi-domain WOZ; tiêu chuẩn đánh giá DST. |
| Ontology-free DST | Không schema | Tạo tên slot và giá trị trực tiếp, không dùng danh sách cố định. |
| Correction | "Thực ra..." | Lượt hội thoại ghi đè lên một slot đã được điền trước đó. |

## Đọc thêm

- [Budzianowski et al. (2018). MultiWOZ — A Large-Scale Multi-Domain Wizard-of-Oz](https://arxiv.org/abs/1810.00278) — benchmark chuẩn.
- [Feng et al. (2023). Towards LLM-driven Dialogue State Tracking (LDST)](https://arxiv.org/abs/2310.14970) — tinh chỉnh LLaMA + LoRA cho DST.
- [Heck et al. (2020). TripPy — A Triple Copy Strategy for Value Independent Neural Dialog State Tracking](https://arxiv.org/abs/2005.02877) — công cụ DST dựa trên copy.
- [King, Flanigan (2024). Unsupervised End-to-End Task-Oriented Dialogue with LLMs](https://arxiv.org/abs/2404.10753) — TOD không giám sát dựa trên EM.
- [MultiWOZ leaderboard](https://github.com/budzianowski/multiwoz) — kết quả DST chuẩn.