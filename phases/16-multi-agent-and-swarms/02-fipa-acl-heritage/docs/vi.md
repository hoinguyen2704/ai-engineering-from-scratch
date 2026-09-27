# Di sản của FIPA-ACL và Speech Acts

> Trước MCP, trước A2A, đã từng có FIPA-ACL. Vào năm 2000, IEEE Foundation for Intelligent Physical Agents đã phê chuẩn một ngôn ngữ giao tiếp tác tử (agent communication language) với hai mươi performative, hai ngôn ngữ nội dung và một tập hợp các giao thức tương tác — contract net, subscribe/notify, request-when. Nó dần mờ nhạt trong ngành vì chi phí ontology quá nặng nề đối với web, nhưng sự hồi sinh của các hệ thống đa tác tử (multi-agent systems) nhờ LLM đang âm thầm triển khai lại chính những ý tưởng đó mà không cần ngữ nghĩa hình thức: các hợp đồng JSON thay thế cho performative, ngôn ngữ tự nhiên thay thế cho ontology. Bài học này nghiên cứu FIPA-ACL một cách nghiêm túc để bạn có thể thấy những quyết định về giao thức năm 2026 nào là sự tái phát minh, cái nào là sự đổi mới, và làn sóng hiện tại sẽ đi đến đâu để khám phá lại những vấn đề mà thập niên 2000 đã giải quyết xong.

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 01 (Why Multi-Agent)
**Time:** ~60 phút

## Vấn đề

Bối cảnh giao thức tác tử năm 2026 rất bận rộn: MCP cho công cụ, A2A cho tác tử, ACP cho kiểm toán doanh nghiệp, ANP cho niềm tin phi tập trung, NLIP cho nội dung ngôn ngữ tự nhiên, cộng với CA-MCP và hàng tá đề xuất nghiên cứu khác. Mỗi đặc tả đều tự xưng là nền tảng.

Cách nhìn nhận trung thực là hầu hết chúng đang khám phá lại một cây quyết định cụ thể có tuổi đời hai mươi năm. Lý thuyết hành động ngôn ngữ (speech-act theory) từ Austin (1962) và Searle (1969) đã cho chúng ta biết "phát ngôn là hành động". KQML (1993) đã biến điều đó thành một giao thức truyền tin. FIPA-ACL (phê chuẩn năm 2000) đã tạo ra tiêu chuẩn tham chiếu: hai mươi performative, ngôn ngữ nội dung SL0/SL1, các giao thức tương tác cho contract-net và subscribe-notify. JADE và JACK là các nền tảng tham chiếu Java. Nỗ lực này mờ nhạt dần vào khoảng năm 2010 vì chi phí ontology quá nặng nề và web đã chiến thắng.

Khi bạn nhìn vào `tools/call` của MCP, vòng đời tác vụ của A2A, hoặc kho lưu trữ ngữ cảnh chia sẻ của CA-MCP, bạn đang nhìn vào một phiên bản JSON-native nhẹ nhàng hơn của các quyết định FIPA. Hiểu về di sản này cho bạn biết hai điều: những "đổi mới" mới nào thực chất là sự tái phát minh, và những kiểu thất bại cũ nào mà các đặc tả mới sẽ khám phá lại.

## Khái niệm

### Speech acts, trong một đoạn văn

Austin nhận thấy rằng một số câu không mô tả thế giới — chúng thay đổi nó. "Tôi hứa." "Tôi yêu cầu." "Tôi tuyên bố." Ông gọi đây là các phát ngôn thực hiện (performative utterances). Searle đã chính thức hóa năm loại: khẳng định (assertive), chỉ dẫn (directive), cam kết (commissive), biểu cảm (expressive), tuyên bố (declarative). KQML (Finin và cộng sự, 1993) đã làm cho điều này trở nên khả thi đối với các tác tử phần mềm: một thông điệp là một performative (hành động) cộng với nội dung (điều mà hành động đó hướng tới). FIPA-ACL đã làm sạch các lỗ hổng của KQML và chuẩn hóa xung quanh hai mươi performative.

### Hai mươi FIPA performative (danh sách một phần)

| Performative | Ý định |
|---|---|
| `inform` | "Tôi nói với bạn P là đúng" |
| `request` | "Tôi yêu cầu bạn làm X" |
| `query-if` | "P có đúng không?" |
| `query-ref` | "Giá trị của X là gì?" |
| `propose` | "Tôi đề xuất chúng ta làm X" |
| `accept-proposal` | "Tôi chấp nhận đề xuất" |
| `reject-proposal` | "Tôi từ chối đề xuất" |
| `agree` | "Tôi đồng ý làm X" |
| `refuse` | "Tôi từ chối làm X" |
| `confirm` | "Tôi xác nhận P là đúng" |
| `disconfirm` | "Tôi phủ nhận P" |
| `not-understood` | "Thông điệp của bạn không thể phân tích" |
| `cfp` | "Kêu gọi đề xuất về X" |
| `subscribe` | "Thông báo cho tôi khi X thay đổi" |
| `cancel` | "Hủy bỏ X đang diễn ra" |
| `failure` | "Tôi đã thử X và thất bại" |

Danh sách đầy đủ nằm trong `fipa00037.pdf` (FIPA ACL Message Structure). Điểm mấu chốt không phải là ghi nhớ nó — mà là mỗi cái trong số này tương ứng với một nguyên thủy mà một giao thức LLM cuối cùng sẽ thêm lại.

### Thông điệp FIPA-ACL chuẩn

```
(inform
  :sender       agent1@platform
  :receiver     agent2@platform
  :content      "((price IBM 83))"
  :language     SL0
  :ontology     finance
  :protocol     fipa-request
  :conversation-id   conv-42
  :reply-with   msg-17
)
```

Bảy trường mang phong bì giao thức; một trường (`content`) mang tải trọng (payload). Các trường còn lại chính xác là những gì bạn tái phát minh mỗi khi bạn thêm các cơ chế thử lại (retries), phân luồng (threading) và ontology vào một giao thức JSON.

### Hai nền tảng di sản

**JADE** (Java Agent DEvelopment framework, 1999–2020s) là runtime tuân thủ FIPA được sử dụng nhiều nhất. Các tác tử mở rộng một lớp cơ sở, trao đổi thông điệp ACL, chạy bên trong các container và phối hợp bằng cách sử dụng "hành vi" (behaviors). Thư viện giao thức tương tác đi kèm với contract-net, subscribe-notify, request-when và propose-accept.

**JACK** (Agent Oriented Software, thương mại) nhấn mạnh lý luận BDI (Belief-Desire-Intention) trên nền tảng các thông điệp FIPA. Trang trọng hơn, nhưng ít được áp dụng hơn.

Cả hai đều suy giảm khi ngăn xếp web chiếm lĩnh các trường hợp sử dụng đa tác tử. MCP và A2A là các "container" runtime của năm 2026.

### Tại sao FIPA mờ nhạt

- **Chi phí Ontology.** FIPA yêu cầu một ontology chia sẻ để phân tích `content`. Việc đồng ý về ontology là một quá trình tiêu chuẩn hóa kéo dài nhiều năm. Web chỉ sử dụng HTTP + JSON.
- **Ngữ nghĩa hình thức không ai dùng.** SL (Semantic Language) đưa ra các điều kiện chân lý nghiêm ngặt, nhưng hầu hết các hệ thống sản xuất đều sử dụng nội dung tự do và bỏ qua tính hình thức đó.
- **Sự khóa chặt vào công cụ.** JADE chỉ dành cho Java; JACK là thương mại. Các nhóm đa ngôn ngữ đã tìm cách tránh cả hai.
- **Internet đã chiến thắng ngăn xếp.** REST, sau đó là JSON-RPC, rồi gRPC đã thay thế phương thức truyền tải của ACL.

### Sự hồi sinh của LLM là FIPA-lite

So sánh một `request` của FIPA với một `tools/call` của MCP:

```
(request                                {
  :sender  agent1                         "jsonrpc": "2.0",
  :receiver tool-server                   "method":  "tools/call",
  :content "(lookup stock IBM)"           "params":  {"name":"lookup_stock",
  :ontology finance                                   "arguments":{"symbol":"IBM"}},
  :conversation-id c42                    "id": 42
)                                        }
```

Cùng một phong bì, cú pháp khác nhau. Cả hai đều mang: ai, cho ai, ý định, tải trọng, correlation id. Không cái nào là một cuộc cách mạng so với cái kia — chúng là những sự đánh đổi khác nhau trên cùng một thiết kế.

Khảo sát năm 2025 của Liu và cộng sự ("A Survey of Agent Interoperability Protocols: MCP, ACP, A2A, ANP", arXiv:2505.02279) làm rõ dòng dõi này: MCP tương ứng với các speech act sử dụng công cụ, A2A với các speech act giữa các tác tử ngang hàng, ACP với các speech act kiểm toán, ANP với các phần mở rộng định danh phi tập trung. Các đặc tả mới là hậu duệ của ACL với cú pháp JSON và ngữ nghĩa lỏng lẻo hơn.

### Sự đánh đổi, được nêu rõ ràng

**Những gì FIPA mang lại và các đặc tả hiện đại loại bỏ:**

- Ngữ nghĩa hình thức — bạn có thể chứng minh `inform` ngụ ý người gửi tin tưởng vào nội dung đó.
- Danh mục chuẩn các performative — bạn không cần phải tranh luận lại "chúng ta có nên có một `cancel` không?".
- Hàng thập kỷ các mẫu giao thức tương tác — contract-net, subscribe-notify, propose-accept — với các thuộc tính đúng đắn đã biết.

**Những gì các đặc tả hiện đại mang lại và FIPA không có:**

- Tải trọng JSON-native tương thích với mọi công cụ hiện đại.
- Nội dung ngôn ngữ tự nhiên mà LLM có thể diễn giải mà không cần ontology viết tay.
- Truyền tải ngăn xếp web (HTTP, SSE, WebSocket).
- Khám phá khả năng thông qua MCP `server/discover` trực tiếp và Thẻ tác tử A2A.

Ngữ nghĩa ý định lỏng lẻo hơn để triển khai dễ dàng hơn. Đó chính xác là sự đánh đổi.

### Các giao thức tương tác đáng để chuyển đổi

FIPA đã cung cấp ~15 giao thức tương tác. Ba trong số đó đáng để đưa vào các hệ thống đa tác tử LLM:

1. **Contract Net Protocol (CNP).** Người quản lý đưa ra `cfp` (kêu gọi đề xuất); các bên đấu thầu phản hồi bằng `propose`; người quản lý chấp nhận/từ chối. Đây là mô hình thị trường tác vụ chuẩn (Phase 16 · 16 Negotiation).
2. **Subscribe/Notify.** Người đăng ký gửi `subscribe`; nhà xuất bản gửi `inform` bất cứ khi nào chủ đề thay đổi. Đây là mọi event-bus vào năm 2026.
3. **Request-When.** "Làm X khi điều kiện Y thỏa mãn." Hành động trì hoãn với các điều kiện tiên quyết. Tương tự năm 2026 là các tác vụ trì hoãn trong các công cụ quy trình làm việc bền vững (Phase 16 · 22 Production Scaling).

Mỗi cái đều ánh xạ rõ ràng vào các hàng đợi thông điệp hiện đại, HTTP + polling, hoặc SSE streaming.

### Điều gì xảy ra khi bạn loại bỏ ontology

Nếu không có ontology chia sẻ, các tác tử suy luận ý nghĩa từ nội dung ngôn ngữ tự nhiên. Kiểu thất bại được ghi nhận vào năm 2026 là **trôi dạt ngữ nghĩa (semantic drift)**: hai tác tử sử dụng cùng một từ (`"customer"`) cho các khái niệm khác nhau một cách tinh vi, tác tử của người nhận hành động dựa trên cách diễn giải sai, không có trình xác thực lược đồ nào bắt được nó. Yêu cầu về ontology của FIPA sẽ từ chối thông điệp tại thời điểm phân tích cú pháp.

Các biện pháp giảm thiểu mà không cần dùng toàn bộ ontology:

- JSON Schema trên `content` — từ chối các lỗi cấu trúc tại đường truyền.
- Các artifact có kiểu (A2A) — từ chối sai phương thức.
- Performative rõ ràng trong phong bì — làm cho ý định trở nên không mơ hồ ngay cả khi nội dung là ngôn ngữ tự nhiên.

### Các đặc tả năm 2026, ánh xạ tới di sản speech-act

| Đặc tả hiện đại | Tương đương FIPA | Những gì giữ lại | Những gì loại bỏ |
|---|---|---|---|
| MCP `tools/call` | `request` | ý định rõ ràng, correlation id | ngữ nghĩa hình thức, ontology |
| MCP `resources/read` | `query-ref` | ý định rõ ràng, correlation id | ngữ nghĩa hình thức |
| Vòng đời tác vụ A2A | contract-net + request-when | vòng đời bất đồng bộ, chuyển đổi trạng thái | đảm bảo tính đầy đủ hình thức |
| Sự kiện streaming A2A | subscribe/notify | đẩy bất đồng bộ | đăng ký vị ngữ có kiểu |
| Ngữ cảnh chia sẻ CA-MCP | blackboard (Hayes-Roth 1985) | bộ nhớ chia sẻ đa người ghi | mô hình nhất quán logic |
| NLIP | nội dung ngôn ngữ tự nhiên | LLM-native | lược đồ |

Đọc bảng từ trên xuống dưới, mô hình là: giữ nguyên nguyên thủy cấu trúc, loại bỏ tính hình thức, để LLM che đậy sự mơ hồ.

```figure
sw-contract-net
```

## Xây dựng

`code/main.py` triển khai một bộ dịch FIPA-ACL sử dụng thư viện chuẩn. Nó mã hóa và giải mã phong bì ACL chuẩn và cho thấy cách mọi hình dạng thông điệp MCP / A2A đều quy về bảy trường tương tự. Bản demo:

- Mã hóa năm thông điệp kiểu MCP và A2A dưới dạng FIPA-ACL.
- Giải mã FIPA-ACL trở lại tương đương hiện đại.
- Chạy một cuộc đàm phán Contract Net thử nghiệm giữa một người quản lý và ba bên đấu thầu sử dụng `cfp`, `propose`, `accept-proposal`, `reject-proposal`.

Chạy:

```
python3 code/main.py
```

Đầu ra là một dấu vết song song hiển thị từng thông điệp hiện đại ở cả dạng JSON năm 2026 và dạng FIPA-ACL, sau đó là một vòng lặp của một giá thầu contract-net. Các nguyên thủy giao thức giống nhau tồn tại sau vòng lặp; chỉ có cú pháp khác nhau.

## Sử dụng

`outputs/skill-fipa-mapper.md` là một kỹ năng đọc bất kỳ đặc tả giao thức tác tử nào và tạo ra ánh xạ FIPA-ACL. Sử dụng nó trước khi áp dụng một giao thức mới để trả lời: "Đây có thực sự là cái mới, hay nó là `inform` với cú pháp JSON?"

## Triển khai

Đừng mang FIPA-ACL trở lại. Hãy mang lại danh sách kiểm tra của nó:

- Nguyên thủy ý định (performative) của mỗi thông điệp là gì?
- Có correlation id cho yêu cầu-phản hồi và hủy bỏ không?
- Có ngôn ngữ nội dung rõ ràng không (JSON-RPC, văn bản thuần, artifact có kiểu)?
- Các giao thức tương tác có phải là hạng nhất không, hay bạn đang triển khai lại contract-net từ đầu?
- Điều gì xảy ra khi hai tác tử không đồng ý về ý nghĩa nội dung (trôi dạt ngữ nghĩa)?

Hãy ghi lại năm câu hỏi này cho bất kỳ giao thức mới nào trước khi bạn đưa nó vào sản xuất.

## Bài tập

1. Chạy `code/main.py`. Quan sát quá trình mã hóa vòng lặp. Xác định performative FIPA nào tương ứng với `tools/call`, `resources/read` và việc tạo tác vụ A2A.
2. Mở rộng bản demo contract-net với một performative `cancel` cho phép người quản lý rút tác vụ giữa chừng khi đang đấu thầu. Trường hợp thất bại nào mà `cancel` giải quyết được mà chỉ riêng việc thử lại không làm được?
3. Đọc FIPA ACL Message Structure (http://www.fipa.org/specs/fipa00037/) các phần 4.1–4.3. Chọn một performative không được đề cập trong bài học này và mô tả tương đương JSON-RPC hiện đại của nó.
4. Đọc Liu và cộng sự, arXiv:2505.02279. Đối với mỗi MCP, A2A, ACP, ANP, hãy liệt kê các họ performative FIPA mà chúng giữ lại và loại bỏ.
5. Thiết kế một JSON-Schema tối thiểu cho trường `content` của một performative `request` trong hệ thống của riêng bạn. Lược đồ đó mang lại cho bạn điều gì mà ngôn ngữ tự nhiên thuần túy không có, và nó tốn kém những gì?

## Thuật ngữ chính

| Thuật ngữ | Mọi người nói gì | Ý nghĩa thực sự |
|------|----------------|------------------------|
| Speech act | "Một phát ngôn làm được điều gì đó" | Austin/Searle: phát ngôn là hành động. Cha đẻ lý thuyết của ACL. |
| FIPA | "Cái thứ XML cũ kỹ đó" | IEEE Foundation for Intelligent Physical Agents. Chuẩn hóa ACL năm 2000. |
| ACL | "Ngôn ngữ giao tiếp tác tử" | Định dạng phong bì của FIPA: performative + nội dung + siêu dữ liệu. |
| Performative | "Động từ" | Lớp ý định của một thông điệp: `inform`, `request`, `propose`, `cfp`, v.v. |
| KQML | "Tiền thân của FIPA" | Knowledge Query and Manipulation Language (1993). Đơn giản hơn, hẹp hơn. |
| Ontology | "Từ vựng chia sẻ" | Định nghĩa chính thức về các khái niệm mà ngôn ngữ nội dung nói đến. |
| SL0 / SL1 | "Ngôn ngữ nội dung FIPA" | Các cấp độ ngôn ngữ ngữ nghĩa 0 và 1 — họ ngôn ngữ nội dung hình thức. |
| Contract Net | "Thị trường tác vụ" | Người quản lý đưa ra cfp; các bên đấu thầu đề xuất; người quản lý chấp nhận. Giao thức tương tác chuẩn. |
| Giao thức tương tác | "Mẫu thông điệp" | Một chuỗi các performative với độ chính xác đã biết: request-when, subscribe-notify, v.v. |

## Đọc thêm

- [Liu và cộng sự — A Survey of Agent Interoperability Protocols: MCP, ACP, A2A, ANP](https://arxiv.org/html/2505.02279v1) — khảo sát chuẩn năm 2025 kết nối các đặc tả hiện đại với di sản FIPA
- [FIPA ACL Message Structure Specification (fipa00037)](http://www.fipa.org/specs/fipa00037/) — định dạng phong bì được phê chuẩn năm 2000
- [FIPA Communicative Act Library Specification (fipa00037)](http://www.fipa.org/specs/fipa00037/) — danh mục performative đầy đủ
- [Đặc tả MCP 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28) — tương đương sử dụng công cụ không trạng thái hiện tại của `request`/`query-ref`
- [Đặc tả A2A](https://a2a-protocol.org/latest/specification/) — tương đương tác tử ngang hàng hiện đại của contract-net và subscribe-notify