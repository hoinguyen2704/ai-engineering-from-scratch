# Capstone 06 — Tác nhân khắc phục sự cố DevOps cho Kubernetes

> DevOps Agent của AWS đã chính thức ra mắt (GA), Resolve AI công bố các playbook K8s của họ, NeuBird trình diễn khả năng giám sát ngữ nghĩa, và Metoro liên kết AI SRE với các SLO theo từng dịch vụ. Hình thái vận hành trong môi trường production đã được định hình: một webhook cảnh báo được kích hoạt, một tác nhân (agent) đọc dữ liệu đo lường (telemetry), duyệt qua đồ thị các đối tượng K8s, xếp hạng các giả thuyết về nguyên nhân gốc rễ, và đăng một bản tóm tắt lên Slack kèm các nút phê duyệt. Mặc định là chỉ đọc (read-only). Mọi hành động khắc phục đều phải được con người phê duyệt. Capstone này chính là việc xây dựng tác nhân đó, được đánh giá trên 20 sự cố giả lập và so sánh với Agent của AWS trên ba trường hợp thực tế.

**Type:** Capstone
**Languages:** Python (agent), TypeScript (Slack integration)
**Prerequisites:** Phase 11 (LLM engineering), Phase 13 (tools and MCP), Phase 14 (agents), Phase 15 (autonomous), Phase 17 (infrastructure), Phase 18 (safety)
**Phases exercised:** P11 · P13 · P14 · P15 · P17 · P18
**Time:** 30 giờ

## Vấn đề

Câu chuyện về SRE giai đoạn 2025-2026 đã trở thành: "Các tác nhân AI phân loại sự cố, con người phê duyệt các hành động khắc phục." AWS DevOps Agent, Resolve AI, NeuBird, Metoro, PagerDuty AIOps đều triển khai mô hình này trong môi trường production. Tác nhân đọc các chỉ số Prometheus, nhật ký Loki, dấu vết Tempo, kube-state-metrics và một đồ thị tri thức (knowledge graph) về các đối tượng K8s. Nó đưa ra giả thuyết về nguyên nhân gốc rễ được xếp hạng kèm theo các trích dẫn dữ liệu đo lường trong vòng chưa đầy năm phút. Nó không bao giờ thực thi các lệnh phá hủy nếu không có sự phê duyệt rõ ràng của con người thông qua Slack.

Phần lớn công việc khó khăn nằm ở phạm vi (scoping) và tính an toàn, không phải ở khả năng suy luận. Tác nhân cần một bề mặt RBAC mặc định là chỉ đọc, một máy chủ công cụ MCP được bảo mật, và nhật ký kiểm toán (audit log) của mọi lệnh được cân nhắc so với lệnh đã thực thi. Nó cần biết khi nào vượt quá khả năng của mình để leo thang (escalate). Và nó phải vận hành đủ tiết kiệm để các chuỗi OOM-kill không tạo ra hóa đơn 5.000 đô la cho tác nhân.

## Khái niệm

Tác nhân hoạt động dựa trên một đồ thị tri thức. Các nút là các đối tượng K8s (Pods, Deployments, Services, Nodes, HPAs, PVCs) cộng với các nguồn dữ liệu đo lường (chuỗi Prometheus, luồng Loki, dấu vết Tempo). Các cạnh mã hóa quyền sở hữu (Pod -> ReplicaSet -> Deployment), lập lịch (Pod -> Node), và quan sát (Pod -> chuỗi Prometheus). Đồ thị được cập nhật thông qua đồng bộ hóa kube-state-metrics và được lấy mẫu lại sau mỗi cảnh báo.

Khi một cảnh báo được kích hoạt, tác nhân tìm nguyên nhân gốc rễ từ đối tượng bị ảnh hưởng. Nó duyệt qua các cạnh, lấy các lát cắt dữ liệu đo lường liên quan (15 phút gần nhất), và soạn thảo một giả thuyết. Giả thuyết được xếp hạng theo bằng chứng: bao nhiêu trích dẫn dữ liệu đo lường hỗ trợ nó, độ gần đây, độ cụ thể. Top 3 giả thuyết sẽ được gửi đến Slack kèm theo hình ảnh trực quan hóa đường dẫn đồ thị và các nút phê duyệt cho các hành động khắc phục.

Hành động khắc phục được kiểm soát. Các hành động mặc định được phép là chỉ đọc. Các hành động phá hủy (giảm quy mô, khôi phục phiên bản cũ, xóa Pods) yêu cầu phê duyệt qua Slack; các hook khôi phục ArgoCD yêu cầu một mã xác thực mà tác nhân không bao giờ nắm giữ. Nhật ký kiểm toán ghi lại mọi lệnh mà tác nhân *đã cân nhắc* — không chỉ là lệnh đã thực thi — để quá trình đánh giá có thể phát hiện các tình huống suýt xảy ra sự cố (near-misses).

## Kiến trúc

```
PagerDuty / Alertmanager webhook
           |
           v
     FastAPI receiver
           |
           v
   LangGraph root-cause agent
           |
           +---- read-only MCP tools ----+
           |                             |
           v                             v
   K8s knowledge graph              telemetry slices
     (Neo4j / kuzu)              Prometheus, Loki, Tempo
   ownership + scheduling          last 15m, scoped
           |
           v
   hypothesis ranking (evidence weight)
           |
           v
   Slack brief + approval buttons
           |
           v (approved)
   ArgoCD rollback hook / PagerDuty escalate
           |
           v
   audit log: considered vs executed, every command
```

## Stack

- Nguồn dữ liệu quan sát: Prometheus, Loki, Tempo, kube-state-metrics
- Đồ thị tri thức: Neo4j (managed) hoặc kuzu (embedded) chứa các đối tượng K8s + các cạnh dữ liệu đo lường
- Tác nhân: LangGraph với danh sách cho phép (allow-list) theo từng công cụ, mặc định là chỉ đọc
- Truyền tải công cụ: FastMCP qua StreamableHTTP; máy chủ riêng cho các công cụ phá hủy nằm sau cổng phê duyệt
- Các mô hình: Claude Sonnet 4.7 cho suy luận nguyên nhân gốc rễ, Gemini 2.5 Flash cho tóm tắt nhật ký
- Khắc phục: Webhook khôi phục ArgoCD, leo thang PagerDuty, thẻ phê duyệt Slack
- Kiểm toán: Nhật ký cấu trúc chỉ ghi thêm (đã cân nhắc, đã thực thi, đã phê duyệt, kết quả)
- Triển khai: Triển khai K8s với vai trò RBAC hạn chế riêng; namespace riêng biệt

```figure
ce-rootcause-walk
```

## Xây dựng

1. **Nạp dữ liệu đồ thị.** Đồng bộ hóa kube-state-metrics vào Neo4j/kuzu mỗi 30 giây. Các nút: Pod, Deployment, Node, Service, PVC, HPA. Các cạnh: OWNED_BY, SCHEDULED_ON, EXPOSES, MOUNTS, SCALES. Các cạnh phủ dữ liệu đo lường: OBSERVED_BY (một Pod được quan sát bởi một chuỗi Prometheus).

2. **Bộ nhận cảnh báo.** Endpoint FastAPI chấp nhận webhook từ PagerDuty hoặc Alertmanager. Trích xuất (các) đối tượng bị ảnh hưởng và vi phạm SLO.

3. **Bề mặt công cụ chỉ đọc.** Bao bọc kubectl, truy vấn Prometheus, logql của Loki, traceql của Tempo thông qua FastMCP. Mỗi công cụ có một động từ RBAC hạn chế ("get", "list", "describe"). Không có "delete", "exec", "scale" trong máy chủ mặc định.

4. **Tác nhân nguyên nhân gốc rễ.** LangGraph với ba nút: `sample` lấy lát cắt dữ liệu đo lường 15 phút gần nhất, `walk` truy vấn đồ thị cho các đối tượng lân cận, `hypothesize` soạn thảo các ứng viên nguyên nhân gốc rễ được xếp hạng kèm trích dẫn dữ liệu đo lường.

5. **Chấm điểm bằng chứng.** Mỗi giả thuyết có điểm số = độ gần đây * độ cụ thể * nghịch đảo độ dài đường dẫn đồ thị * số lượng trích dẫn. Trả về top 3.

6. **Bản tóm tắt Slack.** Đăng một tệp đính kèm với giả thuyết, hình ảnh trực quan hóa đường dẫn đồ thị (một hình ảnh đồ thị con được render phía máy chủ), và các nút phê duyệt cho tối đa một hành động khắc phục.

7. **Cổng khắc phục.** Các công cụ phá hủy (giảm quy mô, khôi phục, xóa) nằm trên máy chủ MCP thứ hai sau một mã phê duyệt. Tác nhân chỉ có thể gọi chúng sau khi thẻ Slack được con người phê duyệt.

8. **Nhật ký kiểm toán.** JSONL chỉ ghi thêm: đối với mỗi lệnh ứng viên, ghi lại xem nó đã được cân nhắc chưa, đã thực thi chưa, ai đã phê duyệt. Gửi đến S3 hàng ngày.

9. **Bộ sự cố giả lập.** Xây dựng 20 kịch bản: Chuỗi OOMKill, lỗi DNS, HPA dao động, đầy PVC, hàng xóm ồn ào (noisy neighbor), sidecar lỗi, triển khai ConfigMap sai, xoay vòng chứng chỉ, image-pull backoff, v.v. Chấm điểm tác nhân dựa trên độ chính xác của nguyên nhân gốc rễ và thời gian đưa ra giả thuyết.

## Sử dụng

```
webhook: alert.pagerduty.com -> checkout-api SLO breach, error rate 14%
[graph]   affected: Deployment checkout-api (3 Pods, Node ip-10-2-3-4)
[walk]    neighbors: ReplicaSet checkout-api-abc, Service checkout-api,
           recent rollout 14m ago
[sample]  prometheus error_rate 14%, up-trend; loki 500s on /api/v2/pay
[hypo]    #1 bad rollout: latest image checkout-api:v2.41 fails /healthz
          citations: deploy.yaml (rev 42), prometheus errorRate, loki 500 stack
[slack]   [ROLL BACK to v2.40]  [ESCALATE]  [IGNORE]
          (approval required; agent does not roll back unilaterally)
```

## Xuất xưởng

`outputs/skill-devops-agent.md` là sản phẩm bàn giao. Với một cụm K8s và nguồn cảnh báo, tác nhân tạo ra các giả thuyết nguyên nhân gốc rễ được xếp hạng và quy trình khắc phục được kiểm soát qua Slack.

| Trọng số | Tiêu chí | Cách đo lường |
|:-:|---|---|
| 25 | Độ chính xác RCA trên bộ kịch bản | ≥80% nguyên nhân gốc rễ chính xác trên 20 sự cố giả lập |
| 20 | An toàn | Cổng bảo vệ hành động phá hủy không bao giờ kích hoạt mà không có phê duyệt Slack trong nhật ký kiểm toán |
| 20 | Thời gian đưa ra giả thuyết | p50 dưới 5 phút từ khi cảnh báo đến khi có bản tóm tắt Slack |
| 20 | Khả năng giải thích | Mỗi giả thuyết đều có đường dẫn đồ thị và trích dẫn dữ liệu đo lường |
| 15 | Độ hoàn thiện tích hợp | PagerDuty, Slack, ArgoCD, Prometheus hoạt động end-to-end |
| **100** | | |

## Bài tập

1. Chạy tác nhân của bạn trên cùng ba sự cố mà AWS DevOps Agent đã trình diễn. Công bố kết quả so sánh song song. Báo cáo nơi tác nhân của bạn khác biệt.

2. Thêm một nhật ký kiểm toán "suýt xảy ra sự cố" (near-miss) để gắn cờ bất kỳ lệnh nào mà tác nhân *đã cân nhắc* nhưng sẽ gây phá hủy nếu không có phê duyệt. Đo lường tỷ lệ này trong một tuần.

3. Thay thế mô hình giả thuyết từ Claude Sonnet 4.7 bằng Llama 3.3 70B tự lưu trữ. Đo lường sự thay đổi về độ chính xác RCA và chi phí trên mỗi sự cố.

4. Xây dựng bộ lọc nhân quả: phân biệt các đỉnh dữ liệu đo lường tương quan với nguyên nhân gốc rễ thực sự. Huấn luyện một bộ phân loại nhỏ trên nhãn của 20 kịch bản.

5. Thêm tính năng chạy thử khôi phục (rollback dry-run): Khôi phục ArgoCD trên cụm staging với cùng manifest. Xác minh kế hoạch khôi phục trong cụm thực tế trước khi nút phê duyệt Slack xuất hiện.

## Thuật ngữ chính

| Thuật ngữ | Cách gọi thông thường | Ý nghĩa thực tế |
|------|-----------------|------------------------|
| Đồ thị tri thức K8s | "Cluster graph" | Các nút = đối tượng K8s + chuỗi dữ liệu đo lường; các cạnh = sở hữu, lập lịch, quan sát |
| Mặc định chỉ đọc | "Scoped RBAC" | Tài khoản dịch vụ của tác nhân chỉ có quyền get/list/describe; các quyền phá hủy nằm ở máy chủ riêng sau cổng phê duyệt |
| Nhật ký kiểm toán | "Considered vs executed" | Bản ghi chỉ ghi thêm mọi lệnh ứng viên, đã chạy hay chưa, ai phê duyệt |
| Xếp hạng giả thuyết | "Evidence score" | Độ gần đây × độ cụ thể × nghịch đảo độ dài đường dẫn đồ thị × số lượng trích dẫn |
| Thẻ phê duyệt Slack | "HITL gate" | Tin nhắn Slack tương tác với các nút khắc phục; tác nhân không thể tiếp tục cho đến khi con người nhấn nút |
| Trích dẫn dữ liệu đo lường | "Evidence pointer" | Một truy vấn Prometheus, bộ chọn Loki, hoặc URL dấu vết Tempo hỗ trợ cho một khẳng định |
| MTTR | "Time to resolution" | Thời gian thực từ khi cảnh báo kích hoạt đến khi SLO phục hồi |

## Đọc thêm

- [AWS DevOps Agent GA](https://aws.amazon.com/blogs/aws/aws-devops-agent-helps-you-accelerate-incident-response-and-improve-system-reliability-preview/) — tài liệu tham khảo chính năm 2026
- [Resolve AI K8s troubleshooting](https://resolve.ai/blog/kubernetes-troubleshooting-in-resolve-ai) — tài liệu tham khảo từ đối thủ cạnh tranh
- [NeuBird semantic monitoring](https://www.neubird.ai) — phương pháp tiếp cận đồ thị ngữ nghĩa
- [Metoro AI SRE](https://metoro.io) — khung vận hành production ưu tiên SLO
- [kube-state-metrics](https://github.com/kubernetes/kube-state-metrics) — nguồn trạng thái cụm
- [LangGraph](https://langchain-ai.github.io/langgraph/) — trình điều phối tác nhân tham chiếu
- [FastMCP](https://github.com/jlowin/fastmcp) — khung máy chủ MCP Python
- [ArgoCD rollback](https://argo-cd.readthedocs.io/en/stable/user-guide/commands/argocd_app_rollback/) — mục tiêu khắc phục được kiểm soát